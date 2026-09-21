"""Sanity checks (E-8): does a method produce convincing-looking maps from a
model that has been deliberately broken? If so, it isn't explaining the
model — it's explaining the input (Adebayo et al., "Sanity Checks for
Saliency Maps", NeurIPS 2018).

Two checks:

  - Cascading parameter randomization: progressively reinitialize layers
    from the output back towards the input (head layers first, then the
    backbone's weighted layers in contiguous groups) and recompute the
    heatmap at each stage. A method
    that's actually reading the model should diverge from the original
    heatmap quickly; near-zero change all the way down is the failure mode
    the test is checking for.
  - Label randomization: fit a copy of the (frozen-backbone) head on
    shuffled labels and compare its heatmap to the real model's. This is run
    on a small subset for a few epochs rather than a full retrain — a speed
    trade-off documented here and in docs/decisions_log.md, not hidden.
"""
from __future__ import annotations

import math
import numpy as np
import tensorflow as tf

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - scipy ships with scikit-learn normally
    spearmanr = None

from ..config import ModelConfig
from ..models.baseline import XAIModel, build_model, get_backbone
from ..explain.registry import explain as registry_explain

#: How many contiguous groups the backbone's weighted layers are split into
#: for the cascade. The original list was VGG16's five named blocks plus the
#: head, which stopped meaning anything the moment the backbone became a
#: config value -- DenseNet121 has no 'block5_*' layers. Splitting the
#: backbone's own weighted layers into N groups from the output downwards is
#: the same test (Adebayo et al. 2018) expressed without naming an
#: architecture.
N_BACKBONE_CASCADE_GROUPS = 5


def _rank_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Rank correlation between two heatmaps, or None when it is undefined.

    Randomizing a BatchNorm layer reliably collapses the heatmap to a constant
    (every activation clipped to the same value, so the ReLU emits a flat
    map). A rank correlation against a constant vector is not "zero
    similarity" -- it is undefined, because the denominator is the product of
    two standard deviations and one of them is 0. Both scipy and numpy signal
    that with NaN.

    Returning NaN here would be wrong twice over: it reads as a number in the
    UI, and bare `NaN` is not valid JSON (`json.dump` emits it happily, but
    `JSON.parse`, `jq` and most non-Python readers reject the file). So the
    degenerate case is detected up front and reported as None -> `null`,
    which every JSON reader accepts and which callers can render as
    "undefined" rather than as a suspiciously round 0.000.
    """
    a, b = a.reshape(-1), b.reshape(-1)
    # float32 heatmaps: guard on an exact-equality check of the range rather
    # than a tolerance, since a collapsed map is bit-identical across pixels.
    if a.size == 0 or b.size == 0 or a.max() == a.min() or b.max() == b.min():
        return None
    if spearmanr is not None:
        corr, _ = spearmanr(a, b)
    else:
        corr = np.corrcoef(a, b)[0, 1]
    return None if not np.isfinite(corr) else float(corr)


def _randomize_layer(layer: tf.keras.layers.Layer, seed: int) -> None:
    """Randomize a layer's learned weights in place.

    Kernels are redrawn from N(0, 0.05) and biases zeroed, which is an ordinary
    re-initialisation.

    BatchNormalization needs separate handling, and getting it wrong silently
    destroys the whole test. Its four weights (gamma, beta, moving_mean,
    moving_variance) are all 1-D, so the "1-D means bias, so zero it" rule
    above sets gamma = 0 -- and a BN layer with gamma = 0 outputs beta for
    every channel, i.e. a constant. Every heatmap from that point down the
    cascade is then flat, every rank correlation against it is undefined, and
    the test reports nothing for the majority of its stages.

    Instead: permute the fitted gamma and beta across channels, and leave the
    moving statistics untouched. Permutation destroys the learned
    per-channel correspondence (which is what the test is probing) while
    preserving the distribution of scales exactly, so activations stay in a
    regime where the explanation is still defined. The moving statistics are
    dataset statistics rather than learned parameters, so Adebayo et al.'s
    "randomize the learned weights" is not a licence to touch them.
    """
    weights = layer.get_weights()
    if not weights:
        return
    rng = np.random.RandomState(seed)

    if isinstance(layer, tf.keras.layers.BatchNormalization):
        new_weights = list(weights)
        # Trainable params first (gamma, beta), in Keras' fixed order; a BN
        # with scale=False or center=False simply has fewer of them.
        for i in range(min(2, len(weights))):
            new_weights[i] = rng.permutation(weights[i]).astype(weights[i].dtype)
        layer.set_weights(new_weights)
        return

    layer.set_weights([
        rng.normal(scale=0.05, size=w.shape).astype(w.dtype) if w.ndim > 1
        else np.zeros_like(w) for w in weights
    ])


def _clone_with_weights(model: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone


def _head_layer_names(model: tf.keras.Model, backbone: tf.keras.Model) -> list[str]:
    """Weighted head layers, output-first -- what the cascade randomizes before
    it starts on the backbone."""
    start = model.layers.index(backbone) + 1
    return [l.name for l in reversed(model.layers[start:]) if l.get_weights()]


def _backbone_cascade_groups(backbone: tf.keras.Model,
                             n_groups: int = N_BACKBONE_CASCADE_GROUPS) -> list[list[str]]:
    """Weighted backbone layers split into `n_groups` contiguous groups,
    ordered from the output end downwards."""
    weighted = [l.name for l in backbone.layers if l.get_weights()]
    if not weighted:
        return []
    n_groups = max(1, min(n_groups, len(weighted)))
    size = math.ceil(len(weighted) / n_groups)
    groups = [weighted[i:i + size] for i in range(0, len(weighted), size)]
    return list(reversed(groups))


def cascading_randomization_test(xai_model: XAIModel, image: np.ndarray, method: str,
                                  img_size: tuple[int, int] = (224, 224), seed: int = 0) -> dict:
    """Randomize the model top-down, one group at a time; a faithful
    explanation's similarity to the original should collapse as it goes."""
    clone = _clone_with_weights(xai_model.model)
    clone_xai = XAIModel(clone, xai_model.last_conv_layer_name)
    clone_backbone = get_backbone(clone)

    original_heatmap = registry_explain(method, xai_model, image, img_size=img_size)
    results = [{'stage': 'original', 'similarity_to_original': 1.0}]

    stages: list[tuple[str, list[str], tf.keras.Model]] = [
        (name, [name], clone) for name in _head_layer_names(clone, clone_backbone)
    ]
    for i, group in enumerate(_backbone_cascade_groups(clone_backbone)):
        stages.append((f'backbone_group_{i + 1}/{N_BACKBONE_CASCADE_GROUPS}', group, clone_backbone))

    for i, (stage_name, layer_names, owner) in enumerate(stages):
        for lname in layer_names:
            _randomize_layer(owner.get_layer(lname), seed + i)

        heatmap = registry_explain(method, clone_xai, image, img_size=img_size)
        results.append({
            'stage': stage_name,
            'similarity_to_original': _rank_similarity(original_heatmap, heatmap),
        })

    return {'method': method, 'stages': results}


def label_randomization_test(xai_model: XAIModel, image: np.ndarray, method: str,
                              train_images: np.ndarray, img_size: tuple[int, int] = (224, 224),
                              epochs: int = 2, seed: int = 0,
                              batch_size: int | None = None) -> dict:
    """Fast proxy for the full label-randomization sanity check: refit only
    the (already-frozen-backbone) head on shuffled labels for a couple of
    epochs over a small subset, then compare heatmaps. See module docstring."""
    rng = np.random.RandomState(seed)
    shuffled_labels = rng.randint(0, 2, size=len(train_images)).astype('float32')

    clone = _clone_with_weights(xai_model.model)
    # Re-initialize just the head (whatever layers that is for this config --
    # attention pooling, batch-norm and the dense stack are all optional) so
    # it isn't starting from weights already fit to real labels.
    for name in _head_layer_names(clone, get_backbone(clone)):
        layer = clone.get_layer(name)
        rng2 = np.random.RandomState(seed + 1)
        layer.set_weights([rng2.normal(scale=0.05, size=w.shape).astype(w.dtype)
                            if w.ndim > 1 else np.zeros_like(w) for w in layer.get_weights()])
    clone.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    # batch_size was hardcoded to 16, which was fine at 224px and OOMs at
    # 320px -- this is a *training* step (the head is refit on shuffled
    # labels), so it costs full backprop memory, not inference memory.
    # Default to the same batch size training itself uses.
    if batch_size is None:
        from ..config import DataConfig
        batch_size = DataConfig.load().batch_size
    clone.fit(train_images, shuffled_labels, epochs=epochs, batch_size=batch_size, verbose=0)

    clone_xai = XAIModel(clone, xai_model.last_conv_layer_name)
    original_heatmap = registry_explain(method, xai_model, image, img_size=img_size)
    shuffled_heatmap = registry_explain(method, clone_xai, image, img_size=img_size)

    return {
        'method': method,
        'similarity_to_original': _rank_similarity(original_heatmap, shuffled_heatmap),
        'n_train_images': len(train_images),
        'epochs': epochs,
        'note': 'proxy: head refit on shuffled labels, backbone frozen and unchanged',
    }
