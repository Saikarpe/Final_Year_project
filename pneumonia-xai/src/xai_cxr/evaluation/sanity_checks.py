"""Sanity checks (E-8): does a method produce convincing-looking maps from a
model that has been deliberately broken? If so, it isn't explaining the
model — it's explaining the input (Adebayo et al., "Sanity Checks for
Saliency Maps", NeurIPS 2018).

Two checks:

  - Cascading parameter randomization: progressively reinitialize layers
    from the output back towards the input (logits -> dense -> conv block 5
    -> ... -> block 1) and recompute the heatmap at each stage. A method
    that's actually reading the model should diverge from the original
    heatmap quickly; near-zero change all the way down is the failure mode
    the test is checking for.
  - Label randomization: fit a copy of the (frozen-backbone) head on
    shuffled labels and compare its heatmap to the real model's. This is run
    on a small subset for a few epochs rather than a full retrain — a speed
    trade-off documented here and in docs/decisions_log.md, not hidden.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - scipy ships with scikit-learn normally
    spearmanr = None

from ..config import ModelConfig
from ..models.baseline import XAIModel, build_model
from ..explain.registry import explain as registry_explain

CASCADE_STAGES = ['logits', 'dense', 'block5', 'block4', 'block3', 'block2', 'block1']


def _rank_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    if spearmanr is not None:
        corr, _ = spearmanr(a, b)
        return float(corr)
    return float(np.corrcoef(a, b)[0, 1])


def _randomize_layer(layer: tf.keras.layers.Layer, seed: int) -> None:
    weights = layer.get_weights()
    if not weights:
        return
    rng = np.random.RandomState(seed)
    new_weights = [rng.normal(scale=0.05, size=w.shape).astype(w.dtype) if w.ndim > 1
                    else np.zeros_like(w) for w in weights]
    layer.set_weights(new_weights)


def _clone_with_weights(model: tf.keras.Model) -> tf.keras.Model:
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    return clone


def cascading_randomization_test(xai_model: XAIModel, image: np.ndarray, method: str,
                                  img_size: tuple[int, int] = (224, 224), seed: int = 0) -> dict:
    clone = _clone_with_weights(xai_model.model)
    clone_xai = XAIModel(clone, xai_model.last_conv_layer_name)

    original_heatmap = registry_explain(method, xai_model, image, img_size=img_size)
    results = [{'stage': 'original', 'similarity_to_original': 1.0}]

    vgg_layer_names_by_block = {
        f'block{b}': [l.name for l in clone_xai.vgg.layers if l.name.startswith(f'block{b}_')]
        for b in range(1, 6)
    }

    for i, stage in enumerate(CASCADE_STAGES):
        if stage == 'logits':
            _randomize_layer(clone.get_layer('logits'), seed + i)
        elif stage == 'dense':
            _randomize_layer(clone.get_layer('dense'), seed + i)
        else:
            for lname in vgg_layer_names_by_block.get(stage, []):
                _randomize_layer(clone_xai.vgg.get_layer(lname), seed + i)

        heatmap = registry_explain(method, clone_xai, image, img_size=img_size)
        results.append({
            'stage': stage,
            'similarity_to_original': _rank_similarity(original_heatmap, heatmap),
        })

    return {'method': method, 'stages': results}


def label_randomization_test(xai_model: XAIModel, image: np.ndarray, method: str,
                              train_images: np.ndarray, img_size: tuple[int, int] = (224, 224),
                              epochs: int = 2, seed: int = 0) -> dict:
    """Fast proxy for the full label-randomization sanity check: refit only
    the (already-frozen-backbone) head on shuffled labels for a couple of
    epochs over a small subset, then compare heatmaps. See module docstring."""
    rng = np.random.RandomState(seed)
    shuffled_labels = rng.randint(0, 2, size=len(train_images)).astype('float32')

    clone = _clone_with_weights(xai_model.model)
    # Re-initialize just the head so it isn't starting from weights already
    # fit to real labels.
    for name in ('dense', 'logits'):
        layer = clone.get_layer(name)
        rng2 = np.random.RandomState(seed + 1)
        layer.set_weights([rng2.normal(scale=0.05, size=w.shape).astype(w.dtype)
                            if w.ndim > 1 else np.zeros_like(w) for w in layer.get_weights()])
    clone.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    clone.fit(train_images, shuffled_labels, epochs=epochs, batch_size=16, verbose=0)

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
