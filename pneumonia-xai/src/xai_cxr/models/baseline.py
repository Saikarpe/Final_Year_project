"""The M-1 classifier: a configurable modern backbone + a linear-probe head.

History of the fixes this file carries (kept, because the model card and the
dashboard both narrate them):

  - the family's own `preprocess_input` instead of a manual /255 rescale
    (ImageNet weights are not trained on [0, 1] RGB);
  - GlobalAveragePooling2D instead of Flatten (fewer parameters, and it keeps
    spatial correspondence sane for the explanation methods);
  - a separate pre-activation `logits` output alongside the sigmoid
    probability, so every explanation method (Grad-CAM in particular) can
    backprop from the true logit rather than a saturated post-sigmoid value;
  - class_weight for the 3:1 imbalance (see xai_cxr.data.class_weights);
  - val_auc as the checkpoint/early-stopping monitor, not val_accuracy;
  - a tuned decision threshold (Youden's J on val) instead of a hardcoded 0.5.

What changed in M-1b:

  - The backbone is a registry lookup (xai_cxr.models.backbones), defaulting
    to DenseNet121 rather than VGG16. Nothing downstream names a backbone.
  - Preprocessing moved from a bare `Lambda` into a registered, serializable
    `Preprocess` layer, so an .h5 round-trip no longer needs the caller to
    pass `custom_objects` and guess which family's function was baked in.
  - `XAIModel` replays whatever head layers the model actually has instead of
    hardcoding gap/dense/dropout/logits, so a head change is not an
    explanation-method change.
  - AdamW + warmup-cosine and optional focal loss / label smoothing, all
    config-driven, to support the two-stage fine-tune in scripts/train.py
    (the old setup froze the backbone forever and never adapted it to
    radiographs).
"""
from __future__ import annotations

import json
import os
import numpy as np
import tensorflow as tf

from ..config import ModelConfig, metrics_path
from .backbones import BACKBONES, build_backbone, get_spec, resolve_cam_layer, set_finetune_trainable
from .heads import AttentionPool2D, LearnableWindow, build_pooler, find_attention_layer

K = tf.keras


# ── Preprocessing as a real, serializable layer ────────────────────────────
@K.utils.register_keras_serializable(package='xai_cxr')
class Preprocess(K.layers.Layer):
    """Applies the backbone family's own `preprocess_input` to raw RGB [0, 255].

    Stored by *name* rather than by function reference, which is the whole
    point: `load_model('pneumonia_model.h5')` reconstructs the correct
    preprocessing for whichever backbone that file was trained with, with no
    custom_objects argument and no way for the caller to pick the wrong one.
    """

    def __init__(self, backbone: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self._fn = get_spec(backbone).preprocess

    def call(self, inputs):
        return self._fn(tf.cast(inputs, tf.float32))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['backbone'] = self.backbone
        return config


# ── Learning-rate schedule ─────────────────────────────────────────────────
@K.utils.register_keras_serializable(package='xai_cxr')
class WarmupCosine(K.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay to `final_lr`.

    Written out rather than using `keras.optimizers.schedules.CosineDecay`'s
    warmup arguments because those only exist in some TF 2.x point releases
    and this repo pins 2.13 for everything else.
    """

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int = 0,
                 final_lr: float = 0.0, name: str = 'warmup_cosine'):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.final_lr = float(final_lr)
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warm_lr = self.base_lr * (step + 1.0) / tf.maximum(warmup, 1.0)

        progress = tf.clip_by_value((step - warmup) / tf.maximum(total - warmup, 1.0), 0.0, 1.0)
        cosine = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
            1.0 + tf.cos(np.pi * progress))

        return tf.where(step < warmup, warm_lr, cosine)

    def get_config(self):
        return {'base_lr': self.base_lr, 'total_steps': self.total_steps,
                'warmup_steps': self.warmup_steps, 'final_lr': self.final_lr, 'name': self.name}


# ── Model construction ─────────────────────────────────────────────────────
def build_loss(cfg: ModelConfig):
    """BCE with label smoothing, or focal BCE when the config asks for it.

    Focal loss is offered because the failure gallery is dominated by a small
    set of hard NORMAL cases rather than by the 3:1 class ratio, which
    class_weight already handles; `focal_gamma: 0` keeps plain BCE.
    """
    if cfg.focal_gamma and cfg.focal_gamma > 0:
        focal = getattr(K.losses, 'BinaryFocalCrossentropy', None)
        if focal is None:
            raise RuntimeError(
                f'focal_gamma={cfg.focal_gamma} needs tf.keras.losses.'
                f'BinaryFocalCrossentropy, which TensorFlow {tf.__version__} does not '
                'provide (added in 2.11). Set focal_gamma: 0 in configs/model.yaml, '
                'or train on TensorFlow >= 2.11.')
        return focal(gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing,
                     from_logits=False)
    return K.losses.BinaryCrossentropy(label_smoothing=cfg.label_smoothing)


def build_metrics() -> list:
    # 'auc' keeps its name because scripts/train.py monitors 'val_auc' and
    # docs/model_card.md quotes it; AUPRC is added because with a 3:1 ratio it
    # moves when AUROC has already saturated.
    return [
        'accuracy',
        K.metrics.AUC(name='auc'),
        K.metrics.AUC(name='auprc', curve='PR'),
        K.metrics.Precision(name='precision'),
        K.metrics.Recall(name='recall'),
    ]


def build_optimizer(cfg: ModelConfig, learning_rate) -> K.optimizers.Optimizer:
    """AdamW, not Adam. Adam's L2-as-gradient-penalty interacts badly with
    adaptive step sizes; decoupled weight decay is what every modern recipe
    for these backbones assumes. `weight_decay: 0` gives plain Adam behaviour.

    `tf.keras.optimizers.AdamW` only exists from TensorFlow 2.11. TF 2.10 is
    the last release with native-Windows GPU support, so anyone training on a
    Windows GPU is pinned below that line and would otherwise be unable to run
    this repo at all. Fall back to tensorflow_addons' AdamW when it is
    installed (same decoupled-decay algorithm), and to plain Adam when it is
    not -- loudly, because silently dropping weight decay changes the recipe.
    """
    if hasattr(K.optimizers, 'AdamW'):
        return K.optimizers.AdamW(learning_rate=learning_rate, weight_decay=cfg.weight_decay)

    if not cfg.weight_decay:
        return K.optimizers.Adam(learning_rate=learning_rate)

    try:
        import tensorflow_addons as tfa
        return tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=cfg.weight_decay)
    except ImportError:
        import warnings
        warnings.warn(
            f'TensorFlow {tf.__version__} has no keras AdamW and tensorflow_addons is not '
            f'installed; falling back to Adam and DROPPING weight_decay='
            f'{cfg.weight_decay}. Results will not match a run on TF >= 2.11. '
            'Install tensorflow-addons, or set weight_decay: 0 to make this explicit.',
            RuntimeWarning, stacklevel=2)
        return K.optimizers.Adam(learning_rate=learning_rate)


def build_model(cfg: ModelConfig | None = None, input_shape: tuple[int, int, int] | None = None,
                weights: str | None = 'imagenet', compile_model: bool = True,
                learning_rate=None) -> K.Model:
    """Learnable window -> preprocess -> backbone (frozen) -> pooling head.

    Stage 2 of training unfreezes the top of the backbone via `finetune()`;
    nothing else about the graph changes, so a stage-1 and a stage-2
    checkpoint are interchangeable everywhere else in the codebase."""
    cfg = cfg or ModelConfig.load()
    input_shape = input_shape or (*cfg.input_size, 3)
    spec = get_spec(cfg.backbone)

    backbone = build_backbone(spec.name, input_shape, weights=weights)
    backbone.trainable = False

    inputs = K.Input(shape=input_shape, name='input_image')
    x = inputs
    if cfg.window_layer:
        x = LearnableWindow(name='window')(x)
    x = Preprocess(spec.name, name='preprocess')(x)
    x = backbone(x, training=False)
    if cfg.spatial_dropout:
        # Channel dropout, not pixel dropout -- see heads.py. Applied before
        # pooling so it actually removes a feature rather than a few of its
        # spatial samples.
        x = K.layers.SpatialDropout2D(cfg.spatial_dropout, name='spatial_dropout')(x)
    x = build_pooler(cfg)(x)
    if cfg.head_batchnorm:
        # On the *pooled* vector, where it is cheap and stabilises stage 2:
        # unfreezing the backbone shifts the feature scale, and without this
        # the head spends its first fine-tuning epochs re-learning that scale.
        x = K.layers.BatchNormalization(name='head_bn')(x)
    if cfg.head_dropout:
        x = K.layers.Dropout(cfg.head_dropout, name='head_dropout')(x)
    if cfg.dense_units:
        x = K.layers.Dense(cfg.dense_units, activation='gelu', name='dense')(x)
        x = K.layers.Dropout(cfg.dropout, name='dropout')(x)
    logits = K.layers.Dense(1, activation=None, name='logits')(x)
    probs = K.layers.Activation('sigmoid', dtype='float32', name='sigmoid')(logits)

    model = K.Model(inputs, probs, name='pneumonia_classifier')
    if compile_model:
        model.compile(
            optimizer=build_optimizer(cfg, cfg.learning_rate if learning_rate is None else learning_rate),
            loss=build_loss(cfg),
            metrics=build_metrics(),
        )
    return model


def get_backbone(model: K.Model) -> K.Model:
    """The nested backbone submodel, by structure rather than by name.

    `build_backbone` renames it to 'backbone', but a model loaded from an
    older checkpoint may still carry the Keras default ('vgg16',
    'densenet121', ...), so fall back to "the one nested Model in the graph".

    Also re-attaches the registry tag. `_xai_backbone_name` is a plain Python
    attribute and does not survive an .h5 round-trip, which would leave a
    reloaded model unable to resolve `cam_layer: hires` or report its own
    architecture -- and the app only ever sees reloaded models. The `Preprocess`
    layer *does* serialize the backbone name, so recover it from there.
    """
    try:
        backbone = model.get_layer('backbone')
    except ValueError:
        nested = [l for l in model.layers if isinstance(l, K.Model)]
        if len(nested) != 1:
            raise ValueError(
                f'Expected exactly one nested backbone model, found {len(nested)}: '
                f'{[l.name for l in nested]}') from None
        backbone = nested[0]

    if getattr(backbone, '_xai_backbone_name', None) is None:
        pre = next((l for l in model.layers if isinstance(l, Preprocess)), None)
        if pre is not None:
            backbone._xai_backbone_name = pre.backbone
    return backbone


def finetune(model: K.Model, cfg: ModelConfig, learning_rate: float | None = None,
             steps_per_epoch: int | None = None, epochs: int | None = None) -> int:
    """Switch the model into stage-2 fine-tuning and recompile.

    Recompiling is mandatory, not stylistic: Keras caches the trainable-weight
    list at compile time, so flipping `layer.trainable` without a recompile
    silently trains nothing new.

    Returns the number of newly-trainable backbone layers.
    """
    backbone = get_backbone(model)
    n_trainable = set_finetune_trainable(backbone, cfg.unfreeze_fraction, cfg.freeze_batchnorm)

    lr = cfg.finetune_learning_rate if learning_rate is None else learning_rate
    if steps_per_epoch and epochs:
        lr = WarmupCosine(base_lr=lr, total_steps=steps_per_epoch * epochs,
                          warmup_steps=int(steps_per_epoch * cfg.warmup_epochs),
                          final_lr=lr * 0.01)

    model.compile(optimizer=build_optimizer(cfg, lr), loss=build_loss(cfg), metrics=build_metrics())
    return n_trainable


def load_trained(path: str) -> K.Model:
    """Load a saved model. `Preprocess`/`WarmupCosine` are registered
    serializables, so no custom_objects plumbing is needed; the legacy
    `Lambda(preprocess_input)` graph from the VGG-only build is still
    accepted for backward compatibility."""
    try:
        return K.models.load_model(path)
    except (TypeError, ValueError):
        return K.models.load_model(path, custom_objects={
            'preprocess_input': K.applications.vgg16.preprocess_input,
        }, compile=False)


def describe(model: K.Model) -> dict:
    """Architecture summary for the dashboard, read off the live graph rather
    than restated as literals in the template."""
    backbone = get_backbone(model)
    name = getattr(backbone, '_xai_backbone_name', None)
    spec = BACKBONES.get(name or '', None)
    head = [l.name for l in model.layers[model.layers.index(backbone) + 1:]]
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return {
        # The registry key, not backbone.name -- that is the literal string
        # 'backbone' for every model this codebase builds.
        'backbone': name or backbone.name,
        'backbone_description': spec.description if spec else (name or backbone.name),
        'input_shape': tuple(model.input_shape[1:]),
        'feature_map': tuple(backbone.output_shape[1:]),
        'head': ' -> '.join(head),
        'total_params': int(model.count_params()),
        'trainable_params': trainable,
    }


# ── The one wrapper every explanation method talks to ──────────────────────
class XAIModel:
    """Wraps a trained model with the pieces every explanation method needs:
    a plain probability/logit forward pass, and (for Grad-CAM-family methods)
    access to the intermediate conv-layer activations inside the nested
    backbone. One implementation, shared by training, evaluation, the app,
    and every method in xai_cxr.explain.

    `last_conv_layer` accepts 'auto' (the final feature map), 'hires' (the
    stride-16 stage, a 2x finer CAM grid), or an explicit layer name.
    """

    def __init__(self, model: K.Model, last_conv_layer: str | None = 'auto'):
        self.model = model
        self.backbone = get_backbone(model)
        self.last_conv_layer_name = resolve_cam_layer(self.backbone, last_conv_layer)

        # Optional; None when configs/model.yaml sets window_layer: false.
        self._window = next((l for l in model.layers if isinstance(l, LearnableWindow)), None)
        self._attention = find_attention_layer(model)

        self.logits_layer = model.get_layer('logits')
        self.logits_model = K.Model(model.input, self.logits_layer.output)

        # Head layers = everything between the backbone and the logits output,
        # in graph order. Replaying this list is what makes the gradient
        # methods faithful to the actual trained head instead of an assumed
        # gap/dense/dropout stack.
        start = model.layers.index(self.backbone) + 1
        end = model.layers.index(self.logits_layer)
        self._head_layers = model.layers[start:end + 1]

        self._grad_base_model = K.Model(
            inputs=self.backbone.input,
            outputs=[self.backbone.get_layer(self.last_conv_layer_name).output,
                     self.backbone.output],
        )

    @property
    def cam_grid_shape(self) -> tuple[int, int]:
        """(h, w) of the CAM before upsampling -- surfaced in the UI so the
        resolution of an explanation is visible rather than implied by the
        smooth interpolation."""
        return tuple(self._grad_base_model.output_shape[0][1:3])

    @property
    def has_attention(self) -> bool:
        return self._attention is not None

    def attention_map(self, images_raw: np.ndarray) -> np.ndarray:
        """The head's own (h, w) pooling weights -- an explanation that comes
        free with the forward pass, rather than being reconstructed after it.

        Raises if the model was built with a non-attention head, instead of
        quietly returning something that looks like an attention map and is
        not."""
        if self._attention is None:
            raise ValueError(
                'This model has no attention-pooling head; attention_map needs '
                "head: attention in configs/model.yaml.")
        conv = self._grad_base_model(self.preprocess(images_raw))[1]
        return self._attention.attention_scores(conv).numpy()

    # -- plain forward passes (raw RGB [0, 255] input) --------------------
    def preprocess(self, images_raw: np.ndarray) -> tf.Tensor:
        x = tf.cast(images_raw, tf.float32)
        if self._window is not None:
            x = self._window(x)
        return self.model.get_layer('preprocess')(x)

    def logits(self, images_raw: np.ndarray) -> np.ndarray:
        return self.logits_model(images_raw, training=False).numpy().reshape(-1)

    def proba(self, images_raw: np.ndarray) -> np.ndarray:
        return self.model(images_raw, training=False).numpy().reshape(-1)

    def head_from_conv(self, backbone_out: tf.Tensor) -> tf.Tensor:
        """Backbone feature map -> logit, by replaying the trained head
        layers in order (dropout explicitly in inference mode)."""
        x = backbone_out
        for layer in self._head_layers:
            # Dropout (incl. SpatialDropout2D) and BatchNormalization both
            # change behaviour with `training`; pinning it False keeps an
            # explanation deterministic and identical to what /predict scored.
            if isinstance(layer, (K.layers.Dropout, K.layers.BatchNormalization)):
                x = layer(x, training=False)
            else:
                x = layer(x)
        return x

    def conv_and_logits_with_tape(self, images_raw: np.ndarray):
        """Returns (tape, conv_outputs, logits) with the tape already having
        recorded conv_outputs -> logits, for Grad-CAM / Grad-CAM++."""
        preprocessed = self.preprocess(images_raw)
        tape = tf.GradientTape()
        with tape:
            conv_outputs, backbone_out = self._grad_base_model(preprocessed)
            tape.watch(conv_outputs)
            logits = self.head_from_conv(backbone_out)
        return tape, conv_outputs, logits

    def conv_features(self, images_raw: np.ndarray) -> np.ndarray:
        """conv activations with no gradient tracking, for Score-CAM."""
        preprocessed = self.preprocess(images_raw)
        conv_outputs, _ = self._grad_base_model(preprocessed)
        return conv_outputs.numpy()

    def logits_grad_wrt_input(self, images_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of the logit w.r.t. the raw input pixels (through the
        preprocessing layer too). Used by Integrated Gradients, and by the
        faithfulness/sanity-check evaluation code."""
        x = tf.convert_to_tensor(images_raw, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            preprocessed = self.preprocess(x)
            conv_outputs, backbone_out = self._grad_base_model(preprocessed)
            logits = self.head_from_conv(backbone_out)
        grads = tape.gradient(logits, x)
        return logits.numpy().reshape(-1), grads.numpy()


# ── Decision threshold ─────────────────────────────────────────────────────
def tune_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Youden's J statistic on the validation split."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def tune_threshold_at_sensitivity(y_true: np.ndarray, y_scores: np.ndarray,
                                  min_sensitivity: float) -> float:
    """Lowest-FPR threshold that still reaches `min_sensitivity`.

    Youden's J weights a missed pneumonia and a false alarm equally, which is
    not the clinical trade-off. This is offered as the alternative operating
    point; configs/model.yaml picks which one scripts/train.py saves.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eligible = np.where(tpr >= min_sensitivity)[0]
    if len(eligible) == 0:
        return tune_threshold(y_true, y_scores)
    return float(thresholds[eligible[int(np.argmin(fpr[eligible]))]])


def save_threshold(threshold: float, path: str | None = None) -> None:
    path = path or metrics_path()
    data = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    data.setdefault('classification', {})['decision_threshold'] = threshold
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_threshold(path: str | None = None, default: float = 0.5) -> float:
    path = path or metrics_path()
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return float(data.get('classification', {}).get('decision_threshold', default))
