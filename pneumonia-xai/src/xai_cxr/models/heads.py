"""Custom input filter and classifier head (M-1b).

Three pieces, all opt-out-able from configs/model.yaml, and all built so that
*at initialisation they reduce exactly to the previous model*. That property
is deliberate: it means turning them on can only cost training signal, never
silently change what the baseline was, and the model-card comparison against
the frozen-VGG16 run stays honest.

  LearnableWindow   a 3-parameter monotone intensity remap in front of the
                    backbone -- a differentiable stand-in for the window/level
                    a radiologist sets by hand. Initialised to the identity.

  SpatialDropout2D  (Keras builtin) drops whole feature-map channels rather
                    than scattered pixels. Adjacent pixels in a conv feature
                    map are strongly correlated, so ordinary Dropout on a
                    4-D tensor barely regularises; channel dropout does.

  AttentionPool2D   gated attention pooling (Ilse et al. 2018) replacing the
                    plain global average. Initialised with zero attention
                    logits, so it *is* GlobalAveragePooling2D on step 0 and
                    only departs from it if that helps.

Why attention pooling for this task: pneumonia is a focal finding. Global
average pooling divides each activation by the full 7x7 grid, so a strong
response over three cells is averaged against 46 cells of clear lung. That is
a poor match for the signal, and it is part of why the old model's saliency
was diffuse. Attention pooling lets the head weight the grid, and the learned
weights are themselves an explanation (`XAIModel.attention_map`) that costs
one forward pass rather than Score-CAM's 64.
"""
from __future__ import annotations

import tensorflow as tf

K = tf.keras


@K.utils.register_keras_serializable(package='xai_cxr')
class LearnableWindow(K.layers.Layer):
    """Differentiable window/level on raw [0, 255] intensities.

        out = x + strength * (255 * sigmoid(k * (x/255 - center)) - x)

    `strength` initialises to 0, which makes the layer an exact identity, so
    the optimiser starts from "no windowing" and buys curvature only if the
    validation AUC pays for it. `k` is parameterised through softplus so the
    curve can never invert (a non-monotone intensity map would mean a brighter
    pixel could become darker -- meaningless for a radiograph).

    Three scalars. The point is not capacity, it is that exposure varies a lot
    across this corpus and the previous pipeline had no way to normalise it.
    """

    def __init__(self, center: float = 0.5, sharpness: float = 6.0, **kwargs):
        super().__init__(**kwargs)
        self.center_init = float(center)
        self.sharpness_init = float(sharpness)

    def build(self, input_shape):
        self.strength = self.add_weight(
            name='strength', shape=(), initializer='zeros', trainable=True)
        self.center = self.add_weight(
            name='center', shape=(), trainable=True,
            initializer=K.initializers.Constant(self.center_init))
        # softplus(raw_sharpness) == sharpness_init at init
        raw = float(tf.math.log(tf.math.expm1(self.sharpness_init)).numpy())
        self.raw_sharpness = self.add_weight(
            name='raw_sharpness', shape=(), trainable=True,
            initializer=K.initializers.Constant(raw))
        super().build(input_shape)

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        k = tf.nn.softplus(self.raw_sharpness)
        windowed = 255.0 * tf.sigmoid(k * (x / 255.0 - self.center))
        out = x + tf.clip_by_value(self.strength, 0.0, 1.0) * (windowed - x)
        return tf.clip_by_value(out, 0.0, 255.0)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'center': self.center_init, 'sharpness': self.sharpness_init})
        return config


@K.utils.register_keras_serializable(package='xai_cxr')
class AttentionPool2D(K.layers.Layer):
    """Gated attention pooling over the (h, w) grid of a conv feature map.

        a     = w^T (tanh(V f) * sigmoid(U f))     per grid cell
        alpha = softmax(a) over the h*w cells
        out   = sum_hw alpha * f

    `w` is zero-initialised, so alpha is exactly uniform at step 0 and the
    layer returns the grid mean -- bit-for-bit GlobalAveragePooling2D.

    `attention_scores()` recomputes alpha as an (h, w) map without running the
    classifier, which is what XAIModel.attention_map serves to the UI.
    """

    def __init__(self, units: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        self.v = K.layers.Dense(self.units, activation='tanh', name='attn_v')
        self.u = K.layers.Dense(self.units, activation='sigmoid', name='attn_u')
        self.w = K.layers.Dense(1, use_bias=False, kernel_initializer='zeros', name='attn_w')
        super().build(input_shape)

    def _alpha(self, inputs):
        shape = tf.shape(inputs)
        channels = inputs.shape[-1]
        flat = tf.reshape(inputs, (shape[0], shape[1] * shape[2], channels))
        scores = self.w(self.v(flat) * self.u(flat))          # (b, hw, 1)
        return tf.nn.softmax(scores, axis=1), flat, shape

    def call(self, inputs):
        alpha, flat, _ = self._alpha(inputs)
        return tf.reduce_sum(alpha * flat, axis=1)

    def attention_scores(self, inputs):
        """(b, h, w) attention map, normalised to sum to 1 per image."""
        alpha, _, shape = self._alpha(inputs)
        return tf.reshape(alpha, (shape[0], shape[1], shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config


@K.utils.register_keras_serializable(package='xai_cxr')
class AvgMaxPool2D(K.layers.Layer):
    """concat(GlobalAveragePooling2D, GlobalMaxPooling2D).

    The cheap middle option between GAP and attention: the max branch keeps
    the peak response of a focal opacity that the average dilutes, at the cost
    of doubling the head's input width and no extra parameters of its own.
    """

    def call(self, inputs):
        return tf.concat([tf.reduce_mean(inputs, axis=(1, 2)),
                          tf.reduce_max(inputs, axis=(1, 2))], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] * 2)


POOLERS = {
    'attention': lambda cfg: AttentionPool2D(cfg.attention_units, name='pool'),
    'avgmax': lambda cfg: AvgMaxPool2D(name='pool'),
    'gap': lambda cfg: K.layers.GlobalAveragePooling2D(name='pool'),
}


def build_pooler(cfg) -> K.layers.Layer:
    if cfg.head not in POOLERS:
        raise ValueError(f'Unknown head {cfg.head!r}. Available: {sorted(POOLERS)}. '
                         'Set `head:` in configs/model.yaml.')
    return POOLERS[cfg.head](cfg)


def find_attention_layer(model: K.Model) -> AttentionPool2D | None:
    for layer in model.layers:
        if isinstance(layer, AttentionPool2D):
            return layer
    return None
