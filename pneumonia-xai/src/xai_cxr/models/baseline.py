"""The M-1 baseline, fixed.

Fixes applied relative to the v1 model (kept at models/pneumonia_model_v1_legacy.h5
as the "here's what was wrong with it" artifact):

  - tf.keras.applications.vgg16.preprocess_input instead of a manual /255
    rescale (the VGG16 ImageNet weights expect BGR, mean-centered inputs —
    /255 is not that).
  - GlobalAveragePooling2D instead of Flatten (fewer parameters, and it keeps
    spatial correspondence sane for the explanation methods).
  - A separate pre-activation `logits` output alongside the sigmoid
    probability, so every explanation method (Grad-CAM in particular) can
    backprop from the true logit rather than a saturated post-sigmoid value.
  - class_weight for the 3:1 imbalance (see xai_cxr.data.class_weights).
  - val_auc as the checkpoint/early-stopping monitor, not val_accuracy (and
    the val split is no longer 16 images — see xai_cxr.data.build_splits).
  - a tuned decision threshold (Youden's J on the val split) instead of a
    hardcoded 0.5, saved into models/metrics.json.
"""
from __future__ import annotations

import json
import os
import numpy as np
import tensorflow as tf

from ..config import ModelConfig, metrics_path

PREPROCESS = tf.keras.applications.vgg16.preprocess_input


def build_model(cfg: ModelConfig | None = None) -> tf.keras.Model:
    cfg = cfg or ModelConfig.load()
    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_image')
    x = tf.keras.layers.Lambda(PREPROCESS, name='preprocess')(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dense(cfg.dense_units, activation='relu', name='dense')(x)
    x = tf.keras.layers.Dropout(cfg.dropout, name='dropout')(x)
    logits = tf.keras.layers.Dense(1, activation=None, name='logits')(x)
    probs = tf.keras.layers.Activation('sigmoid', name='sigmoid')(logits)

    model = tf.keras.Model(inputs, probs, name='pneumonia_classifier')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


def load_trained(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path, custom_objects={'preprocess_input': PREPROCESS})


class XAIModel:
    """Wraps a trained model with the pieces every explanation method needs:
    a plain probability/logit forward pass, and (for Grad-CAM-family methods)
    access to the intermediate conv-layer activations inside the nested VGG16
    submodel. One implementation, shared by training, evaluation, the app,
    and every method in xai_cxr.explain."""

    def __init__(self, model: tf.keras.Model, last_conv_layer: str = 'block5_conv3'):
        self.model = model
        self.last_conv_layer_name = last_conv_layer
        self.vgg = model.get_layer('vgg16')
        self.gap = model.get_layer('gap')
        self.dense = model.get_layer('dense')
        self.dropout = model.get_layer('dropout')
        self.logits_layer = model.get_layer('logits')
        self.logits_model = tf.keras.Model(model.input, self.logits_layer.output)
        self._grad_base_model = tf.keras.Model(
            inputs=self.vgg.input,
            outputs=[self.vgg.get_layer(last_conv_layer).output, self.vgg.output],
        )

    # -- plain forward passes (raw RGB [0, 255] input) --------------------
    def preprocess(self, images_raw: np.ndarray) -> tf.Tensor:
        return PREPROCESS(tf.cast(images_raw, tf.float32))

    def logits(self, images_raw: np.ndarray) -> np.ndarray:
        return self.logits_model(images_raw, training=False).numpy().reshape(-1)

    def proba(self, images_raw: np.ndarray) -> np.ndarray:
        return self.model(images_raw, training=False).numpy().reshape(-1)

    def head_from_conv(self, vgg_out: tf.Tensor) -> tf.Tensor:
        """conv features (post-vgg, pre-GAP) -> logit. Reuses the trained
        head layers so gradient/activation methods stay faithful to the
        actual model rather than an approximation of it."""
        x = self.gap(vgg_out)
        x = self.dense(x)
        x = self.dropout(x, training=False)
        return self.logits_layer(x)

    def conv_and_logits_with_tape(self, images_raw: np.ndarray):
        """Returns (tape, conv_outputs, logits) with the tape already having
        recorded conv_outputs -> logits, for Grad-CAM / Grad-CAM++."""
        preprocessed = self.preprocess(images_raw)
        tape = tf.GradientTape()
        with tape:
            conv_outputs, vgg_out = self._grad_base_model(preprocessed)
            tape.watch(conv_outputs)
            logits = self.head_from_conv(vgg_out)
        return tape, conv_outputs, logits

    def conv_features(self, images_raw: np.ndarray) -> np.ndarray:
        """conv activations with no gradient tracking, for Score-CAM."""
        preprocessed = self.preprocess(images_raw)
        conv_outputs, _ = self._grad_base_model(preprocessed)
        return conv_outputs.numpy()

    def logits_grad_wrt_input(self, images_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of the logit w.r.t. the raw input pixels (through the
        preprocessing Lambda too). Used by Integrated Gradients, and by the
        faithfulness/sanity-check evaluation code."""
        x = tf.convert_to_tensor(images_raw, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            preprocessed = self.preprocess(x)
            conv_outputs, vgg_out = self._grad_base_model(preprocessed)
            logits = self.head_from_conv(vgg_out)
        grads = tape.gradient(logits, x)
        return logits.numpy().reshape(-1), grads.numpy()


def tune_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Youden's J statistic on the validation split."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


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
