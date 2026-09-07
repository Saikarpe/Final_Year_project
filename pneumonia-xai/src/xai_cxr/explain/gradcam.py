"""Grad-CAM (X-1), fixed: backprop from the pre-activation logit of the
*predicted* class, through one shared model implementation
(xai_cxr.models.baseline.XAIModel) rather than a bespoke copy per script."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    images = as_batch(image)
    tape, conv_outputs, logits = xai_model.conv_and_logits_with_tape(images)
    with tape:
        loss = logits[:, 0]  # binary head: the single logit *is* the predicted-class score
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    return resize_to(normalize(heatmap), img_size)
