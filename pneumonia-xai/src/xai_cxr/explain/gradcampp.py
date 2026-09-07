"""Grad-CAM++ (X-2): handles multiple/overlapping instances of a pathology
better than vanilla Grad-CAM by weighting pixels with second/third-order
gradient terms instead of a single global-average pool of the gradient.

Uses the standard closed-form approximation (Chattopadhyay et al. 2018): the
higher-order terms are expressed algebraically from one first-order gradient
tape rather than nesting tapes, which is what most published implementations
do in practice.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    images = as_batch(image)
    tape, conv_outputs, logits = xai_model.conv_and_logits_with_tape(images)
    with tape:
        loss = logits[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0].numpy()          # (h, w, c)
    conv = conv_outputs[0].numpy()                                 # (h, w, c)

    grads2 = grads ** 2
    grads3 = grads ** 3
    conv_sum = conv.sum(axis=(0, 1), keepdims=True)                # (1, 1, c)

    alpha_denom = 2 * grads2 + conv_sum * grads3
    alpha_denom = np.where(alpha_denom != 0, alpha_denom, 1e-10)
    alphas = grads2 / alpha_denom

    alpha_norm = alphas.sum(axis=(0, 1), keepdims=True)
    alpha_norm = np.where(alpha_norm != 0, alpha_norm, 1e-10)
    alphas = alphas / alpha_norm

    weights = np.maximum(grads, 0)
    deep_weights = (weights * alphas).sum(axis=(0, 1))             # (c,)

    heatmap = np.tensordot(conv, deep_weights, axes=([2], [0]))    # (h, w)
    heatmap = np.maximum(heatmap, 0)
    return resize_to(normalize(heatmap), img_size)
