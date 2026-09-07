"""Occlusion (X-5): slow but model-agnostic and conceptually unimpeachable —
slide a grey patch over the image, measure how much the logit drops, and
call that drop the patch's importance. Used as the closest thing to a
ground-truth reference for the faster methods.
"""
from __future__ import annotations

import numpy as np

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224),
            patch: int = 16, stride: int = 8, batch_size: int = 32) -> np.ndarray:
    images = as_batch(image)
    x = images[0]
    h, w, _ = x.shape
    fill_value = float(x.mean())

    orig_logit = xai_model.logits(images)[0]

    coords = [(i, j) for i in range(0, h, stride) for j in range(0, w, stride)]
    occluded = np.empty((len(coords), h, w, 3), dtype='float32')
    for k, (i, j) in enumerate(coords):
        occ = x.copy()
        occ[i:i + patch, j:j + patch, :] = fill_value
        occluded[k] = occ

    logits = np.concatenate([
        xai_model.logits(occluded[start:start + batch_size])
        for start in range(0, len(occluded), batch_size)
    ])

    heatmap = np.zeros((h, w), dtype='float32')
    counts = np.zeros((h, w), dtype='float32')
    for (i, j), logit in zip(coords, logits):
        drop = orig_logit - logit  # occluding the pathology should *reduce* the logit
        heatmap[i:i + patch, j:j + patch] += drop
        counts[i:i + patch, j:j + patch] += 1
    counts[counts == 0] = 1
    heatmap /= counts
    return resize_to(normalize(heatmap), img_size)
