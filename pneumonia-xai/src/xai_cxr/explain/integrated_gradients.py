"""Integrated Gradients (X-4): an axiomatic method with a completeness
guarantee (attributions sum to the logit difference between image and
baseline), included in the Stanford CheXlocalize benchmark so numbers here
are comparable to published ones.

Baseline is a heavily blurred version of the image by default (a black
baseline works too, but on chest X-rays a black image looks like the
"no signal" case for a very different reason — total darkness rather than
absence of pathology — so a blurred baseline is the more defensible default;
both are supported).
"""
from __future__ import annotations

import numpy as np
import cv2

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def _make_baseline(image: np.ndarray, kind: str) -> np.ndarray:
    if kind == 'black':
        return np.zeros_like(image)
    ksize = max(3, (min(image.shape[:2]) // 4) | 1)  # odd kernel, ~1/4 of image size
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224),
            steps: int = 32, baseline: str = 'blur', chunk_size: int = 4) -> np.ndarray:
    images = as_batch(image)
    x = images[0]
    x0 = _make_baseline(x, baseline)

    alphas = np.linspace(0.0, 1.0, steps)
    scaled = np.array([x0 + a * (x - x0) for a in alphas], dtype='float32')  # (steps, H, W, 3)

    # Chunked, not one batch of `steps`. Pushing all interpolation steps
    # through at once made peak memory scale with `steps` x resolution, which
    # OOM'd at 320px with steps=16 on a 4 GB card. Reducing `steps` would have
    # been the wrong fix: the step count IS the quality of the Riemann
    # approximation, and completeness is the axiom IG exists for. Summing over
    # chunks and dividing once is arithmetically identical to meaning over the
    # whole stack, at 1/chunk of the memory.
    grad_sum = np.zeros_like(x, dtype='float64')
    for start in range(0, steps, max(1, chunk_size)):
        _, g = xai_model.logits_grad_wrt_input(scaled[start:start + max(1, chunk_size)])
        grad_sum += g.sum(axis=0)
    avg_grads = (grad_sum / steps).astype('float32')
    ig = (x - x0) * avg_grads
    heatmap = np.sum(np.abs(ig), axis=-1)
    return resize_to(normalize(heatmap), img_size)
