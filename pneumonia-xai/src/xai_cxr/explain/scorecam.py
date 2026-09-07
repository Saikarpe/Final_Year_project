"""Score-CAM (X-3): gradient-free, so it sidesteps the saturation problem
that gradient-based methods (Grad-CAM, Grad-CAM++) can suffer from. Costs
one forward pass per conv channel, batched for CPU throughput — this cost is
exactly the "runtime per explanation" finding the eval suite (E-11) is meant
to surface, not hidden away.
"""
from __future__ import annotations

import numpy as np

from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224),
            batch_size: int = 16, max_channels: int | None = None) -> np.ndarray:
    images = as_batch(image)
    base_image = images[0]  # (H, W, 3), raw [0, 255]

    conv = xai_model.conv_features(images)[0]      # (h, w, c)
    h, w, c = conv.shape
    n_channels = c if max_channels is None else min(max_channels, c)

    # Rank channels by activation energy and keep the strongest — a standard
    # speed optimization; with n_channels == c this is exactly the published method.
    if n_channels < c:
        energy = conv.reshape(-1, c).sum(axis=0)
        channel_idx = np.argsort(-energy)[:n_channels]
    else:
        channel_idx = np.arange(c)

    masks = []
    for k in channel_idx:
        act = conv[:, :, k]
        act = resize_to(act, img_size)
        denom = act.max() - act.min()
        act = (act - act.min()) / denom if denom > 1e-8 else np.zeros_like(act)
        masks.append(act)
    masks = np.stack(masks, axis=0)                 # (n, H, W)

    masked_images = base_image[np.newaxis, ...] * masks[..., np.newaxis]  # (n, H, W, 3)

    scores = []
    for start in range(0, len(masked_images), batch_size):
        batch = masked_images[start:start + batch_size]
        scores.append(xai_model.logits(batch))
    scores = np.concatenate(scores)                 # (n,) logits -> softmax-style weights

    weights = np.exp(scores - scores.max())
    weights = weights / (weights.sum() + 1e-8)

    heatmap = np.tensordot(weights, masks, axes=([0], [0]))  # (H, W)
    return resize_to(normalize(heatmap), img_size)
