"""Shared helpers so every method in the registry returns the same shape and
normalization — one contract the eval suite and the UI can both rely on."""
from __future__ import annotations

import numpy as np
import cv2


def normalize(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    denom = heatmap.max() - heatmap.min()
    if denom < 1e-8:
        return np.zeros_like(heatmap, dtype='float32')
    return ((heatmap - heatmap.min()) / denom).astype('float32')


def resize_to(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # Every method in the registry must reduce to a single-channel (H, W)
    # map before this point. cv2.resize will happily "succeed" on a
    # (H, W, C) array instead of raising -- this caught a real bug once
    # (shap_method.py leaving an unreduced axis in), so it's an assertion,
    # not a comment.
    assert heatmap.ndim == 2, f'resize_to expects a 2D heatmap, got shape {heatmap.shape}'
    # INTER_LINEAR, not CUBIC: cubic interpolation overshoots near edges and
    # can push a [0, 1]-normalized map slightly outside that range.
    resized = cv2.resize(heatmap.astype('float32'), size, interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def as_batch(image: np.ndarray) -> np.ndarray:
    """Accepts (H, W, 3) or (1, H, W, 3); always returns (1, H, W, 3)."""
    if image.ndim == 3:
        return image[np.newaxis, ...]
    return image


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """JET-colormapped heatmap with no blending -- used by the app so the
    opacity slider can composite it over the original image client-side
    (instant, no round-trip per slider move)."""
    colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """image_rgb: (H, W, 3) float or uint8, RGB, [0, 255]. heatmap: (H, W) in
    [0, 1], same spatial size. Returns a uint8 RGB overlay — the one overlay
    implementation the app, the failure gallery, and any script use."""
    img_u8 = np.clip(image_rgb, 0, 255).astype('uint8')
    colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_u8, 1 - alpha, colored_rgb, alpha, 0)
    return blended
