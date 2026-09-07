"""Shortcut audit (A-1): four cheap probes for whether the model is keying
on something that isn't the pathology.

  - Border masking: grey out the outer margin. A model reading anatomy
    shouldn't move much; a big swing implicates the border/vignette.
  - Laterality-marker masking: grey out the four corners specifically,
    where L/R radiograph markers typically sit.
  - Blurred-image source probe: heavily blur the whole image, destroying
    fine anatomical detail while keeping global image statistics (contrast,
    compression-era artifacts) intact. Staying confident under this is a
    signal the model is reading something correlated with image
    source/quality rather than the lesion itself.
  - Attribution-in-lung-field ratio: what fraction of the explanation's
    mass falls inside a plausible lung field? No VinDr masks exist in this
    project (D-2, deferred), so the "lung field" here is a fixed geometric
    approximation — two rectangles roughly where lungs sit on a centred,
    upright paediatric chest film — not a real segmentation. Treat the
    number as directional, not precise.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..models.baseline import XAIModel
from ..explain.registry import explain as registry_explain


def approximate_lung_field_mask(img_size: tuple[int, int]) -> np.ndarray:
    h, w = img_size
    mask = np.zeros((h, w), dtype='float32')
    top, bottom = int(0.12 * h), int(0.92 * h)
    left_start, left_end = int(0.08 * w), int(0.44 * w)
    right_start, right_end = int(0.56 * w), int(0.92 * w)
    mask[top:bottom, left_start:left_end] = 1.0
    mask[top:bottom, right_start:right_end] = 1.0
    return mask


def border_masking_test(xai_model: XAIModel, image: np.ndarray, border_frac: float = 0.08) -> dict:
    h, w, _ = image.shape
    bh, bw = int(h * border_frac), int(w * border_frac)
    fill = float(image.mean())
    masked = image.copy()
    masked[:bh, :], masked[-bh:, :], masked[:, :bw], masked[:, -bw:] = fill, fill, fill, fill
    orig = float(xai_model.proba(image[np.newaxis])[0])
    new = float(xai_model.proba(masked[np.newaxis])[0])
    return {'orig_proba': orig, 'masked_proba': new, 'delta': orig - new}


def laterality_marker_masking_test(xai_model: XAIModel, image: np.ndarray,
                                    corner_frac: float = 0.12) -> dict:
    h, w, _ = image.shape
    ch, cw = int(h * corner_frac), int(w * corner_frac)
    fill = float(image.mean())
    masked = image.copy()
    for (r0, r1, c0, c1) in [(0, ch, 0, cw), (0, ch, w - cw, w),
                              (h - ch, h, 0, cw), (h - ch, h, w - cw, w)]:
        masked[r0:r1, c0:c1] = fill
    orig = float(xai_model.proba(image[np.newaxis])[0])
    new = float(xai_model.proba(masked[np.newaxis])[0])
    return {'orig_proba': orig, 'masked_proba': new, 'delta': orig - new}


def blurred_source_probe(xai_model: XAIModel, image: np.ndarray, blur_frac: float = 0.15) -> dict:
    h, w, _ = image.shape
    ksize = max(3, (int(min(h, w) * blur_frac)) | 1)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    orig = float(xai_model.proba(image[np.newaxis])[0])
    new = float(xai_model.proba(blurred[np.newaxis])[0])
    same_side = (orig > 0.5) == (new > 0.5)
    return {'orig_proba': orig, 'blurred_proba': new, 'delta': orig - new,
            'still_confident_same_class': bool(same_side and abs(new - 0.5) > 0.3)}


def attribution_in_lung_field_ratio(heatmap: np.ndarray) -> float:
    mask = approximate_lung_field_mask(heatmap.shape[:2])
    total = heatmap.sum() + 1e-8
    return float((heatmap * mask).sum() / total)


def run_shortcut_audit(xai_model: XAIModel, image: np.ndarray, method: str = 'gradcam',
                        img_size: tuple[int, int] = (224, 224)) -> dict:
    heatmap = registry_explain(method, xai_model, image, img_size=img_size)
    return {
        'method': method,
        'border_masking': border_masking_test(xai_model, image),
        'laterality_marker_masking': laterality_marker_masking_test(xai_model, image),
        'blurred_source_probe': blurred_source_probe(xai_model, image),
        'attribution_in_lung_field_ratio': attribution_in_lung_field_ratio(heatmap),
        'lung_field_mask_is_approximate': True,
    }
