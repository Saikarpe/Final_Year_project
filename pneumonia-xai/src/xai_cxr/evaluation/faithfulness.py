"""Faithfulness (E-7): is the highlighted region what the model actually
used, independent of whether it's anatomically correct? Two standard
perturbation tests:

  - Deletion: progressively replace the most-important pixels with a flat
    baseline; a faithful heatmap should crash the prediction quickly, so a
    *low* deletion AUC is good.
  - Insertion: start from a blank canvas and progressively reveal the
    most-important pixels; a faithful heatmap should recover the prediction
    quickly, so a *high* insertion AUC is good.

ROAD (Rong et al., 2022) replaces the flat-baseline substitution with an
inpainting-based one, because a constant fill creates an artificial edge the
model can key on regardless of whether the removed region mattered. The
full ROAD procedure also retrains/fine-tunes on imputed inputs; that's out
of scope here, so `road_score` is an approximation — inpainting substitution
without retraining — and is documented as such everywhere it's reported.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..models.baseline import XAIModel


def _step_size(n_pixels: int, n_steps: int) -> int:
    return max(1, n_pixels // n_steps)


def deletion_insertion_auc(xai_model: XAIModel, image: np.ndarray, heatmap: np.ndarray,
                            n_steps: int = 20) -> dict:
    h, w, _ = image.shape
    order = np.argsort(-heatmap.reshape(-1))
    baseline_value = float(image.mean())
    n_pixels = h * w
    step = _step_size(n_pixels, n_steps)

    deletion_img = image.copy()
    flat_del = deletion_img.reshape(-1, 3)
    del_scores = [xai_model.proba(deletion_img[np.newaxis])[0]]

    insertion_img = np.full_like(image, baseline_value)
    flat_ins = insertion_img.reshape(-1, 3)
    flat_src = image.reshape(-1, 3)
    ins_scores = [xai_model.proba(insertion_img[np.newaxis])[0]]

    for i in range(0, n_pixels, step):
        idx = order[i:i + step]
        flat_del[idx] = baseline_value
        del_scores.append(xai_model.proba(deletion_img[np.newaxis])[0])
        flat_ins[idx] = flat_src[idx]
        ins_scores.append(xai_model.proba(insertion_img[np.newaxis])[0])

    fractions = np.linspace(0, 1, len(del_scores))
    return {
        'deletion_auc': float(np.trapz(del_scores, fractions)),
        'insertion_auc': float(np.trapz(ins_scores, fractions)),
        'n_steps': len(del_scores) - 1,
    }


def road_score(xai_model: XAIModel, image: np.ndarray, heatmap: np.ndarray,
                fractions: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)) -> dict:
    """Approximate ROAD: inpaint (not flat-fill) the top-k% most important
    pixels and measure how far the prediction moves. No retraining on
    imputed inputs is performed — see the module docstring."""
    h, w, _ = image.shape
    order = np.argsort(-heatmap.reshape(-1))
    orig_score = float(xai_model.proba(image[np.newaxis])[0])

    drops = {}
    for frac in fractions:
        k = int(frac * h * w)
        mask = np.zeros((h, w), dtype='uint8')
        mask.reshape(-1)[order[:k]] = 255
        inpainted = cv2.inpaint(image.astype('uint8'), mask, 3, cv2.INPAINT_TELEA).astype('float32')
        score = float(xai_model.proba(inpainted[np.newaxis])[0])
        drops[frac] = orig_score - score

    return {'orig_score': orig_score, 'score_drop_by_fraction': drops,
            'mean_drop': float(np.mean(list(drops.values()))),
            'approximate': True}
