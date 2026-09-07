"""Robustness and complexity (E-9): how stable is an explanation under small
input perturbations, and how concentrated is it?

The spec suggests the Quantus toolkit (off-the-shelf, citable) for this.
Quantus is not a pinned dependency here — it pulls in a fairly heavy stack
and its exact call signature varies by version, which isn't something to
guess at without a working install to verify against. What's implemented
below are the same two metrics by name and definition (Yeh et al. 2019's
max-sensitivity; Bhatt et al. 2020's complexity, i.e. the entropy of the
normalized attribution map) — if the team adds `quantus` to requirements.txt
later, swap the call in evaluate.py without touching anything else, since
both return the same {'max_sensitivity': ..., 'complexity': ...} shape.

n_perturbations defaults to 2, not the more typical 10-50: max-sensitivity
recomputes the *entire* explanation per perturbation, and for a multi-step
method (Integrated Gradients) that made this the dominant cost of the whole
evaluation suite (measured: ~200s/image at steps=32, perturbations=4, on
this project's CPU-only dev machine). 2 samples is a noisy estimate of a
max -- treat max_sensitivity as directional, not a precise bound -- and is
the floor that keeps the eval suite tractable without a GPU. See
docs/decisions_log.md.
"""
from __future__ import annotations

import numpy as np

from ..models.baseline import XAIModel
from ..explain.registry import explain as registry_explain


def _max_sensitivity(xai_model: XAIModel, image: np.ndarray, method: str,
                      img_size: tuple[int, int], original_heatmap: np.ndarray,
                      n_perturbations: int, radius: float, seed: int) -> float:
    rng = np.random.RandomState(seed)
    max_diff = 0.0
    for _ in range(n_perturbations):
        noise = rng.normal(scale=radius * 255, size=image.shape).astype('float32')
        perturbed = np.clip(image + noise, 0, 255).astype('float32')
        heatmap = registry_explain(method, xai_model, perturbed, img_size=img_size)
        denom = np.linalg.norm(original_heatmap) + 1e-8
        diff = np.linalg.norm(heatmap - original_heatmap) / denom
        max_diff = max(max_diff, float(diff))
    return max_diff


def _complexity(heatmap: np.ndarray) -> float:
    flat = heatmap.reshape(-1).astype('float64')
    flat = flat / (flat.sum() + 1e-8)
    flat = flat[flat > 0]
    return float(-(flat * np.log(flat)).sum())


def robustness_and_complexity(xai_model: XAIModel, image: np.ndarray, method: str,
                               img_size: tuple[int, int] = (224, 224),
                               n_perturbations: int = 2, radius: float = 0.02,
                               seed: int = 0) -> dict:
    original_heatmap = registry_explain(method, xai_model, image, img_size=img_size)
    return {
        'method': method,
        'max_sensitivity': _max_sensitivity(xai_model, image, method, img_size,
                                             original_heatmap, n_perturbations, radius, seed),
        'complexity': _complexity(original_heatmap),
    }
