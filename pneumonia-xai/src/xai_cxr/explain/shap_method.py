"""SHAP (X-6): kept for the comparison the original proposal promised, but
reported honestly — it is the slowest method in the registry (see
xai_cxr.evaluation.runtime) and its pixel-wise fidelity on the highly
correlated pixels of an X-ray is weaker than the gradient/perturbation
methods it's compared against in the eval suite. GradientExplainer needs a
background sample; one is built lazily per model from the calibration split
(never train/val/test) and cached.
"""
from __future__ import annotations

import os
import numpy as np
import shap

from ..config import DataConfig
from ..data import read_manifest, load_image
from ..models.baseline import XAIModel
from ._common import as_batch, normalize, resize_to

_explainer_cache: dict[int, shap.GradientExplainer] = {}


def _background(cfg: DataConfig, size: int) -> np.ndarray:
    rows = [r for r in read_manifest('calibration') if r[1] == 'NORMAL'][:size]
    if not rows:
        rows = read_manifest('calibration')[:size]
    return np.array([load_image(os.path.join(cfg.dataset_dir, r), cfg.img_size) for r, _ in rows])


def _get_explainer(xai_model: XAIModel, background_size: int) -> shap.GradientExplainer:
    key = id(xai_model)
    if key not in _explainer_cache:
        cfg = DataConfig.load()
        bg = _background(cfg, background_size)
        _explainer_cache[key] = shap.GradientExplainer(xai_model.model, bg)
    return _explainer_cache[key]


def explain(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224),
            background_size: int = 10) -> np.ndarray:
    images = as_batch(image)
    explainer = _get_explainer(xai_model, background_size)
    shap_values = explainer.shap_values(images)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    single = sv[0]  # (H, W, C) for older shap versions, (H, W, C, n_outputs) for newer ones
    if single.ndim == 4:
        # Single-output model -> the trailing axis has size 1; take it, then
        # sum the real channel axis. Getting this wrong silently produces a
        # (H, W, C) "heatmap" that still resizes/normalizes without error and
        # only breaks downstream (e.g. faithfulness's pixel indexing) --
        # this shape branch exists because that exact bug shipped once.
        single = single[..., 0]
    agg = np.sum(np.abs(single), axis=-1)
    assert agg.ndim == 2, f'SHAP heatmap should be 2D before resize, got shape {agg.shape}'
    return resize_to(normalize(agg), img_size)
