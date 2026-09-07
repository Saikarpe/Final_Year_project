"""Runtime per explanation (E-11): decides what can ship in the interactive
case reader vs. what only runs in a batch evaluation job. SHAP being far
slower than Grad-CAM is a finding to report, not an inconvenience to hide.

Timed via xai_cxr.explain.registry.explain (not the raw METHODS dict), so
these numbers reflect the actual configs/explain.yaml settings the app and
the rest of the eval suite use -- not an unconfigured worst case (e.g.
Score-CAM over all 512 channels) that nothing in this codebase actually runs.
"""
from __future__ import annotations

import time
import numpy as np

from ..config import ExplainConfig
from ..models.baseline import XAIModel
from ..explain.registry import METHODS, explain as registry_explain


def time_methods(xai_model: XAIModel, image: np.ndarray, img_size: tuple[int, int] = (224, 224),
                  methods: list[str] | None = None, n_repeats: int = 3,
                  explain_cfg: ExplainConfig | None = None) -> dict:
    methods = methods or list(METHODS)
    explain_cfg = explain_cfg or ExplainConfig.load()
    results = {}
    for name in methods:
        # one untimed warm-up call so lazy graph-tracing/compilation cost
        # (real, but a one-time cost) doesn't get counted against the method
        registry_explain(name, xai_model, image, img_size=img_size, explain_cfg=explain_cfg)
        durations = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            registry_explain(name, xai_model, image, img_size=img_size, explain_cfg=explain_cfg)
            durations.append(time.perf_counter() - start)
        print(f'    timed {name}: mean {sum(durations) / len(durations):.2f}s', flush=True)
        results[name] = {
            'mean_seconds': float(np.mean(durations)),
            'min_seconds': float(np.min(durations)),
            'max_seconds': float(np.max(durations)),
            'n_repeats': n_repeats,
        }
    return results
