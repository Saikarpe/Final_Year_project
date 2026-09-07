"""Abstention policy (U-3): refer to a radiologist whenever the conformal
prediction set isn't exactly one label — either genuine ambiguity (both
labels survive) or, in principle, neither (an out-of-distribution input the
calibration set gives no support for)."""
from __future__ import annotations

import numpy as np

from .conformal import prediction_set


def should_abstain(pred_set: list[str]) -> bool:
    return len(pred_set) != 1


def abstention_curve(labels: np.ndarray, proba: np.ndarray, qhats: dict[float, float]) -> list[dict]:
    """Accuracy-vs-abstention-rate curve swept across the calibrated alpha
    thresholds (U-3's acceptance criterion)."""
    from .conformal import LABELS
    rows = []
    for alpha, qhat in sorted(qhats.items()):
        n_abstain = 0
        n_correct_kept = 0
        n_kept = 0
        for y, p in zip(labels, proba):
            pred_set = prediction_set(float(p), qhat)
            if should_abstain(pred_set):
                n_abstain += 1
                continue
            n_kept += 1
            predicted = pred_set[0]
            true_label = LABELS[int(y)]
            n_correct_kept += int(predicted == true_label)
        rows.append({
            'alpha': alpha,
            'target_coverage': 1 - alpha,
            'abstention_rate': n_abstain / len(labels),
            'accuracy_on_kept': (n_correct_kept / n_kept) if n_kept else None,
            'n_kept': n_kept,
        })
    return rows
