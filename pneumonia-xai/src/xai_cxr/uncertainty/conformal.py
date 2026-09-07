"""Split conformal prediction (U-1, U-2, U-4): distribution-free prediction
sets at a user-set error rate, calibrated on the held-out calibration split
(xai_cxr.data — never touched during training). Roughly the "thirty lines on
top of a trained model" the spec describes.

Nonconformity score for a labelled example is 1 - p_model(true label | x).
Given a target miscoverage rate alpha, qhat is the (n+1)(1-alpha)/n empirical
quantile of calibration scores (the standard finite-sample correction — see
Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction"). A new
example's prediction set contains every label whose nonconformity score is
<= qhat, which is what gives the marginal coverage guarantee: at least
1 - alpha of true labels fall in their prediction set, on average over draws
of the calibration set.
"""
from __future__ import annotations

import numpy as np

LABELS = ('NORMAL', 'PNEUMONIA')  # index 0 / 1, matching xai_cxr.data's label encoding


def _nonconformity(proba: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    """proba = P(PNEUMONIA). Score for the true label is 1 - P(true label)."""
    p_true = np.where(label_idx == 1, proba, 1 - proba)
    return 1 - p_true


def calibrate(calib_labels: np.ndarray, calib_proba: np.ndarray,
              alphas: tuple[float, ...] = (0.05, 0.1, 0.2)) -> dict[float, float]:
    scores = _nonconformity(calib_proba, calib_labels)
    n = len(scores)
    qhats = {}
    for alpha in alphas:
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhats[alpha] = float(np.quantile(scores, level, method='higher'))
    return qhats


def prediction_set(proba: float, qhat: float) -> list[str]:
    """Which labels survive at this qhat. Can be empty, one label
    (a confident, actionable prediction), or both (genuine ambiguity —
    that's the abstention trigger)."""
    result = []
    for idx, label in enumerate(LABELS):
        p = proba if idx == 1 else 1 - proba
        if (1 - p) <= qhat:
            result.append(label)
    return result


def empirical_coverage(labels: np.ndarray, proba: np.ndarray, qhat: float) -> float:
    """Fraction of test examples whose true label is in their prediction
    set — the number that should track the target (1 - alpha) within
    sampling error if the guarantee holds (U-4)."""
    hits = 0
    for y, p in zip(labels, proba):
        pred_set = prediction_set(float(p), qhat)
        true_label = LABELS[int(y)]
        hits += int(true_label in pred_set)
    return hits / len(labels)
