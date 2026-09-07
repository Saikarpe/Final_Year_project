"""Classification-performance metrics (E-1, E-2, E-5). A point estimate with
no interval is not a result — every number here that can carry a confidence
interval does.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


def bootstrap_auroc_ci(y_true: np.ndarray, y_scores: np.ndarray,
                        n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Per-pathology AUROC with a bootstrap confidence interval (E-1). With
    only the Kermany binary label this is a single-row analogue of the
    multi-pathology table the spec describes for CheXpert/VinDr (D-3) — the
    same procedure, applied to the one pathology this dataset has."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = float(roc_auc_score(y_true, y_scores))

    boot_scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b, s_b = y_true[idx], y_scores[idx]
        if len(np.unique(y_b)) < 2:
            continue  # AUROC undefined for a single-class resample
        boot_scores.append(roc_auc_score(y_b, s_b))

    alpha = 1 - ci
    lo, hi = np.percentile(boot_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'auroc': point, 'ci_lower': float(lo), 'ci_upper': float(hi),
            'ci_level': ci, 'n_bootstrap': len(boot_scores)}


def sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sensitivity and specificity reported separately (E-2) — an aggregate
    accuracy figure can hide a specificity collapse (this repo's v1 model
    was 85.4% accurate on 63.7% specificity)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')
    return {'sensitivity': sensitivity, 'specificity': specificity,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def calibration_and_brier(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> dict:
    """Calibration curve + Brier score (E-5) — a model can rank well (good
    AUROC) and still report confidences that don't mean what they claim to,
    which matters directly because the UI shows a confidence number."""
    y_true = np.asarray(y_true).astype(float)
    y_scores = np.asarray(y_scores).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_scores, bins[1:-1])

    curve = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        curve.append({
            'bin_lower': float(bins[b]),
            'bin_upper': float(bins[b + 1]),
            'mean_predicted': float(y_scores[mask].mean()),
            'observed_frequency': float(y_true[mask].mean()),
            'n': int(mask.sum()),
        })
    brier = float(brier_score_loss(y_true, y_scores))
    return {'curve': curve, 'brier_score': brier}
