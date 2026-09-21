"""Temperature scaling (E-5).

The M-1b model ranks better than the M-1 baseline but reports worse-calibrated
probabilities (Brier 0.047 -> 0.063). That is the expected signature of
fine-tuning with label smoothing: the network is pushed towards confident
outputs, so the probability it prints drifts away from the frequency it should
represent. It matters here specifically because the UI shows that number to a
reader as a confidence.

Temperature scaling (Guo et al. 2017) is the minimal fix: divide the logit by
a single scalar T fitted on held-out data by minimising negative log
likelihood. One parameter, fitted on the calibration split, never on test.

What it does and does not change:

- It is strictly monotonic in the score, so AUROC, the ROC curve and every
  ranking-based number are **unchanged**. Only the probability values move.
- It therefore does not fix, and cannot fix, the operating-point question --
  see the decisions log on Youden's J. A threshold tuned on raw probabilities
  must be mapped through the same temperature to keep meaning the same thing.
"""
import numpy as np

__all__ = ['fit_temperature', 'apply_temperature', 'expected_calibration_error']

#: Clamp before the logit so p = 0 or 1 does not produce +-inf.
_EPS = 1e-6


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale sigmoid probabilities by `temperature`.

    T > 1 softens (pulls towards 0.5), T < 1 sharpens. T == 1 is identity.
    """
    if temperature <= 0:
        raise ValueError(f'temperature must be positive, got {temperature}')
    return 1.0 / (1.0 + np.exp(-_to_logit(scores) / float(temperature)))


def _nll(y_true: np.ndarray, scores: np.ndarray, temperature: float) -> float:
    p = np.clip(apply_temperature(scores, temperature), _EPS, 1.0 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_temperature(y_true: np.ndarray, y_scores: np.ndarray,
                    lo: float = 0.05, hi: float = 20.0, tol: float = 1e-4) -> float:
    """T minimising NLL on (y_true, y_scores). Fit this on calibration data.

    Golden-section search rather than a gradient optimiser: the objective is
    one-dimensional and smooth, the bracket is known, and this keeps the
    module free of a torch/scipy-optimize dependency. NLL in T is unimodal for
    a fixed set of scores, so a bracketed search is sufficient.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_scores = np.asarray(y_scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        raise ValueError('cannot fit a temperature on an empty split')
    if len(np.unique(y_true)) < 2:
        raise ValueError('calibration split must contain both classes')

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = _nll(y_true, y_scores, c), _nll(y_true, y_scores, d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _nll(y_true, y_scores, c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _nll(y_true, y_scores, d)
    return float((a + b) / 2.0)


def expected_calibration_error(y_true: np.ndarray, y_scores: np.ndarray,
                                n_bins: int = 10) -> float:
    """Population-weighted mean |mean predicted probability - observed frequency|.

    This is the binned gap between the calibration curve and the diagonal, on
    the positive-class probability -- the same quantity
    `calibration_and_brier` plots, reduced to one number. (Note it is not the
    max(p, 1-p) "confidence vs accuracy" variant of ECE used in the
    multiclass literature; for a binary score the curve-based form is the one
    that matches what the dashboard draws.)

    Reported alongside Brier because they answer different questions. Brier
    decomposes into calibration *and* refinement, so a model that is confidently
    wrong on a few cases is penalised by it no matter how well its probabilities
    are scaled. ECE isolates the part temperature scaling can actually move.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_scores = np.asarray(y_scores, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_scores, edges[1:-1])
    total = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y_scores[mask].mean() - y_true[mask].mean())
    return float(total)
