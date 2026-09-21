"""E-5: temperature scaling. One parameter, so the properties are checkable."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from xai_cxr.evaluation import (
    apply_temperature, expected_calibration_error, fit_temperature,
)


def _miscalibrated(sharpen=2.0, n=20000, seed=3):
    """Labels plus scores miscalibrated by a *known* factor.

    Built the only way that makes "calibrated" mean something: draw p, then
    draw the label from p, so p is calibrated by construction. Multiplying the
    logit by `sharpen` then miscalibrates it by a known amount, and since
    apply_temperature divides the logit by T, the temperature that undoes it
    is exactly `sharpen`.

    (Conditioning the logit on the label instead -- the obvious-looking
    shortcut -- does not do this. It produces scores that are already very
    nearly calibrated, so there is nothing for the fit to correct.)
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.001, 0.999, size=n)
    labels = (rng.uniform(size=n) < p).astype(float)
    logit = np.log(p / (1 - p))
    return labels, 1.0 / (1.0 + np.exp(-logit * sharpen))


def test_temperature_of_one_is_identity():
    _, scores = _miscalibrated()
    assert apply_temperature(scores, 1.0) == pytest.approx(scores, abs=1e-9)


def test_temperature_preserves_ranking_and_auroc():
    """The whole reason this is safe to apply: it cannot change discrimination."""
    labels, scores = _miscalibrated()
    for t in (0.5, 0.9, 1.7, 4.0):
        tempered = apply_temperature(scores, t)
        assert np.array_equal(np.argsort(scores), np.argsort(tempered))
        assert roc_auc_score(labels, tempered) == pytest.approx(
            roc_auc_score(labels, scores), abs=1e-12)


def test_temperature_above_one_softens_below_one_sharpens():
    scores = np.array([0.02, 0.3, 0.5, 0.7, 0.98])
    softer = apply_temperature(scores, 3.0)
    sharper = apply_temperature(scores, 0.4)
    # 0.5 is the fixed point; everything else moves towards or away from it.
    assert np.all(np.abs(softer - 0.5) < np.abs(scores - 0.5) + 1e-12)
    assert np.all(np.abs(sharper - 0.5) > np.abs(scores - 0.5) - 1e-12)


@pytest.mark.parametrize('sharpen', [0.5, 2.0, 3.0])
def test_fit_recovers_a_known_temperature(sharpen):
    """Scores miscalibrated by a known factor should be corrected by it.

    apply_temperature divides the logit by T and the fixture multiplied it by
    `sharpen`, so the temperature that undoes the distortion is `sharpen`
    itself -- including the sharpen < 1 case, where the scores are too timid
    rather than too confident and the fit has to sharpen them back.
    """
    labels, scores = _miscalibrated(sharpen=sharpen)
    assert fit_temperature(labels, scores) == pytest.approx(sharpen, rel=0.08)


def test_fit_improves_calibration_error():
    labels, scores = _miscalibrated(sharpen=2.5)
    t = fit_temperature(labels, scores)
    assert expected_calibration_error(labels, apply_temperature(scores, t)) < \
           expected_calibration_error(labels, scores)


def test_fit_rejects_a_degenerate_split():
    with pytest.raises(ValueError):
        fit_temperature(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        fit_temperature(np.ones(10), np.linspace(0.1, 0.9, 10))


def test_apply_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError):
        apply_temperature(np.array([0.5]), 0.0)


def test_extreme_scores_do_not_produce_infinities():
    """p == 0 or 1 would send the logit to +-inf without clamping."""
    out = apply_temperature(np.array([0.0, 1.0]), 1.3)
    assert np.all(np.isfinite(out))
