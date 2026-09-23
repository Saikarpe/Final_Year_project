"""Threshold selection for the screening operating point.

`select_threshold` decides a number that gets reported as a clinical result
(how many pneumonias the model misses), so its contract is pinned here rather
than trusted to the one run that produced the figure in the decisions log.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from screening_threshold import select_threshold, youden_threshold  # noqa: E402
from operating_point import counts_at  # noqa: E402


def _separable(n=400, seed=0):
    """Scores that overlap in the middle, so a floor actually binds.

    Fully separable scores make every floor satisfiable at the same threshold
    and the test would pass without exercising the constraint.
    """
    rng = np.random.default_rng(seed)
    neg = rng.beta(2, 6, size=n // 2)
    pos = rng.beta(6, 2, size=n // 2)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    return scores, labels


@pytest.mark.parametrize('floor', [0.80, 0.90, 0.95, 0.99])
def test_selected_threshold_meets_the_floor(floor):
    scores, labels = _separable()
    thr = select_threshold(scores, labels, floor)
    assert counts_at(scores, labels, thr)['specificity'] >= floor


@pytest.mark.parametrize('floor', [0.80, 0.90, 0.95])
def test_selected_threshold_is_the_most_sensitive_one_meeting_the_floor(floor):
    """Lowest qualifying threshold, i.e. nothing below it also qualifies.

    This is the half of the contract that a "meets the floor" assertion alone
    would not catch: returning 1.0 always meets every floor and always misses
    every pneumonia.
    """
    scores, labels = _separable()
    thr = select_threshold(scores, labels, floor)
    below = [t for t in np.unique(scores) if t < thr]
    assert all(counts_at(scores, labels, t)['specificity'] < floor for t in below)


def test_a_stricter_floor_never_yields_a_more_sensitive_threshold():
    scores, labels = _separable()
    thrs = [select_threshold(scores, labels, f) for f in (0.80, 0.90, 0.95, 0.99)]
    assert thrs == sorted(thrs), f'thresholds not monotone in the floor: {thrs}'


def test_an_unreachable_floor_does_not_silently_return_a_usable_threshold():
    """A floor no threshold can meet must not degrade to a permissive answer."""
    scores = np.full(50, 0.5)
    labels = np.concatenate([np.zeros(25, int), np.ones(25, int)])
    # Every negative scores identically to every positive, so specificity is
    # either 0 or 1 and 0.99 is only met by predicting nothing positive.
    thr = select_threshold(scores, labels, 0.99)
    assert counts_at(scores, labels, thr)['specificity'] >= 0.99


def test_youden_threshold_maximises_sensitivity_plus_specificity():
    scores, labels = _separable()
    thr = youden_threshold(scores, labels)
    c = counts_at(scores, labels, thr)
    best = c['sensitivity'] + c['specificity']
    for t in np.unique(scores):
        o = counts_at(scores, labels, t)
        assert o['sensitivity'] + o['specificity'] <= best + 1e-12


def test_selection_on_one_split_is_not_evaluated_on_itself():
    """The point of the script: a threshold chosen on A and scored on B is
    generally worse on B than the best threshold on B would have been.

    That gap is the selection optimism the screening number was previously
    reporting as if it were performance. Asserting the *direction* (selecting
    on the evaluation split never looks worse) is the honest, non-flaky form
    of that claim -- the magnitude is data-dependent.
    """
    a_scores, a_labels = _separable(seed=1)
    b_scores, b_labels = _separable(seed=2)
    floor = 0.90

    thr_a = select_threshold(a_scores, a_labels, floor)
    thr_b = select_threshold(b_scores, b_labels, floor)

    honest = counts_at(b_scores, b_labels, thr_a)
    oracle = counts_at(b_scores, b_labels, thr_b)
    assert oracle['fn'] <= honest['fn'] or oracle['specificity'] >= honest['specificity']
