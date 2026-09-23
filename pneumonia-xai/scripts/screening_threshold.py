"""Screening operating point, selected on calibration and reported on test.

`operating_point.py` sweeps thresholds on the test split. That is the right
tool for its question -- "is the sensitivity gap against the VGG16 baseline a
threshold artefact?" -- because it asks what the model *could* do, and both
models get the same treatment.

It is the wrong tool for a different question: "if we adopted a screening
operating point, what sensitivity would we actually get?" A threshold chosen
on test and then reported on test is selected on the same 582 images it is
scored on, so it reports the best of ~582 candidate thresholds rather than an
unbiased estimate. HANDOFF §9 flags this explicitly:

    Caveat: those sweep rows are computed on the test split, so they show the
    achievable trade-off, not a held-out estimate of it. Re-tune on
    calibration.

This script does that. The threshold is chosen on the 563-image calibration
split -- which the model never trained on and which is disjoint by patient from
test -- and then applied, unchanged, to test. The difference between the two is
the selection optimism, and it is reported rather than hidden.

The calibration split is already spent on conformal prediction (U-2), and
reusing it here is deliberate: both uses only need held-out scores, neither
fits parameters the other depends on, and spending val instead would reuse the
split that early stopping already selected on.

Usage: python scripts/screening_threshold.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from operating_point import REPO, counts_at, load_scores

#: Specificity floors to report. 0.90 is the screening-style point HANDOFF §9
#: highlights; 0.95 is the conservative one configs/model.yaml names via
#: min_sensitivity. Both are reported so the trade is visible, not asserted.
SPECIFICITY_FLOORS = (0.90, 0.95)


def select_threshold(scores, labels, floor):
    """Lowest threshold on `scores` whose specificity still clears `floor`.

    Lowest, because among thresholds meeting the constraint the lowest is the
    most sensitive -- the objective is "miss as few pneumonias as possible
    subject to a specificity floor". Candidates are the observed scores
    themselves plus the two endpoints, so every distinct split of the data is
    considered exactly once.
    """
    cand = np.unique(np.concatenate([[0.0], np.sort(scores), [1.0]]))
    ok = [t for t in cand if counts_at(scores, labels, t)['specificity'] >= floor]
    return float(min(ok)) if ok else 1.0


def youden_threshold(scores, labels):
    cand = np.unique(np.concatenate([[0.0], np.sort(scores), [1.0]]))
    best, best_j = 0.5, -np.inf
    for t in cand:
        c = counts_at(scores, labels, t)
        j = c['sensitivity'] + c['specificity'] - 1.0
        if j > best_j:
            best, best_j = float(t), j
    return best


def main():
    cal_scores, cal_labels, _ = load_scores('calibration')
    test_scores, test_labels, _ = load_scores('test')

    n_pos = int((test_labels == 1).sum())
    n_neg = int((test_labels == 0).sum())

    print('=' * 78)
    print('SCREENING THRESHOLD  --  selected on calibration, reported on test')
    print('=' * 78)
    print(f'calibration: {len(cal_scores)} images '
          f'({int((cal_labels == 1).sum())} pneumonia / {int((cal_labels == 0).sum())} normal)')
    print(f'test:        {len(test_scores)} images ({n_pos} pneumonia / {n_neg} normal)')
    print()

    header = (f'{"objective":<34}{"thr":>8}{"sens":>8}{"spec":>8}{"FN":>5}{"FP":>5}'
              f'{"  (selected on)":<16}')
    print(header)
    print('-' * 78)

    rows = {}

    def report(name, thr, selected_on):
        c = counts_at(test_scores, test_labels, thr)
        print(f'{name:<34}{thr:>8.4f}{c["sensitivity"]:>8.1%}{c["specificity"]:>8.1%}'
              f'{c["fn"]:>5}{c["fp"]:>5}  {selected_on:<14}')
        return c

    # The pre-specified headline, for reference: chosen on val before test was
    # touched, so it is already a legitimate held-out number.
    metrics = json.load(open(os.path.join(REPO, 'models', 'metrics.json'), encoding='utf-8'))
    reported_thr = metrics['classification']['decision_threshold']
    report('Youden J (reported headline)', reported_thr, 'val')
    print()

    for floor in SPECIFICITY_FLOORS:
        thr_cal = select_threshold(cal_scores, cal_labels, floor)
        honest = report(f'max sens, spec >= {floor:.0%}', thr_cal, 'calibration')

        # The same objective selected on test: the optimistic number, shown
        # only so the gap between the two is explicit.
        thr_test = select_threshold(test_scores, test_labels, floor)
        oracle = report(f'  (same, selected on test)', thr_test, 'test [optimistic]')

        rows[floor] = {
            'specificity_floor': floor,
            'threshold_selected_on_calibration': thr_cal,
            'test_sensitivity': honest['sensitivity'],
            'test_specificity': honest['specificity'],
            'test_fn': honest['fn'],
            'test_fp': honest['fp'],
            'optimism_fn': oracle['fn'] - honest['fn'],
            'threshold_selected_on_test': thr_test,
            'test_selected_sensitivity': oracle['sensitivity'],
            'test_selected_specificity': oracle['specificity'],
        }
        print()

    print('=' * 78)
    print('READING')
    print('=' * 78)
    for floor, r in rows.items():
        gap = -r['optimism_fn']
        print(f'At a {floor:.0%} specificity floor, a threshold fixed on calibration '
              f'({r["threshold_selected_on_calibration"]:.4f}) misses')
        print(f'  {r["test_fn"]} of {n_pos} pneumonias on test at '
              f'{r["test_sensitivity"]:.1%} sensitivity / {r["test_specificity"]:.1%} '
              f'specificity.')
        if gap == 0:
            print('  Selecting on test instead would have found the same confusion matrix, '
                  'so')
            print('  there is no selection optimism to discount at this floor.')
        else:
            print(f'  Selecting on test instead would have reported {gap:+d} fewer misses -- '
                  f'that')
            print('  difference is selection optimism, not model performance.')
        print()

    print('This is the number to quote for a screening operating point: the threshold')
    print('was fixed on data the test split never saw.')

    out = os.path.join(REPO, 'runs', 'screening_threshold.json')
    payload = {
        'selected_on': 'calibration',
        'reported_on': 'test',
        'n_calibration': len(cal_scores),
        'n_test': len(test_scores),
        'model_hash': metrics['model_hash'],
        'reported_youden_threshold': reported_thr,
        'rows': list(rows.values()),
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nwritten -> {out}')


if __name__ == '__main__':
    main()
