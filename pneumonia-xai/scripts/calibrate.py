"""Fit a temperature on the calibration split and report what it buys (E-5).

M-1b discriminates better than the M-1 baseline (AUROC 0.9947 vs 0.9941) but
reports worse-calibrated probabilities (Brier 0.063 vs 0.047). Temperature
scaling is the one-parameter fix. This script fits T on the *calibration*
split -- never on test -- and then reports test-set calibration before and
after, so the improvement quoted is an honest held-out number.

It writes the result into models/metrics.json under
`classification.calibration.temperature_scaling`. It deliberately does NOT
overwrite the reported `brier_score` or the decision threshold: the shipped
operating point is still defined on the raw probabilities, and silently
restating the headline number on a different scale is exactly the kind of
thing that makes two runs incomparable.

Usage:  python scripts/calibrate.py [--write]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from operating_point import REPO, load_scores  # noqa: E402

from xai_cxr.evaluation import (  # noqa: E402
    apply_temperature, calibration_and_brier, expected_calibration_error, fit_temperature,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='write the fitted temperature into models/metrics.json')
    args = ap.parse_args()

    calib_scores, calib_labels, _ = load_scores('calibration')
    test_scores, test_labels, _ = load_scores('test')

    temperature = fit_temperature(calib_labels, calib_scores)

    def report(scores, labels):
        return {
            'brier_score': calibration_and_brier(labels, scores)['brier_score'],
            'ece': expected_calibration_error(labels, scores),
        }

    before = report(test_scores, test_labels)
    after = report(apply_temperature(test_scores, temperature), test_labels)

    print('=' * 72)
    print('TEMPERATURE SCALING (fitted on calibration, evaluated on test)')
    print('=' * 72)
    print(f'  n_calibration        {len(calib_labels)}')
    print(f'  n_test               {len(test_labels)}')
    print(f'  fitted temperature   {temperature:.4f}'
          f'   ({"softens" if temperature > 1 else "sharpens"} the probabilities)')
    print()
    print(f'{"metric":<22}{"before":>12}{"after":>12}{"change":>14}')
    for name in ('brier_score', 'ece'):
        b, a = before[name], after[name]
        pct = (a - b) / b * 100 if b else float('nan')
        print(f'{name:<22}{b:>12.4f}{a:>12.4f}{pct:>13.1f}%')

    print()
    print('AUROC is unchanged by construction: temperature scaling is strictly')
    print('monotonic in the score, so it moves probabilities without reordering')
    print('any case. The decision threshold is NOT restated here -- it is defined')
    print('on the raw probabilities and stays that way.')

    if args.write:
        path = os.path.join(REPO, 'models', 'metrics.json')
        with open(path, encoding='utf-8') as f:
            metrics = json.load(f)
        metrics['classification']['calibration']['temperature_scaling'] = {
            'temperature': temperature,
            'fitted_on': 'calibration',
            'n_calibration': int(len(calib_labels)),
            'brier_before': before['brier_score'],
            'brier_after': after['brier_score'],
            'ece_before': before['ece'],
            'ece_after': after['ece'],
            'note': ('Fitted on the calibration split, evaluated on test. Strictly '
                     'monotonic, so AUROC and every ranking metric are unchanged. The '
                     'reported brier_score and decision_threshold above are on the RAW '
                     'probability scale and are deliberately left as they were.'),
        }
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(metrics, f, indent=2)
            f.write('\n')
        print(f'\nwrote temperature_scaling -> {path}')


if __name__ == '__main__':
    main()
