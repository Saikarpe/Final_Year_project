"""Recompute only the E-8 sanity-check block in models/metrics.json.

`evaluate.py` rewrites metrics.json wholesale and costs ~40 min, almost all of
it in the faithfulness sweep. When only the sanity checks change -- as when
the BatchNorm randomization bug was fixed -- rerunning the whole pipeline
would burn that time to reproduce numbers that are already correct.

This recomputes the cascading randomization (and, with --label, the label
randomization proxy) for the same image evaluate.py used, and writes just that
sub-tree back, stamping `recomputed_at` so the block is never silently
mistaken for part of the original run.

Usage:  python scripts/refresh_sanity_checks.py [--label] [--write]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_cxr.config import DataConfig, ModelConfig, metrics_path  # noqa: E402
from xai_cxr.data import load_image  # noqa: E402
from xai_cxr.evaluation.sanity_checks import (  # noqa: E402
    cascading_randomization_test, label_randomization_test,
)
from xai_cxr.models.baseline import XAIModel, load_trained  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='write back to metrics.json')
    ap.add_argument('--label', action='store_true',
                    help='also redo the label-randomization proxy (slower; it is '
                         'unaffected by the BatchNorm fix, so off by default)')
    args = ap.parse_args()

    path = metrics_path()
    with open(path, encoding='utf-8') as f:
        metrics = json.load(f)

    data_cfg, model_cfg = DataConfig.load(), ModelConfig.load()
    model = load_trained(model_cfg.model_path)
    xai = XAIModel(model, model_cfg.last_conv_layer)

    sanity = metrics['sanity_checks']
    relpath = sanity['image']
    img = load_image(os.path.join(data_cfg.dataset_dir, relpath), data_cfg.img_size)
    print(f'image: {relpath}')

    for method, block in sanity['methods'].items():
        t0 = time.time()
        block['cascading_randomization'] = cascading_randomization_test(
            xai, img, method, img_size=data_cfg.img_size)
        print(f'  {method}: cascade done ({time.time() - t0:.0f}s)')
        for st in block['cascading_randomization']['stages']:
            v = st['similarity_to_original']
            print(f"     {st['stage']:<24} {'undefined' if v is None else f'{v:+.4f}'}")

        if args.label:
            from xai_cxr.data import read_manifest
            rng = np.random.RandomState(11)
            rows = read_manifest('train')
            idx = rng.choice(len(rows), size=block['label_randomization']['n_train_images'],
                             replace=False)
            images = np.array([load_image(os.path.join(data_cfg.dataset_dir, rows[i][0]),
                                          data_cfg.img_size) for i in idx])
            t0 = time.time()
            block['label_randomization'] = label_randomization_test(
                xai, img, method, images, img_size=data_cfg.img_size,
                epochs=block['label_randomization']['epochs'])
            print(f'  {method}: label randomization done ({time.time() - t0:.0f}s)')

    sanity['recomputed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    sanity['recomputed_note'] = (
        'Only the sanity_checks block was recomputed, after the BatchNorm '
        'randomization fix; every other field is from the run stamped in '
        'generated_at. Cascading randomization previously reported undefined for '
        'every stage from head_bn down, because zeroing a BatchNorm gamma makes '
        'the layer emit a constant and the heatmap degenerate.'
        + ('' if args.label else ' The label-randomization proxy is unchanged by '
                                 'that fix and was not rerun.'))

    if args.write:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(metrics, f, indent=2)
            f.write('\n')
        print(f'\nwrote sanity_checks -> {path}')
    else:
        print('\n(dry run; pass --write to update metrics.json)')


if __name__ == '__main__':
    main()
