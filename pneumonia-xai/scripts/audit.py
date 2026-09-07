"""CLI: shortcut audit (A-1) over a test-set sample + failure gallery (A-3).

Usage: python scripts/audit.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_cxr.config import DataConfig, ModelConfig, metrics_path
from xai_cxr.data import load_image, read_manifest
from xai_cxr.models.baseline import XAIModel, load_trained
from xai_cxr.audits.shortcut_audit import run_shortcut_audit
from xai_cxr.audits.failure_gallery import build_failure_gallery

AUDIT_SAMPLE_N = 30


def _mean(rows, path):
    vals = [row for row in (_get(r, path) for r in rows) if row is not None]
    return float(np.mean(vals)) if vals else None


def _get(d, path):
    for key in path.split('.'):
        d = d.get(key)
        if d is None:
            return None
    return d


if __name__ == '__main__':
    data_cfg = DataConfig.load()
    model_cfg = ModelConfig.load()

    model = load_trained(model_cfg.model_path)
    xai_model = XAIModel(model, model_cfg.last_conv_layer)

    rows = read_manifest('test')
    rng = np.random.RandomState(99)
    idx = rng.choice(len(rows), size=min(AUDIT_SAMPLE_N, len(rows)), replace=False)
    sample = [rows[i] for i in idx]

    print(f'Running shortcut audit over {len(sample)} test images...')
    audit_rows = []
    for i, (relpath, label) in enumerate(sample):
        img = load_image(os.path.join(data_cfg.dataset_dir, relpath), data_cfg.img_size)
        result = run_shortcut_audit(xai_model, img, method='gradcam', img_size=data_cfg.img_size)
        result['relpath'] = relpath
        result['true_label'] = label
        audit_rows.append(result)
        print(f'  {i + 1}/{len(sample)}', end='\r')
    print()

    summary = {
        'n_samples': len(audit_rows),
        'border_masking_mean_delta': _mean(audit_rows, 'border_masking.delta'),
        'laterality_masking_mean_delta': _mean(audit_rows, 'laterality_marker_masking.delta'),
        'blurred_source_mean_delta': _mean(audit_rows, 'blurred_source_probe.delta'),
        'blurred_source_still_confident_rate': float(np.mean(
            [r['blurred_source_probe']['still_confident_same_class'] for r in audit_rows])),
        'attribution_in_lung_field_ratio_mean': _mean(audit_rows, 'attribution_in_lung_field_ratio'),
        'lung_field_mask_is_approximate': True,
    }

    # Merge into metrics.json under "audits" rather than overwriting the whole file.
    mpath = metrics_path()
    metrics = {}
    if os.path.exists(mpath):
        with open(mpath, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    metrics['audits'] = {
        'shortcut_audit_summary': summary,
        'shortcut_audit_rows': audit_rows,
        'subgroup_performance': {
            'status': 'not_available',
            'reason': 'Kermany ships no sex/age metadata in this repo -- deferred (A-2)',
        },
    }
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'Shortcut audit summary written into {mpath}')
    print(json.dumps(summary, indent=2))

    print('\nBuilding failure gallery (A-3)...')
    docs_path = build_failure_gallery(xai_model, data_cfg, split='test', method='gradcam', n=10)
    print(f'Failure gallery written -> {docs_path}')
