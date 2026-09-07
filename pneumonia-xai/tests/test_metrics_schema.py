"""G-7: metrics.json schema. This is the contract the dashboard (§8) and
scripts/evaluate.py both rely on -- if either drifts from it, this test
should catch it before the dashboard silently renders nothing."""
import json
import os

import pytest

from xai_cxr.config import metrics_path

REQUIRED_TOP_LEVEL = {
    'generated_at', 'model_path', 'model_hash',
    'classification', 'uncertainty', 'explanation', 'sanity_checks', 'runtime',
    'localisation', 'scope_note',
}
REQUIRED_CLASSIFICATION = {'auroc', 'sensitivity_specificity', 'calibration',
                            'decision_threshold', 'n_test', 'external_validation'}
REQUIRED_AUROC = {'auroc', 'ci_lower', 'ci_upper', 'ci_level'}

path = metrics_path()


@pytest.mark.skipif(not os.path.exists(path), reason='metrics.json not generated -- run scripts/evaluate.py')
def test_metrics_schema():
    with open(path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    missing = REQUIRED_TOP_LEVEL - set(metrics)
    assert not missing, f'metrics.json missing top-level keys: {missing}'

    missing_cls = REQUIRED_CLASSIFICATION - set(metrics['classification'])
    assert not missing_cls, f'classification missing keys: {missing_cls}'

    missing_auroc = REQUIRED_AUROC - set(metrics['classification']['auroc'])
    assert not missing_auroc, f'auroc missing keys: {missing_auroc}'

    assert isinstance(metrics['explanation'], dict) and len(metrics['explanation']) > 0
    assert isinstance(metrics['runtime'], dict) and len(metrics['runtime']) > 0
