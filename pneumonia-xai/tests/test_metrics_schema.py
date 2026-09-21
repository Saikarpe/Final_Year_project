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


@pytest.mark.skipif(not os.path.exists(path), reason='metrics.json not generated -- run scripts/evaluate.py')
def test_metrics_json_is_strict_json():
    """No bare NaN/Infinity in metrics.json.

    Python's json module writes and reads `NaN` happily, so this file can look
    fine from inside the project while being unparseable by JSON.parse, jq,
    Go, and most other readers. The sanity checks used to emit NaN for the
    degenerate case (a rank correlation against a constant heatmap), which is
    exactly the kind of value that leaks into a published artifact unnoticed.
    """
    def reject(constant):
        raise AssertionError(
            f'metrics.json contains bare {constant}, which is not valid JSON. '
            'Emit null instead -- see _rank_similarity in '
            'src/xai_cxr/evaluation/sanity_checks.py.')

    with open(path, 'r', encoding='utf-8') as f:
        json.load(f, parse_constant=reject)


def test_rank_similarity_is_none_for_a_constant_heatmap():
    """A constant map means undefined, not zero.

    Randomizing a BatchNorm layer collapses the heatmap to a constant, and a
    rank correlation needs a non-zero standard deviation in both inputs. The
    honest answer is "undefined"; reporting 0.0 would read as "the explanation
    is unrelated to the original", which is a much stronger claim.
    """
    import numpy as np

    from xai_cxr.evaluation.sanity_checks import _rank_similarity

    varying = np.linspace(0, 1, 64, dtype='float32').reshape(8, 8)
    constant = np.zeros((8, 8), dtype='float32')

    assert _rank_similarity(varying, constant) is None
    assert _rank_similarity(constant, varying) is None
    assert _rank_similarity(constant, constant) is None

    same = _rank_similarity(varying, varying)
    assert same is not None and same == pytest.approx(1.0)
