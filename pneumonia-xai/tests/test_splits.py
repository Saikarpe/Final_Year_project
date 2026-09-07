"""G-7: splits disjoint."""
import os
import pytest

from xai_cxr.config import SPLITS_DIR, DataConfig
from xai_cxr.data import read_manifest, manifest_patient_ids

SPLIT_NAMES = ['train', 'val', 'calibration', 'test']

manifests_exist = all(os.path.exists(os.path.join(SPLITS_DIR, f'{n}.txt')) for n in SPLIT_NAMES)
skip_reason = 'configs/splits/*.txt not generated -- run scripts/build_splits.py'


@pytest.mark.skipif(not manifests_exist, reason=skip_reason)
def test_splits_are_patient_disjoint():
    ids = {name: manifest_patient_ids(name) for name in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            overlap = ids[a] & ids[b]
            assert not overlap, f'{a} and {b} share patient ids: {sorted(overlap)[:5]}...'


@pytest.mark.skipif(not manifests_exist, reason=skip_reason)
def test_splits_are_non_empty_and_disjoint_rows():
    seen = set()
    for name in SPLIT_NAMES:
        rows = read_manifest(name)
        assert len(rows) > 0, f'{name} split is empty'
        relpaths = {r for r, _ in rows}
        assert not (relpaths & seen), f'{name} shares image paths with an earlier split'
        seen |= relpaths


@pytest.mark.skipif(not manifests_exist, reason=skip_reason)
def test_split_ratios_are_approximately_correct():
    cfg = DataConfig.load()
    counts = {name: len(read_manifest(name)) for name in SPLIT_NAMES}
    total = sum(counts.values())
    for name, ratio in cfg.split_ratios.items():
        assert abs(counts[name] / total - ratio) < 0.05, f'{name} split ratio drifted: {counts[name]}/{total}'
