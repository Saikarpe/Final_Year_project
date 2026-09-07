"""CLI: derive patient-level splits and write configs/splits/*.txt (D-7, G-7, U-1).

Usage: python scripts/build_splits.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_cxr.config import DataConfig
from xai_cxr.data import build_splits, write_manifests, scan_dataset

if __name__ == '__main__':
    cfg = DataConfig.load()
    rows = scan_dataset(cfg.dataset_dir)
    print(f'Scanned {len(rows)} images across train/val/test.')

    splits = build_splits(cfg)
    write_manifests(splits)

    for name, items in splits.items():
        n_normal = sum(1 for _, l in items if l == 'NORMAL')
        n_pneu = sum(1 for _, l in items if l == 'PNEUMONIA')
        print(f'{name:12s}: {len(items):5d} images  (NORMAL={n_normal}, PNEUMONIA={n_pneu})')
    print(f'\nManifests written to configs/splits/')
