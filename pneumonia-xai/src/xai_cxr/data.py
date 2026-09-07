"""Patient-level splitting and manifest-driven data loading (D-7, G-7, U-1).

The Kermany dataset ships as train/val/test folders with no patient-ID
column. Filenames do carry recoverable identity though:

  PNEUMONIA:  person<ID>_bacteria_<n>.jpeg  /  person<ID>_virus_<n>.jpeg
  NORMAL:     IM-<ID>-<n>.jpeg  /  NORMAL2-IM-<ID>-<n>.jpeg

We use those IDs to group images by patient/study before splitting, so no
patient appears in more than one split (the spec's D-7/G-7 acceptance
criterion). This is a best-effort derivation, not ground truth — see
docs/dataset_datasheet.md for the caveat. Filenames that don't match either
pattern fall back to being their own singleton group (documented, not
silently merged with anything else).
"""
from __future__ import annotations

import os
import re
import numpy as np
import tensorflow as tf

from .config import DataConfig, SPLITS_DIR, DATASET_DIR

LABELS = ('NORMAL', 'PNEUMONIA')
SOURCE_SPLITS = ('train', 'val', 'test')  # the original Kermany folders, pooled then re-split

_PNEUMONIA_RE = re.compile(r'^person(\d+)_(?:bacteria|virus)_\d+$', re.IGNORECASE)
_NORMAL_RE = re.compile(r'^(?:normal2-)?im-(\d+)-\d+$', re.IGNORECASE)


def patient_id_for(filename: str, label: str) -> str:
    stem = os.path.splitext(filename)[0]
    if label == 'PNEUMONIA':
        m = _PNEUMONIA_RE.match(stem)
        if m:
            return f'pneumonia_{m.group(1)}'
    else:
        m = _NORMAL_RE.match(stem)
        if m:
            return f'normal_{m.group(1)}'
    # No recognisable pattern: treat as its own group rather than guessing.
    return f'singleton_{label.lower()}_{stem}'


def scan_dataset(dataset_dir: str = DATASET_DIR) -> list[tuple[str, str, str]]:
    """Pool every image across the original train/val/test folders.

    Returns (relpath, label, patient_id) tuples. relpath is relative to
    dataset_dir and stable regardless of which original folder it came from.
    """
    rows: list[tuple[str, str, str]] = []
    for source in SOURCE_SPLITS:
        for label in LABELS:
            folder = os.path.join(dataset_dir, source, label)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                relpath = f'{source}/{label}/{fname}'
                rows.append((relpath, label, patient_id_for(fname, label)))
    return rows


def _split_groups(patient_ids: list[str], ratios: dict, seed: int) -> dict[str, set]:
    groups = sorted(set(patient_ids))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]

    names = list(ratios.keys())
    n = len(groups)
    counts = {name: max(1, round(ratios[name] * n)) for name in names}
    diff = n - sum(counts.values())
    counts[names[0]] += diff  # absorb rounding drift into the largest split

    assign: dict[str, set] = {name: set() for name in names}
    idx = 0
    for name in names:
        c = max(0, counts[name])
        assign[name] = set(groups[idx:idx + c])
        idx += c
    return assign


def build_splits(cfg: DataConfig | None = None) -> dict[str, list[tuple[str, str]]]:
    """Patient-grouped, class-balanced split. Splits per class independently
    (by patient group) so the ~3:1 PNEUMONIA:NORMAL ratio is preserved in
    every split, not just overall."""
    cfg = cfg or DataConfig.load()
    rows = scan_dataset(cfg.dataset_dir)

    result: dict[str, list[tuple[str, str]]] = {name: [] for name in cfg.split_ratios}
    for label in LABELS:
        label_rows = [(r, p) for r, l, p in rows if l == label]
        patient_ids = [p for _, p in label_rows]
        assign = _split_groups(patient_ids, cfg.split_ratios, cfg.seed)
        for relpath, pid in label_rows:
            for name, members in assign.items():
                if pid in members:
                    result[name].append((relpath, label))
                    break
    for name in result:
        result[name].sort()
    return result


def write_manifests(splits: dict[str, list[tuple[str, str]]], out_dir: str = SPLITS_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(out_dir, f'{name}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for relpath, label in rows:
                f.write(f'{relpath}\t{label}\n')


def read_manifest(name: str, splits_dir: str = SPLITS_DIR) -> list[tuple[str, str]]:
    path = os.path.join(splits_dir, f'{name}.txt')
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            relpath, label = line.split('\t')
            rows.append((relpath, label))
    return rows


def manifest_patient_ids(name: str, splits_dir: str = SPLITS_DIR) -> set[str]:
    ids = set()
    for relpath, label in read_manifest(name, splits_dir):
        fname = relpath.rsplit('/', 1)[-1]
        ids.add(patient_id_for(fname, label))
    return ids


def load_image(path: str, img_size: tuple[int, int]) -> np.ndarray:
    """Canonical image loader. Returns raw RGB float32 in [0, 255] — the
    model itself applies VGG16 preprocessing (see models/baseline.py), so
    every caller (training, evaluation, explanation methods, the app) reads
    images the same way, once."""
    img = tf.keras.utils.load_img(path, target_size=img_size)
    return tf.keras.utils.img_to_array(img).astype('float32')


def load_split_dataset(name: str, cfg: DataConfig | None = None,
                        shuffle: bool = False, augment: bool = False) -> tuple[tf.data.Dataset, np.ndarray]:
    """Returns (tf.data.Dataset yielding (image[0,255], label), labels array)."""
    cfg = cfg or DataConfig.load()
    rows = read_manifest(name)
    paths = [os.path.join(cfg.dataset_dir, r) for r, _ in rows]
    labels = np.array([1 if lbl == 'PNEUMONIA' else 0 for _, lbl in rows], dtype='float32')

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, cfg.img_size)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=cfg.seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.05),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels


def class_weights(labels: np.ndarray) -> dict[int, float]:
    n = len(labels)
    n_pos = float(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}
