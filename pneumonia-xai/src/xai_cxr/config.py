"""Tiny YAML-backed config loading (G-4).

No hyperparameter, path or seed should appear as a literal in code — it
should come from configs/*.yaml instead. This module is the only place that
reads those files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_DIR = os.path.join(REPO_ROOT, 'configs')
SPLITS_DIR = os.path.join(CONFIG_DIR, 'splits')
DATASET_DIR = os.path.join(REPO_ROOT, 'dataset', 'chest_xray')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
RUNS_DIR = os.path.join(REPO_ROOT, 'runs')
DOCS_DIR = os.path.join(REPO_ROOT, 'docs')


def _load_yaml(name: str) -> dict[str, Any]:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    dataset_dir: str = DATASET_DIR
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    split_ratios: dict = field(default_factory=lambda: {
        'train': 0.70, 'val': 0.10, 'calibration': 0.10, 'test': 0.10,
    })

    @classmethod
    def load(cls) -> "DataConfig":
        raw = _load_yaml('data.yaml')
        cfg = cls()
        cfg.dataset_dir = raw.get('dataset_dir', cfg.dataset_dir)
        cfg.img_size = tuple(raw.get('img_size', list(cfg.img_size)))
        cfg.batch_size = raw.get('batch_size', cfg.batch_size)
        cfg.seed = raw.get('seed', cfg.seed)
        cfg.split_ratios = raw.get('split_ratios', cfg.split_ratios)
        return cfg


@dataclass
class ModelConfig:
    architecture: str = 'vgg16_baseline'
    learning_rate: float = 1e-4
    epochs: int = 15
    early_stopping_patience: int = 4
    dense_units: int = 256
    dropout: float = 0.5
    last_conv_layer: str = 'block5_conv3'
    model_path: str = os.path.join(MODELS_DIR, 'pneumonia_model.h5')
    legacy_model_path: str = os.path.join(MODELS_DIR, 'pneumonia_model_v1_legacy.h5')

    @classmethod
    def load(cls) -> "ModelConfig":
        raw = _load_yaml('model.yaml')
        cfg = cls()
        for key in ('architecture', 'learning_rate', 'epochs', 'early_stopping_patience',
                    'dense_units', 'dropout', 'last_conv_layer'):
            if key in raw:
                setattr(cfg, key, raw[key])
        return cfg


@dataclass
class ExplainConfig:
    default_method: str = 'gradcam'
    occlusion_patch: int = 32
    occlusion_stride: int = 16
    ig_steps: int = 16
    scorecam_batch: int = 16
    scorecam_max_channels: int = 64
    shap_background_size: int = 5

    @classmethod
    def load(cls) -> "ExplainConfig":
        raw = _load_yaml('explain.yaml')
        cfg = cls()
        for key in ('default_method', 'occlusion_patch', 'occlusion_stride',
                    'ig_steps', 'scorecam_batch', 'scorecam_max_channels', 'shap_background_size'):
            if key in raw:
                setattr(cfg, key, raw[key])
        return cfg


def metrics_path() -> str:
    return os.path.join(MODELS_DIR, 'metrics.json')
