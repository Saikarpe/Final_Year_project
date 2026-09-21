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
    #: Augmentation strengths (D-8). Every entry is a magnitude in the units
    #: the matching keras.layers.Random* layer expects; 0 disables that
    #: transform. `horizontal_flip` defaults to False on purpose -- mirroring
    #: a chest radiograph moves the heart to the right hemithorax and flips
    #: the laterality marker, which is exactly the shortcut the A-1 audit
    #: measures. Enabling it is a deliberate choice, not a default.
    augment: dict = field(default_factory=lambda: {
        'horizontal_flip': False,
        'rotation': 0.05,
        'zoom': 0.10,
        'translation': 0.05,
        'contrast': 0.15,
        'brightness': 0.10,
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
        cfg.augment = {**cfg.augment, **(raw.get('augment') or {})}
        return cfg


@dataclass
class ModelConfig:
    # -- architecture -------------------------------------------------------
    #: A key of xai_cxr.models.backbones.BACKBONES. Changing this is the whole
    #: architecture change: nothing downstream names a backbone.
    backbone: str = 'densenet121'
    #: Grad-CAM target: 'auto' (final feature map), 'hires' (stride-16 stage,
    #: 2x finer grid), or an explicit layer name.
    cam_layer: str = 'auto'
    #: Pooling head: 'attention' (gated attention pooling), 'avgmax'
    #: (concat of global average + global max), or 'gap' (the plain global
    #: average the first version used). See xai_cxr.models.heads.
    head: str = 'attention'
    attention_units: int = 128
    #: A 3-parameter learnable window/level in front of the backbone,
    #: identity-initialised. See heads.LearnableWindow.
    window_layer: bool = True
    #: SpatialDropout2D rate on the backbone feature map (channel dropout).
    spatial_dropout: float = 0.1
    head_batchnorm: bool = True
    dense_units: int = 256
    dropout: float = 0.4
    head_dropout: float = 0.2

    # -- stage 1: head warmup (backbone frozen) -----------------------------
    learning_rate: float = 1e-3
    epochs: int = 8
    warmup_epochs: float = 0.5

    # -- stage 2: fine-tune (top of the backbone unfrozen) ------------------
    #: 0 skips stage 2 entirely and reproduces the old frozen-backbone model.
    finetune_epochs: int = 12
    finetune_learning_rate: float = 1e-5
    #: Fraction of backbone layers (counted from the output) made trainable.
    unfreeze_fraction: float = 0.35
    #: Keep BatchNormalization frozen while fine-tuning -- see
    #: xai_cxr.models.backbones.set_finetune_trainable for why.
    freeze_batchnorm: bool = True

    # -- optimisation -------------------------------------------------------
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    #: > 0 switches BCE to focal BCE with this gamma.
    focal_gamma: float = 0.0
    early_stopping_patience: int = 5

    # -- operating point ----------------------------------------------------
    #: 'youden' weights a missed pneumonia and a false alarm equally;
    #: 'min_sensitivity' instead takes the lowest-FPR threshold that still
    #: reaches `min_sensitivity`.
    threshold_policy: str = 'youden'
    min_sensitivity: float = 0.95

    # -- paths --------------------------------------------------------------
    model_path: str = os.path.join(MODELS_DIR, 'pneumonia_model.h5')
    legacy_model_path: str = os.path.join(MODELS_DIR, 'pneumonia_model_v1_legacy.h5')

    #: Mirrors DataConfig.img_size so the model layer never re-reads data.yaml
    #: and the two can never disagree. Populated by load().
    input_size: tuple[int, int] = (224, 224)

    @property
    def architecture(self) -> str:
        """Back-compat alias used by docs/tracking output."""
        return self.backbone

    @property
    def last_conv_layer(self) -> str:
        """Back-compat alias: `XAIModel(model, cfg.last_conv_layer)` predates
        the rename to `cam_layer` and is still how every script reads it."""
        return self.cam_layer

    @classmethod
    def load(cls) -> "ModelConfig":
        raw = _load_yaml('model.yaml')
        cfg = cls()
        for key in ('backbone', 'cam_layer', 'head', 'attention_units', 'window_layer',
                    'spatial_dropout', 'head_batchnorm', 'dense_units', 'dropout', 'head_dropout',
                    'learning_rate', 'epochs', 'warmup_epochs',
                    'finetune_epochs', 'finetune_learning_rate', 'unfreeze_fraction',
                    'freeze_batchnorm', 'weight_decay', 'label_smoothing', 'focal_gamma',
                    'early_stopping_patience', 'threshold_policy', 'min_sensitivity'):
            if key in raw:
                setattr(cfg, key, raw[key])
        # Pre-M-1b configs said `architecture: vgg16_baseline` /
        # `last_conv_layer: block5_conv3`; honour both rather than silently
        # training a different model than an old config asked for.
        if 'backbone' not in raw and 'architecture' in raw:
            cfg.backbone = str(raw['architecture']).replace('_baseline', '')
        if 'cam_layer' not in raw and 'last_conv_layer' in raw:
            cfg.cam_layer = raw['last_conv_layer']
        cfg.input_size = DataConfig.load().img_size
        return cfg


@dataclass
class ExplainConfig:
    default_method: str = 'gradcam'
    occlusion_patch: int = 32
    occlusion_stride: int = 16
    ig_steps: int = 16
    scorecam_batch: int = 16          # superseded by inference_batch; kept for old configs
    scorecam_max_channels: int = 64
    shap_background_size: int = 5
    #: Forward/backward batch size for every perturbation-style method
    #: (Occlusion, Score-CAM, and IG's interpolation chunks). This is a pure
    #: VRAM constraint and never changes a method's result -- only its peak
    #: memory. Occlusion previously hardcoded 32 and was not config-driven at
    #: all, which OOM'd immediately at 320px.
    inference_batch: int = 4

    @classmethod
    def load(cls) -> "ExplainConfig":
        raw = _load_yaml('explain.yaml')
        cfg = cls()
        for key in ('default_method', 'occlusion_patch', 'occlusion_stride',
                    'ig_steps', 'scorecam_batch', 'scorecam_max_channels',
                    'shap_background_size', 'inference_batch'):
            if key in raw:
                setattr(cfg, key, raw[key])
        return cfg


def metrics_path() -> str:
    return os.path.join(MODELS_DIR, 'metrics.json')
