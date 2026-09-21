"""Backbone registry (M-1b): the architecture is a config value, not a code literal.

Why this module exists
----------------------
The original model hardcoded VGG16 (2014) end-to-end: `build_model` called
`tf.keras.applications.VGG16`, `XAIModel` did `model.get_layer('vgg16')`, and
the Grad-CAM layer name `block5_conv3` was written into configs/model.yaml as
if every network had one. That made "try a better backbone" a multi-file edit
and an explanation-method rewrite.

VGG16 is a poor 2025 default for this task specifically:

  - 14.7M conv parameters with no batch-norm and no residual connections, so
    it needs a small LR and still trains slowly/unstably;
  - ~15 GFLOPs per 224x224 forward pass -- roughly 5x DenseNet121 for worse
    features, which is why a single evaluate.py pass on CPU took hours;
  - its ImageNet features are the weakest of any backbone Keras still ships,
    and the frozen-backbone setup used here never adapted them to radiographs.

The default is now DenseNet121: it is the backbone CheXNet and essentially
every subsequent chest-radiograph paper uses, it is half VGG16's conv
parameter count (7.0M), and dense connectivity + batch-norm make the
two-stage fine-tune in scripts/train.py actually converge.

Spatial-resolution trade-off (documented, not hidden)
-----------------------------------------------------
VGG16's `block5_conv3` sits *before* the last max-pool, so its CAM grid at
224x224 is 14x14. Every modern backbone ends at stride 32, i.e. 7x7 at
224x224. A coarser CAM grid is a real cost for an explainability project, so:

  - `cam_layer: auto` resolves to the last rank-4 activation (the standard,
    CheXNet-equivalent choice), and
  - every backbone also declares `hires_cam_layer`, the stride-16 stage
    output (14x14 at 224x224), selectable from configs/model.yaml when CAM
    resolution matters more than feature depth, and
  - raising `img_size` in configs/data.yaml scales the grid linearly
    (320x320 -> 10x10) for anyone running on a GPU.

See docs/decisions_log.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

K = tf.keras


@dataclass(frozen=True)
class BackboneSpec:
    """Everything the rest of the codebase needs to know about a backbone."""
    name: str
    builder: Callable[..., K.Model]
    preprocess: Callable
    #: stride-16 stage output -- a 2x finer CAM grid than the final block.
    hires_cam_layer: str
    #: human-readable, shown on the dashboard's Model Configuration table.
    description: str
    #: conv-parameter count at 224x224, for the same table.
    approx_params_m: float


#: Every entry takes raw RGB in [0, 255]; the matching `preprocess` does
#: whatever that family actually expects (caffe/torch/tf/identity). Callers
#: never need to know which -- see models.baseline.Preprocess.
BACKBONES: dict[str, BackboneSpec] = {
    'densenet121': BackboneSpec(
        name='densenet121',
        builder=K.applications.DenseNet121,
        preprocess=K.applications.densenet.preprocess_input,
        hires_cam_layer='pool4_relu',
        description='DenseNet121 (CheXNet-standard chest-radiograph backbone)',
        approx_params_m=7.0,
    ),
    'efficientnetv2s': BackboneSpec(
        name='efficientnetv2s',
        builder=K.applications.EfficientNetV2S,
        preprocess=K.applications.efficientnet_v2.preprocess_input,
        hires_cam_layer='block5o_add',
        description='EfficientNetV2-S (strongest accuracy/FLOP trade-off here)',
        approx_params_m=20.3,
    ),
    'efficientnetv2b0': BackboneSpec(
        name='efficientnetv2b0',
        builder=K.applications.EfficientNetV2B0,
        preprocess=K.applications.efficientnet_v2.preprocess_input,
        hires_cam_layer='block5e_add',
        description='EfficientNetV2-B0 (smallest modern option; CPU-friendly)',
        approx_params_m=5.9,
    ),
    'convnext_tiny': BackboneSpec(
        name='convnext_tiny',
        builder=K.applications.ConvNeXtTiny,
        preprocess=K.applications.convnext.preprocess_input,
        hires_cam_layer='convnext_tiny_stage_2_block_8_identity',
        description='ConvNeXt-Tiny (modernised ResNet; strongest features, heaviest)',
        approx_params_m=27.8,
    ),
    'resnet50v2': BackboneSpec(
        name='resnet50v2',
        builder=K.applications.ResNet50V2,
        preprocess=K.applications.resnet_v2.preprocess_input,
        hires_cam_layer='conv4_block6_out',
        description='ResNet50V2 (pre-activation residual baseline)',
        approx_params_m=23.6,
    ),
    # Kept selectable on purpose: docs/model_card.md and the dashboard compare
    # against it, and tests/test_gradcam.py can still pin the old behaviour.
    'vgg16': BackboneSpec(
        name='vgg16',
        builder=K.applications.VGG16,
        preprocess=K.applications.vgg16.preprocess_input,
        hires_cam_layer='block5_conv3',
        description='VGG16 (legacy 2014 baseline -- kept for comparison only)',
        approx_params_m=14.7,
    ),
}

DEFAULT_BACKBONE = 'densenet121'


def get_spec(name: str) -> BackboneSpec:
    key = name.strip().lower()
    if key not in BACKBONES:
        raise ValueError(
            f'Unknown backbone {name!r}. Available: {sorted(BACKBONES)}. '
            'Set `backbone:` in configs/model.yaml.'
        )
    return BACKBONES[key]


def build_backbone(name: str, input_shape: tuple[int, int, int],
                   weights: str | None = 'imagenet') -> K.Model:
    spec = get_spec(name)
    backbone = spec.builder(weights=weights, include_top=False, input_shape=input_shape)
    # One stable handle for every downstream consumer (XAIModel, fine-tuning,
    # the app) regardless of what Keras named the submodel internally.
    backbone._name = 'backbone'
    # Remember which registry entry produced this, so `cam_layer: 'hires'`
    # and the dashboard's architecture table can look the spec back up from
    # a model that has already been saved and reloaded.
    backbone._xai_backbone_name = spec.name
    return backbone


def resolve_cam_layer(backbone: K.Model, requested: str | None) -> str:
    """Turn the config's `cam_layer` into a concrete layer name.

    'auto'/None  -> the last rank-4 (H, W, C) activation, i.e. the final
                    feature map. This is the standard Grad-CAM target and the
                    only choice that is correct for every backbone family.
    'hires'      -> the registry's stride-16 stage output (2x finer grid).
    anything else-> used verbatim, so a config can pin an exact layer.
    """
    if requested in (None, '', 'auto'):
        return _last_rank4_layer(backbone)
    if requested == 'hires':
        name = _spec_for_model(backbone).hires_cam_layer
        _assert_layer(backbone, name, hint="cam_layer: 'hires'")
        return name
    _assert_layer(backbone, requested, hint=f'cam_layer: {requested!r}')
    return requested


def _last_rank4_layer(backbone: K.Model) -> str:
    for layer in reversed(backbone.layers):
        shape = getattr(layer, 'output_shape', None)
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.name
    raise ValueError(
        f'No rank-4 feature map found in backbone {backbone.name!r}; '
        'set an explicit `cam_layer` in configs/model.yaml.'
    )


def _spec_for_model(backbone: K.Model) -> BackboneSpec:
    tag = getattr(backbone, '_xai_backbone_name', None)
    if tag is None:
        raise ValueError(
            "cam_layer: 'hires' needs the backbone to have been built through "
            'build_backbone(); got a bare Keras model.'
        )
    return get_spec(tag)


def _assert_layer(backbone: K.Model, name: str, hint: str) -> None:
    try:
        backbone.get_layer(name)
    except ValueError:
        raise ValueError(
            f'{hint} refers to layer {name!r}, which does not exist in backbone '
            f'{backbone.name!r}. Rank-4 layers available (last 10): '
            f'{_rank4_layer_names(backbone)[-10:]}'
        ) from None


def _rank4_layer_names(backbone: K.Model) -> list[str]:
    return [l.name for l in backbone.layers
            if isinstance(getattr(l, 'output_shape', None), tuple) and len(l.output_shape) == 4]


def set_finetune_trainable(backbone: K.Model, unfreeze_fraction: float,
                           freeze_batchnorm: bool = True) -> int:
    """Unfreeze the top `unfreeze_fraction` of the backbone for stage 2.

    BatchNormalization layers stay frozen by default. With batches of 32 on a
    few thousand paediatric radiographs, letting BN re-estimate its moving
    statistics during fine-tuning is the single most common way transfer
    learning silently degrades -- the running stats drift away from the
    ImageNet ones that the still-frozen lower blocks were calibrated against.

    Returns the number of layers made trainable (logged by scripts/train.py).
    """
    if not 0.0 <= unfreeze_fraction <= 1.0:
        raise ValueError(f'unfreeze_fraction must be in [0, 1], got {unfreeze_fraction}')

    layers = backbone.layers

    # Order matters. Assigning `trainable` on a nested Model propagates
    # recursively to every sublayer, so unlocking the container AFTER setting
    # the per-layer flags would overwrite all of them -- including the frozen
    # BatchNormalization layers, which is the exact failure mode this function
    # exists to prevent, and which is invisible until validation degrades.
    # So: unlock first, then set each layer, and never touch the container again.
    backbone.trainable = True

    cut = len(layers) - int(round(unfreeze_fraction * len(layers)))
    n_trainable = 0
    for i, layer in enumerate(layers):
        if i < cut:
            layer.trainable = False
            continue
        if freeze_batchnorm and isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        n_trainable += 1

    if n_trainable == 0:
        backbone.trainable = False   # recursive, and here that is what we want
    return n_trainable
