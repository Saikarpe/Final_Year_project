"""G-7: explanation-method shape and class-correctness.

Uses freshly-built, *randomly-initialised* models rather than
models/pneumonia_model.h5 -- these tests are about the architecture and the
wiring being correct, not about trained performance, so they don't depend on
a training run having finished.

`weights=None` throughout: CI has no reason to download ImageNet checkpoints
to assert that a heatmap is 224x224 and in [0, 1], and downloading them makes
the job both slow and network-dependent.
"""
import numpy as np
import pytest

from xai_cxr.config import ModelConfig
from xai_cxr.models.backbones import BACKBONES
from xai_cxr.models.baseline import build_model, describe, finetune, XAIModel
from xai_cxr.explain.attention import explain as attention_explain
from xai_cxr.explain.gradcam import explain as gradcam_explain
from xai_cxr.explain.gradcampp import explain as gradcampp_explain
from xai_cxr.explain.registry import available_methods

IMG_SIZE = (224, 224)


def _config(**overrides) -> ModelConfig:
    cfg = ModelConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _build(**overrides) -> XAIModel:
    cfg = _config(**overrides)
    model = build_model(cfg, weights=None, compile_model=False)
    return XAIModel(model, cfg.cam_layer)


@pytest.fixture(scope='module')
def xai_model():
    return _build()


def _random_image(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 255, size=(*IMG_SIZE, 3)).astype('float32')


@pytest.mark.parametrize('explain_fn', [gradcam_explain, gradcampp_explain, attention_explain])
def test_heatmap_shape_and_range(xai_model, explain_fn):
    heatmap = explain_fn(xai_model, _random_image(), img_size=IMG_SIZE)
    assert heatmap.shape == IMG_SIZE
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0 - 1e-6
    assert heatmap.max() <= 1.0 + 1e-6


def test_logit_and_proba_agree_on_predicted_class(xai_model):
    """The logit's sign must agree with which side of 0.5 the sigmoid
    probability falls on -- this is the pre-activation/post-activation
    consistency the M-1 fix (X-1) depends on."""
    for seed in range(5):
        image = _random_image(seed)
        logit = xai_model.logits(image[np.newaxis])[0]
        proba = xai_model.proba(image[np.newaxis])[0]
        assert (logit > 0) == (proba > 0.5)


def test_gradcam_is_deterministic_for_same_weights(xai_model):
    """Dropout and SpatialDropout2D are in the graph; if the explanation path
    ever ran them in training mode, the same image would give a different map
    each call."""
    image = _random_image(42)
    h1 = gradcam_explain(xai_model, image, img_size=IMG_SIZE)
    h2 = gradcam_explain(xai_model, image, img_size=IMG_SIZE)
    np.testing.assert_allclose(h1, h2)


# ── M-1b: the architecture is a config value ───────────────────────────────
@pytest.mark.parametrize('backbone', sorted(BACKBONES))
def test_every_registered_backbone_builds_and_explains(backbone):
    """The registry's whole promise is that swapping `backbone:` is a config
    change. That only holds if every entry actually resolves a CAM layer and
    produces a well-formed heatmap -- including the legacy vgg16 entry the
    model card compares against."""
    xai = _build(backbone=backbone)
    heatmap = gradcam_explain(xai, _random_image(), img_size=IMG_SIZE)
    assert heatmap.shape == IMG_SIZE
    assert len(xai.cam_grid_shape) == 2 and all(d > 1 for d in xai.cam_grid_shape)


def test_hires_cam_layer_doubles_the_grid():
    """`cam_layer: hires` is only worth offering if it really is the stride-16
    stage; a silent fallback to the final block would make the UI's
    resolution note wrong."""
    default_grid = _build().cam_grid_shape
    hires_grid = _build(cam_layer='hires').cam_grid_shape
    assert hires_grid[0] == default_grid[0] * 2
    assert hires_grid[1] == default_grid[1] * 2


def test_unknown_backbone_fails_loudly():
    with pytest.raises(ValueError, match='Unknown backbone'):
        _build(backbone='resnet9000')


# ── M-1b: the custom head reduces to the old one at initialisation ─────────
def test_attention_head_equals_gap_at_init():
    """AttentionPool2D zero-initialises its attention logits, so an untrained
    attention head must be bit-for-bit GlobalAveragePooling2D. This is what
    makes turning the head on a safe, comparable change rather than a
    different model."""
    import tensorflow as tf
    from xai_cxr.models.heads import find_attention_layer

    xai = _build(head='attention')
    features = xai._grad_base_model(xai.preprocess(_random_image()[np.newaxis]))[1]
    pooled = find_attention_layer(xai.model)(features).numpy()
    np.testing.assert_allclose(pooled, tf.reduce_mean(features, axis=(1, 2)).numpy(), atol=1e-5)


def test_learnable_window_is_identity_at_init():
    """LearnableWindow's `strength` starts at 0, so it must pass raw pixels
    straight through until training moves it."""
    xai = _build(window_layer=True)
    image = _random_image()[np.newaxis]
    np.testing.assert_allclose(xai.model.get_layer('window')(image).numpy(), image, atol=1e-3)


def test_attention_map_sums_to_one_and_matches_cam_grid():
    xai = _build(head='attention')
    alpha = xai.attention_map(_random_image()[np.newaxis])
    assert alpha.shape == (1, *xai.cam_grid_shape)
    np.testing.assert_allclose(alpha.sum(axis=(1, 2)), [1.0], atol=1e-5)


@pytest.mark.parametrize('head', ['gap', 'avgmax'])
def test_attention_method_is_unavailable_without_an_attention_head(head):
    """The registry must not offer a method this checkpoint cannot run -- the
    app's method picker is built from available_methods()."""
    xai = _build(head=head)
    assert 'attention' not in available_methods(xai)
    assert not xai.has_attention
    with pytest.raises(ValueError, match='head: attention'):
        attention_explain(xai, _random_image(), img_size=IMG_SIZE)


@pytest.mark.parametrize('head', ['gap', 'avgmax', 'attention'])
def test_every_head_produces_a_valid_heatmap(head):
    """XAIModel replays the head layers rather than assuming a fixed stack, so
    a head change must not require an explanation-method change."""
    heatmap = gradcam_explain(_build(head=head), _random_image(), img_size=IMG_SIZE)
    assert heatmap.shape == IMG_SIZE


# ── M-1b: two-stage fine-tuning ────────────────────────────────────────────
def test_finetune_unfreezes_backbone_but_not_batchnorm():
    """Stage 2's entire purpose is that backbone weights start moving. If
    freeze_batchnorm leaked, the moving statistics would drift away from the
    still-frozen lower blocks -- the classic silent transfer-learning
    degradation."""
    import tensorflow as tf
    from xai_cxr.models.baseline import get_backbone

    cfg = _config(unfreeze_fraction=0.35, freeze_batchnorm=True)
    model = build_model(cfg, weights=None)
    backbone = get_backbone(model)
    assert not any(l.trainable for l in backbone.layers), 'stage 1 must start fully frozen'

    n_trainable = finetune(model, cfg, steps_per_epoch=4, epochs=2)
    assert n_trainable > 0
    assert not any(l.trainable for l in backbone.layers
                   if isinstance(l, tf.keras.layers.BatchNormalization))
    # Recompile must have picked the new weights up, or stage 2 trains nothing.
    assert len(model.trainable_weights) > 0


def test_finetune_fraction_zero_keeps_everything_frozen():
    from xai_cxr.models.baseline import get_backbone

    cfg = _config(unfreeze_fraction=0.0)
    model = build_model(cfg, weights=None)
    assert finetune(model, cfg) == 0
    assert not any(l.trainable for l in get_backbone(model).layers)


def test_describe_reports_the_live_graph():
    """The dashboard's architecture table reads this; it must not be able to
    disagree with the model that is actually loaded."""
    info = describe(build_model(_config(), weights=None, compile_model=False))
    assert info['input_shape'] == (*IMG_SIZE, 3)
    assert len(info['feature_map']) == 3
    assert info['total_params'] > info['trainable_params'] > 0
    assert 'logits' in info['head']
