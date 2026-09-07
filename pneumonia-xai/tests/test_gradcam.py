"""G-7: Grad-CAM shape and class-correctness. Uses a freshly-built (untrained)
model rather than models/pneumonia_model.h5 -- this test is about the
architecture/wiring being correct, not about trained performance, so it
doesn't depend on a training run having finished."""
import numpy as np
import pytest

from xai_cxr.config import ModelConfig
from xai_cxr.models.baseline import build_model, XAIModel
from xai_cxr.explain.gradcam import explain as gradcam_explain
from xai_cxr.explain.gradcampp import explain as gradcampp_explain

IMG_SIZE = (224, 224)


@pytest.fixture(scope='module')
def xai_model():
    cfg = ModelConfig()
    model = build_model(cfg)
    return XAIModel(model, cfg.last_conv_layer)


def _random_image(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 255, size=(*IMG_SIZE, 3)).astype('float32')


@pytest.mark.parametrize('explain_fn', [gradcam_explain, gradcampp_explain])
def test_heatmap_shape_and_range(xai_model, explain_fn):
    image = _random_image()
    heatmap = explain_fn(xai_model, image, img_size=IMG_SIZE)
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
    image = _random_image(42)
    h1 = gradcam_explain(xai_model, image, img_size=IMG_SIZE)
    h2 = gradcam_explain(xai_model, image, img_size=IMG_SIZE)
    np.testing.assert_allclose(h1, h2)
