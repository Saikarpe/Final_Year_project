"""E-8: the label-randomization check must report the same number twice.

Every explicit random draw in `label_randomization_test` was already seeded
(shuffled labels, head re-initialisation), which made the test *look*
deterministic. It was not: the head's Dropout/SpatialDropout2D layers draw
their masks from TensorFlow's global RNG during the refit, and nothing seeded
that. Two runs of the real suite reported Grad-CAM similarities of 0.266 and
0.090 -- a swing wider than the Grad-CAM/IG gap the check is cited to support.

These tests pin the fix. They build a small randomly-initialised model rather
than loading the trained checkpoint, so they run in CI without a training run.
"""
import numpy as np
import pytest

from xai_cxr.config import ModelConfig
from xai_cxr.evaluation.sanity_checks import label_randomization_test
from xai_cxr.models.baseline import build_model, XAIModel

IMG_SIZE = (96, 96)


@pytest.fixture(scope='module')
def small_model():
    """Smallest backbone at a small input size, with dropout turned *up*.

    High dropout rates are the point: if the global RNG were still unseeded,
    a 0.5 rate makes the resulting heatmaps differ obviously rather than in
    the last decimal place, so the test fails loudly instead of flaking.
    """
    cfg = ModelConfig()
    cfg.backbone = 'efficientnetv2b0'
    cfg.input_size = IMG_SIZE
    cfg.spatial_dropout = 0.5
    cfg.dropout = 0.5
    cfg.head_dropout = 0.5
    model = build_model(cfg, weights=None, compile_model=False)
    return XAIModel(model, cfg.cam_layer)


@pytest.fixture(scope='module')
def train_images():
    rng = np.random.RandomState(0)
    return rng.rand(4, *IMG_SIZE, 3).astype('float32')


@pytest.fixture(scope='module')
def image():
    rng = np.random.RandomState(1)
    return rng.rand(*IMG_SIZE, 3).astype('float32')


def _run(xai_model, image, train_images, seed):
    return label_randomization_test(
        xai_model, image, 'gradcam', train_images,
        img_size=IMG_SIZE, epochs=1, seed=seed, batch_size=2,
    )['similarity_to_original']


def test_same_seed_gives_the_same_similarity(small_model, image, train_images):
    a = _run(small_model, image, train_images, seed=0)
    b = _run(small_model, image, train_images, seed=0)
    if a is None and b is None:
        pytest.skip('degenerate heatmap on this random model; similarity undefined')
    assert a == pytest.approx(b, abs=1e-6), (
        f'label randomization is not reproducible at a fixed seed: {a} vs {b}'
    )


def test_the_reported_value_is_a_number_or_explicitly_undefined(small_model, image,
                                                                train_images):
    """Never NaN -- the E-8 block is consumed by JSON readers and the dashboard."""
    v = _run(small_model, image, train_images, seed=0)
    assert v is None or (isinstance(v, float) and -1.0 <= v <= 1.0 and v == v)


def test_repeated_run_reports_the_spread_not_just_one_draw(small_model, image,
                                                            train_images):
    """The aggregate is the point: a single draw of this check is not usable.

    Two seeds keep the test fast; the contract being pinned is the shape and
    the internal consistency of the summary, not the size of the spread on a
    randomly-initialised model.
    """
    from xai_cxr.evaluation.sanity_checks import label_randomization_repeated

    r = label_randomization_repeated(
        small_model, image, 'gradcam', train_images,
        img_size=IMG_SIZE, epochs=1, seeds=(0, 1), batch_size=2,
    )

    assert r['seeds'] == [0, 1]
    assert len(r['similarities']) == 2
    # similarity_to_original is retained for older readers, and must be the
    # first seed's value rather than the mean -- silently changing its meaning
    # would misreport every metrics.json written before this change.
    assert r['similarity_to_original'] == r['similarities'][0]

    defined = [s for s in r['similarities'] if s is not None]
    if len(defined) == 2:
        assert r['mean'] == pytest.approx(float(np.mean(defined)))
        assert r['sd'] == pytest.approx(float(np.std(defined, ddof=1)))
        assert r['min'] <= r['mean'] <= r['max']
