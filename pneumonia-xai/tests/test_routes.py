"""G-7: every Flask route returns 200. Skipped until models/pneumonia_model.h5
has been trained with the M-1 architecture (i.e. after scripts/train.py has
completed at least one checkpoint) -- app/app.py loads the model at import
time, so there's no way to test the routes without a compatible model file
on disk.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from xai_cxr.config import ModelConfig

model_cfg = ModelConfig.load()


def _model_is_ready() -> bool:
    if not os.path.exists(model_cfg.model_path):
        return False
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(model_cfg.model_path, compile=False)
        return 'gap' in [l.name for l in m.layers]
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _model_is_ready(),
                                 reason='models/pneumonia_model.h5 not trained with the M-1 architecture yet')


@pytest.fixture(scope='module')
def client():
    app_module = importlib.import_module('app')
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.parametrize('route', ['/', '/dashboard', '/dataset', '/audit', '/study'])
def test_route_returns_200(client, route):
    resp = client.get(route)
    assert resp.status_code == 200
