"""G-7: every Flask route renders.

This used to skip itself entirely unless models/pneumonia_model.h5 already
existed, because app.py loaded the model at import time and crashed without
one -- which meant the route tests never ran in CI, on a fresh clone, or
before the first training run. app.py now loads lazily and renders a "model
not loaded" state instead, so the pages are testable with no checkpoint at
all, which is exactly the state a new contributor's repo is in.

The prediction path itself still needs a model, so those tests build a small
randomly-initialised one into tmp_path. It predicts nonsense, which is fine:
what is under test is the request/response contract, not the diagnosis.
"""
import importlib
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

PAGES = ['/', '/dashboard', '/dataset', '/audit', '/study']


@pytest.fixture(scope='module')
def app_module():
    module = importlib.import_module('app')
    module.app.config['TESTING'] = True
    return module


@pytest.fixture(scope='module')
def client(app_module):
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(scope='module')
def trained_client(app_module, tmp_path_factory):
    """A client whose checkpoint exists but is untrained."""
    from xai_cxr.config import ModelConfig
    from xai_cxr.models.baseline import build_model

    path = str(tmp_path_factory.mktemp('model') / 'pneumonia_model.h5')
    build_model(ModelConfig.load(), weights=None, compile_model=False).save(path)

    original = app_module.model_cfg.model_path
    original_state = dict(app_module._state)
    app_module.model_cfg.model_path = path
    app_module._state.update({'model': None, 'xai': None, 'hash': None, 'error': None})
    with app_module.app.test_client() as c:
        yield c
    app_module.model_cfg.model_path = original
    app_module._state.update(original_state)


def _jpeg(size=(300, 300)) -> io.BytesIO:
    buf = io.BytesIO()
    rng = np.random.RandomState(0)
    Image.fromarray(rng.randint(0, 255, (*size, 3), dtype='uint8')).save(buf, 'JPEG')
    buf.seek(0)
    return buf


@pytest.mark.parametrize('route', PAGES)
def test_page_renders_without_a_trained_model(client, route):
    assert client.get(route).status_code == 200


def test_healthz_reports_the_model_state(client):
    payload = client.get('/healthz').get_json()
    assert 'ok' in payload and 'backbone' in payload


def test_sample_xray_serves_a_test_image_or_hides_the_button(client, app_module, tmp_path, monkeypatch):
    # No dataset (a fresh clone / CI): 404, and the home page drops the button.
    monkeypatch.setattr(app_module, '_sample_rows', [])
    assert client.get('/sample-xray').status_code == 404
    assert b'id="sampleBtn"' not in client.get('/').data

    # Dataset present: an image comes back, and its filename -- which encodes
    # the ground-truth label in this corpus -- is not leaked in the response.
    path = tmp_path / 'person1_bacteria_1.jpeg'
    path.write_bytes(_jpeg().getvalue())
    monkeypatch.setattr(app_module, '_sample_rows', [str(path)])
    resp = client.get('/sample-xray')
    assert resp.status_code == 200 and resp.mimetype == 'image/jpeg'
    assert b'bacteria' not in b''.join(f'{k}: {v}'.encode() for k, v in resp.headers.items())
    assert b'id="sampleBtn"' in client.get('/').data


def test_audit_csv_downloads(client):
    resp = client.get('/audit.csv')
    assert resp.status_code == 200
    assert 'attachment' in resp.headers['Content-Disposition']


@pytest.mark.parametrize('route', PAGES)
def test_page_renders_with_a_model(trained_client, route):
    assert trained_client.get(route).status_code == 200


def test_predict_returns_a_calibrated_result(trained_client):
    data = trained_client.post('/api/predict', data={'xray': (_jpeg(), 'case.jpg')},
                               content_type='multipart/form-data').get_json()
    assert data['label'] in ('NORMAL', 'PNEUMONIA')
    assert 0.0 <= data['proba'] <= 1.0
    # The operating point travels with the result: the UI cannot render the
    # probability meaningfully without the threshold it is compared against.
    assert 0.0 < data['threshold'] < 1.0
    assert len(data['cam_grid']) == 2


def test_result_flags_images_smaller_than_the_model_input(trained_client, app_module):
    side = min(app_module.data_cfg.img_size)
    small = trained_client.post('/api/predict', data={'xray': (_jpeg((side - 1, side - 1)), 's.jpg')},
                                content_type='multipart/form-data').get_json()
    big = trained_client.post('/api/predict', data={'xray': (_jpeg((side, side + 40)), 'b.jpg')},
                              content_type='multipart/form-data').get_json()
    assert small['low_resolution'] is True and small['original_size'] == [side - 1, side - 1]
    assert big['low_resolution'] is False


def test_every_result_page_carries_the_paediatric_scope_warning(trained_client):
    # _jpeg takes (rows, cols), so this is a 120 px wide, 114 px tall image
    html = trained_client.post('/predict', data={'xray': (_jpeg((114, 120)), 't.jpg')},
                               content_type='multipart/form-data').data
    assert b'Trained on children aged 1&ndash;5 only.' in html
    assert b'Low-resolution image (120&times;114' in html


def test_public_mode_never_exposes_a_full_case_uid(trained_client, app_module, monkeypatch):
    # A full uid opens /static/uploads/<uid>_input.jpg, i.e. someone else's X-ray.
    uid = trained_client.post('/api/predict', data={'xray': (_jpeg(), 'p.jpg')},
                              content_type='multipart/form-data').get_json()['uid']
    assert trained_client.get(f'/static/uploads/{uid}_input.jpg').status_code == 200

    monkeypatch.setattr(app_module, 'PUBLIC_MODE', True)
    for route in ('/audit', '/audit.csv'):
        body = trained_client.get(route).data
        assert uid.encode() not in body and uid[:8].encode() in body

    monkeypatch.setattr(app_module, 'PUBLIC_MODE', False)
    assert uid.encode() in trained_client.get('/audit.csv').data


def test_explain_endpoint_serves_each_supported_method(trained_client, app_module):
    uid = trained_client.post('/api/predict', data={'xray': (_jpeg(), 'case.jpg')},
                              content_type='multipart/form-data').get_json()['uid']
    # The fast ones only -- occlusion and SHAP are minutes per call by design.
    for method in ('attention', 'gradcam', 'gradcam++'):
        resp = trained_client.get(f'/explain/{uid}/{method}')
        assert resp.status_code == 200, method
        assert resp.get_json()['url'].endswith('.png')


def test_explain_rejects_unknown_methods_and_uids(trained_client):
    uid = 'a' * 32
    assert trained_client.get(f'/explain/{uid}/not_a_method').status_code == 400
    assert trained_client.get(f'/explain/{uid}/gradcam').status_code == 404
    # A uid is a uuid4 hex string and nothing else -- a path-shaped one must
    # not reach the filesystem.
    assert trained_client.get('/explain/..%2f..%2fsecrets/gradcam').status_code in (400, 404)


@pytest.mark.parametrize('payload,label', [
    (b'this is not an image', 'undecodable'),
    (b'', 'empty'),
])
def test_predict_rejects_bad_uploads(trained_client, payload, label):
    """The 10 MB / JPG-PNG limit the upload panel advertises is enforced, not
    decorative."""
    resp = trained_client.post('/api/predict', data={'xray': (io.BytesIO(payload), 'x.jpg')},
                               content_type='multipart/form-data')
    assert resp.status_code == 400, label
    assert 'error' in resp.get_json()


def test_predict_rejects_oversized_uploads(trained_client):
    big = io.BytesIO(b'0' * (11 * 1024 * 1024))
    resp = trained_client.post('/api/predict', data={'xray': (big, 'x.jpg')},
                               content_type='multipart/form-data')
    assert resp.status_code == 413


def test_predict_without_a_file_is_a_client_error(trained_client):
    assert trained_client.post('/api/predict').status_code == 400
