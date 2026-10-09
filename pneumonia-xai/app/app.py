"""The Flask case reader (§8). Reads the model, every metric, and every
split through xai_cxr -- nothing here duplicates preprocessing, Grad-CAM, or
metric computation; that all lives in src/xai_cxr and is exercised by
scripts/*.py and tests/*.py the same way.

Changes in this revision:
  - the model is loaded lazily and a missing/incompatible checkpoint renders
    a setup page instead of crashing the process at import time (which is
    what made tests/test_routes.py skip itself);
  - uploads are size- and content-checked, rather than the 10 MB limit being
    a claim in the UI that nothing enforced;
  - /api/predict returns JSON so the page can show progress instead of
    blocking on a form POST for the duration of a forward pass;
  - METHOD_INFO is merged with the measured runtime and faithfulness numbers
    from metrics.json, so the method picker can warn that SHAP takes minutes
    before the user picks it rather than after;
  - the dashboard's architecture table is read off the live graph
    (models.baseline.describe) instead of restating VGG16 as a literal.
"""
import hashlib
import json
import os
import sys
import threading
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response

from xai_cxr.config import DataConfig, ModelConfig, metrics_path, REPO_ROOT
from xai_cxr.data import load_image, read_manifest, manifest_patient_ids
from xai_cxr.models.baseline import XAIModel, describe, load_trained, load_threshold
from xai_cxr.explain.registry import METHODS, METHOD_INFO, available_methods, explain as registry_explain
from xai_cxr.explain._common import colorize
from xai_cxr.uncertainty.conformal import prediction_set, LABELS
from xai_cxr.uncertainty.abstention import should_abstain
from xai_cxr import auditlog

# ── Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024        # the 10 MB the upload panel advertises
ALLOWED_FORMATS = {'JPEG', 'PNG'}
DEFAULT_ALPHA = 0.1
DEFAULT_METHOD = 'gradcam'
# Set on a public deployment (the Hugging Face Space's Dockerfile does). A
# full case uid is enough to open that case's upload under static/uploads/,
# so a publicly readable audit log must only show a prefix of it.
PUBLIC_MODE = os.environ.get('PNEUMOSCAN_PUBLIC') == '1'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

data_cfg = DataConfig.load()
model_cfg = ModelConfig.load()
audit_conn = auditlog.get_connection(os.path.join(REPO_ROOT, 'audit.db'))

_model_lock = threading.Lock()
_state = {'model': None, 'xai': None, 'hash': None, 'threshold': 0.5, 'error': None}


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def get_model():
    """Load the checkpoint on first use, once, behind a lock.

    Loading at import time meant an untrained repo could not even start the
    app, and every route test had to skip. Deferring it also keeps `flask
    --reload` usable while training is still running.
    """
    if _state['model'] is not None or _state['error'] is not None:
        return _state
    with _model_lock:
        if _state['model'] is not None or _state['error'] is not None:
            return _state
        path = model_cfg.model_path
        if not os.path.exists(path):
            _state['error'] = (
                f'No trained model at {path}. Run `python scripts/train.py` first '
                '(and `python scripts/build_splits.py` before that).')
            return _state
        try:
            model = load_trained(path)
            _state['model'] = model
            _state['xai'] = XAIModel(model, model_cfg.cam_layer)
            _state['hash'] = _file_hash(path)
            _state['threshold'] = load_threshold()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            _state['error'] = (
                f'Could not load {path}: {exc}. If this checkpoint predates the '
                'current configs/model.yaml backbone, retrain with scripts/train.py.')
    return _state


def _load_metrics():
    path = metrics_path()
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _qhat_for(alpha, metrics):
    if not metrics:
        return None
    qhats = metrics.get('uncertainty', {}).get('qhats', {})
    return qhats.get(str(alpha))


def method_catalog(xai_model=None) -> dict:
    """METHOD_INFO plus whatever the last evaluation run measured.

    The UI needs to tell someone that Score-CAM costs 15 s and SHAP costs
    minutes *before* they pick one -- those numbers already exist in
    metrics.json (E-11), they were just never surfaced where the choice is
    made. Methods the current checkpoint cannot support are dropped, so the
    picker never offers something that will 400.
    """
    metrics = _load_metrics() or {}
    runtime = metrics.get('runtime', {})
    quality = metrics.get('explanation', {})
    names = available_methods(xai_model) if xai_model is not None else list(METHODS)

    catalog = {}
    for name in names:
        info = dict(METHOD_INFO[name])
        seconds = (runtime.get(name) or {}).get('mean_seconds')
        info['seconds'] = seconds
        info['speed_label'] = _speed_label(seconds)
        row = quality.get(name) or {}
        info['insertion_auc'] = row.get('insertion_auc_mean')
        info['deletion_auc'] = row.get('deletion_auc_mean')
        catalog[name] = info
    return catalog


def _speed_label(seconds) -> str:
    if seconds is None:
        return 'not yet timed'
    if seconds < 2:
        return f'~{seconds:.1f}s'
    if seconds < 60:
        return f'~{seconds:.0f}s'
    return f'~{seconds / 60:.0f} min'


def _save_validated_upload(file_storage, uid: str) -> tuple[str, tuple[int, int]]:
    """Write the upload to disk only after Pillow confirms it decodes.

    `file.save()` straight to disk accepted anything with a filename -- a
    renamed PDF, a zero-byte file, a decompression bomb -- and the failure
    surfaced later as an opaque traceback from the image loader.
    """
    raw = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f'File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.')
    if not raw:
        raise ValueError('The uploaded file is empty.')

    import io
    try:
        probe = Image.open(io.BytesIO(raw))
        probe.verify()                      # header/structure check, cheap
        fmt = (probe.format or '').upper()
    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError('That file is not a readable JPG or PNG image.') from None
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f'Unsupported image format {fmt or "unknown"}; upload a JPG or PNG.')

    # verify() consumes the file object, so re-open to actually convert.
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    path = os.path.join(UPLOAD_DIR, f'{uid}_input.jpg')
    image.save(path, format='JPEG', quality=95)
    return path, image.size


def _heatmap_path(uid: str, method: str) -> str:
    return os.path.join(UPLOAD_DIR, f'{uid}_{method}_heatmap.png')


def _render_heatmap(xai_model, img, uid: str, method: str) -> str:
    out_path = _heatmap_path(uid, method)
    if not os.path.exists(out_path):
        heatmap = registry_explain(method, xai_model, img, img_size=data_cfg.img_size)
        Image.fromarray(colorize(heatmap)).save(out_path)
    return url_for('static', filename=f'uploads/{uid}_{method}_heatmap.png')


def _run_case(file_storage) -> dict:
    """Upload -> probability -> conformal set -> default heatmap. The single
    code path behind both the no-JS form POST and /api/predict, so the two can
    never drift apart."""
    state = get_model()
    if state['error']:
        raise RuntimeError(state['error'])
    xai_model = state['xai']

    uid = uuid.uuid4().hex
    img_path, original_size = _save_validated_upload(file_storage, uid)
    input_hash = hashlib.sha256(open(img_path, 'rb').read()).hexdigest()[:16]
    # An image smaller than the network input has to be upscaled, and the
    # model never saw upscaled images in training. Measured on the test split:
    # shrinking NORMAL radiographs to ~120 px thumbnails turned 31% of them
    # into false PNEUMONIA calls. Flagged, not refused -- the call stays the
    # user's, but they should know the number is less trustworthy.
    low_resolution = min(original_size) < min(data_cfg.img_size)

    img = load_image(img_path, data_cfg.img_size)
    proba = float(xai_model.proba(img[np.newaxis])[0])
    threshold = state['threshold']
    predicted_label = 'PNEUMONIA' if proba > threshold else 'NORMAL'

    metrics = _load_metrics()
    qhat = _qhat_for(DEFAULT_ALPHA, metrics)
    if qhat is not None:
        pred_set = prediction_set(proba, qhat)
        abstain = should_abstain(pred_set)
    else:
        pred_set = [predicted_label]   # conformal predictor not calibrated yet -- fall back honestly
        abstain = False

    heatmap_url = _render_heatmap(xai_model, img, uid, DEFAULT_METHOD)

    auditlog.log_event(
        audit_conn, 'inference', case_uid=uid, input_hash=input_hash, model_hash=state['hash'],
        proba=proba, predicted_label=predicted_label, prediction_set=pred_set, abstained=abstain,
        alpha=DEFAULT_ALPHA, decision_threshold=threshold, method=DEFAULT_METHOD,
        original_size=list(original_size), low_resolution=low_resolution,
    )

    return {
        'uid': uid,
        'label': predicted_label,
        'proba': proba,
        'threshold': threshold,
        'confidence': round(proba * 100 if predicted_label == 'PNEUMONIA' else (1 - proba) * 100, 1),
        'pred_set': pred_set,
        'abstain': abstain,
        'alpha': DEFAULT_ALPHA,
        'coverage_pct': round((1 - DEFAULT_ALPHA) * 100, 1),
        'conformal_calibrated': qhat is not None,
        'img_url': url_for('static', filename=f'uploads/{uid}_input.jpg'),
        'heatmap_url': heatmap_url,
        'default_method': DEFAULT_METHOD,
        'cam_grid': list(xai_model.cam_grid_shape),
        'original_size': list(original_size),
        'low_resolution': low_resolution,
    }


def _hero_stats(catalog: dict) -> dict:
    """The three numbers on the landing page, read from metrics.json.

    The design mockup printed "95%" and "< 5s" as literals; here each one is
    derived from the last evaluation run, or comes back None so the template
    says "not evaluated yet" instead of inventing a figure.
    """
    metrics = _load_metrics() or {}
    cls = metrics.get('classification', {})
    ss = cls.get('sensitivity_specificity') or {}
    counts = [ss.get(k) for k in ('tp', 'tn', 'fp', 'fn')]
    accuracy = None
    if all(isinstance(c, int) for c in counts) and sum(counts):
        accuracy = (counts[0] + counts[1]) / sum(counts)
    gradcam = catalog.get(DEFAULT_METHOD) or {}
    return {
        'accuracy': accuracy,
        'n_test': cls.get('n_test'),
        'n_methods': len(catalog),
        'default_seconds': gradcam.get('seconds'),
        'default_label': gradcam.get('label', DEFAULT_METHOD),
    }


def _dataset_root() -> str:
    root = data_cfg.dataset_dir
    return root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)


_sample_rows = None


def _sample_candidates() -> list:
    """Held-out test images that are actually on disk. The dataset is not in
    git, so on a fresh clone this is empty and the sample button is hidden."""
    global _sample_rows
    if _sample_rows is None:
        try:
            rows = read_manifest('test')
        except OSError:
            rows = []
        root = _dataset_root()
        _sample_rows = [os.path.join(root, rel) for rel, _ in rows
                        if os.path.exists(os.path.join(root, rel))]
    return _sample_rows


def _page_context(**extra) -> dict:
    state = get_model()
    catalog = method_catalog(state['xai'])
    ctx = {
        'method_info': catalog,
        'model_error': state['error'],
        'cam_grid': list(state['xai'].cam_grid_shape) if state['xai'] else None,
        'input_size': list(data_cfg.img_size),
        'hero': _hero_stats(catalog),
        'has_samples': bool(_sample_candidates()),
    }
    ctx.update(extra)
    return ctx


# ── Routes: case reader ─────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', active='home', **_page_context())


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """No-JS fallback. The page normally posts to /api/predict instead so it
    can render a progress state."""
    if request.method == 'GET':
        # Reloading or bookmarking a result URL is a normal thing to do, and
        # a raw Flask 405 page is a dead end. Results are per-upload and not
        # addressable, so send them back to the upload form.
        return redirect(url_for('index'))

    file = request.files.get('xray')
    if not file or not file.filename:
        return render_template('index.html', active='home',
                               **_page_context(error='No file uploaded.')), 400
    try:
        result = _run_case(file)
    except (ValueError, RuntimeError) as exc:
        return render_template('index.html', active='home',
                               **_page_context(error=str(exc))), 400
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return render_template('index.html', active='home',
                               **_page_context(error=f'Prediction failed: {exc}')), 500
    return render_template('index.html', active='home', **_page_context(**result))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get('xray')
    if not file or not file.filename:
        return jsonify({'error': 'No file uploaded.'}), 400
    try:
        return jsonify(_run_case(file))
    except (ValueError, RuntimeError) as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


@app.route('/sample-xray')
def sample_xray():
    """A random held-out test radiograph for "Use sample image". Test split
    only, so a demo never shows the model an image it was trained on."""
    import random
    candidates = _sample_candidates()
    if not candidates:
        return jsonify({'error': 'No sample images: the dataset is not installed.'}), 404
    path = random.choice(candidates)
    with open(path, 'rb') as f:
        raw = f.read()
    mime = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
    # The original filename is deliberately not sent: names like
    # person101_bacteria_483.jpeg give the ground-truth label away.
    return Response(raw, mimetype=mime, headers={'Cache-Control': 'no-store'})


@app.route('/explain/<uid>/<method>')
def explain_endpoint(uid, method):
    state = get_model()
    if state['error']:
        return jsonify({'error': state['error']}), 503
    if method not in METHODS:
        return jsonify({'error': f'unknown method {method}'}), 400
    catalog = method_catalog(state['xai'])
    if method not in catalog:
        return jsonify({'error': f'{method} is not supported by this checkpoint'}), 400

    # uuid4().hex only -- keeps a crafted uid from walking out of UPLOAD_DIR.
    if not (len(uid) == 32 and all(c in '0123456789abcdef' for c in uid)):
        return jsonify({'error': 'invalid case uid'}), 400
    img_path = os.path.join(UPLOAD_DIR, f'{uid}_input.jpg')
    if not os.path.exists(img_path):
        return jsonify({'error': 'unknown case uid'}), 404

    img = load_image(img_path, data_cfg.img_size)
    url = _render_heatmap(state['xai'], img, uid, method)

    auditlog.log_event(audit_conn, 'explanation_viewed', case_uid=uid,
                       model_hash=state['hash'], method=method)
    info = catalog[method]
    return jsonify({'url': url, 'label': info['label'], 'notes': info['notes'],
                    'seconds': info['seconds']})


@app.route('/audit/action', methods=['POST'])
def audit_action():
    data = request.get_json(force=True, silent=True) or {}
    uid = data.get('uid')
    action = data.get('action')
    reason = data.get('reason', '')
    if not uid or action not in ('agree', 'disagree', 'abstain_acknowledged'):
        return jsonify({'error': 'invalid action'}), 400
    auditlog.log_event(audit_conn, f'clinician_action:{action}', case_uid=uid,
                       model_hash=get_model()['hash'], reason=reason)
    return jsonify({'ok': True})


# ── Routes: dashboard / dataset / audit / study ─────────────────────────────
@app.route('/dashboard')
def dashboard():
    metrics = _load_metrics()
    state = get_model()

    config = {
        'Input Size': f'{data_cfg.img_size[0]} x {data_cfg.img_size[1]} x 3',
        'Backbone': model_cfg.backbone,
        'Pooling Head': model_cfg.head,
        'Learnable Window Filter': 'on' if model_cfg.window_layer else 'off',
        'Spatial Dropout': model_cfg.spatial_dropout,
        'Head Dropout / Dense Dropout': f'{model_cfg.head_dropout} / {model_cfg.dropout}',
        'Optimizer': f'AdamW (weight_decay={model_cfg.weight_decay})',
        'LR Schedule': 'warmup-cosine',
        'Stage 1 LR / Epochs': f'{model_cfg.learning_rate} / {model_cfg.epochs}',
        'Stage 2 LR / Epochs': f'{model_cfg.finetune_learning_rate} / {model_cfg.finetune_epochs}',
        'Unfreeze Fraction (stage 2)': model_cfg.unfreeze_fraction,
        'Batch Size': data_cfg.batch_size,
        'Loss Function': ('Focal BCE (gamma=%s)' % model_cfg.focal_gamma if model_cfg.focal_gamma
                          else 'Binary Cross-Entropy') + f', label_smoothing={model_cfg.label_smoothing}, class-weighted',
        'Threshold Policy': model_cfg.threshold_policy,
    }
    # Anything read off the live graph goes in only when a model is loadable,
    # so the page degrades to "config says" rather than lying.
    if state['model'] is not None:
        arch = describe(state['model'])
        config.update({
            'Architecture': arch['backbone_description'],
            'Feature Map (CAM grid)': '%s x %s x %s' % arch['feature_map'],
            'Head Graph': arch['head'],
            'Total Parameters': f'{arch["total_params"]:,}',
            'Trainable Parameters': f'{arch["trainable_params"]:,}',
        })
    config['Explainability'] = ', '.join(m['label'] for m in method_catalog(state['xai']).values())

    return render_template('dashboard.html', active='dashboard', metrics=metrics, config=config,
                           live_model_hash=state['hash'], **_page_context())


@app.route('/dataset')
def dataset():
    split_names = ['train', 'val', 'calibration', 'test']
    splits = []
    total_normal = total_pneumonia = 0
    missing = []
    for name in split_names:
        try:
            rows = read_manifest(name)
        except FileNotFoundError:
            missing.append(name)
            continue
        n_normal = sum(1 for _, l in rows if l == 'NORMAL')
        n_pneu = sum(1 for _, l in rows if l == 'PNEUMONIA')
        total_normal += n_normal
        total_pneumonia += n_pneu
        splits.append({'name': name.capitalize(), 'normal': n_normal, 'pneumonia': n_pneu,
                       'total': n_normal + n_pneu})

    dataset_stats = {
        'total': total_normal + total_pneumonia,
        'normal': total_normal,
        'pneumonia': total_pneumonia,
        'classes': 2,
    }

    patient_disjoint = None
    if not missing:
        ids = {name: manifest_patient_ids(name) for name in split_names}
        patient_disjoint = all(
            len(ids[a] & ids[b]) == 0
            for i, a in enumerate(split_names) for b in split_names[i + 1:]
        )

    return render_template('dataset.html', active='dataset', dataset=dataset_stats, splits=splits,
                           patient_disjoint=patient_disjoint, missing_splits=missing,
                           **_page_context())


@app.route('/audit')
def audit_view():
    rows = auditlog.fetch_recent(audit_conn, limit=200)
    if PUBLIC_MODE:
        for row in rows:
            row['case_uid'] = auditlog.redact_uid(row['case_uid'])
    return render_template('audit.html', active='audit', rows=rows, **_page_context())


@app.route('/audit.csv')
def audit_csv():
    csv_data = auditlog.to_csv(audit_conn, redact_uids=PUBLIC_MODE)
    return Response(csv_data, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=audit_log.csv'})


@app.route('/study')
def study():
    return render_template('study.html', active='study', **_page_context())


@app.route('/healthz')
def healthz():
    """Model-aware liveness check -- the Dockerfile's HEALTHCHECK and CI both
    want to know the checkpoint loaded, not just that Flask is listening."""
    state = get_model()
    ok = state['error'] is None
    return jsonify({'ok': ok, 'model_hash': state['hash'], 'error': state['error'],
                    'backbone': model_cfg.backbone}), (200 if ok else 503)


@app.errorhandler(413)
def too_large(_):
    message = f'File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.'
    if request.path.startswith('/api/'):
        return jsonify({'error': message}), 413
    return render_template('index.html', active='home', **_page_context(error=message)), 413


if __name__ == '__main__':
    # host 0.0.0.0 so this also works unmodified inside the Dockerfile's container.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
