"""The Flask case reader (§8). Reads the model, every metric, and every
split through xai_cxr -- nothing here duplicates preprocessing, Grad-CAM, or
metric computation; that all lives in src/xai_cxr and is exercised by
scripts/*.py and tests/*.py the same way.
"""
import hashlib
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for, jsonify, Response

from xai_cxr.config import DataConfig, ModelConfig, metrics_path, REPO_ROOT
from xai_cxr.data import load_image, read_manifest, manifest_patient_ids
from xai_cxr.models.baseline import XAIModel, load_trained, load_threshold
from xai_cxr.explain.registry import METHODS, METHOD_INFO, explain as registry_explain
from xai_cxr.explain._common import colorize
from xai_cxr.uncertainty.conformal import prediction_set, LABELS
from xai_cxr.uncertainty.abstention import should_abstain
from xai_cxr import auditlog

# ── Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
data_cfg = DataConfig.load()
model_cfg = ModelConfig.load()
model = load_trained(model_cfg.model_path)
xai_model = XAIModel(model, model_cfg.last_conv_layer)
decision_threshold = load_threshold()

_model_hash = None
if os.path.exists(model_cfg.model_path):
    _h = hashlib.sha256()
    with open(model_cfg.model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            _h.update(chunk)
    _model_hash = _h.hexdigest()[:12]

DEFAULT_ALPHA = 0.1
audit_conn = auditlog.get_connection(os.path.join(REPO_ROOT, 'audit.db'))


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


# ── Routes: case reader ─────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', method_info=METHOD_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('xray')
    if not file:
        return render_template('index.html', error='No file uploaded.', method_info=METHOD_INFO)

    try:
        uid = uuid.uuid4().hex
        img_path = os.path.join(UPLOAD_DIR, f'{uid}_input.jpg')
        file.save(img_path)

        with open(img_path, 'rb') as f:
            input_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        img = load_image(img_path, data_cfg.img_size)
        proba = float(xai_model.proba(img[np.newaxis])[0])
        predicted_label = 'PNEUMONIA' if proba > decision_threshold else 'NORMAL'

        metrics = _load_metrics()
        qhat = _qhat_for(DEFAULT_ALPHA, metrics)
        if qhat is not None:
            pred_set = prediction_set(proba, qhat)
        else:
            pred_set = [predicted_label]  # conformal predictor not calibrated yet -- fall back honestly
        abstain = should_abstain(pred_set) if qhat is not None else False

        default_method = 'gradcam'
        heatmap = registry_explain(default_method, xai_model, img, img_size=data_cfg.img_size)
        colored = colorize(heatmap)
        heatmap_path = os.path.join(UPLOAD_DIR, f'{uid}_{default_method}_heatmap.png')
        Image.fromarray(colored).save(heatmap_path)

        auditlog.log_event(
            audit_conn, 'inference', case_uid=uid, input_hash=input_hash, model_hash=_model_hash,
            proba=proba, predicted_label=predicted_label, prediction_set=pred_set, abstained=abstain,
            alpha=DEFAULT_ALPHA, decision_threshold=decision_threshold, method=default_method,
        )

        return render_template(
            'index.html',
            method_info=METHOD_INFO,
            uid=uid,
            label=predicted_label,
            confidence=round(proba * 100 if predicted_label == 'PNEUMONIA' else (1 - proba) * 100, 1),
            pred_set=pred_set,
            abstain=abstain,
            alpha=DEFAULT_ALPHA,
            coverage_pct=round((1 - DEFAULT_ALPHA) * 100, 1),
            conformal_calibrated=qhat is not None,
            img_url=url_for('static', filename=f'uploads/{uid}_input.jpg'),
            heatmap_url=url_for('static', filename=f'uploads/{uid}_{default_method}_heatmap.png'),
            default_method=default_method,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f'Prediction failed: {e}', method_info=METHOD_INFO)


@app.route('/explain/<uid>/<method>')
def explain_endpoint(uid, method):
    if method not in METHODS:
        return jsonify({'error': f'unknown method {method}'}), 400
    img_path = os.path.join(UPLOAD_DIR, f'{uid}_input.jpg')
    if not os.path.exists(img_path):
        return jsonify({'error': 'unknown case uid'}), 404

    out_path = os.path.join(UPLOAD_DIR, f'{uid}_{method}_heatmap.png')
    if not os.path.exists(out_path):
        img = load_image(img_path, data_cfg.img_size)
        heatmap = registry_explain(method, xai_model, img, img_size=data_cfg.img_size)
        colored = colorize(heatmap)
        Image.fromarray(colored).save(out_path)

    auditlog.log_event(audit_conn, 'explanation_viewed', case_uid=uid, model_hash=_model_hash, method=method)
    return jsonify({'url': url_for('static', filename=f'uploads/{uid}_{method}_heatmap.png'),
                    'label': METHOD_INFO[method]['label'], 'notes': METHOD_INFO[method]['notes']})


@app.route('/audit/action', methods=['POST'])
def audit_action():
    data = request.get_json(force=True, silent=True) or {}
    uid = data.get('uid')
    action = data.get('action')
    reason = data.get('reason', '')
    if not uid or action not in ('agree', 'disagree', 'abstain_acknowledged'):
        return jsonify({'error': 'invalid action'}), 400
    auditlog.log_event(audit_conn, f'clinician_action:{action}', case_uid=uid,
                        model_hash=_model_hash, reason=reason)
    return jsonify({'ok': True})


# ── Routes: dashboard / dataset / audit / study ─────────────────────────────
@app.route('/dashboard')
def dashboard():
    metrics = _load_metrics()
    config = {
        'Base Architecture': 'VGG16 (frozen, transfer learning)',
        'Head': 'GlobalAveragePooling2D -> Dense(256, relu) -> Dropout -> Dense(1, logit) -> Sigmoid',
        'Input Size': f'{data_cfg.img_size[0]} x {data_cfg.img_size[1]} x 3',
        'Preprocessing': 'tf.keras.applications.vgg16.preprocess_input',
        'Optimizer': 'Adam',
        'Learning Rate': model_cfg.learning_rate,
        'Batch Size': data_cfg.batch_size,
        'Max Epochs': model_cfg.epochs,
        'Loss Function': 'Binary Cross-Entropy (class-weighted)',
        'Explainability': ', '.join(m['label'] for m in METHOD_INFO.values()),
    }
    return render_template('dashboard.html', metrics=metrics, config=config, method_info=METHOD_INFO)


@app.route('/dataset')
def dataset():
    split_names = ['train', 'val', 'calibration', 'test']
    splits = []
    total_normal = total_pneumonia = 0
    for name in split_names:
        rows = read_manifest(name)
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

    ids = {name: manifest_patient_ids(name) for name in split_names}
    patient_disjoint = all(
        len(ids[a] & ids[b]) == 0
        for i, a in enumerate(split_names) for b in split_names[i + 1:]
    )

    return render_template('dataset.html', dataset=dataset_stats, splits=splits,
                            patient_disjoint=patient_disjoint)


@app.route('/audit')
def audit_view():
    rows = auditlog.fetch_recent(audit_conn, limit=200)
    return render_template('audit.html', rows=rows)


@app.route('/audit.csv')
def audit_csv():
    csv_data = auditlog.to_csv(audit_conn)
    return Response(csv_data, mimetype='text/csv',
                     headers={'Content-Disposition': 'attachment; filename=audit_log.csv'})


@app.route('/study')
def study():
    return render_template('study.html')


if __name__ == '__main__':
    # host 0.0.0.0 so this also works unmodified inside the Dockerfile's container.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
