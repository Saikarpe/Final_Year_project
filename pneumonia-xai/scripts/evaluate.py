"""CLI: the evaluation suite (E-1, E-2, E-4[deferred], E-5, E-7, E-8, E-9, E-11)
plus conformal calibration (U-2, U-4). Writes everything into
models/metrics.json, which is the *only* thing the dashboard reads — no
literal numbers anywhere downstream of this script.

Usage: python scripts/evaluate.py
"""
import json
import os
import sys
import time
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from xai_cxr.config import DataConfig, ModelConfig, metrics_path
from xai_cxr.data import load_split_dataset, read_manifest, load_image
from xai_cxr.models.baseline import XAIModel, load_trained, load_threshold
from xai_cxr.explain.registry import METHODS, available_methods, explain as registry_explain
from xai_cxr.evaluation.metrics import bootstrap_auroc_ci, sensitivity_specificity, calibration_and_brier
from xai_cxr.evaluation.faithfulness import deletion_insertion_auc, road_score
from xai_cxr.evaluation.robustness import robustness_and_complexity
from xai_cxr.evaluation.sanity_checks import cascading_randomization_test, label_randomization_test
from xai_cxr.evaluation.runtime import time_methods
from xai_cxr.evaluation.calibration import (
    apply_temperature, expected_calibration_error, fit_temperature,
)
from xai_cxr.uncertainty.conformal import calibrate, empirical_coverage
from xai_cxr.uncertainty.abstention import abstention_curve
from xai_cxr import tracking

# Full method set is timed and faithfulness/ROAD-tested on FAST_SAMPLE_N
# images. The three slow methods (scorecam, occlusion, shap) are additionally
# checked on a smaller sample -- their per-image cost is exactly the E-11
# finding, and running them at the same N as Grad-CAM would make this script
# take hours rather than minutes. Sanity checks (E-8, expensive: they rerun
# the method many times per image) are restricted to the fast/gradient
# methods for the same reason. All of this is recorded in metrics.json under
# "scope_note" rather than left implicit.
# 2026-09-07: cut down further after a first CPU run of this script (with
# the old, unwired-to-config occlusion/scorecam defaults) ran for 17+ hours
# without finishing. configs/explain.yaml now caps occlusion/scorecam/shap
# cost directly (see xai_cxr.explain.registry.config_kwargs); these sample
# sizes are a second, independent lever on top of that. See decisions_log.md.
# 'attention' is in FAST_METHODS because it is a single forward pass -- it is
# cheaper than Grad-CAM, and it is exactly the method whose faithfulness we
# should NOT take on trust just because it comes from inside the head.
# available_methods() drops it automatically on a checkpoint with a gap/avgmax
# head, so this list stays valid across configs.
FAST_METHODS = ['attention', 'gradcam', 'gradcam++', 'integrated_gradients']
SLOW_METHODS = ['scorecam', 'occlusion', 'shap']
FAST_SAMPLE_N = 12
SLOW_SAMPLE_N = 4
SANITY_CHECK_METHODS = ['gradcam', 'integrated_gradients']
LABEL_RAND_N = 60
LABEL_RAND_EPOCHS = 1


def _join(names: list[str]) -> str:
    """'a, b and c' -- the scope note is shown to a reader, not logged."""
    if not names:
        return 'no methods'
    if len(names) == 1:
        return names[0]
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def _mean_of(dicts: list[dict], key: str) -> float:
    vals = [d[key] for d in dicts if d.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _sample_rows(cfg: DataConfig, n: int, seed: int = 42) -> list[tuple[str, str]]:
    rows = read_manifest('test')
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in idx]


def _model_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    data_cfg = DataConfig.load()
    model_cfg = ModelConfig.load()

    model = load_trained(model_cfg.model_path)
    xai_model = XAIModel(model, model_cfg.last_conv_layer)
    threshold = load_threshold()

    print('=== Classification performance (E-1, E-2, E-5) ===')
    test_ds, test_labels = load_split_dataset('test', data_cfg)
    test_scores = model.predict(test_ds, verbose=1).reshape(-1)
    test_pred = (test_scores > threshold).astype(int)

    classification = {
        'auroc': bootstrap_auroc_ci(test_labels, test_scores),
        'sensitivity_specificity': sensitivity_specificity(test_labels, test_pred),
        'calibration': calibration_and_brier(test_labels, test_scores),
        'decision_threshold': threshold,
        'n_test': len(test_labels),
        'external_validation': {
            'status': 'not_available',
            'reason': 'Needs D-4 (a held-out external source) -- deferred, see docs/decisions_log.md',
        },
    }

    print('\n=== Conformal prediction (U-2, U-4) ===')
    calib_ds, calib_labels = load_split_dataset('calibration', data_cfg)
    calib_scores = model.predict(calib_ds, verbose=0).reshape(-1)
    qhats = calibrate(calib_labels, calib_scores)
    coverage = {str(a): empirical_coverage(test_labels, test_scores, q) for a, q in qhats.items()}
    abst_curve = abstention_curve(test_labels, test_scores, qhats)
    # Temperature scaling (E-5). Fitted here rather than in a separate pass
    # because the calibration split has just been scored -- refitting it would
    # mean a second inference run over 563 images for one scalar. Fitted on
    # calibration, reported on test, and deliberately NOT folded back into
    # `classification.calibration.brier_score` or the decision threshold:
    # those are defined on the raw probabilities and restating them on a
    # different scale would make runs incomparable.
    temperature = fit_temperature(calib_labels, calib_scores)
    tempered_test = apply_temperature(test_scores, temperature)
    classification['calibration']['temperature_scaling'] = {
        'temperature': temperature,
        'fitted_on': 'calibration',
        'n_calibration': int(len(calib_labels)),
        'brier_before': classification['calibration']['brier_score'],
        'brier_after': calibration_and_brier(test_labels, tempered_test)['brier_score'],
        'ece_before': expected_calibration_error(test_labels, test_scores),
        'ece_after': expected_calibration_error(test_labels, tempered_test),
        'note': ('Strictly monotonic, so AUROC and every ranking metric are '
                 'unchanged. brier_score and decision_threshold above are on the '
                 'RAW probability scale.'),
    }
    print(f'  temperature {temperature:.4f} '
          f'(ECE {classification["calibration"]["temperature_scaling"]["ece_before"]:.4f}'
          f' -> {classification["calibration"]["temperature_scaling"]["ece_after"]:.4f})',
          flush=True)

    uncertainty = {
        'qhats': {str(a): q for a, q in qhats.items()},
        'empirical_coverage': coverage,
        'abstention_curve': abst_curve,
        'n_calibration': len(calib_labels),
    }

    # Coverage plot (U-4's "coverage plot at 3+ target levels, committed as a
    # report figure"), written next to the other dashboard plots.
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    alphas_sorted = sorted(qhats)  # sort by alpha so targets (1 - alpha) come out ascending
    targets = [1 - a for a in alphas_sorted]
    empirical = [coverage[str(a)] for a in alphas_sorted]
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect calibration')
    plt.scatter(targets, empirical, color='#00d4ff', zorder=3, label='Observed')
    for t, e in zip(targets, empirical):
        plt.annotate(f'{t:.0%}', (t, e), textcoords='offset points', xytext=(6, 6), fontsize=8)
    plt.xlabel('Target coverage (1 - alpha)')
    plt.ylabel('Empirical coverage on test')
    plt.title('Conformal coverage verification (U-4)')
    plt.legend(loc='lower right')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'conformal_coverage.png'))
    plt.close()

    print('\n=== Explanation quality (E-7 faithfulness/ROAD, E-9 robustness) ===', flush=True)
    explanation = {}
    supported = available_methods(xai_model)
    fast_methods = [m for m in FAST_METHODS if m in supported]
    slow_methods = [m for m in SLOW_METHODS if m in supported]

    for method in fast_methods + slow_methods:
        n = FAST_SAMPLE_N if method in fast_methods else SLOW_SAMPLE_N
        rows = _sample_rows(data_cfg, n)
        t_method_start = time.time()
        faith_rows, road_rows, robust_rows = [], [], []
        for i, (relpath, _label) in enumerate(rows):
            img = load_image(os.path.join(data_cfg.dataset_dir, relpath), data_cfg.img_size)
            heatmap = registry_explain(method, xai_model, img, img_size=data_cfg.img_size)
            faith_rows.append(deletion_insertion_auc(xai_model, img, heatmap))
            road_rows.append(road_score(xai_model, img, heatmap))
            if method in fast_methods:
                robust_rows.append(robustness_and_complexity(xai_model, img, method, img_size=data_cfg.img_size))
            print(f'  {method}: {i + 1}/{len(rows)} ({time.time() - t_method_start:.0f}s elapsed)', flush=True)

        explanation[method] = {
            'n_samples': len(rows),
            'deletion_auc_mean': _mean_of(faith_rows, 'deletion_auc'),
            'insertion_auc_mean': _mean_of(faith_rows, 'insertion_auc'),
            'road_mean_drop': _mean_of(road_rows, 'mean_drop'),
            'road_approximate': True,
            'max_sensitivity_mean': _mean_of(robust_rows, 'max_sensitivity') if robust_rows else None,
            'complexity_mean': _mean_of(robust_rows, 'complexity') if robust_rows else None,
        }

    print('\n=== Sanity checks (E-8) ===', flush=True)
    sanity_image_relpath, _ = _sample_rows(data_cfg, 1, seed=7)[0]
    sanity_image = load_image(os.path.join(data_cfg.dataset_dir, sanity_image_relpath), data_cfg.img_size)
    label_rand_rows = _sample_rows(data_cfg, LABEL_RAND_N, seed=11)
    label_rand_images = np.array([
        load_image(os.path.join(data_cfg.dataset_dir, r), data_cfg.img_size) for r, _ in label_rand_rows
    ])

    sanity = {'image': sanity_image_relpath, 'methods': {}}
    for method in [m for m in SANITY_CHECK_METHODS if m in supported]:
        t0 = time.time()
        cascade = cascading_randomization_test(xai_model, sanity_image, method, img_size=data_cfg.img_size)
        print(f'  {method}: cascading randomization done ({time.time() - t0:.0f}s)', flush=True)
        t0 = time.time()
        label_rand = label_randomization_test(xai_model, sanity_image, method, label_rand_images,
                                                img_size=data_cfg.img_size, epochs=LABEL_RAND_EPOCHS)
        sanity['methods'][method] = {'cascading_randomization': cascade, 'label_randomization': label_rand}
        print(f'  {method}: label randomization done ({time.time() - t0:.0f}s)', flush=True)

    print('\n=== Runtime per explanation (E-11) ===', flush=True)
    timing_image, _ = _sample_rows(data_cfg, 1, seed=3)[0]
    timing_image = load_image(os.path.join(data_cfg.dataset_dir, timing_image), data_cfg.img_size)
    runtime = time_methods(xai_model, timing_image, img_size=data_cfg.img_size,
                           methods=supported, n_repeats=2)

    metrics = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model_path': model_cfg.model_path,
        'model_hash': _model_hash(model_cfg.model_path),
        'classification': classification,
        'uncertainty': uncertainty,
        'explanation': explanation,
        'sanity_checks': sanity,
        'runtime': runtime,
        'localisation': {
            'status': 'not_available',
            'reason': 'Needs D-2 (VinDr-CXR boxes) -- deferred, see docs/decisions_log.md',
        },
        # Rendered straight onto the dashboard, so it is prose, not a repr:
        # "['gradcam', 'gradcam++']" is not something to show a reader.
        'scope_note': (
            f'Faithfulness/ROAD computed on {FAST_SAMPLE_N} test images for '
            f'{_join(fast_methods)} and {SLOW_SAMPLE_N} for {_join(slow_methods)} '
            f'(cost-scoped, see scripts/evaluate.py). Robustness/complexity computed only for '
            f'{_join(fast_methods)}. Sanity checks (cascading + label randomization) computed '
            f'only for {_join([m for m in SANITY_CHECK_METHODS if m in supported])}.'
        ),
    }

    path = metrics_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nmetrics.json written -> {path}')

    run_dir = tracking.start_run('evaluate', {'model_path': model_cfg.model_path})
    tracking.log_metrics(run_dir, {
        'auroc': classification['auroc']['auroc'],
        'sensitivity': classification['sensitivity_specificity']['sensitivity'],
        'specificity': classification['sensitivity_specificity']['specificity'],
        'brier_score': classification['calibration']['brier_score'],
    })


if __name__ == '__main__':
    main()
