# Explainable AI for Medical Diagnosis (Pneumonia Detection)

A chest X-ray pneumonia classifier used as a testbed for explainability
methodology: six post-hoc explanation methods behind one registry, split
conformal prediction with an abstention policy, an evaluation suite that
scores explanation *quality* (not just classification accuracy), a
shortcut-learning audit, and a Flask case-reader UI that reads every number
live from computed metrics -- never a hardcoded literal.

See `../What Done Looks Like.pdf` for the full build spec this repo
implements the engineering-feasible slice of, and `docs/decisions_log.md`
for what's deferred and why (new credentialed datasets, a real clinician
study, and academic-writing/legal deliverables all need action outside a
codebase).

## Setup

1. Download the dataset from
   <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
   and place it at `dataset/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`.
   (Not committed to git -- see `docs/dataset_datasheet.md` for licence/
   provenance.)
2. `pip install -e ".[dev]"` (or `pip install -r requirements.txt`).
3. `python scripts/build_splits.py` -- derives patient-grouped
   train/val/calibration/test splits into `configs/splits/*.txt`. Run this
   before anything else; every other script reads those manifests, not the
   raw Kaggle folders.

Or with the Makefile (`make setup && make splits`) -- see the Makefile for
Windows notes if `make` isn't installed.

## Run order

1. `python scripts/train.py` -- trains the M-1 baseline (VGG16, corrected
   preprocessing/pooling/class-weighting; see `docs/model_card.md`).
2. `python scripts/evaluate.py` -- classification metrics, conformal
   calibration, explanation-quality/faithfulness/sanity-check suite. Writes
   `models/metrics.json`, which the dashboard reads live.
3. `python scripts/explain.py` -- renders all six registered methods on a
   sample NORMAL/PNEUMONIA pair to `models/explanations_grid.png`.
4. `python scripts/audit.py` -- shortcut audit + failure gallery
   (`docs/failure_gallery.md`).
5. `python app/app.py` -- the case-reader UI at <http://localhost:5000>.

Optional: `python scripts/dataset_analysis.py` regenerates the `/dataset`
page's class-distribution/pie/sample-image plots from the current split
manifests (only needed again if you re-run `build_splits.py` with different
ratios).

`src/train.py`, `src/evaluate.py`, `src/gradcam.py`, `src/shap_explain.py`
are kept as thin wrappers around the `scripts/` versions above, so this run
order also works exactly as originally documented.

`pytest tests/` runs the test suite (patient-disjointness, Grad-CAM
shape/class-correctness, `metrics.json` schema, Flask routes) -- tests that
need a trained model or generated metrics skip themselves with a clear
reason until those exist.

## Docker

```
docker build -t pneumonia-xai .
docker run -p 5000:5000 -v "$(pwd)/models:/app/models" pneumonia-xai
```

## Layout

- `src/xai_cxr/` -- the package: config, patient-level splitting, the fixed
  model architecture, the explanation-method registry, conformal prediction/
  abstention, the evaluation suite, the shortcut/failure audits, and the
  audit-log database. Everything else imports from here.
- `scripts/` -- CLI entry points.
- `configs/` -- every hyperparameter/path/seed, plus the split manifests.
- `app/` -- the Flask case reader, dashboard, dataset explorer, audit log,
  and study-mode placeholder.
- `docs/` -- model card, dataset datasheet, decisions log, EU AI Act
  mapping, and the generated failure gallery.
- `CLAUDE.md` -- the hard rules for not reintroducing fixed defects.
