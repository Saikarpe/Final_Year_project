# Explainable AI for Medical Diagnosis (Pneumonia Detection)

A chest X-ray pneumonia classifier used as a testbed for explainability
methodology: seven explanation methods behind one registry, split conformal
prediction with an abstention policy, an evaluation suite that scores
explanation *quality* (not just classification accuracy), a shortcut-learning
audit, and a Flask case-reader UI that reads every number live from computed
metrics -- never a hardcoded literal.

The classifier itself is deliberately swappable: `backbone:` in
`configs/model.yaml` selects from DenseNet121 (default), EfficientNetV2-S/B0,
ConvNeXt-Tiny, ResNet50V2 or VGG16, and nothing outside
`src/xai_cxr/models/` names an architecture. See
`docs/decisions_log.md` for why the default moved off VGG16 and what it cost
in CAM resolution.

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
   provenance.) The Mendeley original
   (<https://data.mendeley.com/datasets/rscbjbr9sj/2>, `ChestXRay2017.zip`,
   CC BY 4.0) is the same 5,856 images and needs no Kaggle account; it ships
   `train/`+`test/` only, with no `val/` folder, which `build_splits.py`
   handles -- the derived splits come out identical either way.

   If the repo lives inside a synced folder (OneDrive/Dropbox), put the
   images elsewhere and link them in, so 1.2 GB of radiographs are not
   uploaded to a cloud drive:
   `mklink /J dataset C:\path\outside\sync` (Windows) or
   `ln -s /path/outside/sync dataset` (Linux/macOS).
2. `pip install -e ".[dev]"` (or `pip install -r requirements.txt`).

   **Training on a Windows GPU needs a second environment.** TensorFlow
   dropped native-Windows GPU support after 2.10, and TF 2.10 caps at Python
   3.10 -- so the app/tests (TF 2.13) and GPU training (TF 2.10) cannot share
   one env on Windows. See `requirements-gpu.txt`, which documents the whole
   setup including the Windows DLL-path gotcha that otherwise makes TF report
   no GPU even with CUDA correctly installed. Checkpoints trained under TF
   2.10 load fine under TF 2.13 (verified). On Linux/WSL2, one env does both.
3. `python scripts/build_splits.py` -- derives patient-grouped
   train/val/calibration/test splits into `configs/splits/*.txt`. Run this
   before anything else; every other script reads those manifests, not the
   raw Kaggle folders.

Or with the Makefile (`make setup && make splits`) -- see the Makefile for
Windows notes if `make` isn't installed.

## Run order

1. `python scripts/train.py` -- trains the M-1 classifier in two stages
   (head warmup with the backbone frozen, then fine-tuning the top of the
   backbone). See `docs/model_card.md` for the full architecture and
   `configs/model.yaml` for every knob.
2. `python scripts/evaluate.py` -- classification metrics, conformal
   calibration, explanation-quality/faithfulness/sanity-check suite. Writes
   `models/metrics.json`, which the dashboard reads live.
3. `python scripts/explain.py` -- renders the registered methods on a
   sample NORMAL/PNEUMONIA pair to `models/explanations_grid.png`.
4. `python scripts/audit.py` -- shortcut audit + failure gallery
   (`docs/failure_gallery.md`).
5. `python app/app.py` -- the case-reader UI at <http://localhost:5000>.

Steps 2 and 4 are ordered, not interchangeable: `evaluate.py` rewrites
`models/metrics.json` wholesale and `audit.py` only appends its `audits` key,
so running the audit first means the evaluation wipes it. Run it in the order
above, or re-run `audit.py` afterwards -- `evaluate.py` leaves a
`not_available` marker in `audits` saying exactly that, and the dashboard
surfaces it.

Optional:

- `python scripts/dataset_analysis.py` regenerates the `/dataset` page's
  class-distribution/pie/sample-image plots from the current split manifests
  (only needed again if you re-run `build_splits.py` with different ratios).
- `python scripts/operating_point.py` sweeps decision thresholds on the test
  split to show whether a sensitivity difference between two models is a
  ranking difference or just a threshold placement.
- `python scripts/screening_threshold.py` picks a screening operating point on
  the **calibration** split and reports it on test, so the quoted sensitivity
  is a held-out estimate rather than the best of ~582 thresholds tried on the
  images it is scored on. This is the number to cite for "how many pneumonias
  would a screening threshold miss"; see `docs/decisions_log.md`.
- `python scripts/calibrate.py` fits temperature scaling on calibration and
  reports Brier/ECE on test.

`src/train.py`, `src/evaluate.py`, `src/gradcam.py`, `src/shap_explain.py`
are kept as thin wrappers around the `scripts/` versions above, so this run
order also works exactly as originally documented.

`pytest tests/` runs the test suite: patient-disjointness, explanation
shape/class-correctness across every registered backbone and head, the
identity-at-initialisation property of the custom head and window layer, the
two-stage fine-tuning freeze/unfreeze contract, the `metrics.json` schema,
and every Flask route (including upload validation). The route tests build a
throwaway untrained checkpoint, so they no longer skip on a fresh clone;
only the `metrics.json` schema test still waits on `scripts/evaluate.py`.

## Docker

```
docker build -t pneumonia-xai .
docker run -p 5000:5000 -v "$(pwd)/models:/app/models" pneumonia-xai
```

## Layout

- `src/xai_cxr/` -- the package: config, patient-level splitting, the model
  (`models/backbones.py` registry, `models/heads.py` custom pooling/filter
  layers, `models/baseline.py` assembly + the `XAIModel` wrapper every
  explanation method talks to), the explanation-method registry, conformal
  prediction/abstention, the evaluation suite, the shortcut/failure audits,
  and the audit-log database. Everything else imports from here.
- `scripts/` -- CLI entry points.
- `configs/` -- every hyperparameter/path/seed, plus the split manifests.
- `app/` -- the Flask case reader, dashboard, dataset explorer, audit log,
  and study-mode placeholder. Templates share `_head.html` / `_nav.html` /
  `_foot.html` / `_style.html`; the UI has a light/dark toggle, keyboard
  navigation, and honours `prefers-reduced-motion`.
- `docs/` -- model card, dataset datasheet, decisions log, EU AI Act
  mapping, and the generated failure gallery.
- `CLAUDE.md` -- the hard rules for not reintroducing fixed defects.

## Changing the model

Everything is a config edit; no code change is needed to try a different
architecture.

```yaml
# configs/model.yaml
backbone: efficientnetv2s   # or densenet121 / convnext_tiny / resnet50v2 / vgg16
cam_layer: hires            # stride-16 stage: 14x14 CAM grid instead of 7x7
head: attention             # or avgmax / gap
finetune_epochs: 0          # skip stage 2 and keep the backbone frozen
```

To reproduce the original frozen-VGG16 baseline for comparison:

```yaml
backbone: vgg16
cam_layer: block5_conv3
head: gap
window_layer: false
spatial_dropout: 0
head_batchnorm: false
finetune_epochs: 0
```
