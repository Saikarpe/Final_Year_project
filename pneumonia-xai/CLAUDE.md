# CLAUDE.md

Hard rules for this repo, so a coding agent (or a human moving fast) doesn't
reintroduce the defects the "What Done Looks Like" pass fixed. If you're
about to violate one of these, stop and reconsider first.

## Non-negotiable

- **Never hardcode a metric, dataset count, or model config value in a
  template or route.** `app/app.py` reads `models/metrics.json` and
  `configs/splits/*.txt` at request time. If a number needs to appear on a
  page and there's no computed source for it yet, render a "not available"
  state (see `dashboard.html`'s pattern) -- don't invent a number to fill
  the gap.
- **Never train or evaluate against `image_dataset_from_directory` on the
  raw `dataset/chest_xray/{train,val,test}` folders.** Those are the
  *original* Kermany folders, not patient-disjoint splits. Always go through
  `xai_cxr.data.load_split_dataset()` / `configs/splits/*.txt`, which are
  patient-grouped (see `xai_cxr/data.py`'s docstring for the caveat on how
  patient IDs are derived).
- **Never feed raw `/255`-rescaled images into a VGG16 backbone.** Use
  `tf.keras.applications.vgg16.preprocess_input` (already wired into
  `xai_cxr.models.baseline.build_model()` as a `Lambda` layer) -- this was
  the root cause of the v1 model's border-attribution failure.
- **Never use `Flatten` on the VGG16 output in this architecture.** Use
  `GlobalAveragePooling2D`, and keep the pre-activation `logits` output
  alongside the sigmoid -- every Grad-CAM-family method backprops from the
  logit, not the post-sigmoid probability.
- **Never add a new explanation method outside `xai_cxr/explain/registry.py`
  `METHODS`.** The eval suite and the app's method selector both iterate
  that dict; a method that isn't registered there is invisible to both.
- **Never write to the calibration split (`configs/splits/calibration.txt`
  contents) during training or hyperparameter search.** It exists only to
  calibrate the conformal predictor (`xai_cxr.uncertainty.conformal`).
- **Never call `metrics_path()`'s file "final" or "the report figure"
  without a `generated_at` timestamp and `model_hash` -- both are already in
  the schema `scripts/evaluate.py` writes; keep them if you touch that file.
- **The audit log (`xai_cxr.auditlog`) is append-only by database trigger,
  not just convention.** Don't add an UPDATE/DELETE path against
  `audit_log` -- if you need to correct a row, log a new event referencing
  the old one's id instead.
- **Don't claim something is "not available" is actually available, or vice
  versa.** Several UI states and `metrics.json` fields (`external_validation`,
  `localisation`, `subgroup_performance`, Study Mode) are deliberately marked
  `not_available` with a `reason` because the underlying data doesn't exist
  in this repo (see docs/decisions_log.md). If you add the missing data
  (VinDr-CXR, concept labels, an IEC-approved study), update the reason
  *and* the code path together -- don't leave a stale disclaimer next to
  real numbers, and don't remove a disclaimer without the data to back it.

## Where things live

- `src/xai_cxr/` -- the one implementation of everything (config, data
  splitting, model, explanation methods, conformal prediction, evaluation,
  audits). `app/app.py` and every `scripts/*.py` import from here; they
  should never duplicate logic that belongs in the package.
- `scripts/*.py` -- thin CLIs. `src/train.py`, `src/evaluate.py`,
  `src/gradcam.py`, `src/shap_explain.py` are thinner wrappers around those,
  kept only so the README's original run order still works.
- `configs/*.yaml` -- every hyperparameter, path, and seed. If you're typing
  a learning rate or image size into a `.py` file, it belongs here instead.
- `docs/decisions_log.md` -- append an entry for every non-obvious call you
  make (an approximation, a scoping cut, a deferred component). Don't let a
  future reader rediscover the same trade-off from scratch.
