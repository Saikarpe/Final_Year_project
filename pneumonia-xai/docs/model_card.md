# Model card (P-7) -- PneumoScan AI baseline (M-1 / M-1b)

Follows the spirit of Mitchell et al.'s "Model Cards for Model Reporting".
Covers the M-1 classifier only -- M-2/M-3/M-4 (multi-label, concept
bottleneck, patch self-explainable) are all deferred pending D-3/D-5 (see
docs/decisions_log.md).

## Model details

- **Architecture (M-1b, current):** a *configurable* backbone selected by
  `backbone:` in `configs/model.yaml`, defaulting to **DenseNet121**
  (ImageNet weights), then:

      LearnableWindow -> <family>.preprocess_input -> backbone
        -> SpatialDropout2D(0.1) -> AttentionPool2D(128) -> BatchNorm
        -> Dropout(0.2) -> Dense(256, GELU) -> Dropout(0.4)
        -> Dense(1, logit) -> Sigmoid

  Implemented in `xai_cxr.models.baseline.build_model` and
  `xai_cxr.models.heads`. The backbone registry
  (`xai_cxr.models.backbones.BACKBONES`) also carries `efficientnetv2s`,
  `efficientnetv2b0`, `convnext_tiny`, `resnet50v2` and `vgg16`.
- **Architecture (M-1, superseded):** VGG16 (ImageNet, frozen forever) ->
  GlobalAveragePooling2D -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(1,
  logit) -> Sigmoid. Reproduce it exactly with `backbone: vgg16`,
  `cam_layer: block5_conv3`, `head: gap`, `window_layer: false`,
  `spatial_dropout: 0`, `head_batchnorm: false`, `finetune_epochs: 0`. That
  is the configuration any before/after comparison should use.
- **Head, initialisation:** `AttentionPool2D` zero-initialises its attention
  logits and `LearnableWindow.strength` starts at 0, so at step 0 the M-1b
  head is bit-for-bit the M-1 head (global average pooling, raw pixels). The
  additions have to earn their keep during training; they cannot silently
  change what the baseline was. Asserted in `tests/test_gradcam.py`.
- **Input:** 320x320x3 RGB (`img_size` in `configs/data.yaml`). The backbone
  family's own `preprocess_input` is applied *inside the model graph*, by a
  registered serializable `Preprocess` layer, so every caller passes raw
  [0, 255] pixels and a saved checkpoint reconstructs its own correct
  preprocessing on load.
- **Explanation target:** `cam_layer: auto` resolves to the final feature map.
  Every modern backbone ends at stride 32, so the CAM grid is `img_size/32`:
  **10x10** at the 320x320 used here (7x7 if run at 224). VGG16's 14x14 came
  from `block5_conv3` sitting before the last pool at 224x224. `cam_layer:
  hires` selects the stride-16 stage to double the grid again, at the cost of
  shallower features. The case reader displays the live grid size, because a
  bilinearly upsampled map looks far more precise than the evidence behind it.
- **Training:** two stages. Stage 1 warms the head with the backbone frozen
  (AdamW, warmup-cosine from 5e-4); stage 2 unfreezes the top 60% of backbone
  layers at 5e-6 with BatchNormalization kept frozen. The M-1 model had no
  stage 2 at all, so no part of its feature extractor ever saw a radiograph.
  Batch size is 8 and the learning rates are scaled by sqrt(8/32) from the
  batch-32 values -- both are 4 GB VRAM constraints on the training machine,
  not modelling choices; see docs/decisions_log.md before reusing them.
- **Loss:** class-weighted binary cross-entropy with label smoothing 0.05
  (optionally focal, `focal_gamma > 0`) for the ~3:1 imbalance.
- **Augmentation:** rotation, zoom, translation, contrast and brightness.
  Horizontal flip is **off by default** -- mirroring a chest radiograph moves
  the heart to the right hemithorax and flips the laterality marker, which is
  the exact shortcut the A-1 audit measures.
- **Output:** P(PNEUMONIA), thresholded at a value tuned on the validation
  split (see `models/metrics.json`'s `classification.decision_threshold`), not
  a fixed 0.5. `threshold_policy` selects Youden's J (equal cost either way)
  or the lowest-FPR threshold meeting `min_sensitivity`.
- **Training data:** Kermany paediatric chest X-ray corpus (D-1), re-split
  patient-grouped into train/val/calibration/test (see
  `docs/dataset_datasheet.md`).
- **Versioning:** the v1 model (manual `/255` rescale, `Flatten`, no class
  weighting, val monitored on 16 images) is kept at
  `models/pneumonia_model_v1_legacy.h5` as the "here's what was wrong with
  it" artifact, not deleted. Checkpoints trained before M-1b are not loadable
  by the current graph and must be retrained; the app says so explicitly
  rather than failing obscurely.

## Intended use

Research prototype for explainability methodology (this project's actual
subject) -- comparing post-hoc explanation methods, conformal prediction for
abstention, and shortcut-learning audits on a chest X-ray classifier. **Not**
intended, validated, or authorised for clinical use, and not a substitute
for a radiologist's read (this disclaimer is also shown in the app's footer
on every page).

## Out-of-scope use

- Any adult patient population -- the training data is exclusively
  paediatric (ages 1-5). Performance on adult radiographs is unknown and not
  claimed.
- Any pathology other than pneumonia vs. normal -- this is a binary model;
  the multi-label extension (M-2, 8-14 findings) is deferred pending D-3.
- Any deployment decision made from this model alone, without the
  abstention policy (U-3) in the loop -- the conformal prediction set and
  its abstention trigger exist specifically so an ambiguous case is referred
  rather than force-classified.

## Metrics

Live numbers -- AUROC with bootstrap CI, sensitivity/specificity reported
separately, calibration/Brier score, per-method faithfulness/robustness, and
conformal coverage -- are in `models/metrics.json`, generated by
`scripts/evaluate.py`, and rendered on `/dashboard`. This card intentionally
does not duplicate them as static text: a model card with stale numbers is
worse than one that points at the live source. Re-run
`python scripts/evaluate.py` after any retrain; `metrics.json`'s
`generated_at` and `model_hash` fields let you confirm which run a number
came from.

### Decision threshold and operating point

The reported threshold (`classification.decision_threshold`) is **Youden's J,
tuned on the validation split** before the test split was touched. J weights a
missed pneumonia exactly as heavily as a false alarm, which is *not* the
clinical trade-off for a triage tool -- and on this model it costs real
sensitivity.

`python scripts/operating_point.py` sweeps the threshold on cached test scores
and shows the size of the effect: at the reported J threshold the model misses
26 of 431 pneumonias, but the same unchanged model re-tuned to hold
specificity at 90% misses 4. The full analysis, including the like-for-like
comparison against the M-1 baseline, is in `docs/decisions_log.md`
("the sensitivity gap is a threshold artefact", 2026-09-21).

Two consequences for anyone reading a sensitivity number off this model:

- **Do not compare sensitivities across models at their own Youden points.**
  Compare AUROC, or compare at a matched operating point. Against the M-1
  VGG16 baseline the two models are indistinguishable once matched, despite a
  seven-case difference in raw false negatives.
- **Any deployment must re-tune the threshold** on a calibration split against
  an explicit cost ratio. `tune_threshold_at_sensitivity` in
  `src/xai_cxr/models/` implements the screening policy;
  `configs/model.yaml` selects which policy `scripts/train.py` saves, and the
  choice is recorded per run in `runs/results_table.csv`.

## Performance by subgroup

**Not available.** The Kermany corpus as distributed here carries no
sex/age/other demographic metadata at the per-image level (see
docs/dataset_datasheet.md) -- see `models/metrics.json`'s
`audits.subgroup_performance` field for the same statement in machine-
readable form (A-2).

## Known failure modes

- See `docs/failure_gallery.md` (generated by `scripts/audit.py`) for the
  worst-confidence errors on the test split, with their explanation overlay
  and a one-line note each (A-3).
- See `models/metrics.json`'s `audits.shortcut_audit_summary` (A-1) for
  whether the model's prediction is sensitive to border content, corner
  markers, or heavy blurring -- all signals of shortcut learning rather than
  genuine pathology detection.
- See `models/metrics.json`'s `sanity_checks` (E-8) for whether each
  explanation method's output actually depends on the trained weights, or
  would look similar from an untrained/randomly-relabelled network.

### Which explanation method to trust

The seven methods are **not** interchangeable, and the benchmark in
`models/metrics.json` disagrees with the intuition that a more expensive
method is a better one.

- **Grad-CAM passes both sanity checks (E-8) and has the best deletion AUC.**
  It is also the cheapest of the post-hoc methods (~0.7 s). It is the default
  for good reasons, not historical ones.
- **Integrated Gradients fails the cascading-randomization check.** With every
  layer of the network randomized, its attribution map still correlates +0.63
  with the map from the trained model (Grad-CAM: +0.23, falling to -0.34
  mid-cascade). An explanation that survives randomizing the model it explains
  is substantially describing the input -- the known failure mode for
  gradient-and-input methods (Adebayo et al. 2018). IG also has the worst
  deletion AUC in the suite. Do not present it as evidence that the model
  looked at the right region.

See `docs/decisions_log.md`, "The cascading randomization test was broken by
its own BatchNorm handling", for the full table and for why earlier runs
reported these stages as undefined.

### Calibration

`classification.calibration.temperature_scaling` records a temperature fitted
on the calibration split (T = 0.718 for the current checkpoint -- it sharpens,
because label smoothing leaves the model under-confident). It cuts expected
calibration error by ~15% but barely moves the Brier score, because the
residual is refinement rather than calibration: confidently-wrong cases cannot
be rescued by a monotonic rescaling.

The reported `brier_score` and `decision_threshold` are on the **raw**
probability scale. If you apply the temperature, you must map the threshold
through it as well.

## Caveats and recommendations

- No external validation exists yet (E-4/D-4 deferred) -- every number above
  is internal to a single-source, paediatric, retrospective dataset.
  Treat any AUROC/sensitivity/specificity figure as an upper bound on
  likely real-world performance, not an estimate of it.
- The conformal prediction set's coverage guarantee (U-2/U-4) is marginal,
  not conditional -- it holds on average over draws of the calibration set,
  not for every individual subgroup or case.
