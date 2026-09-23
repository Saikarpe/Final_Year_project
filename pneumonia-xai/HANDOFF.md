# Session handoff — resume point

**Last updated:** 2026-09-23 ~23:50 IST
**Branch:** `main` — §9 has the results, **§11 corrects two numbers in §9/§10
and is the one to read before writing anything up**

Read this first, then `docs/decisions_log.md` for the *why* behind every choice.

---

## 1. Where things stand in one line

The model is **trained, evaluated and audited**. The pipeline is finished,
`models/metrics.json` describes the current DenseNet121 (`2a279eeff356`) and
is single-provenance, all 73 tests pass and the app serves the real model. The
headline finding is that M-1b **ties** the VGG16 baseline rather than beating
it — see §9.

Two numbers in §9/§10 were corrected on 2026-09-23 and the old ones must not
be quoted: the screening operating point misses **8** pneumonias, not 4 (the
old figure was selected on the split it was scored on), and the E-8 label
randomization is **0.24 ± 0.55** for Grad-CAM against **0.71 ± 0.01** for IG
over five seeds, not the single draws previously reported. Both corrections
strengthen rather than weaken the conclusions — see §11.

---

## 2. What was done this session

### Architecture (M-1b) — replaced the frozen VGG16
- Backbone is now a **registry key** (`src/xai_cxr/models/backbones.py`):
  `densenet121` (default), `efficientnetv2s`, `efficientnetv2b0`,
  `convnext_tiny`, `resnet50v2`, `vgg16`. Nothing outside `models/` names an
  architecture.
- **Two-stage training**: head warmup (backbone frozen) → fine-tune the top
  60% of the backbone with BatchNorm frozen. The old model never fine-tuned at
  all, which is the single biggest change.
- **Custom head** (`src/xai_cxr/models/heads.py`), all identity-initialised so
  they reduce exactly to the old model at step 0 (asserted in tests):
  `LearnableWindow` (3-param differentiable window/level), `SpatialDropout2D`,
  `AttentionPool2D` (gated attention pooling replacing GAP).
- Attention weights registered as a **7th explanation method** (`attention`) —
  the only intrinsic one in the suite.
- AdamW + warmup-cosine, label smoothing, optional focal loss, AUPRC.
- Augmentation rewritten; **horizontal flip off by default** (mirroring a chest
  X-ray moves the heart and flips the laterality marker — the exact shortcut
  the A-1 audit measures).

### Training run (completed)
```
DenseNet121, 320x320, batch 8, LR 5e-4 / 5e-6, unfreeze 0.60
stage 1: 8 epochs   best val_auc 0.99294
stage 2: 6 epochs   best val_auc 0.99491  (early stopped, restored epoch 3)
threshold (Youden's J on val): 0.3447
wall clock ~2h14m on a Quadro M1200
checkpoint: models/pneumonia_model.h5 (76.7 MB, gitignored)
run record: runs/20260920T162909Z_train_baseline/
```

### UI/UX
Rebuilt the Flask front end: clinical design system with light/dark toggle,
accessibility (focus rings, skip link, `prefers-reduced-motion`), async
progress states, per-method runtime shown *before* you pick a slow method,
window/level + invert + synced pan/zoom viewer, probability-vs-threshold
meter, stale-metrics banner.

---

## 3. Metrics freshness — RESOLVED

`models/metrics.json` was an inconsistent hybrid (old VGG16 metrics + new
threshold + new audits) while the pipeline was mid-run. It no longer is:
`evaluate.py` rewrote it at `2026-09-21T04:55:18Z` and `audit.py` appended to
it afterwards. Verify at any time with:

```bash
python -c "import json; m=json.load(open('models/metrics.json')); print(m['generated_at'], m['model_hash'])"
```

Expected now: `2026-09-21T04:55:18Z 2a279eeff356`. The `model_hash` is
`sha256(models/pneumonia_model.h5)[:12]`, so it can be checked independently
against the checkpoint on disk. The dashboard's stale-metrics banner compares
the same two values and is currently **clear** (verified by fetching
`/dashboard` and `/healthz`, both reporting `2a279eeff356`).

If you ever see `2026-09-07T04:35:01Z` / `63a339ab677f` again, that is the
archived VGG16 baseline's metrics and the pipeline did not finish.

---

## 4. Resuming — exact commands

### Environments (there are two, on purpose)
| purpose | interpreter |
|---|---|
| app, tests, everything CPU | `python` (system, 3.11, TF 2.13) |
| **GPU training/eval** | `C:\Users\karpe\mamba\envs\tfgpu\python.exe` (3.10, TF 2.10) |

GPU work **must** go through the wrapper, or TensorFlow will not see the card:
```
C:\Users\karpe\AppData\Local\Temp\claude\rungpu.bat <script.py>
```
That wrapper sets a real Windows `PATH` to `<env>\Library\bin` first. Python
3.8+ does not resolve extension-module DLLs via `PATH`, and
`os.add_dll_directory` is *not* sufficient for TF's loader. See
`requirements-gpu.txt`.

> If the scratchpad has been cleared, recreate `rungpu.bat`:
> ```bat
> @echo off
> set "PATH=C:\Users\karpe\mamba\envs\tfgpu\Library\bin;C:\Users\karpe\mamba\envs\tfgpu\Library\lib;C:\Users\karpe\mamba\envs\tfgpu\Scripts;C:\Users\karpe\mamba\envs\tfgpu;%PATH%"
> "C:\Users\karpe\mamba\envs\tfgpu\python.exe" %*
> ```

### To finish the pipeline (this is the next step)
Order is load-bearing: `evaluate.py` rewrites `metrics.json` wholesale,
`audit.py` only appends. Running audit first means evaluate wipes it.
```
rungpu.bat scripts/evaluate.py     # ~40 min at 320px
rungpu.bat scripts/audit.py        # ~7 min
```
`scripts/explain.py` already succeeded — `models/explanations_grid.png` is
current (regenerated 2026-09-21 08:53).

### Then
```
python -m pytest tests/ -q          # 47 passing as of last full run
python app/app.py                   # http://localhost:5000
```

### Data
`dataset/` is a **junction** → `C:\Users\karpe\pneumonia-data`, so 1.2 GB of
images stay out of OneDrive sync. If it is missing:
```
mklink /J dataset C:\Users\karpe\pneumonia-data
```
Source: Mendeley `rscbjbr9sj` v2, `ChestXRay2017.zip`, CC BY 4.0, 5,856 images.

---

## 5. Verified facts worth not re-deriving

- **The test split is identical to the archived VGG16 baseline's.** Checked
  image-by-image: same 582 test images, same patients. Only 16 *path strings*
  differ (Kaggle ships a 16-image `val/` folder; Mendeley distributes those
  inside `train/`/`test/`). **So the before/after IS a controlled comparison
  on data** — a caveat I wrote earlier and then disproved.
- **Checkpoint portability TF 2.10 → 2.13 works.** Verified: backbone
  densenet121, input (320,320,3), feature map (10,10,1024), Grad-CAM and the
  attention map both functional after reload.
- **Baseline to beat** (archived, `models/baseline_vgg16_2026-09-07/`):
  AUROC 0.9941 [0.989, 0.998], sens 95.6%, spec 96.7%, Brier 0.047,
  **19 false negatives / 5 false positives of 582**. The FN count is the
  number that actually matters clinically.
- **Occlusion returns an all-zero map on confidently-NORMAL images.** This is
  correct, not a bug: there is no pneumonia evidence to remove, so every
  occlusion raises the logit and the ReLU zeroes the map. Verified it
  localises properly on PNEUMONIA cases (max 1.0, 21% nonzero).

---

## 6. Bugs fixed (16) — all but one verified

| # | Bug | Verified by |
|---|---|---|
| 1 | Nested `trainable=True` silently unfroze BatchNorm | test |
| 2 | Cascade sanity check named VGG blocks → no-op on DenseNet | test |
| 3 | Failure gallery judged errors at 0.5, not the tuned threshold | code |
| 4 | App crashed at import without a checkpoint (route tests always skipped) | 19 tests |
| 5 | 10 MB upload limit advertised but unenforced | test |
| 6 | Uploads not content-validated | test |
| 7 | `/explain/<uid>` path traversal | test |
| 8 | JET overlay painted over the anatomy it explained | browser |
| 9 | Image viewer trapped the page scroll | browser |
| 10 | Dashboard didn't flag metrics from a different model | browser |
| 11 | `GET /predict` → raw Flask 405 | curl |
| 12 | Deletion-AUC bar longer when the method was worse | browser |
| 13 | Data pipeline re-decoded every JPEG each epoch (12.7 h → 3.4 h) | measured |
| 14 | Missing `shap` killed all 3 downstream scripts | pipeline |
| 15 | **5x hardcoded batch sizes OOMing at 320px** (IG, Score-CAM, Occlusion, SHAP, sanity checks) | all verified on GPU |
| 16 | Checkpoint could regress across training stages | **NOT YET EXERCISED** — only a future 2-stage run hits that path |

**Bug 15 is the one to understand**: `img_size` was config-driven, but every
constant whose memory cost is a *function* of `img_size` was a literal. Peak
memory scales with `batch x H x W`, so each OOM hid the next and it surfaced
five times in a row. Now one knob — `inference_batch` in
`configs/explain.yaml` — drives them all.

---

## 7. What is NOT done

- [x] ~~Finish `evaluate.py` + `audit.py`~~ — done, see §3
- [x] ~~Report test AUROC vs the baseline~~ — done, see §9
- [x] ~~Restart the app on the real model; confirm the stale banner clears~~
- [x] ~~Final full test-suite run~~ — **73 passed** (2026-09-23, 8m00s)
- [x] ~~Mobile layout never verified visually~~ — verified at 360px on all
      five pages with headless Chrome; one real overflow bug found and fixed
- [x] ~~Temperature scaling~~ — implemented; it does **not** close the Brier
      gap, see §9
- [ ] **Nothing is deployed anywhere.** Local Flask only.
- [ ] Pre-existing deferrals, unchanged and out of scope: external validation
      (D-4), localisation mIoU (D-2), subgroup analysis (A-2), study mode
      (D-6/P-1), concept bottleneck (M-3/D-5). These need data-use agreements,
      ethics approval or recruited clinicians — they are not coding tasks.
- [x] ~~`evaluate.py` has not been rerun end-to-end since the BatchNorm fix~~ —
      done 2026-09-23. `metrics.json` is single-provenance;
      `sanity_checks.recomputed_at` is gone. See §11 for the three defects the
      rerun exposed.
- [ ] **Uncommitted.** §11's work is in the working tree, not in git.
- [ ] Bug 16 (checkpoint regression across training stages) is still **not
      exercised** — only a future two-stage training run reaches that path.
      Unchanged from §6; re-running the evaluation suite does not test it.

---

## 8. Honest caveats for the write-up

- `val_auc 0.99491` is **mildly optimistic by construction** — it drove early
  stopping and threshold tuning. The test split is untouched; quote test
  numbers.
- The comparison is an **A/B between two complete configurations**, not an
  ablation. Backbone, head, resolution, optimiser and augmentation all changed
  together. It shows *that* M-1b differs, not *which change* caused it.
- At n=582 with AUROC already ~0.994, check whether the new CI lower bound
  clears the old point estimate before claiming a significant improvement.
- `batch_size: 8` and the scaled learning rates are **4 GB VRAM constraints**,
  not modelling choices. On a bigger card use batch 32 at 320px with LR 1e-3.

---

## 9. Final results (2026-09-21)

### The headline: M-1b ties the baseline, it does not beat it

Same 582 test images, verified identical split membership
(`python scripts/compare_to_baseline.py`):

| | VGG16 (frozen, M-1) | DenseNet121 (M-1b) |
|---|---|---|
| AUROC | 0.9941 [0.9892, 0.9976] | 0.9947 [0.9908, 0.9980] |
| Sensitivity | 95.6% | 94.0% |
| Specificity | 96.7% | 98.7% |
| Brier | 0.047 | 0.063 |
| **False negatives** | **19** | **26** |
| False positives | 5 | 2 |

**Do not write this up as "M-1b misses 7 more pneumonias."** It does not.
The AUROC CIs overlap heavily, so the two models rank cases equally well; the
sensitivity difference is entirely *where the threshold landed*. Youden's J
chose 0.2174 for the baseline and 0.3447 for M-1b.

`python scripts/operating_point.py` (caches per-image test scores to
`runs/test_scores.npz`, then sweeps) makes the comparison like-for-like:

| operating point | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|
| M-1b @ Youden J (reported) | 0.3447 | 94.0% | 98.7% | 26 | 2 |
| M-1b @ baseline's specificity | 0.2224 | 95.4% | 96.7% | 20 | 5 |
| M-1b @ baseline's sensitivity | 0.2220 | 95.6% | 96.7% | 19 | 5 |
| VGG16 @ its own Youden J | 0.2174 | 95.6% | 96.7% | 19 | 5 |

Matched on specificity the gap is +1 of 431 (one image). Matched on
sensitivity the confusion matrices are **identical**. So: no significant
difference either way, which is the honest claim.

### The interesting finding: Youden's J is the wrong objective

J weights a missed pneumonia exactly as heavily as a false alarm. For triage
it should not. The same unchanged model, re-tuned:

Threshold selected on **calibration**, reported on **test**
(`python scripts/screening_threshold.py`, cached to
`runs/screening_threshold.json`):

| objective | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|
| max sensitivity, specificity >= 95% | 0.1819 | 97.2% | 96.7% | 12 | 5 |
| max sensitivity, specificity >= 90% | 0.0957 | 98.1% | 91.4% | **8** | 13 |

**8** missed of 431, against the baseline's 19, for 8 extra false positives out
of 151 normals. `tune_threshold_at_sensitivity` already implements this policy
and `configs/model.yaml` selects it; the reported threshold was deliberately
left at J because it was fixed on val before test was touched.

> **Corrected 2026-09-23.** This table previously read **4** missed at the 90%
> floor. That row was produced by `operating_point.py`, which selects the
> threshold on the test split and then scores it on the same split -- the best
> of ~582 candidates on the very images it is evaluated on. Re-selected on the
> calibration split and applied unchanged to test, the honest figure is **8**.
> The selection optimism was therefore 4 pneumonias, i.e. the old number was
> exactly twice as good as reality. **Quote 8, not 4.**
>
> The argument is unaffected -- 8 against the baseline's 19 is still the
> finding, and still shows Youden's J is the wrong objective for triage. Only
> the number moves, and it moves from unquotable to quotable.
> `operating_point.py` is unchanged and still correct for *its* question
> (is the M-1b/VGG16 sensitivity gap a threshold artefact?), which is a
> comparison both models enter on identical terms. See `docs/decisions_log.md`,
> entry "The screening operating point was selected on test".

Full reasoning: `docs/decisions_log.md`, entry
"M-1b vs the VGG16 baseline: the sensitivity gap is a threshold artefact".

### Other numbers worth knowing

- **Conformal coverage holds** at all three alphas (94.5 / 89.9 / 80.8% against
  95 / 90 / 80% targets).
- **Calibration regressed** (Brier 0.047 -> 0.063). Better discrimination with
  worse calibration is the expected signature of fine-tuning with label
  smoothing. The UI relies on the conformal layer, not raw probabilities, so
  this was not treated as a blocker. Temperature scaling is the cheap fix.
- **Explanation faithfulness:** Grad-CAM has the best deletion AUC (0.610) and
  the second-best insertion (0.851) at 0.7 s; SHAP and IG score worst on
  deletion (0.928 / 0.908) despite costing 30.6 s and 3.5 s. The intrinsic
  `attention` map is mid-pack at 0.4 s.
- **Sanity checks:** both Grad-CAM and IG depend on the trained weights.
  Label randomization is the cleaner evidence (Grad-CAM 0.266, IG 0.683 --
  Grad-CAM falls further). Cascading randomization agrees over the stages
  where it is defined, but only the first three are; see below.
  > **Superseded by §11.** "Grad-CAM 0.266" is one draw from a distribution
  > with sd 0.55; two later runs of the same check gave 0.090 and 0.734. The
  > conclusion holds but this is not the number that supports it.

### Fixed this session: the sanity checks were emitting NaN

`sanity_checks.*.cascading_randomization.stages[].similarity_to_original` was
`NaN` for **7 of 10 stages** on both methods -- everything from `head_bn`
onward, not just the final stage:

```
original 1.000 | logits 0.477 | dense -0.335 | head_bn .. backbone_group_5/5  NaN
```

Two separate problems, both now fixed:

1. **It was reported as a number.** Randomizing a BatchNorm layer collapses
   the heatmap to a constant, and a rank correlation against a constant is
   *undefined* (zero standard deviation in the denominator), not zero.
   `_rank_similarity` now detects the degenerate case up front and returns
   `None`; the dashboard renders "undefined (degenerate map)" instead of a bar
   of length `NaN` px.
2. **`metrics.json` was not valid JSON.** Python's `json` module writes and
   reads bare `NaN` happily, so the file looked fine from inside the project
   while being unparseable by `JSON.parse`, `jq`, Go and most other readers --
   a real defect in a file that is meant to be the reproducibility artifact.
   The existing file was migrated in place (14 non-finite values -> `null`,
   values unchanged, see its `encoding_note`); no re-run was needed.

Two regression tests were added in `tests/test_metrics_schema.py`
(`test_metrics_json_is_strict_json`, `test_rank_similarity_is_none_for_a_constant_heatmap`).

> **Superseded by §10.** The paragraph below concluded that the degenerate
> stages were inherent to the test. They were not — they were a bug in
> `_randomize_layer`, which zeroed BatchNorm's gamma. Fixing it made all 10
> stages defined and produced a result that matters (IG fails E-8). Read
> §10 instead; this is kept only so the reasoning trail is visible.

**What this costs the E-8 claim:** cascading randomization is only informative
for the first three stages. Both methods *do* fall away from the original
there (Grad-CAM 1.000 -> 0.477 -> -0.335, IG 1.000 -> 0.773 -> 0.751), and
label randomization is defined throughout, so the checks still support the
conclusion. But "similarity falls to zero as the model is randomized" is not
a claim this run can make past `dense`. Making the deeper stages informative
would need a randomization scheme that preserves activation scale -- e.g.
resampling BatchNorm gamma/beta from their fitted distribution rather than
from a fresh initializer. Not attempted.

### Verified this session

```
python -c "import json; m=json.load(open('models/metrics.json')); print(m['generated_at'], m['model_hash'])"
  -> 2026-09-21T04:55:18Z 2a279eeff356   (hash matches sha256 of the .h5)
python -m pytest tests/ -q          -> 47 passed in 233.80s
curl /healthz                       -> {"backbone":"densenet121","model_hash":"2a279eeff356","ok":true}
curl /dashboard                     -> 200, stale banner absent, "loaded now 2a279eeff356"
curl / /dataset /audit /study       -> 200
POST /api/predict  test NORMAL      -> NORMAL, p=0.046
POST /api/predict  test PNEUMONIA   -> PNEUMONIA, p=0.972
```

---

## 10. Second pass, 2026-09-21 afternoon

Three things closed, and two of them changed what the project can claim.

### Integrated Gradients fails the E-8 sanity check

The cascading randomization test was reporting `undefined` for 7 of its 10
stages. That turned out to be a **bug in the test**, not a property of it:
`_randomize_layer` zeroed every 1-D weight, and BatchNorm's gamma is 1-D, so
it set gamma = 0 and the layer emitted a constant. Every heatmap from that
point down was flat.

With BatchNorm handled properly (gamma/beta permuted across channels, moving
statistics left alone) all 10 stages are defined, and the result matters:

| | Grad-CAM | Integrated Gradients |
|---|---|---|
| logits | +0.477 | +0.773 |
| dense | **-0.335** | +0.751 |
| fully randomized | +0.230 | **+0.629** |
| label randomization (one run -- see §11) | +0.266 | +0.683 |

**Grad-CAM passes. IG largely fails.** With every layer randomized, IG's
attribution still correlates +0.63 with the trained model's. An explanation
that survives the destruction of the model it explains is substantially
describing the *input*, which is the known failure mode for
gradient-and-input methods (Adebayo et al. 2018). IG also has the worst
deletion AUC in the benchmark (0.908 vs Grad-CAM's 0.610) — two independent
diagnostics agreeing.

**Do not present IG as a trustworthy explanation for this model.** Grad-CAM
is the cheapest method *and* the one that passes both checks.

### Temperature scaling does not fix the Brier gap

Fitted T = **0.718** on the calibration split. It *sharpens*, meaning the
model is under-confident — the signature of label smoothing, not the
over-confidence usually assumed.

| metric (test) | before | after |
|---|---|---|
| Brier | 0.0630 | 0.0624 (-1.0%) |
| ECE | 0.1249 | 0.1065 (-14.8%) |

ECE improves usefully; Brier barely moves. The residual is *refinement*, not
calibration: the 26 missed pneumonias are confidently wrong, and no monotonic
rescaling rescues a case on the wrong side of the boundary. So the Brier gap
against the baseline is **not** a calibration-scale artefact. The headline
`brier_score` and `decision_threshold` are deliberately left on the raw
probability scale; the temperature is recorded beside them.

### Mobile layout verified

All five pages at a true 360px. One real bug: the dashboard's absolute
`model_path` overflowed horizontally (a Windows path breaks only at
backslashes). Fixed with `overflow-wrap:anywhere` scoped to `.meta code`.

**If you need to redo this:** `chrome --headless --window-size=360,...` does
**not** give a 360px viewport — Chrome clamps the window near 500px and then
crops the screenshot, which looks precisely like an overflow bug and wasted a
cycle here. Render the app inside a 360px-wide `<iframe>` on a wider page
instead; the inner document then gets a genuine 360px viewport.

### New in this pass

```
src/xai_cxr/evaluation/calibration.py   temperature scaling + ECE
scripts/calibrate.py                    fit on calibration, report on test
scripts/operating_point.py              threshold sweep on cached scores
scripts/refresh_sanity_checks.py        recompute only the E-8 block
tests/test_calibration.py               10 tests
```

Test count went 47 -> 59.

---

## 11. Third pass, 2026-09-23 evening -- the rerun, and what it caught

**Goal:** make `models/metrics.json` single-provenance. Since 2026-09-21 it
had been a hybrid again: `generated_at 04:55:18Z`, a `sanity_checks` block
surgically refreshed at `06:07:31Z`, and `audits` appended after that.

`evaluate.py` was rerun end-to-end on the GPU and `audit.py` after it. That
part succeeded on the first attempt. The rerun then caught **three defects**,
two of which would have put a wrong number in the write-up.

### What reproduced exactly

Every classification and faithfulness number, bit-for-bit: AUROC 0.9947 and
its CI, the 26/2 confusion matrix, Brier 0.0630, the threshold 0.3447, all
seven deletion AUCs, the shortcut-audit deltas, and the failure gallery
(regenerated byte-identical). The 20 cascading-randomization stages reproduce
to ~5 decimal places; the residual drift is GPU convolution nondeterminism.

This is the result that makes the rest of the section trustworthy: the
pipeline *is* reproducible, so the things that moved, moved for a reason.

### Defect 1 -- the screening claim was twice as good as reality

§9 above quoted "4 missed of 431" for a screening operating point. That
threshold was chosen on the test split and then scored on the test split --
the best of ~582 candidates on the very images it is evaluated on.

Re-selected on the **calibration** split and applied unchanged to test
(`scripts/screening_threshold.py`, cached to `runs/screening_threshold.json`):

| objective | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|
| max sens, spec >= 90% | 0.0957 | 98.1% | 91.4% | **8** | 13 |
| max sens, spec >= 95% | 0.1819 | 97.2% | 96.7% | 12 | 5 |

**Quote 8, not 4.** The argument is untouched -- 8 against the VGG16
baseline's 19 is still the finding, and still shows Youden's J is the wrong
objective for triage -- but the number was a selection artefact and would not
have survived a viva.

### Defect 2 -- a "seeded, deterministic" check that was neither

`label_randomization_test` seeds its shuffled labels and its head
re-initialisation. It does not seed the head's three dropout layers, which
draw from TensorFlow's *global* RNG during the refit. So the one number in
the suite that depends on a training step was the one number that moved:
**0.266 (Sept 21), 0.090, then 0.734** across three runs of the same check.

Fixed with `tf.random.set_seed(seed)`. `tests/test_sanity_check_determinism.py`
pins it and was verified to fail against the unfixed code.

### Defect 3 -- seeding was necessary but not sufficient: sd = 0.55

A reproducible arbitrary number is still arbitrary. Measured over five seeds:

| method | mean | sd | min | max |
|---|---|---|---|---|
| Grad-CAM | +0.239 | **0.552** | -0.438 | +0.812 |
| Integrated Gradients | +0.712 | **0.009** | +0.702 | +0.723 |

Grad-CAM ranges over 1.25 of the 2.0 this statistic can span. The three
historical values (0.266 / 0.090 / 0.734) are simply three draws from that,
and they invite three different conclusions.

**The spread is the result, not noise around it.** Adebayo et al.'s test asks
whether an explanation changes when the model stops being the trained model:

- **Grad-CAM passes.** Its map swings from anti-correlated to strongly
  correlated depending only on what the shuffled-label refit did to the head.
  That is an explanation tracking the model.
- **IG fails, and more clearly than before.** It returns +0.70 to +0.72 *every
  time*, sd 0.009. Whatever happens to the head, IG's attribution is unmoved
  -- it is substantially reading the input. "IG scored 0.683 once" was a much
  weaker way to say this.

`evaluate.py` now runs the check over `LABEL_RAND_SEEDS = (0,1,2,3,4)` and
records `mean`/`sd`/`min`/`max`/`similarities`. `similarity_to_original` is
kept as the first seed's value so older readers keep working. The dashboard
shows `mean +/- sd [min, max]`, falling back to the single value (labelled as
such) for an older file. Cost: about +6 minutes on the suite.

**For the write-up:** cite Grad-CAM 0.24 +/- 0.55 against IG 0.71 +/- 0.01
over five seeds, and make the argument about the sd. Where one precise number
is needed, use the cascading-randomization stages instead -- those are fully
seeded and reproduce across runs.

### Also fixed

- **The evaluate/audit ordering is now in the artifact, not just in prose.**
  `evaluate.py` writes `audits: {status: not_available, reason: "...re-run
  scripts/audit.py"}`, in the same idiom the genuinely-deferred sections use.
  The dashboard previously crashed on that shape (`'%.3f'|format(None)`); it
  is now guarded and was rendered in all three states to confirm.
- README documents the ordering constraint and the optional scripts.

### New in this pass

```
scripts/screening_threshold.py          select on calibration, report on test
tests/test_screening_threshold.py       11 tests
tests/test_sanity_check_determinism.py  3 tests
src/xai_cxr/evaluation/sanity_checks.py tf.random.set_seed + label_randomization_repeated
```

Test count 59 -> 73.

### State

`models/metrics.json` is single-provenance again: one `evaluate.py` run
stamped `2026-09-23T18:36:49Z` against model `2a279eeff356`, with `audits`
appended by the `audit.py` run immediately after and nothing patched in
between. `sanity_checks.recomputed_at` is **absent**, which is the marker to
check -- if it reappears, the file is a hybrid again.

**Nothing is committed.** All of the above is in the working tree.
