# Decisions log (P-10)

One entry per real choice, written as it's made -- this file exists so April's
methodology-justification writing and the viva don't require reconstructing
why something was done a particular way. Append to it; don't rewrite history
in it.

## 2026-09-06 -- Engineering-feasible slice of "What Done Looks Like"

**Decision:** Implement every component of the 69-item spec that's pure
engineering and runs against the Kermany dataset already in the repo; defer
everything that needs a new credentialed dataset (VinDr-CXR, CheXpert),
concept labels, a real clinician study, or an academic-writing/legal
deliverable (papers, ethics application, copyright filing).
**Why:** Those deferred items need external institutional action (data-use
agreements, IEC approval, recruited radiologists, journal submission) that
can't happen inside a coding session. Building the engineering substrate now
means the moment D-2/D-5/D-6 land, the multi-label model, concept
bottleneck, and localisation metrics are additive work on top of a working
pipeline, not a rewrite of one.
**See also:** the "Deferred" table in the session's implementation plan;
each deferred item is also marked `not_available` with a `reason` in
`models/metrics.json` and in the relevant template, rather than silently
dropped.

## Patient-level split derivation (D-7, G-7)

**Decision:** Derive patient/study groups from filenames --
`person<ID>_bacteria|virus_*.jpeg` for PNEUMONIA, `IM-<ID>-*.jpeg` /
`NORMAL2-IM-<ID>-*.jpeg` for NORMAL -- rather than treating every image as
its own patient.
**Why:** The Kermany folders as distributed carry no explicit patient-ID
column. The PNEUMONIA filenames do encode a patient id directly (confirmed:
`person1000_bacteria_2931.jpeg` and `person1000_virus_1681.jpeg` are the same
patient). The NORMAL-side grouping (by `IM-<ID>`) is a best-effort proxy --
there's no equivalent confirmation that two NORMAL images sharing an `IM`
prefix are the same patient rather than sequential accession numbers. Treat
the NORMAL grouping as directionally correct, not verified.
**How to apply:** If VinDr-CXR (D-2) or another source with real patient IDs
is added later, prefer its ID column outright and only fall back to filename
parsing for Kermany.

## ROAD faithfulness metric is an approximation (E-7)

**Decision:** Implement ROAD (Rong et al. 2022) as inpainting-based pixel
substitution without the paper's retrain-on-imputed-inputs step.
**Why:** Full ROAD retrains (or fine-tunes) the classifier on inpainted
images at each masking fraction to avoid the substitution itself being an
out-of-distribution signal the model keys on. That's a training run per
fraction per method -- not affordable in this pass. The inpainting
substitution (`cv2.INPAINT_TELEA`) still avoids the flat-fill artifact that
motivated ROAD over plain deletion, so it's a meaningful improvement over
deletion/insertion alone, just not the full published procedure. Every
place this number is surfaced (`metrics.json`, the dashboard) carries a
`road_approximate: true` flag next to it.

## Lung-field ROI for the shortcut audit is geometric, not segmented (A-1)

**Decision:** `xai_cxr.audits.shortcut_audit.approximate_lung_field_mask`
is two fixed rectangles positioned by eyeballed proportion of a centred,
upright paediatric chest film, not a real lung segmentation.
**Why:** A real segmentation needs either a trained segmentation model or
VinDr/similar mask annotations (D-2, deferred). The geometric approximation
is enough to catch a gross failure mode (all attribution mass outside any
plausible lung location) but should not be read as a precise localisation
metric -- that's what E-6 (deferred, needs D-2) is for.

## Label-randomization sanity check is a head-refit proxy, not a full retrain (E-8)

**Decision:** `xai_cxr.evaluation.sanity_checks.label_randomization_test`
refits only the (frozen-backbone) head on shuffled labels, on a 150-image
subset, for 2 epochs, rather than retraining the whole classifier from
scratch on shuffled labels for a full run.
**Why:** A full retrain is the same ~15-epoch, ~17-minutes-per-epoch cost as
the real training run, and running the sanity-check suite would double that
just to get one comparison heatmap. Refitting the head still genuinely
breaks the model's real learned mapping from image to label; it's a speed
trade-off on how much of the network's history is destroyed, not a fake
result.

## Score-CAM channel sampling (X-3)

**Decision:** `xai_cxr.explain.scorecam` supports an optional
`max_channels` parameter to use only the highest-energy conv channels
instead of all 512.
**Why:** Score-CAM's cost is one forward pass per channel; at 512 channels
per image this is the slowest method in the registry by a wide margin (see
`metrics.json.runtime`). The evaluation suite calls it with the full channel
count by default (so the reported runtime is the real, honest cost) --
`max_channels` exists for anyone who wants a faster approximate version for
interactive use, not to hide the cost in the eval numbers.

## configs/explain.yaml wasn't actually wired to the explanation methods (fixed 2026-09-07)

**What happened:** The first real `scripts/evaluate.py` run (started 2026-09-06
14:07) was killed after 17.5 hours wall-clock (~11.25 CPU-hours) without
finishing. Root cause: `occlusion.py` and `scorecam.py` took `patch`/`stride`/
`max_channels` as function arguments with hardcoded defaults (patch=16,
stride=8 -> 784 positions; `max_channels=None` -> all 512 conv channels), and
every caller (`evaluate.py`, the app, the audits) invoked them without ever
passing those arguments -- `configs/explain.yaml`'s `occlusion_patch`,
`occlusion_stride`, and a (missing) channel cap were never actually read by
anything. A literal config-vs-code drift, exactly what CLAUDE.md's rules
exist to prevent, that I introduced and didn't catch until it cost most of a
day.
**Fix:** `xai_cxr.explain.registry.explain()` now resolves per-method kwargs
from `ExplainConfig` via a new `config_kwargs()` helper before calling the
method function, so every call site (already routing through this one
function) picks up the config automatically -- no call site needed to
change. Added `scorecam_max_channels` (default 64, was uncapped) to
`ExplainConfig`; widened `occlusion_patch`/`stride` to 32/16 (4x fewer
positions); dropped `shap_background_size` to 5. Verified against a real
timing test on this machine: Score-CAM 17s, Occlusion 64s, SHAP 151s per
image (previously: unbounded/hours). `evaluate.py`'s own sample sizes
(`FAST_SAMPLE_N`, `SLOW_SAMPLE_N`, label-randomization subset) were also cut
further as a second, independent lever.
**How to apply:** If a future method needs its own speed knob, add it to
`ExplainConfig` *and* to `config_kwargs()` in the same change -- don't add a
function parameter with a bare default and assume a caller will pass it.

**Follow-up, same day:** the re-run still surfaced a second, unrelated cost
cliff -- `robustness_and_complexity` (E-9) recomputes the full explanation
once per perturbation to measure max-sensitivity. For a single-backward-pass
method (Grad-CAM) that's cheap; for Integrated Gradients (32 gradient steps
per call) it made one image take ~200s (1 base + 4 perturbation recomputes,
each a full 32-step IG call), which alone would have pushed the run past 90
minutes. Fixed by dropping `ig_steps` 32 -> 16 and
`robustness_and_complexity`'s default `n_perturbations` 4 -> 2 -- both
documented as accuracy/speed trade-offs in their own docstrings rather than
silently changed.

## SHAP heatmap shape bug (fixed 2026-09-07)

**What happened:** `xai_cxr.explain.shap_method.explain` returned a
`(224, 224, 3)` array instead of `(224, 224)` -- the code assumed
`GradientExplainer.shap_values()` on a single-sigmoid-output model always
returns either a list or a `(batch, H, W, C)` array, but shap 0.51.0
sometimes keeps a trailing `n_outputs` axis, making the real shape
`(batch, H, W, C, 1)`. `resize_to` (`cv2.resize`) silently accepts a
3-channel array and "succeeds", so the bug didn't surface until
`deletion_insertion_auc` tried to index a flattened-pixel array with
indices computed from the wrong (150,528-element) flattened shape --
`IndexError: index 138415 is out of bounds for axis 0 with size 50176`,
41 minutes into an evaluate.py run, after gradcam/gradcam++/IG/scorecam/
occlusion had all already finished.
**Fix:** `shap_method.explain` now checks `single.ndim` and drops a
trailing size-1 axis before summing channels, instead of assuming the
shape. Added an `assert heatmap.ndim == 2` in the shared `resize_to()`
helper so any future method with the same class of bug fails immediately,
at the method call, instead of silently producing a wrong-shaped heatmap
that only breaks a downstream consumer minutes or hours later.
**How to apply:** when wrapping a third-party explainability library,
don't trust its documented output shape across versions -- assert the
shape you actually need at the boundary where your code takes over.

## SHAP is compared, not favoured (X-6)

**Decision:** SHAP is registered like every other method, but its
`METHOD_INFO` note and the dashboard both call out that it's typically the
slowest method and has weaker fidelity on the X-ray's spatially-correlated
pixels than the gradient/perturbation methods it's compared against.
**Why:** The original proposal promised a SHAP comparison; the spec (X-6)
asks that it be "reported honestly, including its cost and its poor
fidelity" rather than presented as interchangeable with the others.

## 2026-09-20 -- Backbone is a registry key; VGG16 is no longer the default (M-1b)

**Decision:** Replace the hardcoded VGG16 backbone with
`xai_cxr.models.backbones.BACKBONES`, a registry of
`{densenet121, efficientnetv2s, efficientnetv2b0, convnext_tiny, resnet50v2,
vgg16}` selected by `backbone:` in `configs/model.yaml`. Default:
**DenseNet121**. VGG16 stays selectable.
**Why:** VGG16 was already an odd choice in 2019 and is indefensible now for
this task. It carries 14.7M convolutional parameters with no batch-norm and
no residual connections, costs roughly 5x DenseNet121's FLOPs per forward
pass for weaker features, and the weakness compounded here because the
backbone was *permanently frozen* -- the model was matching ImageNet natural-
image texture statistics and leaving all the adaptation to a 256-unit head.
DenseNet121 is the backbone CheXNet and most subsequent chest-radiograph work
uses, at half the conv parameters. The registry matters as much as the
default: the previous code named VGG16 in `build_model`, named `vgg16` in
`XAIModel.__init__`, and wrote `block5_conv3` into the config, so trying a
different backbone was a multi-file edit that touched the explanation methods.
**Cost, stated plainly:** VGG16's `block5_conv3` sits *before* the final
max-pool, so its CAM grid at 224x224 is 14x14. Every modern backbone ends at
stride 32, i.e. **7x7** -- a real loss of localisation resolution for a
project whose subject is explanation quality. Three mitigations, all in-repo:
`cam_layer: hires` selects the stride-16 stage (14x14, matching the old grid,
at the cost of shallower features); raising `img_size` scales the grid
linearly (320x320 -> 10x10); and the case reader now *states* the CAM grid
size on screen instead of letting bilinear upsampling imply a precision the
evidence does not have.
**How to apply:** `backbone: vgg16` + `cam_layer: block5_conv3` reproduces the
old feature extractor exactly, which is how the model-card comparison should
be run. Any checkpoint trained before this change must be retrained -- the
graph differs, and `app.py` now says so rather than failing obscurely.

## 2026-09-20 -- Two-stage training: the backbone was never fine-tuned (M-1b)

**Decision:** `scripts/train.py` now runs stage 1 (head warmup, backbone
frozen, LR 1e-3) then stage 2 (top `unfreeze_fraction` of the backbone
trainable, LR 1e-5, BatchNormalization kept frozen), each with its own
warmup-cosine schedule. `finetune_epochs: 0` restores the old behaviour.
**Why:** The previous recipe froze the backbone and never unfroze it, so no
part of the feature extractor ever saw a radiograph. That is the single
largest accuracy and saliency-sharpness left on the table, and it is also why
the old Grad-CAM maps were diffuse -- generic ImageNet edge/texture filters
have no notion of a consolidation.
**Why BN stays frozen:** with batches of 32 over a few thousand images,
letting BatchNormalization re-estimate its moving statistics during
fine-tuning drifts them away from the ImageNet statistics the still-frozen
lower blocks are calibrated against. It degrades silently -- training loss
looks fine, validation does not.
**Why two schedules, not one:** a single cosine across both stages would
decay the LR to nearly zero at exactly the epoch the backbone unfreezes.
**Also changed:** Adam -> AdamW (decoupled weight decay), added label
smoothing and optional focal loss, added AUPRC (AUROC saturates at this class
ratio), and `threshold_policy: min_sensitivity` as an alternative to Youden's
J -- Youden weights a missed pneumonia and a false alarm equally, which is not
the clinical trade-off, and that choice should be explicit rather than
implied.

## 2026-09-20 -- Custom input filter and attention-pooling head (M-1b)

**Decision:** Add three optional pieces, all **identity-initialised**, in
`xai_cxr.models.heads`: `LearnableWindow` (3 parameters, a differentiable
window/level in front of the backbone), `SpatialDropout2D` on the feature map,
and `AttentionPool2D` (gated attention pooling) replacing global average
pooling. All configurable; `window_layer: false, head: gap, spatial_dropout: 0`
reproduces the previous graph exactly.
**Why identity-initialised:** each piece reduces to the old model at step 0 --
`AttentionPool2D` zero-initialises its attention logits so it *is* GAP on the
first batch, and `LearnableWindow.strength` starts at 0. So turning them on
cannot silently change what the baseline was; it can only add capacity the
optimiser has to earn. Both properties are asserted in `tests/test_gradcam.py`
rather than claimed here.
**Why attention pooling specifically:** pneumonia is a focal finding. Global
average pooling divides every activation by the full 7x7 grid, averaging a
strong 3-cell response against 46 cells of clear lung -- a poor match for the
signal. Attention pooling lets the head weight the grid instead.
**Why spatial (channel) dropout:** ordinary Dropout on a conv tensor barely
regularises, because neighbouring pixels in a feature map are strongly
correlated and the surviving ones reconstruct the dropped ones. Dropping whole
channels does not have that escape hatch.
**Bonus, and a caveat:** the attention weights are registered as a seventh
explanation method (`attention`). It is the only *intrinsic* one in the suite
-- the weights are literally what the classifier used, so there is no
approximation step to be unfaithful. But attention says which cells were
*pooled*, not how the logit would move if a cell changed: it is unsigned, and
it can highlight a region the head then weights negatively. It is therefore
scored by the same deletion/insertion/ROAD suite as everything else rather
than trusted because it came from inside the model.

## 2026-09-20 -- Augmentation: horizontal flip is now opt-in (D-8)

**Decision:** Replace `RandomFlip('horizontal') + RandomRotation(0.05)` with a
config-driven stack (rotation, zoom, translation, contrast, brightness) and
default `horizontal_flip: false`.
**Why:** Mirroring a chest radiograph puts the heart in the right hemithorax
and flips the laterality marker -- it manufactures anatomically impossible
images, and the laterality marker is the exact shortcut the A-1 audit measures.
Training on mirrored images while auditing for laterality dependence is
working against ourselves. The added contrast/brightness jitter is the most
clinically realistic axis of variation the JPEGs still carry (exposure differs
across the machines and techs that produced this corpus), and nothing in the
pipeline normalised for it.
**Note:** augmentation now runs *after* batching, where the `Random*` layers
vectorise over the batch axis.

## 2026-09-20 -- Cascading randomization no longer names VGG blocks (E-8)

**Decision:** `sanity_checks.CASCADE_STAGES` (`['logits', 'dense', 'block5',
... 'block1']`) is replaced by: randomize the weighted *head* layers
output-first, then the backbone's weighted layers in
`N_BACKBONE_CASCADE_GROUPS` contiguous groups from the output down.
**Why:** The stage list was VGG16's block names. With any other backbone the
`block*` lookups matched nothing, so the test would have quietly randomized
only the head and then reported reassuring similarity scores for the untouched
backbone -- a sanity check that passes without checking anything is worse than
no sanity check. The head-layer list is also derived now, because `dense` is
optional (`dense_units: 0`) and `pool`/`head_bn` may or may not carry weights.

## 2026-09-20 -- Explanation overlays are alpha-ramped, not opaque (§8)

**Decision:** `xai_cxr.explain._common.colorize` now returns RGBA with the
alpha channel set to the attribution value (gamma 0.8), instead of opaque RGB.
**Why:** found by actually looking at the case reader rather than at the
markup. JET's low end is a saturated dark blue, so an opaque overlay painted
every *un*attributed region at the slider's opacity -- the parts of the
radiograph the model ignored were exactly the parts the reader could no longer
see. An explanation overlay that hides the anatomy it is explaining inverts
its own purpose. With alpha tied to attribution, zero-attribution regions are
fully transparent. The hue at a given pixel is unchanged, so the colour bar
still means what it says.
**Not changed:** `overlay()` (used by the failure gallery and
scripts/explain.py) still alpha-blends server-side at a fixed weight, because
those are flat PNGs for the docs, not an interactive layer.

## 2026-09-20 -- The image viewer no longer swallows the page scroll (§8)

**Decision:** Zoom is `Ctrl`/`Cmd` + wheel, the `+`/`-` keys, or explicit
toolbar buttons. A plain wheel scrolls the page.
**Why:** the viewer bound `wheel` with `preventDefault()`, so the result page
stopped scrolling whenever the cursor crossed either image -- and on a page
where the two images are the tallest elements, the cursor crosses them
constantly. Trapping the page scroll is only acceptable when the user asked
for it with a modifier. A zoom percentage readout and +/- buttons were added
because a modifier-gated gesture is not discoverable on its own.
**Found by:** the automation harness timing out on `Page.captureScreenshot`
after a wheel scroll -- the same symptom a user experiences as "the page is
stuck", which is worth noting as a reason to drive the real UI rather than
assert on rendered HTML.

## 2026-09-20 -- The dashboard says when its metrics describe another model (§8)

**Decision:** `/dashboard` compares `metrics.json`'s `model_hash` against the
hash of the checkpoint the app actually loaded, and banners a mismatch.
**Why:** every number on that page comes from the last `scripts/evaluate.py`
run. After a retrain, a backbone change, or a fresh clone with a shipped
`metrics.json`, those numbers describe a *different model* and nothing said
so. The page's own claim -- "every number is read live" -- was true and
misleading at the same time. Also relevant to the M-1b switch specifically:
anyone pulling this change has a metrics.json describing the old frozen VGG16.
**Related:** the deletion-AUC column's in-cell bar is drawn inverted
(`1 - value`), so a longer bar means a better method in both the deletion and
insertion columns. Drawn naively, the worst method had the most emphatic bar.

## 2026-09-20 -- Local GPU training stack: TF 2.10 on a 4 GB Quadro M1200

**Decision:** Train on the machine's NVIDIA Quadro M1200 via a separate
micromamba environment: Python 3.10 + CUDA 11.2 + cuDNN 8.1 + TensorFlow
2.10.1 + tensorflow-addons. The repo's own pin (TF 2.13, Python 3.11) is
unchanged and remains what the app, the tests and CI use.
**Why a second environment:** TensorFlow dropped native-Windows GPU support
after 2.10, so TF 2.13 sees no GPU on this machine no matter what CUDA is
installed. 2.10 in turn requires Python <= 3.10. WSL2 would keep TF 2.13 but
needs Administrator, which this account does not have.
**Consequences handled in code:** TF 2.10 has no `keras.optimizers.AdamW` and
no `BinaryFocalCrossentropy`. `build_optimizer` now falls back to
`tensorflow_addons.optimizers.AdamW` (same decoupled-decay algorithm) and, if
addons is absent, to plain Adam with an explicit RuntimeWarning -- silently
dropping weight decay would change the recipe without saying so.
`build_loss` raises a clear error rather than mis-training if focal loss is
requested on a TF that lacks it.
**Three obstacles worth recording:** (1) the Miniconda NSIS installer was
blocked outright ("Access is denied", no Mark-of-the-Web, so antivirus);
micromamba is a plain executable and worked. (2) `opencv-python` 4.12 requires
numpy >= 2, which TF 2.10 cannot use -- pinned to opencv 4.8.1.78 + numpy
1.23.5. (3) TF could not find any CUDA DLL despite all of them being present
in the env, because Python 3.8+ no longer resolves extension-module DLLs via
PATH; neither a Git-Bash PATH nor `os.add_dll_directory` fixed it. A small
batch wrapper that sets a real Windows `PATH` before launching python did.

## 2026-09-20 -- 320px at batch 8, and the learning rate that follows from it

**Decision:** `img_size: [320, 320]`, `batch_size: 8`, `learning_rate: 0.0005`,
`finetune_learning_rate: 0.000005`, `unfreeze_fraction: 0.60`.
**Measured, each in a fresh process** (sequential benchmarking in one process
lets an earlier OOM's fragmentation cause spurious later OOMs -- the first
pass wrongly reported 224px/batch-32 as OOM for exactly that reason):

    320px batch 8    4.31 h   peak 2662 MB   CAM grid 10x10
    224px batch 16   1.93 h   peak 2680 MB   CAM grid 7x7
    288px batch 12   OOM
    320px batch 12   hard crash

**Why 320px:** it doubles the pixel count and restores the CAM grid to 10x10,
better than the 7x7 the move off VGG16 cost us and better than VGG16's own
14x14-at-224 in absolute terms per unit of image. For a project whose subject
is explanation quality, paying 2.4 hours for that is the right trade.
**Why batch 8 is forced:** batch, not resolution, is the binding constraint on
4 GB -- DenseNet's concatenation-heavy activations scale harshly with batch.
288px/batch-12 OOMs while 320px/batch-8 fits comfortably at 2.66 GB.
**Why the LR moved:** 0.001 was chosen for batch 32. At batch 8 that is 4x more
and noisier steps at a rate tuned for 4x larger batches. Scaled by
sqrt(8/32) = 0.5 (square-root rather than linear, the usual choice for
Adam-family optimisers) for both stages. Leaving the mismatch in place would
have quietly cost accuracy.
**Note for anyone re-running on different hardware:** these three values are
hardware-derived, not principled defaults. On a 16 GB card, batch 32 at 320px
and `learning_rate: 0.001` is the configuration to use.

## 2026-09-21 -- Bugs found by actually running the full pipeline

Four defects that only surfaced once the thing ran end-to-end on real data.
Recorded because each is a class of mistake, not a one-off typo.

**1. A missing optional dependency took down the entire eval suite.**
`shap` was dropped from the GPU environment while resolving a numpy conflict
(opencv 4.12 needs numpy >= 2; TF 2.10 needs numpy < 1.24). Because
`explain/registry.py` imported `shap_method` at module scope, and that module
imports `shap` at module scope, `import xai_cxr.explain` raised
ImportError -- so evaluate.py, explain.py AND audit.py each died in about
four seconds, after a 2h14m training run had completed overnight. The cost of
one missing optional package should be one method out of seven, not the whole
suite. The import is now guarded and the registry degrades to six methods
with a warning. **Rule: a heavy optional dependency must never be imported at
the top of a module that everything else imports.**

**2. The checkpoint could regress across training stages.**
`ModelCheckpoint(save_best_only=True)` tracks its best value per `fit()` call.
With two stages this resets to -inf between them, so stage 2's first epoch
overwrote the file regardless of whether it beat stage 1.
`EarlyStopping(restore_best_weights=True)` does not cover this -- it also only
tracks within one `fit()`. Harmless in the run that exposed it (stage 2 epoch
1 scored 0.99416 against stage 1's best 0.99294) which is exactly why it is
worth recording: it only bites when fine-tuning *hurts*, i.e. precisely when
the earlier checkpoint matters most. Fixed by passing stage 1's best through
`initial_value_threshold`.

**3. Benchmarking several configs in one process gives false OOMs.**
TensorFlow does not reliably release VRAM between configurations, so an
earlier OOM fragments the allocator and causes spurious later ones. The first
benchmark pass reported 224px/batch-32 as OOM; re-run in a fresh process it
completed in 1.93 h. Every capacity benchmark now runs one config per
process.

**4. A synthetic-tensor benchmark does not predict real training speed.**
Measured 1.05 s/step on in-memory tensors; real training ran at 3.25 s/step
because the input pipeline re-decoded and re-resized all 4,120 JPEGs every
epoch, leaving the GPU ~50% idle and projecting 12.7 h instead of 4. Adding
`.cache()` after decode/resize (as uint8: 1.3 GB rather than 5.1 GB) brought
it to 1.12 s/step, within 7% of the compute-only figure. **Benchmark the real
`tf.data` pipeline, not synthetic batches.**

## 2026-09-21 -- Checkpoint portability across TensorFlow versions (verified)

**Checked, not assumed:** a model trained under TF 2.10.1 / Python 3.10 loads
and runs correctly under TF 2.13.0 / Python 3.11, which is what the Flask app
and the test suite use. `Preprocess`, `AttentionPool2D` and `LearnableWindow`
all round-trip through `.h5` because they are registered serializables, and
`get_backbone()` recovers the registry tag from the `Preprocess` layer (the
tag itself is a plain Python attribute and does not survive serialisation).
Verified outputs after reload: backbone densenet121, input (320, 320, 3),
feature map (10, 10, 1024), 7,566,788 params, Grad-CAM and the attention map
both functional. This mattered because the two-environment split on Windows
would otherwise have produced a model the app could not serve.

## 2026-09-21 -- `img_size` was configurable; everything that scales with it was not

**The pattern.** Raising `img_size` from 224 to 320 broke five separate places,
one after another, each surfacing only when the previous one was fixed:

| where | what was wrong |
|---|---|
| `integrated_gradients.explain` | pushed all `steps` interpolations as one batch |
| `scorecam.explain` | `scorecam_batch: 16` in configs, tuned at 224px |
| `occlusion.explain` | `batch_size=32` **hardcoded, not config-driven at all** |
| `shap_method` | `shap.GradientExplainer`'s own `batch_size=50` default |
| `sanity_checks.label_randomization_test` | `batch_size=16` hardcoded, and this one is a *training* step |

**Why it kept happening.** The project treated `img_size` as a config value
(G-4, "no hyperparameter as a literal in code") but left every constant whose
memory cost is a function of `img_size` as a literal. A resolution change is
not a local change: peak memory scales with `batch x H x W`, so every batch
constant in the codebase is implicitly a function of the resolution. Making
one of the two configurable and not the other guarantees this failure, and
guarantees it surfaces one site at a time, because each OOM hides the next.

**Fix.** One knob, `inference_batch` in configs/explain.yaml, now drives
Occlusion, Score-CAM, SHAP and IG's chunking; the label-randomization refit
uses `DataConfig.batch_size` (it is training, not inference, so it belongs on
the training budget). None of these change a method's output -- only how the
work is batched. That distinction drove every choice here: the tempting
shortcuts (`ig_steps` down, SHAP `nsamples` down) would have "fixed" the OOM
by degrading the estimates, which for IG means giving up the completeness
axiom the method exists for.

**Diagnostic note worth keeping.** SHAP's failure arrived as
`NotFoundError: No algorithm worked!`, which reads like a missing CUDA kernel.
The real cause was in the sub-messages: `RESOURCE_EXHAUSTED` from every cuDNN
engine in turn, i.e. plain OOM. Reading the error *class* rather than the full
traceback would have sent this down a version-incompatibility rabbit hole.

**If you re-run at a different resolution:** `inference_batch` and
`batch_size` are the two knobs to revisit, and the occluder geometry
(`occlusion_patch`/`occlusion_stride`) should scale with `img_size` to keep
the occluded fraction and the position count constant.

## 2026-09-21 -- M-1b vs the VGG16 baseline: the sensitivity gap is a threshold artefact

The headline comparison on the same 582 test images
(`python scripts/compare_to_baseline.py`):

| | VGG16 (frozen, M-1) | DenseNet121 (M-1b) |
|---|---|---|
| AUROC | 0.9941 [0.9892, 0.9976] | 0.9947 [0.9908, 0.9980] |
| Sensitivity | 95.6% | 94.0% |
| Specificity | 96.7% | 98.7% |
| Brier | 0.047 | 0.063 |
| **False negatives** | **19** | **26** |
| False positives | 5 | 2 |
| Youden threshold | 0.2174 | 0.3447 |

Read naively this says M-1b **misses seven more pneumonias** -- the one number
that matters clinically -- in exchange for three fewer false alarms. That
reading is wrong, and the reason is worth recording because it is a trap any
before/after comparison of two independently-thresholded models falls into.

**AUROC is threshold-free; sensitivity is not.** The two AUROCs are
statistically indistinguishable (the CIs overlap heavily; M-1b's lower bound
0.9908 does not clear the baseline's point estimate 0.9941, so per the caveat
in §8 of HANDOFF.md we do *not* claim an improvement). Identical ranking
quality plus different sensitivity means the models differ in *where the
threshold landed*, not in what they can tell apart.

`scripts/operating_point.py` caches the per-image test scores
(`runs/test_scores.npz`) and sweeps the threshold to compare like for like:

| operating point | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|
| M-1b @ Youden J (reported) | 0.3447 | 94.0% | 98.7% | 26 | 2 |
| M-1b @ baseline's specificity (96.7%) | 0.2224 | 95.4% | 96.7% | 20 | 5 |
| M-1b @ baseline's sensitivity (95.6%) | 0.2220 | 95.6% | 96.7% | 19 | 5 |
| VGG16 @ its own Youden J | 0.2174 | 95.6% | 96.7% | 19 | 5 |

Matched on specificity the gap collapses from +7 missed pneumonias to +1 of
431 (+0.23%, one image). Matched on sensitivity the confusion matrices are
*identical* (19 FN / 5 FP). The seven-case gap was entirely the threshold.

**The real finding is that Youden's J is the wrong objective here.** J
maximises `sensitivity + specificity - 1`, which weights a missed pneumonia
exactly as heavily as a false alarm. For a triage model those costs are not
remotely equal: a false positive costs a radiologist's second look, a false
negative sends an untreated pneumonia home. M-1b's better-separated scores
let J buy specificity it did not need (98.7%, against 5 false positives
avoided) at the cost of sensitivity it did need.

Re-tuned against a screening objective -- maximise sensitivity subject to a
specificity floor -- the same model, unchanged, gives:

| objective | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|
| max sensitivity, specificity >= 95% | 0.1465 | 97.7% | 95.4% | 10 | 7 |
| max sensitivity, specificity >= 90% | 0.0692 | 99.1% | 90.1% | **4** | 15 |

**4 missed of 431** against the baseline's 19, for ten extra false positives
out of 151 normals. That is the trade a triage tool should be making.

**Decision.** The reported threshold stays Youden's J (it was fixed on val
before the test split was touched, and moving it now, after seeing test
results, would be threshold-hacking the headline number). `tune_threshold_at_
sensitivity` already exists in `src/xai_cxr/models/` for the screening policy;
`threshold_policy` is recorded per run in `runs/results_table.csv` so the two
are never confused. Any clinical deployment should re-tune on a calibration
split against an explicit cost ratio, not J.

**Caveat on the sweep.** The rows above are computed *on the test split*, so
they are an illustration of the achievable trade-off, not honest held-out
estimates of it. A deployed threshold must be tuned on val/calibration data.

**Also visible in the table:** M-1b's Brier score is worse (0.063 vs 0.047)
despite the better ranking. Better discrimination with worse calibration is
the expected signature of fine-tuning with label smoothing; the conformal
layer (U-2) is what the UI actually relies on for uncertainty, and its
empirical coverage holds at all three alphas, so this was not treated as a
blocker. A temperature-scaling pass on the calibration split is the obvious
cheap fix and is not done.

## 2026-09-21 -- Sanity checks reported NaN, and metrics.json was not valid JSON

`sanity_checks.*.cascading_randomization.stages[].similarity_to_original` came
back `NaN` for 7 of the 10 cascade stages on both Grad-CAM and IG -- every
stage from `head_bn` downwards.

**Cause.** Randomizing a BatchNorm layer draws new gamma/beta (and, from a
fresh initializer, moving statistics that do not match the data), which drives
the activations into a regime where the post-ReLU map is *constant*. A rank
correlation needs a non-zero standard deviation in both inputs; against a
constant vector the denominator is 0 and both scipy and numpy return NaN.
Once the head BN is randomized the map stays degenerate for every later stage,
so the NaN propagates down the rest of the cascade.

**Why NaN was the wrong output, twice.**

1. *It is not zero.* A similarity of 0.000 asserts "the explanation is
   unrelated to the original", which is a strong and checkable claim. The
   truth here is weaker: the comparison has no defined value. Reporting a
   number where there is none is the sort of thing a reviewer catches.
2. *Bare `NaN` is not valid JSON.* Python's `json` module emits and accepts it
   by default, so the file round-tripped cleanly inside the project while
   being rejected by `JSON.parse`, `jq`, Go's `encoding/json` and essentially
   every non-Python reader. For a file whose whole purpose is to be the
   portable reproducibility artifact, that is a real defect, and it was
   invisible precisely because every consumer in-repo was Python.

**Fix.** `_rank_similarity` now checks for a constant input before correlating
and returns `None` (-> JSON `null`), and also maps any non-finite correlation
to `None`. The dashboard renders `null` as "undefined (degenerate map)" rather
than a bar of length `NaN` px; `compare_to_baseline.py` prints `undef` and
counts how many stages were undefined. Two regression tests were added in
`tests/test_metrics_schema.py`: one asserts the whole file parses with
`parse_constant` raising, one asserts `_rank_similarity` returns `None` for a
constant map and 1.0 for a map against itself.

The existing `models/metrics.json` was migrated in place rather than
regenerated -- 14 non-finite values rewritten as `null`, every other value
byte-identical, with an `encoding_note` field recording what changed. A full
`evaluate.py` re-run costs ~40 min and would have produced the same numbers.

**What this costs the E-8 claim.** Cascading randomization is only informative
for `logits` and `dense`. Both methods do fall away there (Grad-CAM
1.000 -> 0.477 -> -0.335, IG 1.000 -> 0.773 -> 0.751) and label randomization
is defined throughout, so the conclusion that both methods depend on the
trained weights still holds. But the stronger textbook statement -- similarity
decays monotonically to zero as randomization propagates backwards through the
network -- is **not** something this run demonstrates, and the write-up should
not claim it. Recovering the deeper stages needs a randomization scheme that
preserves activation scale, e.g. resampling BatchNorm parameters from their
fitted distribution instead of from a fresh initializer. Not attempted.

## 2026-09-21 -- The cascading randomization test was broken by its own BatchNorm handling

Fixing the NaN reporting (entry above) exposed that the NaNs were not an
inherent property of the test. They were a bug in `_randomize_layer`.

**Cause.** The randomizer redrew any weight with `ndim > 1` from N(0, 0.05)
and zeroed everything else -- a reasonable rule for kernels and biases.
BatchNormalization's four weights (gamma, beta, moving_mean, moving_variance)
are all 1-D, so the rule set **gamma = 0**. A BN layer with gamma = 0 emits
beta for every channel, i.e. a constant. From the first BN in the cascade
downwards every heatmap was flat, so every rank correlation was undefined --
7 of 10 stages, on both methods.

**Fix.** BatchNormalization is now handled separately: gamma and beta are
**permuted across channels**, and the moving statistics are left alone.
Permutation destroys the learned per-channel correspondence (which is what the
test probes) while preserving the distribution of scales exactly, so the
activations stay in a regime where an explanation is still defined. The moving
statistics are dataset statistics, not learned parameters, so Adebayo et al.'s
"randomize the learned weights" gives no licence to touch them.

Re-running the cascade (`python scripts/refresh_sanity_checks.py --write`,
which recomputes only this block rather than burning a 40 min full evaluate)
gives all 10 stages defined for both methods:

| stage | Grad-CAM | Integrated Gradients |
|---|---|---|
| original | +1.000 | +1.000 |
| logits | +0.477 | +0.773 |
| dense | **-0.335** | +0.751 |
| head_bn | -0.332 | +0.751 |
| pool | -0.382 | +0.688 |
| backbone 1/5 | +0.208 | +0.632 |
| backbone 5/5 | +0.230 | **+0.629** |

**This is now a real result, and it is not a flattering one for IG.**

- **Grad-CAM passes.** Similarity collapses immediately and stays near zero or
  negative for the rest of the cascade. Its explanation depends on the trained
  weights, which is what E-8 is asking.
- **Integrated Gradients largely fails.** With *every layer in the network
  randomized* its attribution map still correlates at +0.63 with the map from
  the trained model. An explanation that survives the destruction of the model
  it claims to explain is substantially describing the input, not the model --
  this is exactly the failure Adebayo et al. (2018) report for
  gradient-and-input-style methods, which behave partly as edge detectors.

The label-randomization proxy told the same story all along (Grad-CAM 0.266,
IG 0.683) and is unaffected by this fix, so it was not rerun.

**Consequence for the write-up.** IG should not be presented as a trustworthy
explanation for this model on the strength of its faithfulness numbers. Note
that it also has the *worst* deletion AUC in the benchmark (0.908, against
Grad-CAM's 0.610) -- two independent diagnostics agreeing. Grad-CAM is both
the cheapest method (0.7 s vs 3.5 s) and the one that passes both.

## 2026-09-21 -- Temperature scaling recovers calibration error, but not the Brier gap

M-1b's Brier score regressed against the baseline (0.047 -> 0.063), so the
obvious cheap fix was temperature scaling (Guo et al. 2017): one scalar T
dividing the logit, fitted by NLL on held-out data.

Implemented in `src/xai_cxr/evaluation/calibration.py`, fitted on the
**calibration** split and evaluated on test (`scripts/calibrate.py`), and
wired into `evaluate.py` so future runs produce it without a second inference
pass over the 563 calibration images.

| metric (test) | before | after | change |
|---|---|---|---|
| Brier | 0.0630 | 0.0624 | **-1.0%** |
| ECE | 0.1249 | 0.1065 | **-14.8%** |

Fitted **T = 0.718**, which *sharpens* rather than softens. That direction is
the informative part: the model is **under**-confident, which is the expected
consequence of training with label smoothing. The usual intuition that a
fine-tuned network is over-confident does not hold here.

**The honest conclusion is that this did not fix what it was reached for.**
ECE -- the part temperature scaling can actually move -- improves by a useful
15%. Brier barely moves. Brier decomposes into calibration plus refinement,
and the residual is refinement: the 26 missed pneumonias are confidently
wrong, and no monotonic rescaling of the probability axis can help a case that
is on the wrong side of the decision boundary. So the Brier gap against the
baseline is **not** a calibration-scale artefact, and should not be reported
as one.

**Deliberately not applied to the headline numbers.** The reported
`brier_score` and `decision_threshold` stay on the raw probability scale;
the fitted temperature is recorded alongside them under
`classification.calibration.temperature_scaling`. Restating the headline on a
rescaled axis would make this run incomparable with the archived baseline, and
the threshold would silently stop meaning what it meant when it was tuned.
Temperature scaling is monotonic, so AUROC and every ranking metric are
unchanged by construction -- asserted in `tests/test_calibration.py`.

## 2026-09-21 -- Mobile layout, finally verified (and one real bug)

The mobile layout had been "written but unseen" across two sessions because
the browser tooling was unavailable. Verified this session with headless
Chrome instead, which needs one trick worth recording:
`chrome --headless --window-size=360,...` **does not** give a 360px viewport.
Chrome clamps the window to roughly 500px wide on Windows and then simply
crops the screenshot, which looks exactly like a horizontal-overflow bug and
sent this investigation down a false trail once already. Rendering the app
inside a `360px`-wide `<iframe>` on a wider page gives the inner document a
genuine 360px viewport, and media queries respond to it correctly.

All five pages check out at 360px: the stats grid collapses 4 -> 2 -> 1, plots
go single-column, the nav wraps and its link row scrolls, and every table sits
in an `overflow:auto` card.

**One real bug found and fixed.** The dashboard's "Evaluated model" line
prints an absolute `model_path`. A Windows path offers break opportunities
only at its backslashes, and one segment is wider than a 360px viewport, so
the line overflowed horizontally.

The first fix -- `overflow-wrap:anywhere` on `code` globally -- was **wrong**
and is worth recording as a near-miss: the audit log's timestamps are
monospace too, and breaking those anywhere shattered each one into seven
lines down a narrow column instead of wrapping cleanly at the hyphen. The
shipped rule is scoped to `.meta code`, which is used on exactly that one
dashboard line.

## 2026-09-23 -- The screening operating point was selected on test; the honest number is 8, not 4

**Decision:** Select the screening threshold on the calibration split and
report it on test (`scripts/screening_threshold.py`), and correct the
headline screening claim from 4 missed pneumonias to **8**.

**Why:** The threshold sweep in `scripts/operating_point.py` picks the
threshold on the test split and then scores it on that same split. That is
the right procedure for the question that script asks -- "is the sensitivity
gap against the VGG16 baseline a threshold artefact?" -- because it asks what
each model *could* achieve and treats both identically. It is the wrong
procedure for the different claim §9 of HANDOFF went on to make, that a
screening operating point "misses 4 of 431". Choosing the best of ~582
candidate thresholds on the same 582 images it is then evaluated on reports
the maximum of a noisy quantity, not an estimate of it.

Re-selected on the 563-image calibration split -- never trained on, disjoint
by patient from test -- and applied unchanged to test:

| objective | selected on | thr | sens | spec | FN | FP |
|---|---|---|---|---|---|---|
| max sens, spec >= 90% | **calibration** | 0.0957 | 98.1% | 91.4% | **8** | 13 |
| max sens, spec >= 90% | test (optimistic) | 0.0692 | 99.1% | 90.1% | 4 | 15 |
| max sens, spec >= 95% | **calibration** | 0.1819 | 97.2% | 96.7% | **12** | 5 |
| max sens, spec >= 95% | test (optimistic) | 0.1465 | 97.7% | 95.4% | 10 | 7 |

So the selection optimism is **4 pneumonias at the 90% floor** -- the
previously quoted figure was exactly twice as good as the held-out result.

**What survives:** the argument, entirely. 8 missed of 431 against the
baseline's 19 at a comparable specificity is still the substantive finding,
and it is still evidence that Youden's J is the wrong objective for triage.
Only the number changes, and it changes from an unquotable one to a quotable
one.

**How to apply:** quote 8/431 at the 90% floor, and cite
`runs/screening_threshold.json` (which records both the honest and the
optimistic row, deliberately, so the gap is auditable). `operating_point.py`
is unchanged and still correct for its own question; the two scripts answer
different questions and both are kept.

**Note on split reuse:** the calibration split is already spent on conformal
prediction (U-2). Reusing it here is deliberate -- both uses need only
held-out scores and neither fits parameters the other consumes -- and it is
preferable to using val, which early stopping has already selected on.

## 2026-09-23 -- The label-randomization check was not reproducible, and it looked like it was

**Decision:** Seed TensorFlow's global RNG inside
`label_randomization_test` (`tf.random.set_seed(seed)`), and re-run
`evaluate.py` so `metrics.json` records a value the current code actually
reproduces.

**Why:** Re-running the evaluation suite end-to-end reproduced every
classification and faithfulness number bit-for-bit -- AUROC, the confusion
matrix, Brier, all seven deletion AUCs, and all 20 cascading-randomization
stages. One number did not: Grad-CAM's label-randomization similarity came
out **0.090** against the **0.266** recorded on 2026-09-21.

The function looked deterministic and was not. Both of its explicit random
draws -- the shuffled labels and the head re-initialisation -- take
`RandomState(seed)`. But the refit calls `clone.fit(...)`, and the head
carries three dropout layers (`spatial_dropout`, `dropout`, `head_dropout`,
rates 0.1/0.4/0.2). Dropout draws its masks from TensorFlow's *global* RNG,
which nothing seeded. So the one number in the suite that depends on a
training step was the one number that moved.

**Why it matters more than the size of the gap suggests:** the swing
(0.266 -> 0.090) is wider than the Grad-CAM/IG difference this check is cited
to establish. The conclusion is unaffected in direction -- IG sits at 0.683
and 0.685 across the two runs while Grad-CAM moves between 0.09 and 0.27, so
Grad-CAM falls much further either way -- but a reviewer who re-ran the suite
would have got a different headline number from the one in the write-up, out
of a file whose entire purpose is reproducibility.

**How to apply:** quote the E-8 label-randomization figures as "Grad-CAM
falls far below IG" and cite both values, rather than leaning on the third
decimal place. The cascading-randomization stages are fully seeded and *are*
reproducible to the bit; prefer them where a precise number is needed.

**Residual nondeterminism:** GPU convolution backprop is still
nondeterministic at the last decimal or two (TF is not run with
`TF_DETERMINISTIC_OPS`). Seeding removes the dominant term, not all of it.
`tests/test_sanity_check_determinism.py` pins the contract -- and was checked
against the unfixed code, where it fails, so it is not a test that would pass
regardless.

**Note on scope:** `tf.random.set_seed` mutates global state. That is safe
for the one caller: `evaluate.py` runs the sanity checks second-to-last, and
the only step after them (E-11 runtime timing) does not consume the RNG
stream.

> **Extended below.** Seeding makes the number reproducible, which is
> necessary but not sufficient: measuring the spread across seeds showed the
> check itself has an sd of ~0.55 for Grad-CAM, so *any* single seeded value
> is an arbitrary pick from a wide distribution. See the next entry.

## 2026-09-23 -- The label-randomization check has an sd of 0.55, and that spread IS the result

**Decision:** Report the label-randomization check as a distribution over five
seeds (`label_randomization_repeated`, new `mean`/`sd`/`min`/`max`/
`similarities` fields) rather than as one number, and state the E-8 conclusion
in terms of *stability* rather than in terms of which method scores lower.

**Why:** Seeding the check (previous entry) made it reproducible but not
meaningful. Running it across five seeds on the trained model:

| method | mean | sd | min | max |
|---|---|---|---|---|
| Grad-CAM | +0.239 | **0.553** | -0.440 | +0.812 |
| Integrated Gradients | +0.712 | **0.009** | +0.702 | +0.723 |

Grad-CAM's similarity ranges over 1.25 of the 2.0 the statistic can span. Any
single run is therefore close to uninformative -- and the three runs this
project actually performed reported 0.266, 0.090 and 0.734, which read as
"Grad-CAM clearly passes", "Grad-CAM emphatically passes" and "Grad-CAM is
indistinguishable from IG" respectively. The write-up would have quoted
whichever run happened to be last.

**What the spread means -- this is the part worth understanding.** Adebayo et
al.'s test asks whether an explanation *changes* when the model stops being
the trained model. It does not ask for a low correlation as such; it asks for
dependence on the weights.

- Grad-CAM's map swings from anti-correlated (-0.44) to strongly correlated
  (+0.81) depending only on what the shuffled-label refit did to the head.
  That is an explanation tracking the model. It **passes**, and the variance
  is the evidence, not a defect in the measurement.
- IG returns +0.70 to +0.72 **every time**, sd 0.009. Whatever the refit does
  to the head, IG's attribution is unmoved. That is an explanation that is
  substantially reading the *input*, which is precisely the failure mode the
  test exists to detect, and it is a far stronger statement of it than "IG
  scored 0.683 once".

So the original conclusion survives intact; the *statistic* supporting it was
wrong. Quote mean and sd.

**How to apply:** cite Grad-CAM 0.24 +/- 0.55 against IG 0.71 +/- 0.01 over
five seeds, and make the argument about sd, not about which mean is lower.
Where a single precise number is wanted, use the cascading-randomization
stages instead: those are fully seeded and reproduce to five decimal places
across runs (the residual drift is GPU convolution nondeterminism).

**Cost:** five refits per method instead of one, about +6 minutes on the whole
suite. `similarity_to_original` is retained as the first seed's value so older
readers of `metrics.json` and the dashboard keep working; the dashboard shows
`mean +/- sd [min, max]` when the aggregate is present and falls back to the
single value, labelled as such, when it is not.

**Not done:** five seeds is enough to separate sd 0.55 from sd 0.01, which is
the only distinction being drawn. It is not enough to put a tight interval on
either mean, and no significance test is claimed on them.
