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
