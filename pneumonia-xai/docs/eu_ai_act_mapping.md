# EU AI Act mapping (A-4)

Maps this build's design to Articles 12 (record-keeping), 13 (transparency),
and 14 (human oversight) of the EU AI Act, whose high-risk-system
obligations became applicable 2 August 2026. **This is design evidence for a
research prototype, not a conformity assessment or certification** -- a
diagnostic aid used in an actual clinical pathway would need a full
high-risk-system conformity assessment (which is out of scope for a
capstone project and not something this document claims).

## Article 12 -- Record-keeping / logging

**Requirement (paraphrased):** automatic logging of events while the system
operates, sufficient to identify situations that may affect its behaviour
and facilitate post-market monitoring.

**What exists:** `xai_cxr.auditlog` -- one SQLite table, `audit_log`,
append-only by database trigger (not just convention: an `UPDATE` or
`DELETE` against it raises `ABORT`). Every inference logs its input hash,
model hash, predicted label, full conformal prediction set, and abstention
flag; every clinician action (agree/disagree-with-reason/referral
acknowledgement) logs against the same case UID. Viewable at `/audit`,
exportable as CSV.

**Gap:** no log retention/rotation policy, no tamper-evidence beyond the
append-only trigger (e.g. no hash chaining), and no operator-side
authentication on who is performing an action -- all reasonable next steps
for a system actually approaching deployment, not implemented here.

## Article 13 -- Transparency and provision of information to users

**Requirement (paraphrased):** the system must be sufficiently transparent
for users to interpret its output and use it appropriately; instructions for
use must cover intended purpose, performance metrics, and known
limitations.

**What exists:** `/dashboard` renders live classification, explanation-
quality, sanity-check, and conformal-coverage metrics straight from
`models/metrics.json` -- including explicit `not_available` states with a
`reason` for anything not yet computed, rather than a stale or fabricated
number. `docs/model_card.md` states intended use, out-of-scope use, and
known failure modes. The case reader shows a calibrated prediction set
(with its coverage level) instead of a bare, uncalibrated confidence
percentage, and every page carries the "research/educational use only, not
a substitute for professional diagnosis" disclaimer.

**Gap:** no user-facing instructions-for-use document separate from the
model card/README, and no versioned "what changed" release notes -- both
reasonable for a real deployment, not built here.

## Article 14 -- Human oversight

**Requirement (paraphrased):** the system must be designed so natural
persons can effectively oversee it, including the ability to decide not to
use an output, or to override/reverse it.

**What exists:** the abstention banner (visually distinct from either
prediction outcome -- amber, not red/green) fires whenever the conformal
prediction set contains more than one label, i.e. whenever the model has no
statistically defensible single answer at the chosen coverage level, and
explicitly recommends referral rather than presenting a forced guess. The
case reader's "I agree" / "I disagree" (with a required free-text reason)
capture writes directly to the Article-12 log, so an override is both
possible and recorded. The concept-intervention panel that would let a
clinician directly edit the model's reasoning is not implemented (it
depends on M-3, deferred).

**Gap:** no role-based access control distinguishing "clinician" from
"anyone with the URL" -- oversight actions here are unauthenticated in this
prototype.
