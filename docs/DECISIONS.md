# Architectural Decisions

Each record states the decision, the reasoning, the alternatives rejected, and
the trade-off accepted. A decision with no stated cost has not been thought
through.

Add new records at the end. Do not delete superseded ones — mark them.

---

## D-001 — Python 3.11 as the target interpreter

**Status:** Accepted · Milestone 2

**Decision.** Target Python 3.11 in a project-local `.venv/`.

**Reasoning.** The detection stack lags new interpreter releases in prebuilt
wheel availability. Newer versions can force source builds or block installs
entirely. 3.11 has the broadest support across every dependency here.

**Rejected.** The newest available interpreter — no benefit to this project,
material compatibility risk.

**Trade-off.** Newer language features are unavailable. None are needed.

---

## D-002 — Configuration read from environment at import time

**Status:** Accepted · Milestone 2

**Decision.** `config.py` resolves paths and constants once at import, exposing
them as module-level `Final` values, each overridable by an `MD_*` variable.

**Reasoning.** Module constants are simple to read and cheap to access. Env-var
overrides mean the same code runs against different data with no edits — which
the smoke test exercised directly by redirecting all output to a temporary
location.

**Rejected.** A config object threaded through every call — more ceremony than
this project needs. A config file — a second format to parse when environment
variables already cover the need.

**Trade-off.** Values are fixed at import, so changing the environment
mid-process has no effect, and tests must `monkeypatch` module attributes
rather than set variables. Accepted: the tests do exactly that, and it keeps
the common path simple.

---

## D-003 — Resource verification is separate from resource use

**Status:** Accepted · Milestone 2

**Decision.** `utils/resources.py` reports availability without raising.
Callers decide what to do.

*Amended in the Milestone 2 review (D-015):* this record originally described a
`require()` helper that raised on missing resources. It was never called in
production and has been removed. The principle stands unchanged — checks report,
callers decide.

**Reasoning.** The model and dataset are not available during development, so
"resource missing" is a normal state, not an exception. Separating the question
from the reaction lets a health check report **all** problems at once instead
of surfacing them one failed run at a time.

**Rejected.** Raising on first missing resource — forces users to discover
problems serially. Silently continuing — hides real errors.

**Trade-off.** Callers must remember to check. Mitigated by the CLI running the
full check before any operation, and by `Detector.load()` re-checking and
raising `ResourceNotFoundError` if weights are absent.

---

## D-004 — Model discovery falls back to any `*.pt`

**Status:** Accepted · Milestone 2

**Decision.** Resolve weights as: explicit path → configured path → newest
`*.pt` in the models directory.

**Reasoning.** Training runs emit varied filenames. Requiring one exact name
creates a pointless failure mode where the file is present but "missing".

**Trade-off.** With several files present, selection is implicit. Mitigated by
choosing deterministically (newest), logging the choice at DEBUG, and always
reporting the selected file in the health report.

---

## D-005 — Detections are converted to library-agnostic objects immediately

**Status:** Accepted · Milestone 2

**Decision.** Convert the detector's output tensors into frozen `Detection`
dataclasses at the boundary, in absolute-pixel `xyxy` form.

**Reasoning.** This is the core extensibility decision. Analysis code that
consumed library tensors would be permanently coupled to that library, and
supporting another detector would mean rewriting every consumer. With a
normalised type, a new detector is a new producer and nothing else changes.

It also removes device concerns: values are plain Python floats, so no
downstream module handles GPU memory.

**Rejected.** Passing results through directly — faster to write, permanently
coupling. Dictionaries — no type safety, no IDE support.

**Trade-off.** A conversion pass per image, and the wrapper must be maintained.
Negligible against inference cost, and the isolation is the point.

---

## D-006 — Predictions and ground truth share one coordinate convention

**Status:** Accepted · Milestone 2

**Decision.** All annotations use absolute pixels in `xyxy`. Normalised label
values are converted once, on read.

*Amended by D-016:* this now covers polygons as well as boxes — an outline is
stored in the same absolute-pixel space as the box derived from it. The
type originally named `GroundTruthBox` is now `ObjectAnnotation`.

**Reasoning.** Comparing predictions to ground truth is the project's entire
purpose. Mixed conventions — normalised vs pixel, centre vs corner — are the
most common source of silent errors in detection code, because the result is
plausible-looking wrong numbers rather than an exception.

**Trade-off.** Conversion needs image dimensions at read time, so
`load_ground_truth` requires width and height. A small, explicit cost.

---

## D-007 — Shared geometry via mixin, not base dataclass

**Status:** Accepted · Milestone 2

**Decision.** `BoxGeometryMixin` provides derived geometry to both box types.

**Reasoning.** The prediction and ground-truth types had independent
implementations of `xyxy` and `area` — duplication that drifts, and drift here
silently corrupts every failure classification built on top.

A base **dataclass** owning the corner fields would force them first in every
subclass constructor, turning `Detection(class_id, class_name, confidence, ...)`
into `Detection(x1, y1, x2, y2, class_id, ...)`. A mixin contributes behaviour
without contributing fields, so each class keeps a natural signature.

**Rejected.** A base dataclass — reorders constructor arguments for no gain.
A free function `area(box)` — loses attribute-style access and reads worse.
Leaving the duplication — violates the no-duplication standard.

**Trade-off.** The contract is structural rather than enforced by the type
system: a class must expose `x1..y2`. Mitigated by the mixin declaring those
names as class-level annotations, and by a test asserting neither class
re-implements the shared properties.

*Amended in the Milestone 2 review (D-015):* a `HasCorners` protocol originally
documented this contract, but nothing in production used it and the mixin's own
annotations already state it. It was removed. See D-014 for the rule on when
geometry belongs on the mixin versus in a free function.

---

## D-008 — Exceptions live in a dependency-free module

**Status:** Accepted · Milestone 2

**Decision.** All project exceptions live in `utils/exceptions.py`, which
imports nothing.

**Reasoning.** Previously the base class lived in `utils/resources.py`, so
`utils/dataset.py` imported the resource checker purely to raise an error —
coupling with no functional basis, which would have propagated to every future
module.

**Trade-off.** One more small module. Worth it: it makes the dependency graph
honest.

---

## D-009 — Default confidence threshold is low

**Status:** Accepted · Milestone 2

**Decision.** Default to `0.25`, not a production-style `0.50`.

**Reasoning.** A wrong detection at low confidence is exactly the evidence this
tool exists to surface. A high threshold hides the failures we are built to
diagnose.

Generally: **filter late, never early.** Analysis can always raise a threshold;
it can never recover detections discarded at inference time.

**Trade-off.** More detections to process and more visual noise in annotated
output. Correct for a diagnostic tool; a deployment would choose differently,
which is why it is configurable.

---

## D-010 — The project owns its output paths

**Status:** Accepted · Milestone 2

**Decision.** Render annotated images ourselves into the configured predictions
directory rather than using the library's built-in save.

**Reasoning.** Library-managed saving writes to a directory of its choosing with
its own run-numbering. Owning the path keeps the project self-contained,
configurable, and predictable for later modules that must locate these files.

**Trade-off.** A few extra lines, and we handle write failure. Deliberately
non-fatal: a rendering problem must not discard numeric results already
computed.

---

## D-011 — Parallel tensor alignment is enforced, not assumed

**Status:** Accepted · Milestone 2

**Decision.** Zip the detector's coordinate, confidence, and class tensors with
`strict=True`.

**Reasoning.** Those three tensors are aligned only by position. Plain `zip`
truncates to the shortest input, so a length mismatch would silently discard
detections and produce confident, wrong output with no error. `strict=True`
converts that into an immediate, loud failure.

Found by lint (`B905`) in code whose own comments described the hazard —
evidence that documenting an invariant is not the same as enforcing it.

**Trade-off.** A malformed result now raises instead of degrading. That is the
desired behaviour for a diagnostic tool.

---

## D-012 — Line length of 88

**Status:** Accepted · Milestone 2

**Decision.** 88 characters, enforced by lint.

**Reasoning.** PEP 8 explicitly permits teams to raise its 79-character
guidance. 88 is the modern default and keeps two files readable side by side
while letting a descriptive name plus a type hint fit on one line.

**Trade-off.** Not the literal PEP 8 number. Documented here so the deviation is
a decision rather than drift.

---

## D-013 — Evaluation metrics are normalised at the detector boundary

**Status:** Accepted · Milestone 2 review

**Decision.** `Detector.validate()` returns a `ValidationMetrics` dataclass.
`ValidationMetrics.from_raw()` is the only code that knows the evaluation
library's attribute names.

**Reasoning.** `validate()` previously returned the library's own object typed
as `Any`, and the CLI reached into it — `getattr(metrics, "box")`, then
`box.mp`, `box.map50`. That leaked a third-party object two layers outward,
which is exactly what D-005 established must not happen for detections. The
same reasoning applies to metrics; the inconsistency was an oversight, not a
deliberate exception.

Three concrete gains:

1. Presentation code no longer duck-types a foreign object.
2. Version fragility is centralised. A library rename now breaks one function
   with a clear cause, rather than a `print` statement in the CLI.
3. **The code became testable.** `from_raw()` reads attributes off a plain
   object, so stubs exercise it with no model, dataset, or network. Nine tests
   now cover a surface that was previously untestable until a dataset existed.

**Rejected.** Returning the raw object with a typed alias — documents the
coupling without removing it. Waiting until a dashboard needs it — the cost of
extraction grows with each new consumer.

**Trade-off.** A conversion layer to maintain, and every field is `float | None`
so consumers must handle absence. Accepted: a metric that a library version
stops reporting should degrade to "not reported", never abort a validation run
that has already done the expensive work.

---

## D-014 — Mixin for unary geometry, free functions for binary operations

**Status:** Accepted · Milestone 2 review

**Decision.** `BoxGeometryMixin` provides properties derived from a *single*
box. Operations over *two* boxes — IoU, containment, distance — will be free
functions in the same module, not mixin methods.

**Reasoning.** Raised by the "prefer composition over inheritance" standard,
which warrants an explicit answer since D-007 chose inheritance.

The mixin is defensible in its narrow role: it is stateless, declares no
fields, and adds no constructor behaviour — the least entangling form of
inheritance, closer to a shared trait than a type hierarchy. It buys natural
attribute access (`det.area`) and eliminates real duplication.

It stops being the right tool for binary operations. `a.iou(b)` implies an
asymmetry that does not exist — IoU is symmetric, and neither argument is
privileged. `iou(a, b)` states that plainly and composes better.

Recording the boundary now prevents the predictable drift where a later
milestone adds `iou()` as a method because the mixin was already there.

**Trade-off.** Two idioms in one module. Justified: they model genuinely
different relationships, and the rule for choosing between them is stated.

---

## D-015 — Speculative public API is removed, not retained

**Status:** Accepted · Milestone 2 review

**Decision.** Deleted five public symbols never called in production:
`resources.require()`, `Detector.is_loaded`, `HasCorners`, and `to_dict()` on
both result containers. Wired `dataset.describe()` into `--check` rather than
deleting it.

**Reasoning.** Each was written for an anticipated need rather than a real one.
Unused public API is a liability: it must be maintained and tested, it widens
the surface a future change can break, and it constrains design by making a
guessed shape look settled.

`to_dict()` is the clearest case. Its schema was a guess at what a future
persistence layer might want. When Milestone 3 persists failure records, the
shape it actually needs will be driven by that module's requirements — and
would likely have differed from the guess, leaving a misleading method in
place.

`describe()` was different: genuinely useful, merely unwired. Connecting it to
`--check` — where it now prints discovered classes and per-split image counts —
converted dead code into the feature that confirms a dataset is set up
correctly. Deleting it would have discarded working, tested code that answers a
real question.

**Consistency note.** `dataset.load_ground_truth()` is also currently uncalled
in production and was *kept*. It is not speculative: parsing the annotation
format was an explicit Milestone 2 objective, and it is fully tested. The rule
is "delete API written for an imagined future need", not "delete anything not
yet called".

**Trade-off.** Some of these will be rewritten later. Rewriting twenty lines
against a known requirement beats maintaining a wrong guess and then having to
migrate its consumers.

---

## D-016 — One generic annotation model for detection and segmentation

**Status:** Accepted · Milestone 2.5 (pre-Milestone 3 refactor)

**Decision.** Ground truth and predictions are both
`utils.annotations.ObjectAnnotation`: a class id, a class name, a bounding box,
an optional polygon, and an optional confidence. `Detection` is a subclass that
requires a confidence. When a polygon is present the bounding box is *derived*
from it, never stored independently.

**Reasoning.** The first real dataset turned out to be segmentation: every one
of its annotations is a polygon, and the previous box-only parser skipped all
of them. It did not crash — it returned zero ground truth for every image,
which would have made every prediction look like a false positive. Silent and
confident wrongness is the worst failure mode for a diagnosis tool.

Modelling boxes and polygons as unrelated types would fork every future
consumer into "the box path" and "the polygon path". One type with an optional
polygon means analysis is written once and works on both.

Deriving the box rather than storing it makes "the box and the polygon agree"
a property of the type, not a rule someone must remember. It also matches how
the detection ecosystem converts segments to boxes, so Model Doctor and the
model's own evaluator agree about where the ground truth is — verified against
`ultralytics/data/utils.py`, which applies the same min/max extent.

Three dataset shapes are supported *by construction*, decided per annotation
rather than per dataset: detection-only, segmentation-only, and both mixed in
one file.

**Rejected.** Adding a parallel `polygon` field to two separate classes —
reintroduces the duplication D-007 removed, and it would drift. A single class
with `confidence: float | None` and no `Detection` type — simpler, but erases
the ground-truth/prediction distinction that error analysis is built on.
A converter that turns polygons into boxes at read time and discards the
outline — loses the data mask analysis will need, and would force a reparse.

**Trade-off.** Field order is inherited, so `Detection`'s constructor changed
and its call sites were updated. A one-time cost against a permanent one: the
alternative is maintaining two models that must be kept in step by hand.

Deliberately *not* included, per the milestone boundary: mask IoU, polygon
area, and any segmentation analysis. Prediction outlines are also not yet read
from model output — the field exists, the extraction does not. The model is
ready for those; it does not pre-empt them.

**Verified.** Against the real 1,291-image segmentation dataset: 2,151/2,151
annotations parsed, class counts matching an independent audit exactly, zero
degenerate boxes.

---

## D-017 — A weak overlap is poor localisation, not a false positive plus a miss

**Status:** Accepted · Milestone 3

**Decision.** When a prediction overlaps a ground-truth annotation by less than
the match threshold but by more than `LOCALIZATION_IOU_FLOOR`, the diagnosis
engine reports **one** finding — poor localisation, or wrong class if the
labels disagree — rather than a false positive *and* a false negative.

Implemented as two matching passes: the first at `MATCH_IOU_THRESHOLD` (0.50),
the second over the leftovers at `LOCALIZATION_IOU_FLOOR` (0.10). Anything
still unpaired after the second pass is a genuine invention or a genuine miss.

**Reasoning.** Evaluation metrics take the other route: below threshold means
no match, so both sides count as errors. That is correct *for scoring* — mAP
must penalise a bad box twice or a model could game it.

It is wrong for *explaining*. "Found the object, outlined it badly" names a
cause an engineer can act on. "One spurious detection and one miss" describes a
single object as two unrelated failures and destroys the causal link between
them. The project exists to explain, so it takes the explanatory reading.

The floor exists so the reading cannot be abused. Overlap below 0.10 is not a
near miss, and excusing it would let genuinely wild predictions hide inside a
sympathetic category. Below the floor, a false positive and a false negative
are exactly what happened.

A weak overlap with *disagreeing* classes is reported as wrong class, not poor
localisation: naming the object incorrectly is the more consequential error,
and folding it into a localisation bucket would conceal it.

**Rejected.** Matching the mAP convention — reproduces a benchmark this project
is explicitly not trying to reproduce, and loses the causal link. A single
threshold with no second pass — makes every near miss look like two unrelated
errors. Reporting both a poor-localisation finding *and* FP/FN records —
double-counts, so no total would reconcile.

**Trade-off, and it must be stated wherever these numbers appear: the
false-positive and false-negative counts produced here will not equal those
implied by mAP.** They are lower, because near misses are reclassified. The two
answer different questions and must not be compared directly. A second
difference compounds it: diagnosis runs at the configured confidence threshold
(0.25 by default), whereas mAP integrates across all thresholds.

**Verified** on 136 real images: every prediction (218) and every ground truth
(238) is accounted for exactly once, with no double counting.

---

## D-018 — Matching is greedy on similarity, and similarity is a parameter

**Status:** Accepted · Milestone 3

**Decision.** `utils/matching.py` pairs annotations one-to-one, accepting the
strongest overlap first. The comparison function is an argument, defaulting to
box IoU.

**Reasoning — strongest-first rather than confidence-first.** Evaluation
metrics order by confidence because they simulate a detector's own ranking.
Model Doctor is not scoring the model; it is explaining a specific image, where
the question is which prediction genuinely describes which object. The best
geometric correspondence answers that. A confident prediction that overlaps
poorly does not become the right pairing by being confident.

**Reasoning — greedy rather than optimal.** Hungarian assignment is globally
optimal but differs from greedy only in crowded scenes with heavy mutual
overlap. Greedy is deterministic and obvious to read. If crowding ever becomes
a real limitation, `match_annotations` is the only function that changes.

**Reasoning — similarity as a parameter.** This is what "architecture-ready for
masks" means concretely. Mask IoU becomes a different argument at the call site
rather than a rewrite of the engine, and the seam is exercised by tests today
so it cannot rot. No mask overlap is implemented.

**Ties** are broken by higher confidence, then by lower index. Without a
defined order, identical input could produce different pairings between runs
and diagnoses would not be reproducible.

**Trade-off.** Greedy can be suboptimal where several objects overlap heavily.
Accepted for determinism and readability, and recorded here so the limitation
is known rather than discovered.

---

## D-019 — Persistence lives in `app/`, and the schema is the contract

**Status:** Accepted · Week 2 completion

**Decision.** SQLite persistence is `app/storage.py`, not `utils/storage.py`.
`docs/SCHEMA.md` is the published interface; the Python module is one
implementation of it.

**Reasoning.** Persistence needs the diagnosis domain types — `Finding`,
`Outcome`, `ImageDiagnosis` — which live in `app.diagnosis`. A `utils` module
importing from `app` would be the first upward dependency in the project and
would break the layering held since Milestone 2. Direction is one-way and
enforced by review:

    app/storage.py → app/diagnosis.py → app/inference.py → utils/* → config.py

`app.diagnosis` must never import `app.storage`, or the direction becomes a
cycle. The CLI orchestrates instead: diagnose, then persist. The import in
`main()` is deliberately function-local for exactly this reason.

The schema is documented as a **contract** rather than described as an
implementation detail because a second developer is building a dashboard
against the database without reading this codebase. If SCHEMA.md is not
sufficient on its own, the deliverable has failed regardless of whether the
code works.

**Trade-off.** Two artefacts must stay in step — the DDL and the document. The
schema statements carry comments pointing at SCHEMA.md, and every example query
in the document is executed against a real database as part of verification, so
drift shows up as a failing query rather than as silent staleness.

---

## D-020 — The findings table never changes shape; later milestones add tables

**Status:** Accepted · Week 2 completion

**Decision.** `runs`, `images`, and `findings` are frozen. Clustering,
root-cause analysis, and recommendations will each arrive as their own table
keyed on `finding_id`, not as new columns.

**Reasoning.** The roadmap's storage sketch listed `cluster_id`, `root_cause`,
and `recommendation` alongside the failure columns. Adding them now would mean
three columns that nothing writes for weeks — exactly the speculative surface
D-015 removed.

But a database schema is not ordinary code. Altering a table another developer
is already querying is far more disruptive than adding a function, so
"implement it later" cannot mean "change the shape later". Normalising resolves
both: nothing speculative exists today, and nothing existing has to change when
it does.

**Rejected.** Nullable columns for future features — dead surface now, and it
would invite queries written against columns that are always null. Deferring
the schema until those features exist — the dashboard cannot start.

**Trade-off.** Consumers will need joins once those tables land. Cheap, and the
alternative is breaking published queries.

---

## D-021 — Runs are events, not idempotent writes

**Status:** Accepted · Week 2 completion

**Decision.** Every `--save` creates a new run. Nothing is overwritten or
de-duplicated. Only schema creation is idempotent.

**Reasoning.** A run records what a *specific* model produced under *specific*
thresholds at a *specific* time. Overwriting would destroy the history that
makes model-versus-model comparison possible — which the roadmap schedules
explicitly. Re-running the same configuration twice is legitimately two
observations, not one repeated.

Each run stores the model's SHA-256, not just its path, because weight files
get overwritten in place. Without the hash, two runs that disagree look like a
regression when they may simply be different models.

**Trade-off.** The database grows with every run and nothing prunes it.
Acceptable — rows are small, and deleting a run cascades cleanly to its images
and findings.
