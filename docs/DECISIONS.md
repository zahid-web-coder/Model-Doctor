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

---

## D-022 — Mask-level diagnosis is deferred, not abandoned

**Status:** Accepted (deferral) · Recorded during Week 5 planning

**Decision.** Mask-based failure diagnosis is deferred until after the core
reasoning modules are complete. It is scheduled as a post-roadmap enhancement,
not dropped. The frozen schema and backend contract are **not** modified now.

**The gap being deferred.** Measured on the real segmentation model and test
split, box-level diagnosis reports the two classes failing almost identically,
while the model's own mask metrics say otherwise:

| Signal | `door` | `door_frame` |
| --- | --- | --- |
| Diagnosis engine, mean box IoU | 0.878 | 0.877 |
| Mask mAP50-95 | 0.599 | **0.246** |
| Box mAP50-95 | 0.630 | 0.521 |

The model locates door frames about as well as it locates doors; what it cannot
do is trace their outline. A door frame is a thin rectangular annulus, so a few
pixels of boundary error destroys mask IoU while barely moving box IoU.

**Consequence, stated plainly: for a segmentation dataset, the diagnosis engine
currently under-reports the dominant failure mode of thin-structure classes.**
Its box-level findings are correct; they are simply blind to this axis.

**Why defer rather than fix now.**

1. The schema was published this week and a second developer is building
   against it. Mask findings cannot be added without either altering `findings`
   (breaking D-020 and that contract) or introducing a new table and a schema
   version bump. Doing that mid-onboarding trades a real correctness
   improvement for a real coordination cost.
2. The core reasoning modules — feature extraction, clustering, root-cause
   analysis, recommendations — are not finished. Adding a second diagnosis axis
   before they exist widens the surface each of them must handle.
3. The gap is *known and documented*, which is materially different from
   unknown. A documented limitation can be stated alongside results; an
   undiscovered one silently misleads.

**Schema options, recorded now so the analysis is not repeated later.**

| Option | Cost |
| --- | --- |
| `mask_iou` column on `findings` | Alters the frozen table. Violates D-020. Rejected. |
| New `mask_findings` table keyed on `finding_id` | Additive, respects D-020, needs a schema version bump. **Preferred.** |
| Separate run with a different similarity function | Needs a `similarity` column on `runs` — also an alteration. |

The preferred option is the second. It leaves `findings` untouched, so every
query written against the published contract keeps working.

**What is already in place, so this is an addition rather than a rewrite.**

- `utils.matching.match_annotations` takes the comparison as a **parameter**
  (D-018), defaulting to `box_iou`. A mask comparison is a different argument.
- `app.diagnosis.diagnose_image` forwards that parameter through untouched.
- `ObjectAnnotation.polygon` already carries ground-truth outlines in absolute
  pixels (D-016), and they are already persisted in `findings.truth_polygon`.

**What is genuinely missing.**

- A mask IoU function — polygon rasterisation or polygon intersection.
- Prediction outlines. The inference layer does not extract masks from model
  output, so `Detection.polygon` is always `None`. This is the real work:
  without predicted outlines there is nothing to compare ground-truth outlines
  against.

**Trade-off accepted.** Until this lands, mask-sensitive failures are invisible
to the diagnosis engine, and any report covering a segmentation dataset should
say so. That statement belongs anywhere these numbers are presented.

---

## D-023 — Feature extraction is a database pass with two injected seams

**Status:** Accepted · Week 5

**Decision.** `app/features.py` reads findings back **from the database**,
crops a region per finding, encodes it, and writes vectors to a new
`embeddings` table. Both the region extractor and the encoder are parameters.

**Why a database pass rather than an in-memory step.** The obvious design is to
embed during diagnosis, while the findings are still objects. That would have
required threading vectors through `save_dataset_diagnosis` and returning the
ids it assigns — changing a write path that a second developer is already
building against. Reading back instead means the write path is untouched,
findings already carry their primary keys, and extraction can run over a run
saved days earlier without re-running inference. Week 6 clustering will take
the same shape.

**Why the region extractor is a parameter.** This is the seam mask-level
diagnosis will use (D-022). Today `region_from_box` crops a finding's bounding
box; a polygon extractor replaces it at one call site, and the outlines it
needs are already persisted in `findings.truth_polygon`. The parameter is
exercised by a test today, so it cannot rot before it is needed.

**Why the encoder is injected.** Not speculation: a CLIP checkpoint is several
hundred megabytes, so a suite requiring one would be untestable in practice and
would quietly stop being run. A fake backend is the only way to test the
pipeline at all. It also keeps the dependency at one boundary, as D-005 does
for the detector.

**Which region is encoded.** Ground truth when present, prediction otherwise —
matching how `Finding.class_name` attributes a failure. A false positive
therefore encodes what the model *thought* it saw; a false negative encodes
what it *missed*. Each finding encodes the thing it is actually about.

**Padding.** Regions are expanded by 15% before cropping. A box tight against
an object discards its surroundings, and the surroundings are frequently the
explanation — an object may be missed *because* of what is beside it.

**Vectors are L2-normalised on write**, so cosine similarity is a dot product
and every consumer gets comparable vectors without knowing which metric was
intended. Stored as raw float32 bytes rather than JSON: exact, and 2 KB for 512
dimensions.

**Trade-off.** Extraction requires a saved run, so it cannot be part of a
single in-memory pipeline. Accepted deliberately — that constraint is what
keeps the published write path stable.

**Observed on real data, and stated rather than glossed.** Across 26 embedded
failures from the segmentation dataset, nearest-neighbour class agreement was
at chance. This is not yet evidence of a defect: the sample is small, and
`door` and `door_frame` crops are close to the same pixels, since a padded box
around a frame contains the door. Whether that matters depends on what
clustering is *for* — grouping by failure cause rather than by class label may
be the desired behaviour. It is an open question for the clustering milestone,
recorded here so it is evaluated rather than assumed.

---

## D-024 — Grad-CAM targets raw class logits, not decoded scores

**Status:** Accepted · Week 3

**Decision.** Explanation backpropagates a **raw pre-sigmoid class logit** from
the head's secondary return value, selected from the anchors falling inside the
finding's region. Target layers are the three feature-pyramid tensors the head
consumes, fused.

**Why the obvious approach fails — measured, not assumed.** Backpropagating a
YOLO26 detection's *decoded* class score produces gradients that are **exactly
zero at every layer**. The decoded output has passed through a sigmoid which
saturates. This is the failure mode that matters most: a saturated scalar
yields a uniformly zero map, which renders as a plausible-looking heatmap
rather than an error. Nothing about it looks wrong.

The head's own source shows the way out. In evaluation mode it returns
``(decoded, raw_predictions)``, and the one-to-one branch is **not** detached
when not training, so gradients flow through the raw logits.

Verified before any code was written: raw logits give live gradients at all
three pyramid levels.

**Why fuse all three levels.** The coarsest carries the strongest per-element
gradient but only ~3% of it is non-zero, and a 21x21 map upscaled to full
resolution is unreadable. The finest supplies spatial detail. Betting on one
level would make heatmap quality depend on object size, which is exactly the
variable this project studies.

**Why anchors inside the region.** Selecting only anchors whose centre lands in
the finding's box makes the map answer "what made the model respond *here*"
rather than "what does this class look like in general". A finding is about one
object; a global saliency map would not explain it.

**Which box.** Prediction first, ground truth otherwise — the reverse of
feature extraction's preference (D-023). Here the question is what drove the
model, so where it actually responded is the faithful region. Where it
responded to nothing, the ground-truth location is the next best question, and
those false-negative maps turn out to be informative: they are visibly diffuse,
which is the signature of "not attending here".

**Detector-agnostic by construction.** ``GradCAM`` hooks layers and weights
activations by gradients; it knows nothing about YOLO. ``CamAdapter`` supplies
the two architecture-specific answers. A new detector is a new adapter — the
same seam as ``SimilarityFn`` (D-018) and ``RegionExtractor`` (D-023).

*Amended during the pre-commit review:* the generic engine was originally
written in ``app/explainability.py``. It contained no reference to any
detector, so it was moved to ``utils/cam.py``. Because ``utils`` may not import
``app``, that move converts "detector-agnostic by discipline" into
"detector-agnostic enforced by the layering rule" — the engine can no longer
acquire a detector dependency even by accident. It also matches the existing
split between ``utils/matching.py`` and ``app/diagnosis.py``.

**A defect this work surfaced.** The first implementation reported "no gradient
reached any target layer" whenever no map survived, but a Grad-CAM map can also
collapse to zero when the pooled channel weights are negative and ReLU removes
everything. That is a real result, not an error, and conflating the two would
have sent a future reader hunting for a gradient problem that did not exist.
The two cases are now distinguished, and a collapsed map is kept rather than
dropped.

**Trade-off.** Explanation needs gradients, so it cannot reuse the inference
path: the network is loaded separately in float precision with gradients
enabled. Slower and more memory-hungry than inference, and unavoidable.

---

## D-025 — Heatmaps are files, recorded in an additive table

**Status:** Accepted · Week 3

**Decision.** Heatmap images are written to ``results/heatmaps/run_<id>/`` and
recorded in a new ``heatmaps`` table. Schema version 2 to 3, additive.

**Reasoning.** A dashboard needs to know which findings have an explanation and
where it lives. A filesystem naming convention would avoid the version bump but
force a consumer to stat the disk per row, and would record nothing about how a
map was produced. The table answers both with one left join and carries the
method and layer set, so a heatmap generated with different settings is
identifiable rather than silently mixed in.

``runs``, ``images``, ``findings`` and ``embeddings`` are all unchanged —
verified column-by-column against the previous commit. This is the third
demonstration of the stability guarantee in SCHEMA.md: new capability arrives
as a new table, never as an alteration.

**Trade-off.** Images live outside the database, so a database copied without
its ``results/`` directory has dangling paths. Accepted: storing image blobs in
SQLite would bloat the file a consumer is meant to query cheaply, and the paths
are as reproducible as the run itself.

---

## D-026 — Root causes attach to findings, not to clusters

**Status:** Accepted · Week 7

**Decision.** Attributed factors are recorded against a `finding_id` in a new
`root_causes` table. Nothing references a cluster.

**Reasoning.** The roadmap describes root-cause analysis as summarising
findings *per cluster*, and clustering is a later milestone owned by another
developer. Waiting for it would block this work; designing around a table that
does not exist would guess at its shape.

Attaching to findings avoids both. Once clusters exist, a per-cluster summary
is a join and a `GROUP BY` over rows that are already there:

```sql
SELECT c.cluster_id, rc.factor, COUNT(*)
FROM root_causes rc JOIN clusters c ON c.finding_id = rc.finding_id
GROUP BY c.cluster_id, rc.factor;
```

No schema change, no second pipeline, no rework. A test constructs a stand-in
clusters table and runs exactly that query, so the property is verified rather
than asserted.

Attaching to clusters instead would have inverted the dependency: root causes
could not be computed until clustering existed, and re-clustering would
invalidate every attribution.

**Trade-off.** A factor that is genuinely a property of a *group* rather than
of an individual finding — "this cluster is all night-time images" — is stored
redundantly, once per member. Accepted: the redundancy is small, and the
alternative is a table that cannot be written until another milestone lands.

---

## D-027 — Two detector scopes, one output shape

**Status:** Accepted · Week 7

**Decision.** `FindingFactor` examines a single finding; `RunFactor` examines
all of them. Both emit the same `FactorEvidence` and are stored identically.

**Reasoning.** Some conditions are visible in one finding — a region is dark,
a box is tiny, an object runs off the frame. Others do not exist at that scale
at all: a class cannot be under-represented in a single instance. Forcing both
through one interface would mean passing whole-run statistics into a
per-finding detector that does not need them.

Keeping the output identical is what matters to consumers: a dashboard reading
`root_causes` never has to know which kind produced a row.

Detectors carry `needs_pixels`, so the runner can skip image-dependent ones
when images are unavailable rather than having each detector rediscover that
its input is missing. Images are read once per file and shared across every
finding on it, which is also what makes neighbours — and therefore crowding —
available at all.

**Trade-off.** Two protocols rather than one. Justified: they model genuinely
different scopes, and collapsing them would make the common case carry the rare
case's parameters.

---

## D-028 — Confusion pairs are reported undirected, and the limit is documented

**Status:** Accepted · Week 7

**Decision.** The roadmap asks for "repeated confusion pairs". The engine
reports **recurring misclassification per true class** instead: which classes
are repeatedly named wrongly, without the direction of the confusion.

**Reasoning.** A directed pair needs both the true class and the predicted one.
The stored contract records only the class that *should* have been found —
deliberately, so per-class statistics charge a miss to the class that was
missed (SCHEMA.md, `findings`). The predicted class is not stored.

The options were: add a column, which breaks a contract another developer is
already querying; skip the factor; or report the half that is derivable. The
undirected form carries most of the value — "`door_frame` is misidentified 11
times" names the class whose labelling needs attention — and on a two-class
dataset the direction is implied.

**Trade-off, stated rather than hidden.** On a dataset with many classes the
direction matters and is unavailable. Recorded here and in SCHEMA.md so the
limitation is known rather than discovered. If it becomes important, the fix is
a `predicted_class_id` column on `findings` and a schema version bump — an
alteration to a published table, which is exactly the kind of change that
should require a deliberate decision rather than happening incidentally.

---

## D-029 — The dashboard is a read-only consumer of the published schema

**Status:** Accepted · Dashboard workstream

**Decision.** The Streamlit dashboard lives in `app/dashboard.py` and opens
SQLite using a read-only URI. It queries only `runs`, `images`, and `findings`
as documented in `SCHEMA.md`; it does not import the diagnosis engine or the
persistence implementation.

**Reasoning.** The schema exists specifically to decouple the dashboard from
backend implementation. A direct read-only consumer keeps the user interface
small, prevents accidental modification of diagnosis history, and lets saved
runs be inspected even when model dependencies cannot be exercised. Image
paths are treated as external resources: a missing image yields a clear viewer
state while its stored findings remain usable.

Grad-CAM has no published storage contract yet. Adding a visual explanation
placeholder would create a misleading feature surface and couple this module to
an unknown future schema, so it is deferred until Milestone 4.

**Trade-off.** A single Streamlit module keeps the first dashboard slice easy
to review, but page-level decomposition can be introduced once visual
explanations, clusters, or recommendations add enough independent UI surface to
justify it.


---

## D-030 — Failures are grouped by root-cause signature, not by clustering embeddings

**Status:** Accepted · Milestone 5

**Decision.** Group a run's failures deterministically by the set of
root-cause factors attributed to each one. Two failures share a group when
they share a factor set. The group's label *is* that set — `blur +
edge_truncation` — so a group needs no separate naming step. Failures with no
attributed factor form an `unexplained` group rather than being dropped.

Do not call this clustering in any user-facing surface. It is *failure
grouping*, and the distinction is not cosmetic: nothing is learned or fitted,
and the same input always produces the same partition.

**Reasoning.** K-means over the CLIP embeddings was tried first, on the
136-image run with 126 embedded failures, and measured before it was judged:

| Projection | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCA-10 | 0.228 | 0.206 | 0.192 | 0.179 | 0.194 | 0.199 | 0.184 |
| PCA-20 | 0.178 | 0.153 | 0.132 | 0.150 | 0.143 | 0.142 | 0.139 |

Silhouette below 0.25 indicates no substantial structure. Every value is
below it. PCA needed 20 components to retain 68% of variance, so there is no
low-dimensional shape to exploit either. The embedding space itself is not
degenerate — cosine distances run 0.00 to 0.78, a spread ratio of 2.01 — so
this is a real absence of cluster structure, not distance concentration.

Inspecting what k-means did produce settled it. At k=5 one cluster was 78%
false-negative and 75% `door_frame`, another 100% `door`. The clusters were
substantially re-encoding class and outcome, both already columns on
`findings`, and presenting that as discovery.

Factor-signature grouping on the same 127 failures gives 22 groups, with 74%
of failures in groups of five or more and only six singletons. Every group
arrives named.

**Rejected.** *K-means or HDBSCAN over embeddings* — measured above; it would
have produced a feature that appears to work and silently misleads, which is
the exact failure mode this project exists to expose. *Grouping by outcome or
class alone* — that is a `GROUP BY` on an existing table, not a milestone.
*Dropping unexplained failures* — group sizes would stop summing to the failure
count, and the failures no detector can account for are the most interesting
ones.

**Trade-off.** Group count is not controllable: it is whatever the factor
combinations produce, 22 here, and a run with more factors would fragment
further. Groups are only as good as the factors feeding them — grouping
inherits every blind spot of the seven detectors, and adding a factor
re-partitions every group. Accepted, because a group that cannot be explained
is not worth acting on, and this method cannot produce one.

This also rules out discovering a failure mode nobody has written a detector
for. That gap is real, and `unexplained` — 12 of 127 failures here — is
deliberately where it surfaces rather than being hidden inside a numbered
cluster.

**Kept, not repurposed.** The embeddings remain untouched and now serve
nearest-neighbour retrieval in `app/similarity.py`. Ranking by similarity to
one query point needs no cluster structure — it is a sort, and it degrades
honestly, reporting a low score rather than asserting a cluster membership.
Neighbours are computed on demand and not stored: the answer depends on which
finding is asked about, so a table of them would be answers to questions
nobody has asked.

**Naming.** The tables are `clusters` and `cluster_members`, and the module is
`app/clustering.py`, because a second method — genuinely unsupervised — may be
added later under the same `method` seam. Every human-readable surface says
*failure group*. The mismatch is deliberate and documented here so it does not
read as an oversight.

---

## D-031 — A factor's count is reported against its base rate, never alone

**Status:** Accepted · Milestone 5.5

**Decision.** Measure every per-finding factor over correct findings as well as
failures, and store both rates with a lift and a p-value in a new
`factor_rates` table. A factor count is never presented without its control
rate.

Correct findings receive **no** `root_causes` rows. That table is documented as
covering failures, and every query written against it must keep returning what
it always did. Only the aggregate goes into the new table.

**Reasoning.** `root_cause.py` only ever loaded failures. It could therefore
report, truthfully, that `edge_truncation` described 90 of 127 failures — 71% —
while being structurally incapable of noticing that it also described 115 of 151
correct detections, which is 76%. Measured over the reference run:

| Factor | Failures | Correct | Lift | p |
| --- | --- | --- | --- | --- |
| `crowding` | 57% | 45% | 1.28x | **0.041** |
| `low_light` | 9% | 7% | 1.43x | 0.504 |
| `edge_truncation` | 71% | 76% | 0.93x | 0.340 |
| `blur` | 30% | 37% | 0.81x | 0.252 |
| `small_object` | 2% | 0% | undefined | 0.208 |

The largest failure group in the product was built on a condition that is *no
more common among failures than among successes*. Objects touch the frame edge
in this dataset because the photographs are taken close up; the factor was
describing the room, not the failure. Recommending "reduce edge truncation"
would have been confident, specific, and unfounded — which is precisely the
behaviour this project exists to expose in other tools.

To state it correctly: p = 0.34 is not evidence that edge truncation helps. It
is an absence of evidence that it matters either way. And with five factors
tested, a Bonferroni-corrected threshold is 0.01, so **`crowding` at 0.041 does
not survive correction either.** It is a lead, not a finding.

**Rejected.** *Reporting counts alone* — the status quo, and the reason a
meaningless factor became the headline. *Dropping the weak factors* — they
still describe conditions accurately, and a factor with no lift on this dataset
may have lift on another; the fix is to report the denominator, not to hide the
measurement. *Attributing factors to correct findings in `root_causes`* — that
would change what every existing query returns, for a number better stored as
an aggregate.

**Trade-off.** The pass now reads pixels for every finding rather than only
failures, roughly doubling that work — about one extra second on the 136-image
run. Accepted without hesitation: the alternative is a faster wrong answer.

Run-level factors (`class_imbalance`, `recurring_misclassification`) get no
base rate. They are defined in terms of mistakes, so a correct finding has no
analogous condition and inventing one would fabricate a comparison.

**A second bug this exposed.** Building contexts from failures alone also meant
a failure's `neighbours` contained only *other failures*. An object sitting
beside a correctly-detected one appeared to have no neighbours at all, so
`crowding` was under-reported — 49 attributions where the true figure is 73, a
third missing. Crowding is a property of what is physically nearby; whether the
neighbour happened to be detected correctly has no bearing on whether this
object was occluded. Contexts are now built from every finding, while run-level
factors still receive failures only, because both derive a denominator from the
list they are handed and widening it would silently redefine their evidence
strings.

**Consequence for Milestone 6.** Recommendations cannot be built on raw factor
counts. Any advice must cite lift and significance, and must be able to say
"this condition is common but does not distinguish failures" — which, on this
dataset, is the honest verdict for the two largest factors.

---

## D-032 — Size and shape thresholds are derived from the data, not from constants

**Status:** Accepted · Milestone 5.6

**Decision.** Calibrate `small_object` from the run's own distribution of
object areas, at the 25th percentile, and add a `thin_structure` factor
calibrated at the 75th percentile of aspect ratios. Both are computed over
**every** finding, correct ones included. The absolute factors — blur, light,
edge proximity, overlap — are unchanged, because they measure physical
quantities that do not depend on what else is in the dataset.

**Reasoning.** D-031 established that no factor in the set distinguished a
failure from a success. The follow-up question was whether that is a property
of the dataset or of the factors. It was the factors.

`SMALL_OBJECT_AREA_FRACTION` was 0.0012 — the COCO convention for "small". On
a dataset photographed close up, where objects fill the frame, it fired on 2 of
278 findings on one split and **0 of 496** on another. It was not measuring
anything. Yet size is the strongest predictor of failure in the data, once
measured relatively:

| Threshold | test lift | test p | val lift | val p |
| --- | --- | --- | --- | --- |
| area below p10 | all failures | <0.001 | 8.62x | <0.001 |
| area below p25 | 2.21x | <0.001 | 2.67x | <0.001 |
| area below p50 | 1.66x | <0.001 | 1.81x | <0.001 |
| area below p75 | 1.25x | 0.006 | 1.29x | <0.001 |

A clean dose-response: the smaller the object, the more likely the failure,
monotonically across both splits. Thinness behaves the same way, with the
effect concentrated at the extreme — 1.71x/2.21x above the 75th percentile,
nothing below the median.

As implemented, measured over the same two splits:

| Factor | test | val |
| --- | --- | --- |
| `small_object` (calibrated) | **2.77x**, p < 0.001 | **3.45x**, p < 0.001 |
| `thin_structure` (new) | **2.14x**, p < 0.001 | **2.32x**, p < 0.001 |
| `crowding` | 1.28x, p = 0.041 | 1.12x, p = 0.310 |
| `edge_truncation` | 0.93x, p = 0.340 | 1.04x, p = 0.542 |

Both new factors replicate across splits at effect sizes no previous factor
approached, and both survive Bonferroni correction with room to spare.

**Why thinness.** `door_frame` mask mAP50-95 is 0.246 against `door` at 0.599 —
a gap known since training and invisible to box-level analysis. A door frame is
a thin rectangle. Nothing in the factor set could name that. It is not a proxy
for size: within `door_frame` alone, small objects still fail at 1.94x on test
and 1.99x on val (both p < 0.001), so the two conditions are separable.

**Rejected.** *Keeping the COCO constant* — it is a convention for reporting
object scale, not a threshold for this dataset, and it measured nothing here.
*Calibrating from failures alone* — the reference distribution would be the
thing being measured, and the factor would fire on a fixed share of failures by
construction. *A fixed aspect-ratio threshold* — the same transferability
problem the size constant had; the configured value survives only as a fallback
when no distribution is available.

**Trade-off, and it is real.** A calibrated factor fires on a fixed share of
the run by construction, so "small" is relative and two runs over different
datasets are not directly comparable on it. `factor_rates` is what makes this
safe: the share is fixed, but the *lift* is not, and lift is what says whether
the condition matters.

**Grouping fragmented.** Adding two factors took failure groups from 22 to 35
on test and 43 on val, with median group size falling to 2. `unexplained` fell
from 10 to 1, so coverage improved, but 35 groups over 127 failures is close to
one group per failure and defeats the purpose. D-030 predicted exactly this.

Measured, though not implemented: grouping on only the factors with lift > 1
and p < 0.05 gives **8 groups with a median size of 11** on test, which is far
more actionable. It is not adopted here because the qualifying set differs
between runs — `crowding` qualifies on test but not val — so groups would stop
being comparable across runs. Milestone 6 should decide this, likely by fixing
the factor set once from pooled evidence rather than per run.

---

## D-033 — Failures are grouped twice: once to describe, once to act on

**Status:** Accepted · Milestone 5.7

**Decision.** Store two partitions of the same failures under
`clusters.method`:

- `factor-signature` — every attributed factor. The complete descriptive record.
- `discriminating-signature` — only factors listed in
  `config.DISCRIMINATING_FACTORS`, currently `small_object` and
  `thin_structure`. **This is what a consumer should show by default.**

The qualifying set is a **configured decision, not a per-run computation**. A
factor joins only if it shows lift > 1 and p < 0.05 on at least two splits.

**Reasoning.** Adding two genuinely discriminating factors in D-032 fragmented
the grouping, because grouping on every factor means grouping on factors that
carry no information. Measured on the reference runs:

| Grouping | groups (test / val) | median size | singletons | unexplained |
| --- | --- | --- | --- | --- |
| every factor | 35 / 43 | 2 / 3 | 11 / 10 | 1% / 2% |
| per-run significant | 8 / 4 | 10 / 44 | 0 | 9% / 47% |
| **fixed replicated set** | **4 / 4** | 24 / 44 | 0 | 50% / 47% |

Grouping on every factor explains 99% of failures using groups that mean
nothing — median size 2, eleven singletons, and labels built largely from
`edge_truncation`, whose pooled lift is 1.00x at p = 1.000. It is
comprehensive and it is false comfort.

The fixed set produces the same four groups on both splits, in close
proportions:

| Group | test | val |
| --- | --- | --- |
| `unexplained` | 50.4% | 46.8% |
| `small_object + thin_structure` | 24.4% | 27.9% |
| `small_object` | 15.4% | 14.2% |
| `thin_structure` | 11.0% | 10.0% |

The composition is diagnostic in a way the fragmented version was not. On both
runs, `small_object + thin_structure` is dominated by false negatives and by
`door_frame` — 26 of 31 misses on test, 24 of 31 `door_frame`. `thin_structure`
alone leans the other way, toward false positives. Those are two different
problems, and the previous grouping split them across dozens of tiny buckets.

**Rejected.** *Per-run significance* — the qualifying set differed between runs
(`crowding` qualified on test at p = 0.041 and failed on val at p = 0.310), so
groups stopped being comparable and run comparison, the feature that makes
regressions visible, silently broke. *Replacing the full signature* — the
complete attribution is still the honest record of what was measured, and
discarding it would hide the basis for the restriction. *Recomputing the set
automatically* — silent changes to what counts as a cause is precisely the
failure mode D-031 was written about.

**Trade-off.** Half of failures land in `unexplained` — 50.4% and 46.8%. That
is the honest cost and it is stated rather than softened. Those failures do not
have a known cause today; under the previous grouping they had a *label*, built
from a factor that describes correct detections equally well. A named group
that means nothing is worse than an honest gap, because only one of them
prompts someone to look further.

The set also needs a human to revisit it when factors or classes change. That
is deliberate: see the membership rule above, and re-measure before editing it.

---

## D-034 — Recommendations attach to failure groups, and ordering is a rule, not a score

**Status:** Accepted · Milestone 6

**Decision.** A recommendation belongs to a **cluster** — one row per
`(cluster_id, rule)` — not to a finding and not to a factor. Generation is
restricted to the `discriminating-signature` partition. The foreign key
cascades on delete.

Display order is fixed and documented: **`actionable DESC, priority DESC, id`**.
`priority` carries the number of failures addressed and nothing else.

**Reasoning.** A recommendation is about a pattern, and the failure group is
already the unit that means "failures sharing a cause" (D-030). Attaching per
finding would write 127 near-identical rows and lose exactly the property that
makes advice worth acting on — that it covers many failures at once. Attaching
per factor is too coarse in the other direction: `small_object` and
`small_object + thin_structure` rest on the same factor but behave differently,
72–80% missed for the first against a different mix for the second, and advice
keyed on the factor would flatten that.

The cascade is deliberate rather than incidental. `save_clusters` deletes and
reinserts, so cluster ids are not stable across regrouping. Advice derived from
a partition that no longer exists is stale, and stale advice is worse than
none — it looks current.

**On ordering.** An obvious design is a weighted score blending size, lift and
significance. It was rejected for the same reason `root_causes.score` is
documented as comparable within a factor but not across: a number nobody can
decompose is a number nobody can check. Two explicit terms — is it actionable,
and how many failures does it address — are reproducible and can be explained
in one sentence.

Ranking by size alone would put `unexplained` first on both reference runs, at
64 and 94 failures, and it is the one group nothing can be done about.
Non-actionable rows keep their true `affected` count and sort below actionable
ones, so they stay visible without heading a to-do list.

`actionable` is derived from `status` and stored anyway, so ordering is a plain
column sort rather than a `CASE` every consumer has to reproduce correctly. The
same denormalisation `clusters.size` already uses.

**Rejected.** *Attaching to findings* — loses the pattern. *Attaching to
factors* — too coarse, and would let a factor's frequency alone drive advice.
*A composite priority score* — unauditable. *Omitting non-actionable rows from
the ordering* — see D-035.

**Trade-off.** One recommendation per rule per group means a group can produce
several rows, and a run with many groups produces a long list. Accepted: the
alternative is choosing for the reader which of two valid actions to hide.
Regrouping also discards recommendations and they must be regenerated, which is
correct but means the two passes have to be run in order.

---

## D-035 — A recommendation states its evidential status, and absence of evidence is stored

**Status:** Accepted · Milestone 6

**Decision.** Every recommendation carries one of four statuses:

| Status | Meaning | Actionable |
| --- | --- | --- |
| `replicated` | Factors qualify **and** runs of the same model agree on the outcome mix | yes |
| `provisional` | Only one run has the evidence | yes, flagged |
| `conflicting` | Factors qualify but runs disagree about what the group does | **no** |
| `insufficient_evidence` | Group too small, unexplained, mixed, or its factors do not qualify | **no** |

Groups with nothing to recommend still get a row. Replication is judged only
against runs sharing the same `model_sha256`, and only those that have measured
`factor_rates`.

**Reasoning.** This exists because of a mistake made during the milestone. On
the test split `thin_structure` was 85% false positive, and it was reported —
in this conversation, confidently — as "thin objects are hallucinated." On the
validation split the same group is evenly divided, differing at p = 0.011. The
group replicates; **what it means does not.** Without a status for that, the
engine would have emitted a precision fix for a pattern that exists on one
split and not the other.

Membership replicating is not the same as the pattern replicating, and only the
second justifies an action.

**Why absence is stored rather than omitted.** A missing row is
indistinguishable from a pass that never ran. `images.error` exists for exactly
this reason — so "processed and found nothing" stays separable from "never
processed" — and the same argument applies to advice. "We examined 94 failures
and no measured condition accounts for them" is a finding. Silence is not.

**Why missing data cannot confirm.** A run analysed before `factor_rates`
existed has no rates. Counting it as agreement would manufacture replication
out of a gap: run 1 in the reference database is exactly this case, and it is
excluded with a logged note rather than silently treated as consenting.

**Why the qualification check is repeated.** The discriminating grouping has
already filtered to factors with measured lift, so re-checking in the rule
engine is redundant by construction. It is done anyway because the requirement
— no recommendation from a factor's raw frequency — should hold even if the
grouping method changes, and a redundant check that costs one dictionary lookup
is cheaper than the failure it prevents (D-031).

**Rejected.** *A single confidence percentage* — collapses four genuinely
different situations into one number, and "60% confident" invites acting anyway.
*Filtering non-actionable rows* — the refusals are the honest part of the
output. *Inferring replication from agreement in group size* — that is the
error this record exists to prevent.

**Trade-off.** On the reference run only 2 of 4 groups yield actionable advice.
A tool that always has an answer would show four. This one shows two actions,
one investigation, and one explicit refusal — which is the accurate picture,
and the minimum group size (`MIN_RECOMMENDATION_GROUP_SIZE`, default 10) is
configurable so the bar can be examined rather than assumed.

---

## D-036 — Mask diagnosis re-measures existing pairs, and never re-pairs on outlines

**Status:** Accepted · Milestone 8.5. Closes the deferral recorded in D-022.

**Decision.** Outline-level diagnosis takes the pairing the box pass already
made and measures how well the two outlines agree. It does **not** re-run
matching with mask IoU as the similarity function. Results land in
`mask_findings`, keyed on `finding_id`, exactly as D-022 planned.

`mask_iou` and `mask_outcome` are nullable. An outline that was never produced
is not an outline that scored zero.

**Reasoning.** Re-matching on mask IoU would produce a second, different set of
findings — different pairs, different counts, a different story. The useful
output is not "here is another diagnosis" but "*this* finding, which the box
pass called correct, has an outline that is not". Keeping the pair fixed is
what makes that sentence expressible.

Measured on the reference run, this is not a hypothetical:

| Class | Mean box IoU | Mean mask IoU |
| --- | --- | --- |
| `door` | 0.878 | 0.797 |
| `door_frame` | 0.877 | **0.629** |

Box IoU reports the two classes as indistinguishable, to three decimal places.
Outlines separate them by 0.168. On the validation split the same gap appears —
`door` 0.889 against 0.866, `door_frame` 0.859 against 0.686 — so `door_frame`'s
box-to-outline drop is roughly three times `door`'s on both. **14 findings on
the test split are correct by box and not by outline**, and box-level diagnosis
cannot see any of them.

**Why rasterise rather than intersect polygons analytically.** An outline traced
from a predicted mask follows pixel boundaries, doubles back on itself, and
sometimes encloses no area. Analytic intersection is undefined on such input;
the libraries that offer it either raise or silently repair the shape into
something the model did not predict. Rasterising asks which pixels are inside,
which is well defined for any vertex list and is what the model's own mask
metric is computed from. OpenCV is already a dependency; Shapely is not.

Rasterisation happens inside the two outlines' shared bounding box rather than
the full image — identical answer, a canvas of hundreds of pixels rather than
millions.

**Why this pass re-runs inference.** Every other analysis module is a pure pass
over saved data. This one cannot be: `findings` stores ground-truth outlines
but not predicted ones, and `findings` is frozen (D-020). Rather than alter it,
the model is run again and its outlines are matched back to stored findings by
box identity at IoU ≥ 0.98. Inference is deterministic given the same weights,
size and threshold — all recorded on the run — so a true match scores far above
that bar. A process configured for a different image size than the run recorded
**refuses to run** rather than measuring predictions the run never made.

**Rejected.** *A `mask_iou` column on `findings`* — alters the frozen table
(D-022 already rejected this). *Re-matching on mask IoU* — loses the
disagreement, which is the finding. *Analytic polygon intersection* — wrong on
self-intersecting outlines. *Storing zero for an unmeasurable pair* — would
turn every false positive and false negative into a total outline failure and
drag every average down with fabricated data.

**Trade-off.** The pass is slower than the others because it runs the model
again — a few seconds on 136 images, and it needs the images and weights
present, unlike the pure database passes. Accepted: the alternative is changing
a published table.

100 of 278 findings on the reference run are unmeasurable, and that is correct
rather than a gap: false negatives have no prediction and false positives have
no ground truth, so there is only one outline in each case. Every *paired*
finding was measured.

**The limitation D-022 asked to be stated is now closed.** Reports covering a
segmentation dataset no longer need to warn that thin-structure failures are
invisible — they are measured, and the class comparison prints both numbers
side by side.

---

## D-037 — The API is a read-only projection of the published schema

**Status:** Accepted · API layer

**Decision.** `app/api.py` serves the schema over HTTP and adds nothing. Every
endpoint is a documented query answered by an existing `app/storage.py` reader.
Connections are opened read-only. Files are addressed by **id**, never by path,
and every stored path is verified against configured roots before it is opened.
Responses are schema rows, not view-models.

**Reasoning.** A browser cannot open a SQLite file, so a Next.js front end needs
a service where Streamlit needed none. The question was how much that service
should know.

The answer is nothing. `docs/SCHEMA.md` has survived eight versions without
breaking a query, and two independent consumers have now been built against it.
If the API reimplemented those queries, a disagreement between the dashboard and
the web UI would become possible — and the first symptom would be two different
numbers on two screens with no way to tell which was right. Seventeen storage
readers already return exactly what the endpoints serve; they are reused
verbatim.

**Why `storage.connect()` is unusable here.** It calls `initialise_database()`,
which creates tables and can bump `schema_info.version`. Correct for a CLI pass
that may open a fresh database; wrong for a request handler, where concurrent
workers could race on a schema upgrade and where analysis history is evidence
that must not change because someone loaded a page. The API opens
`mode=ro` URI connections instead and passes them to the same readers, which
issue only `SELECT`s. A reader that ever began writing now fails loudly at this
boundary rather than silently mutating history.

**Why files are served by id.** A `?path=` parameter would be a straight
traversal hole. But the *stored* path is not automatically trustworthy either:
it was written by whichever process ran the diagnosis, into a database file the
operator selected. It is input. Each path is resolved — following symlinks —
and checked against `config.API_FILE_ROOTS` before opening.

Three outcomes are kept distinct rather than collapsed:

| Situation | Status |
| --- | --- |
| Resolves outside every allowed root | **403** |
| Allowed, but the file is gone | **404** |
| Allowed and present | **200** |

Collapsing 403 into 404 would hide a misconfiguration behind a message about a
missing file. Collapsing either into 500 would report an ordinary state as a
server fault — a run diagnosed from a temporary directory outlives it, and that
is normal.

**Why rows rather than view-models.** A response shaped for a screen encodes
assumptions about a front end that does not exist yet, and both consumers would
then have to share those assumptions. Rows are what the contract documents;
composition belongs in the client.

**Rejected.** *Next.js route handlers with `better-sqlite3`* — duplicates every
query in TypeScript, which is the coupling this decision exists to avoid.
*Static JSON export per run* — goes stale, and cannot answer
`/findings/{id}/neighbours`, which is computed. *Write endpoints* — analysis
history is evidence; running a diagnosis stays a CLI action, for the same reason
the dashboard is read-only (D-029). *Authentication* — speculative surface on a
single-user local tool (D-015).

**Trade-off.** The API must be started with the same `MD_DATASETS_DIR` the runs
were diagnosed with, or their images resolve outside every allowed root and are
refused with 403. That is a real operational constraint, and it is stated in the
403 body rather than left to be discovered. `MD_API_FILE_ROOTS` exists for
datasets that live in several places.

Serving image bytes through the API is slower than a static mount. Accepted: a
mount would be a second path-validation surface, and one security boundary that
is definitely correct beats two that are probably correct.

**One capability this unlocks.** `app/similarity.py` has been built and tested
since Milestone 5 and has never been reachable, because nearest-neighbour search
is Python and the dashboard reads only SQL. `/findings/{id}/neighbours` is the
first thing the new front end can do that the old one could not.

---

## D-038 — The compute device is resolved on first access, not at import

**Status:** Accepted · API deployment readiness

**Decision.** `config.DEVICE` is computed the first time it is read and cached,
via a module-level `__getattr__` (PEP 562), rather than assigned at import.
Every caller reads `config.DEVICE` exactly as before.

**Reasoning.** `resolve_device()` imports torch to probe for CUDA and MPS. It
was called at import time, and **every module in the project imports
`config`** — so any process that merely wanted a path loaded the entire ML
stack.

The read-only API is precisely such a process. It touches no model, runs no
inference, and reads SQLite. Yet `import app.api` loaded torch before serving a
request. Measured on this machine:

| | Installed |
| --- | --- |
| torch | 498 MB |
| cv2, scipy, pandas, scikit-learn, numpy, matplotlib, ultralytics | ~395 MB |
| What the API genuinely needs | **~46 MB** |

A deployable container went from roughly 100 MB to 2.5 GB for a device string
that is never read on that path.

**This completes an intent already stated rather than changing one.** The
docstring on `resolve_device()` says it imports torch lazily *"so that config
stays importable in an environment where the ML stack is not installed"*. The
eager assignment defeated that whenever the stack **was** installed. Only three
call sites read `DEVICE` — `features.py`, `inference.py` and the resources
health check — all of them ML paths that load torch regardless.

**Rejected.** *Turning `DEVICE` into a function* — changes three call sites and
every future one, for no benefit over deferring the value. *Shipping torch in
the API image* — a 25× size penalty to avoid a six-line change. *Duplicating a
cut-down config for the API* — two sources of truth for the same settings, and
they would drift.

**Trade-off.** `DEVICE` can no longer be annotated `Final`, since it is not a
module-level assignment. It is still resolved once and cached, so it is
constant for the life of the process exactly as before; a `TYPE_CHECKING` block
declares it for type checkers. This is a narrow deviation from D-002's
"configuration read at import time", and it applies to this one value — the
only one whose computation costs anything.

Deferring also means a torch import failure now surfaces at first device read
rather than at startup. `resolve_device()` already catches `ImportError` and
returns `"cpu"`, so the failure mode is unchanged.

**Guarded by test.** `test_importing_the_api_does_not_load_torch` spawns a
fresh interpreter and asserts neither torch nor ultralytics is loaded after
`import app.api`. It runs in a subprocess deliberately: the test suite has
already imported torch through other modules, so checking `sys.modules` in
process would prove nothing.
