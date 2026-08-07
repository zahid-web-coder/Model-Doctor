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

**Decision.** Both `Detection` and `GroundTruthBox` use absolute pixels in
`xyxy`. Normalised label values are converted once, on read.

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

**Reasoning.** `Detection` and `GroundTruthBox` had independent implementations
of `xyxy` and `area` — duplication that drifts, and drift here silently
corrupts every failure classification built on top.

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
