# Changelog

Updated after each completed milestone. Entries describe what changed and, for
anything non-obvious, why — with a link to the decision record.

Status vocabulary matches [ROADMAP.md](ROADMAP.md).

---

## Milestone 2 — Inference Foundation · 2026-08-07

First working code. Establishes the configuration, resource, dataset, geometry,
and inference layers that every later milestone builds on.

### Added

**Configuration**
- `config.py` — single source of truth for paths, device selection, and
  thresholds. Paths derive from the module location rather than the working
  directory, and every value is overridable by an `MD_*` environment variable
  (D-002).
- Device auto-detection, CUDA → MPS → CPU, with override.
- `pyproject.toml` — lint and test configuration, making the coding standards
  mechanically enforceable.

**Utilities**
- `utils/exceptions.py` — project exception hierarchy in a dependency-free
  module (D-008).
- `utils/logging_utils.py` — root logger configured exactly once.
- `utils/geometry.py` — `BoxGeometryMixin` and the `HasCorners` protocol, so
  predictions and ground truth share one geometry implementation (D-007).
- `utils/resources.py` — model discovery with fallback (D-004) and
  non-raising verification that reports every missing resource at once (D-003).
- `utils/dataset.py` — `data.yaml` parsing with dynamically discovered class
  names, split resolution, and YOLO label parsing that converts normalised
  centre-form to absolute pixel corners on read (D-006).

**Application**
- `app/inference.py` — lazy model loading, single-image and batch inference,
  library-agnostic `Detection` / `ImagePrediction` types (D-005), annotated
  image output to a project-owned location (D-010), metrics validation, and a
  CLI with a `--check` health command.

**Tests**
- `tests/test_resources.py`, `tests/test_geometry.py` — 27 tests covering
  resource absence, dataset parsing, label handling, geometry equivalence, and
  CLI parsing. None require a model, a dataset, or network access.

**Documentation**
- Full `docs/` set established.
- `README.md` — setup, expected resource layout, usage.

### Fixed

- **Parallel tensor alignment.** The detector's coordinate, confidence, and
  class tensors were zipped without `strict=`, so a length mismatch would have
  silently truncated to the shortest and discarded detections with no error.
  Now enforced (D-011). Found by lint in code whose own comments described the
  hazard — documenting an invariant is not enforcing it.

### Changed

- Exception hierarchy moved out of `utils/resources.py`, which had forced
  `utils/dataset.py` to depend on the resource checker purely to raise an error
  (D-008).
- Duplicated geometry on the two box types replaced by a shared mixin (D-007).
- Model-discovery logging lowered to `DEBUG`; the selected file is already
  reported once in the health report.

### Verified

- Test suite passes.
- Lint passes with zero findings under the project ruleset.
- Single-image inference runs end to end and writes an annotated image.
- All missing-resource paths exit cleanly with actionable messages and correct
  exit codes.

### Not verified

- **Batch inference** and **metrics validation** are implemented and
  unit-tested but have never been run against a real dataset, because none is
  available. They are written, not proven, and are reported as such until a
  dataset exists.

### Notes

- Environment: Python 3.11 in a project-local `.venv/` (D-001). No global
  installs; no other project on the machine was modified.
- End-to-end verification used a publicly available pretrained checkpoint held
  entirely outside the project tree, with paths redirected by environment
  variable. No model or dataset was placed in `models/` or `datasets/`, and
  none was fabricated.

---

## Milestone 2 — Quality & Maintainability Review · 2026-08-07

A structured review of the code produced in Milestone 2, conducted before
starting Milestone 3. Findings were determined by measurement — call-site
analysis and lint — not by inspection alone.

### Fixed

- **Leaky abstraction in validation.** `Detector.validate()` returned the
  evaluation library's own object typed as `Any`, and the CLI reached into it
  via `getattr` to read `box.mp`, `box.map50`, and friends. A third-party
  object was escaping two layers outward — the exact coupling D-005 forbids for
  detections. Now returns `ValidationMetrics`, with `from_raw()` as the single
  point of coupling (D-013).

  Side benefit: the extraction is testable with plain stubs, so nine tests now
  cover a surface that was previously untestable until a dataset existed.

### Removed

Five public symbols never called in production, identified by call-site
analysis (D-015):

- `resources.require()`
- `Detector.is_loaded`
- `geometry.HasCorners`
- `Detection.to_dict()` and `ImagePrediction.to_dict()` — their schema was a
  guess at a future persistence format; the real shape will be driven by the
  module that actually needs it.

### Changed

- `dataset.describe()` was also uncalled, but useful. Rather than deleting it,
  it is now wired into `--check`, which prints the discovered class list and
  per-split image counts — the fastest way to confirm a dataset is set up as
  intended.
- `try/except/pass` around dataset description replaced with
  `contextlib.suppress`.

### Documented

- D-013 — metrics normalised at the detector boundary.
- D-014 — mixin for single-box geometry, free functions for two-box operations.
  Written now to pre-empt the predictable drift of adding `iou()` as a method
  in Milestone 3 simply because the mixin exists.
- D-015 — speculative public API is removed rather than retained, including the
  reasoning for why `load_ground_truth()` was kept under the same rule.
- D-003 and D-007 amended, since both referenced code this review deleted.

### Verified

- Lint passes with zero findings.
- 34 tests pass (up from 27).
- Single-image inference unchanged: identical detections and output.
- All four missing-resource paths exit `1` with actionable messages.
- `--check` against a temporary descriptor correctly reports discovered classes
  and per-split counts.

### Assessment

The architecture still fits. The layering held under review — no cycles, no
upward dependencies — and the changes above were consistency fixes within the
existing design rather than a restructuring. No refactor was undertaken for
elegance alone.

---

## Milestone 2.5 — Generic Annotation Model · 2026-08-08

Prompted by auditing the first real dataset, which turned out to be
segmentation: 2,151 polygon annotations, zero bounding boxes. The box-only
parser skipped every one of them without crashing — returning zero ground truth
for all 1,291 images, which would have made every prediction look like a false
positive.

### Added

- `utils/annotations.py` — the generic object-annotation model (D-016).
  - `ObjectAnnotation`: class id, class name, bounding box, optional polygon,
    optional confidence. One type for ground truth and predictions.
  - `polygon_to_bbox()`: derives a box from an outline's extent, matching how
    the detection ecosystem converts segments to boxes.
  - `from_polygon()` / `from_box()` factories, so call sites state which kind
    of source they are handling.
- `tests/test_annotations.py` — 19 tests covering all three dataset shapes,
  polygon derivation, and malformed-input rejection.

### Changed

- `load_ground_truth()` now reads **both** YOLO label forms, chosen per *line*
  rather than per file, so a dataset containing both needs no configuration.
  It also accepts a `class_names` mapping so annotations carry readable names.
- `Detection` is now a subclass of `ObjectAnnotation` that requires a
  confidence, enforced at construction. Predictions and ground truth are the
  same kind of thing, differing only in whether certainty is known.
- `GroundTruthBox` retired — the name asserted box-only, which is no longer
  true. Its role is filled by `ObjectAnnotation`.
- Constructor call sites updated in `tests/test_geometry.py` and
  `tests/test_resources.py`. Every assertion was preserved; only construction
  changed.

### Not included, deliberately

Mask IoU, polygon area, and segmentation analysis are out of scope for this
change. Prediction outlines are also not yet read from model output — the field
exists, the extraction does not. The model is ready for those; it does not
pre-empt them.

### Verified

- Lint clean; **53 tests pass** (up from 34).
- Against the real 1,291-image segmentation dataset, read-only:
  **2,151 / 2,151 annotations parsed** in 0.5 s, class counts matching an
  independent audit exactly (1,126 / 1,025), zero degenerate boxes, every
  ground-truth annotation correctly carrying no confidence.
- Before this change the same dataset yielded **0** annotations.

---

## Milestone 3 — Diagnosis Engine · 2026-08-09

The first genuinely diagnostic capability: explaining *why* predictions fail,
rather than reporting how much they do.

### Added

- `utils/geometry.box_iou()` — overlap between two boxes, as a free function
  because the operation is symmetric (D-014, decided in advance).
- `utils/matching.py` — generic one-to-one assignment between two sets of
  annotations. Independent of YOLO, of failure taxonomies, and of what an
  unmatched prediction means. Greedy on similarity; ties broken deterministically
  so diagnoses are reproducible (D-018). The comparison function is a
  **parameter** — the seam mask IoU will use, exercised by tests today.
- `app/diagnosis.py` — the diagnosis engine.
  - `Outcome`: correct, wrong class, poor localisation, false positive,
    false negative.
  - `Finding`: one outcome plus **both** annotations and their overlap, so every
    conclusion carries its evidence.
  - `ImageDiagnosis`, `DatasetDiagnosis`, `ClassStatistics`.
  - `worst_images()` — turns "recall is low" into a ranked list of images to
    open.
  - CLI: `python -m app.diagnosis --split test`.
- `config.LOCALIZATION_IOU_FLOOR` — the floor for the second matching pass.
- `tests/test_matching.py`, `tests/test_diagnosis.py` — 38 tests covering IoU
  edge cases (perfect, partial, none, contained, touching edges and corners,
  degenerate), matching behaviour, all five outcomes, and the accounting
  invariant.

### Design decisions

- **D-017** — a weak overlap with the correct class is **one poor localisation**,
  not a false positive plus a false negative. Explanation over benchmark parity.
  Consequence: these FP/FN counts will not equal mAP's, and that is documented
  wherever the numbers appear.
- **D-018** — matching is greedy on *similarity*, not confidence, because the
  question is which prediction describes which object, not how the model ranked
  its own guesses.

### Not implemented, deliberately

Mask IoU, visual explanation, clustering, recommendations, dashboard, LLM.
Mask support is architecture-ready — the matcher takes a similarity function —
but nothing computes mask overlap.

### Verified

- Lint clean; **91 tests pass** (53 from Milestone 2, unchanged, plus 38 new).
- Milestone 2 capabilities confirmed intact: inference, validation (box and
  mask metrics identical to before), polygon parsing, segmentation support.
- Run end to end against the real model and the real test split: every
  prediction and every ground truth accounted for exactly once, with no double
  counting.

---

## Week 2 completion — Persistence Layer & Backend Contract · 2026-08-09

Diagnosis results were previously in-memory only and died with the process.
This makes them durable, queryable, and — the actual point — consumable by a
second developer building a dashboard without reading this codebase.

### Added

- `app/storage.py` — SQLite persistence.
  - Tables `runs`, `images`, `findings`, normalised so later milestones add
    tables rather than altering ones already being queried (D-020).
  - `runs` records the model SHA-256, dataset, split, all three thresholds and
    image size — everything needed to reproduce a run (D-021).
  - Foreign keys enforced per connection; SQLite ignores them otherwise.
  - Writes occur in the caller's transaction, so an interrupted save leaves no
    partial run.
  - API: `save_run`, `save_image`, `save_findings`, `save_dataset_diagnosis`,
    `load_run`, `list_runs`, `load_findings`, `outcome_counts`.
- **`docs/SCHEMA.md`** — the published backend contract. Schema, relationships,
  outcome enum with its null patterns, nine worked queries, data-flow diagram,
  stability guarantees, and the explicit warning that these counts are not COCO
  mAP.
- `--save`, `--db`, and `--list-runs` on the diagnosis CLI.
- `config.DB_DIR` / `config.DB_PATH`.
- `tests/test_storage.py` — 20 tests covering round trips, schema integrity,
  foreign-key enforcement, cascade deletes, multiple runs, idempotent schema
  creation, and transaction rollback.

### Changed

- `ValidationMetrics` gained scalar **F1** for boxes and masks, closing the last
  Week 1 metric gap. Derived, not stored, so it cannot disagree with the
  precision and recall it comes from.
- `ImageDiagnosis` carries `image_width` / `image_height`. Additive with
  defaults — no existing call site changed. Needed because a pixel box is
  meaningless without the frame it sits in, and the database must be
  interpretable without re-opening images.
- `.gitignore` excludes `db/` and any `*.db`. The database records absolute
  image paths and filenames, which carry dataset identifiers; the schema is
  committed, the data is not.

### Not implemented, deliberately

Grad-CAM, dashboard, clustering, root-cause analysis, recommendations, mask IoU.
No tables exist for them — they arrive with the milestone that produces them.

### Verified

- Lint clean; **111 tests pass** (91 from Milestone 3, unchanged, plus 20 new).
- Milestone 2 and 3 capabilities intact: inference, validation, diagnosis,
  polygon parsing, segmentation support.
- Two real runs saved from the real model and dataset (test and val splits).
  SQL outcome counts match the engine's printed report exactly; the accounting
  guarantee holds for both runs (218/218 predictions, 238/238 truths on run 1;
  392/392 and 434/434 on run 2). 672 ground-truth polygons round-tripped.
- **Every example query in SCHEMA.md was executed verbatim against the live
  database.** All ten valid. Documentation drift would surface as a failure
  rather than as silent staleness.
