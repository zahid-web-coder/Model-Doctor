# Architecture

Describes the system **as built**. Planned components are marked as such and
carry no implementation claims.

---

## Layering

Dependencies point downward only. A lower layer never imports a higher one.

```
                 ┌──────────────────────────────┐
   Presentation  │  CLI (inference · diagnosis) │   Planned: dashboard
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
   Application   │  Detector · Diagnosis engine │   Planned: clustering
                 │  Storage (SQLite)            │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
   Utilities     │  resources · dataset ·       │
                 │  annotations · geometry ·    │
                 │  matching · exceptions ·     │
                 │  logging_utils               │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
   Configuration │  config.py                   │
                 └──────────────────────────────┘
```

`config.py` imports nothing from the project. `utils/exceptions.py` imports
nothing at all. Those two are the stable foundation everything else rests on.

---

## Modules

### `config.py` — Completed

Single source of truth for paths, device selection, and tunable constants.

- Paths derive from `__file__`, never the working directory, so behaviour does
  not change with launch location.
- Every value is overridable by environment variable (`MD_*`).
- Importing it performs no I/O and cannot fail. Verification is separate and
  explicit.
- Creates output directories only. It never creates `models/` or `datasets/`,
  because a conjured empty directory turns a clear "not provided yet" into a
  confusing "empty".

### `utils/exceptions.py` — Completed

The project's exception hierarchy, in a module with **zero imports**.

Any module can raise a project error without acquiring an unrelated dependency.
`ModelDoctorError` is the common base, which lets callers distinguish
anticipated failures from genuine bugs.

```
ModelDoctorError
├── ResourceNotFoundError
├── ModelLoadError
└── DatasetConfigError
```

### `utils/logging_utils.py` — Completed

Root logger configured exactly once, guarded by a module flag. Modules call
`get_logger(__name__)` and never `basicConfig`, which prevents duplicated or
suppressed output.

### `utils/geometry.py` — Completed

`BoxGeometryMixin` supplies derived geometry (`xyxy`, `width`, `height`,
`area`, `center`) to any class exposing pixel corners. `box_iou` is a free
function computing overlap between two boxes.

A mixin rather than a base dataclass, so subclasses keep their own field
ordering (D-007). Geometry over a *single* box belongs on the mixin; operations
over *two* boxes will be free functions in this module (D-014).

### `utils/annotations.py` — Completed

The generic object-annotation model: `ObjectAnnotation` (class id, class name,
bounding box, optional polygon, optional confidence) and `polygon_to_bbox`.

One type serves detection and segmentation data. When a polygon is present the
box is *derived* from it, so box-based analysis works on segmentation data with
no special-casing and the two can never disagree. `Detection` subclasses it and
requires a confidence. See [DECISIONS.md](DECISIONS.md) D-016.

Deliberately absent: mask IoU, polygon area, segmentation analysis.

### `utils/matching.py` — Completed

One-to-one assignment between two sets of annotations. Knows nothing about
YOLO, about failure categories, or about what an unmatched prediction *means* —
it reports what paired with what and how strongly, and interpretation is the
caller's job.

- Greedy on similarity, strongest pair first (D-018).
- Ties broken by confidence then index, so runs are reproducible.
- The comparison is a **parameter**, defaulting to `box_iou`. This is the seam
  mask IoU will use; it is exercised by tests today so it cannot rot.

### `utils/resources.py` — Completed

Answers "is it here?" — once, in one place.

- `discover_model` resolves weights: explicit path → configured path → newest
  `*.pt` in the models directory. Real training runs emit varied filenames;
  users should not have to rename files.
- `check_*` functions each return a `ResourceStatus` and **never raise**.
- `verify_all` + `format_report` produce the health report, listing **every**
  missing resource at once so problems are fixed in one pass rather than one
  failed run each.

### `utils/dataset.py` — Completed

Owns everything about ground truth.

- Parses `data.yaml`; accepts `names:` as list or dict.
- Resolves split paths and drops splits that do not exist on disk.
- Warns when declared `nc` disagrees with the length of `names`.
- `label_path_for_image` swaps the **last** `images` path segment.
- `load_ground_truth` reads both YOLO label forms — 5-field boxes and
  polygons of 1+2n fields — choosing per *line*, so a file may mix them. Both
  are converted to absolute pixels on read; malformed lines are skipped with a
  warning rather than aborting the file.
- A missing label file returns `[]` — a legitimate negative sample.

### `app/diagnosis.py` — Completed

The diagnosis engine. Classifies every prediction and every ground truth into
one of five outcomes: correct, wrong class, poor localisation, false positive,
false negative.

- Two matching passes. The first at the match threshold finds hits; the second,
  at a much lower floor, catches predictions that landed on an object but
  outlined it badly (D-017).
- `Finding` keeps **both** annotations plus the overlap, so every conclusion
  carries its evidence.
- `ImageDiagnosis` accounts for every prediction and every ground truth exactly
  once — verified on real data, not only in unit tests.
- `DatasetDiagnosis` aggregates, computes per-class statistics, and ranks the
  worst images by failure count.
- CLI: `python -m app.diagnosis --split test`.

Not implemented: mask IoU, clustering, recommendations, visual explanation.

### `app/storage.py` — Completed

SQLite persistence for diagnosis results, and the point at which this project
becomes consumable by code that is not this project.

- Three tables: `runs`, `images`, `findings`. Normalised so later milestones add
  tables rather than altering ones a dashboard already queries (D-020).
- `runs` stores the model SHA-256 and every threshold, so a finding has
  provenance and two runs can be compared (D-021).
- Foreign keys enforced per connection — SQLite ignores them otherwise.
- Writes happen in the caller's transaction, so an interrupted save leaves no
  partial run.
- **The published interface is [SCHEMA.md](SCHEMA.md), not this module.** Every
  example query in that document is executed against a real database during
  verification, so documentation drift surfaces as a failure.

Lives in `app/` because it depends on the diagnosis domain types; a `utils`
module importing `app` would be the first upward dependency in the project
(D-019).

### `app/inference.py` — Completed

Converts a detector into structured, inspectable data.

- `Detection` / `ImagePrediction` / `ValidationMetrics`: library-agnostic
  result types. Precision, recall, F1, mAP50 and mAP50-95 for boxes and masks;
  F1 is derived rather than stored so it cannot disagree with its inputs. `ValidationMetrics.from_raw` is the sole point of coupling to
  the evaluation library's object shape (D-013).
- `Detector`: lazy loading, idempotent `load()`, per-image error isolation.
- `predict_many` is a generator, so memory stays flat and progress streams.
- `validate()` wraps the detector's own metrics evaluation and normalises the
  result, so no third-party object escapes this module.
- CLI entry point returning exit codes rather than calling `sys.exit`, so it is
  testable.

---

## Data flow

```
  data.yaml ──► DatasetConfig ──► class names, split paths
                                          │
  image ──────────────────────────────────┤
                                          ▼
                     Detector.predict_image()
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                 list[Detection]              annotated image
                 (pixels, xyxy)               results/predictions/
                            │
                            ▼
                  diagnose_image()  ◄── list[ObjectAnnotation]
                            │              ground truth (boxes or polygons)
                            ▼
              Finding × (predictions + truths)
              correct · wrong class · poor localisation
              false positive · false negative
                            │
                            ▼
                    DatasetDiagnosis
              per-class statistics · worst images
                            │
                            ▼  --save
                    db/model_doctor.db
              runs · images · findings   ◄── see SCHEMA.md
```

Predictions and ground truth are the *same type* — `Detection` is an
`ObjectAnnotation` that carries a confidence — sharing one coordinate
convention (**absolute pixels, `xyxy`**) and one geometry implementation. That
is what allows them to be compared directly, and is the most consequential
decision in the system so far.

---

## Key boundaries

### Detector isolation

Everything specific to the detection library lives inside
`Detector._extract_detections` and `ValidationMetrics.from_raw`. Nothing outside
`app/inference.py` imports the detection library or handles its objects.
Supporting another architecture means adding a sibling that produces the same
`Detection` objects — analysis code is untouched.

This is why raw tensors are converted to plain Python floats at that boundary
rather than passed onward.

### Output ownership

Annotated images are written by us into the configured predictions directory,
not by the library into a directory of its choosing. The project stays
self-contained and its output location stays configurable.

### Resource verification

Exactly one module decides whether a resource exists. Callers ask; they do not
re-implement checks. This keeps missing-resource behaviour consistent and
testable.

---

## Extension points

| Extension | Mechanism | Status |
| --- | --- | --- |
| Additional detector family | New module producing `Detection` | Planned |
| Failure classification | `app/diagnosis.py` | **Available** |
| Segmentation analysis | Pass a mask similarity fn to the matcher | Planned |
| Prediction outlines | Read masks in `_extract_detections` | Planned |
| New image format | One entry in `config.IMAGE_EXTENSIONS` | Available |
| Alternative output location | `MD_*` environment variables | Available |

---

## Testing strategy

Tests target behavioural contracts, not implementation details.

- Resource absence is the **primary** contract while the model and dataset are
  unavailable, so it carries the most coverage.
- Geometry tests assert that predictions and ground truth agree, and include a
  guard against re-introducing duplicated geometry.
- Inference tests cover the containers, error states, and CLI parsing. They do
  not require a model.
