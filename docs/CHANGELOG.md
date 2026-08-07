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
