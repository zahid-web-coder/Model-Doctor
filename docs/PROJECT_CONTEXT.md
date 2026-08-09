# Project Context

**This document is the project's source of truth.** Where any other document
disagrees with it, this one wins and the other should be corrected.

Last updated: 2026-08-09 (Milestone 3 + persistence layer)

---

## What this project is

Model Doctor is a Vision AI **failure diagnosis** platform for object detection
models. It explains *why* a model fails, rather than only reporting how much it
fails. See [VISION.md](VISION.md) for the reasoning behind that goal.

Version 1 targets YOLO. Support for other detector families is an explicit
architectural concern but is **not implemented**.

---

## Current state

| Area | Status |
| --- | --- |
| Environment and dependencies | **Completed** |
| Configuration layer | **Completed** |
| Resource verification | **Completed** |
| Dataset descriptor + label parsing | **Completed** — boxes *and* polygons |
| Shared box geometry | **Completed** |
| Generic annotation model (box + polygon) | **Completed** |
| YOLO inference + structured output | **Completed** |
| Batch inference over a split | **Completed** |
| Metrics validation (box + mask) | **Completed** |
| Milestone 2 quality review | **Completed** |
| Box IoU + generic matching engine | **Completed** |
| Failure classification (5 outcomes) | **Completed** |
| Per-image and dataset diagnosis | **Completed** |
| SQLite persistence + published schema | **Completed** |
| Mask IoU / segmentation analysis | **Planned** — architecture-ready, not built |
| Everything beyond error analysis | **Planned** — see [ROADMAP.md](ROADMAP.md) |

### Verified by execution

- Full test suite passes; lint clean under the project ruleset.
- Inference, validation, and diagnosis all run against a real trained model and
  a real labelled dataset.
- Validation reports both box and mask metric families.
- The diagnosis engine accounts for every prediction and every ground truth
  exactly once — checked on real data, not only in unit tests.
- Every missing-resource path exits cleanly with an actionable message and a
  non-zero exit code.

### Not yet verified by execution

Nothing implemented is currently unproven.

---

## Resource situation

The project depends on two resources it does not own and does not create:

| Resource | Expected location | Status |
| --- | --- | --- |
| Trained detector weights | `models/` (any `*.pt`) | **Available** — a nano segmentation model |
| Dataset + descriptor | `datasets/` incl. `data.yaml` | **Available** — a 2-class segmentation dataset |

The codebase is written to operate in this state. Absence of a resource is a
supported condition, reported through the health check, not an error state that
crashes.

**Rule: never fabricate a model or dataset to make code appear to work.** If a
capability cannot be exercised, say so.

Check current status at any time:

```bash
./.venv/bin/python -m app.inference --check
```

---

## Expected resource layout

```
models/
  <any>.pt                 # newest is selected when several are present

datasets/
  data.yaml                # must declare `names:`
  <split>/
    images/<name>.<ext>
    labels/<name>.txt      # matching stem; images/ -> labels/
```

Label format, one object per line, geometry normalised to `0..1`. Both forms
are supported, chosen per line, so a file may contain either or both:

```
<class_id> <x_center> <y_center> <width> <height>     detection
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...          segmentation
```

A segmentation annotation also yields a bounding box, derived from the
outline's extent, so box-based analysis needs no special-casing.

---

## Environment

| Item | Value | Reason |
| --- | --- | --- |
| Python | 3.11 | Broadest wheel support across the detection stack |
| Environment | `.venv/` in project root | Isolated; no global installs |
| Compute | Auto-detected: CUDA → MPS → CPU | Overridable via `MD_DEVICE` |

Setup:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

---

## Working agreements

Recorded in full in [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md). The ones that
most often get violated:

- Never hardcode class names, class counts, dataset paths, or model paths.
- Configurable values belong in `config.py`; reusable logic belongs in `utils/`.
- Version control is not managed by tooling on this project. Do not initialise
  a repository, create a remote, commit, or push.
- Do not implement beyond the current milestone.

---

## Where to look

| Question | Document |
| --- | --- |
| Why does this project exist? | [VISION.md](VISION.md) |
| How is the code organised? | [ARCHITECTURE.md](ARCHITECTURE.md) |
| What is built, and what is next? | [ROADMAP.md](ROADMAP.md) |
| Why was it built this way? | [DECISIONS.md](DECISIONS.md) |
| What are the process rules? | [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) |
| How should code be written? | [CODING_STANDARDS.md](CODING_STANDARDS.md) |
| What changed, and when? | [CHANGELOG.md](CHANGELOG.md) |
| **What is the database contract?** | **[SCHEMA.md](SCHEMA.md)** |

---

## Reading the diagnosis numbers

The failure counts this project reports **do not equal those implied by mAP**,
by design (D-017):

- A prediction that lands on an object but overlaps it weakly is reported as
  **one** poor localisation, not as a false positive plus a false negative.
- Diagnosis runs at the configured confidence threshold; mAP integrates across
  all thresholds.

Both are correct answers to different questions. Do not compare them directly.
