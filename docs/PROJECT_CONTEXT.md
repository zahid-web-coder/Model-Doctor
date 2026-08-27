# Project Context

**This document is the project's source of truth.** Where any other document
disagrees with it, this one wins and the other should be corrected.

Last updated: 2026-08-25 (Milestone 10 closed — evaluation and benchmark
persistence complete). Schema version 9.

---

## What this project is

Model Doctor is a Vision AI **failure diagnosis** platform for object detection
models. It explains *why* a model fails, rather than only reporting how much it
fails. See [VISION.md](VISION.md) for the reasoning behind that goal.

YOLO is the default detector family. RF-DETR is supported as an **optional**
second family: detector-specific code is isolated behind `app/inference.py`,
`app/detectors.py` and `app/rfdetr_adapter.py`, and every analysis stage runs
unchanged on either. Explainability is the one stage that does not yet cover
both — see *Current state* below.

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
| SQLite persistence + published schema | **Completed** — schema version 2 |
| Feature extraction (CLIP embeddings) | **Completed** |
| Grad-CAM explanation backend | **Completed** — schema version 3 |
| Root-cause analysis | **Completed** — schema version 4 |
| Failure grouping (deterministic, by root cause) | **Completed** — schema version 5 |
| Factor base rates + significance | **Completed** — schema version 6 |
| Discriminating failure groups (two partitions) | **Completed** — no schema change |
| Recommendations with evidence status | **Completed** — schema version 7 |
| Mask-level diagnosis | **Completed** — schema version 8 |
| Read-only HTTP API | **Completed** — no schema change; D-037 |
| Data-calibrated size and shape factors | **Completed** — replicated on two splits |
| Similar-failure retrieval (nearest neighbour) | **Completed** — reads existing embeddings |
| Developer dashboard | **Completed** — eight tabs, including recommendations and outline comparison |
| Mask IoU / segmentation analysis | **Completed** — schema version 8; D-022 closed by D-036 |
| Second detector family (RF-DETR) | **Completed** — optional dependency; family read from the checkpoint, no schema change |
| Web dashboard (Next.js, over the HTTP API) | **Completed** — overview, failures, root causes, clusters, heatmaps, comparison |
| Model comparison | **Completed** — no new endpoint, no schema change |
| Prediction and ground-truth overlays | **Completed** — drawn from stored geometry |
| Shared COCO evaluation + persisted mAP | **Completed** — schema version 9; one evaluator for every family |
| Persisted per-device benchmarks | **Completed** — schema version 9; device recorded per measurement |
| RF-DETR explainability | **Not implemented** — Grad-CAM refuses rather than mis-applying; see [ROADMAP.md](ROADMAP.md) |
| Everything beyond the above | **Planned** — see [ROADMAP.md](ROADMAP.md) |

### Verified by execution

- Full test suite passes; lint clean under the project ruleset.
- Inference, validation, and diagnosis all run against a real trained model and
  a real labelled dataset.
- Validation reports both box and mask metric families.
- The diagnosis engine accounts for every prediction and every ground truth
  exactly once — checked on real data, not only in unit tests.
- Every missing-resource path exits cleanly with an actionable message and a
  non-zero exit code.
- The dashboard reads the SQLite schema directly and never modifies diagnosis
  history. It reports missing databases and unavailable source images clearly.
- Failure grouping was run on the 136-image reference run: 127 failures, all
  127 grouped, 22 groups, largest `edge_truncation` at 28.3%, `unexplained` at
  9.4%. Grouped total equals the failure count.
- K-means over the embeddings was implemented and measured before being
  rejected — silhouette 0.13–0.23 across every *k* tried (D-030). The rejection
  is evidence-based, not assumed.
- Factor base rates measured on the same run: `edge_truncation` describes 71%
  of failures and 76% of correct detections (lift 0.93x, p = 0.34), so the
  largest failure group rests on a factor with no demonstrated association.
  `crowding` is the only factor with a positive signal, and at p = 0.041 it
  does not survive correction for testing five factors (D-031). It then failed
  to replicate on the val split entirely — 1.12x at p = 0.310.
- Two calibrated factors do replicate, on 136-image and 258-image splits:
  `small_object` at 2.77x / 3.45x and `thin_structure` at 2.14x / 2.32x, all
  p < 0.001. Size shows a monotonic dose-response and holds within a single
  class, so it is not class in disguise (D-032).
- Both detector families run the whole pipeline on the same dataset, and are
  compared through one evaluator. RF-DETR is the more accurate model there and
  markedly the more expensive: mAP@50 0.879 against 0.836 for boxes, at 1.80 FPS
  against 21.57 on the CPU.
- Box mAP from `app/evaluation.py` reproduces the standalone comparison script
  exactly, which is what establishes the persisted pipeline as equivalent to the
  one already validated.
- Comparison alignment was checked against the API: 228 ground-truth objects
  paired, none unmatched on either side, and every unpaired finding a false
  positive — which is why false positives are counted per image rather than
  paired to ground truth.
- A benchmark records the device it actually ran on. RF-DETR binds its device at
  construction rather than per call, and until that was passed through, CPU-
  labelled measurements were running on MPS — 588 ms recorded as 162 ms. Found
  by review, fixed, and covered by regression tests.

### Not yet verified by execution

Nothing implemented is currently unproven.

---

## Resource situation

The project depends on two resources it does not own and does not create:

| Resource | Expected location | Status |
| --- | --- | --- |
| Trained detector weights | `models/` (any `*.pt`) | **Available** — nano segmentation models for both families |
| Dataset + descriptor | `datasets/` incl. `data.yaml` | **Available** — segmentation datasets; the current reference run is single-class |

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
- Work happens on `feature/<name>` branches, merged to `main` by pull request.
  Never push directly to `main`.
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

## Dashboard scope

There are two readers, both read-only, both rendering nothing but the tables
published in [SCHEMA.md](SCHEMA.md).

The **web dashboard** (`web/`, Next.js) reads through the HTTP API and is where
current work lands: overview, failures, root causes, clusters, heatmaps,
reports, and model comparison. Comparison covers metric provenance, failure
alignment, prediction overlays, persisted mAP and per-device compute.

The **Streamlit explorer** (`app/dashboard.py`) predates it and opens a SQLite
file directly, without needing the API running. It is kept for that reason.

Every stage now has a schema contract, so nothing is withheld for lack of one.
Where a figure genuinely is not available — a run never evaluated, or two runs
benchmarked on different devices — it is labelled as unavailable with the
reason, never rendered as zero.
