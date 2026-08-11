# Roadmap

Status vocabulary, used consistently across all documentation:

- **Completed** — implemented, executed, and tested.
- **In Progress** — actively being built.
- **Planned** — agreed direction, no implementation. Nothing may be documented
  elsewhere as though it exists.

A milestone is complete only when the Definition of Done in
[DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) is fully satisfied.

---

## Milestone 1 — Project initiation · **Completed**

Project scope and mission agreed. See [VISION.md](VISION.md).

---

## Milestone 2 — Inference foundation · **Completed**

Understand a trained detector and build the inference foundation.

| Item | Status |
| --- | --- |
| Environment verified, dependencies installed | Completed |
| Project structure established | Completed |
| Configuration layer (`config.py`) | Completed |
| Exception hierarchy | Completed |
| Centralised logging | Completed |
| Resource discovery + verification | Completed |
| `data.yaml` parsing, dynamic class names | Completed |
| YOLO label parsing, normalised → pixel | Completed |
| Shared box geometry | Completed |
| Model loading + single-image inference | Completed |
| Structured detection output | Completed |
| Annotated image output | Completed |
| CLI with health check | Completed |
| Batch inference over a split | Completed (code) — **unrun** |
| Metrics validation | Completed (code) — **unrun** |
| Test suite | Completed |
| Lint clean under project ruleset | Completed |
| Documentation set | Completed |
| Quality & maintainability review | Completed |
| Generic annotation model (box + polygon) | Completed |

**Deliberately excluded** from this milestone: explanation methods, embeddings,
failure clustering, recommendations, dashboard, database, reports, and
additional detector families.

**Carried forward.** Batch inference and metrics validation are implemented and
unit-tested but have never run against a real dataset, because no dataset is
available. They are written, not proven. This must be stated as such until a
dataset exists — see [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

---

## Milestone 3 — Error analysis · **Completed**

The first genuinely diagnostic capability, and the reason the inference layer
was shaped as it is.

| Item | Status |
| --- | --- |
| Box IoU (`utils/geometry.box_iou`) | Completed |
| Generic matching engine (`utils/matching.py`) | Completed |
| Five-outcome failure classification | Completed |
| Per-image diagnosis | Completed |
| Dataset summary + per-class statistics | Completed |
| Worst-image ranking | Completed |
| CLI (`python -m app.diagnosis`) | Completed |
| Test suite | Completed |
| Verified on a real model and dataset | Completed |
| SQLite persistence (`app/storage.py`) | Completed |
| `docs/SCHEMA.md` — published backend contract | Completed |
| Scalar F1 (box + mask) | Completed |

The three open design questions were resolved:

1. *Weak overlap, correct class* — reported as **one poor localisation**, not a
   false positive plus a miss (D-017). Explanation beats benchmark parity.
2. *Matching strategy* — greedy on **similarity**, not confidence (D-018).
3. *Threshold or sweep* — a single configurable threshold, plus a second
   low-overlap pass. A sweep is a reporting feature, not a matching one, and
   was left out.

**Deliberately excluded:** mask IoU, visual explanation, clustering,
recommendations, dashboard, LLM. Mask support is architecture-ready — the
matcher takes a similarity function — but no mask overlap is implemented.

Depends on: Milestone 2.

---

## Week 3 — Explainability (Grad-CAM backend) · **Completed**

Per-finding heatmaps showing where the model looked.

| Item | Status |
| --- | --- |
| Grad-CAM support reviewed for the segmentation model | Completed |
| Target layers determined by measurement | Completed |
| Detector-agnostic CAM engine + adapter seam | Completed |
| Raw-logit scalar selection (decoded score saturates) | Completed |
| Letterboxing and overlay rendering | Completed |
| `heatmaps` table, schema version 3 | Completed |
| CLI (`python -m app.explainability`) | Completed |
| Verified on the real model and dataset | Completed |

**Deliberately excluded:** the dashboard rendering of these heatmaps, which is
Week 4 and belongs to the dashboard developer. This milestone is the backend.

---

## Week 7 — Root-cause analysis · **Completed**

Attributes failures to measurable conditions, turning "recall is low" into
hypotheses that can be checked.

| Item | Status |
| --- | --- |
| Blur, low light, small object, edge truncation, crowding | Completed |
| Class imbalance, recurring misclassification | Completed |
| Pluggable detector seam (both scopes) | Completed |
| `root_causes` table, schema version 4 | Completed |
| Cluster-ready by attachment point, verified by test | Completed |
| CLI (`python -m app.root_cause`) | Completed |
| Verified on the real model and dataset | Completed |

**Documented limitation:** directed confusion pairs need the predicted class,
which the contract does not store. Reported undirected instead — see
[DECISIONS.md](DECISIONS.md) D-028.

**Deliberately excluded:** clustering, recommendations, dashboard rendering.

---

## Week 5 — Feature extraction · **Completed**

Encode each failed region into a vector so failures become comparable, which is
the input the failure-grouping milestone needs.

| Item | Status |
| --- | --- |
| Region extraction from bounding boxes | Completed |
| Pluggable region extractor (mask seam) | Completed |
| CLIP encoder, lazily loaded | Completed |
| Injectable encoder so tests need no download | Completed |
| `embeddings` table, schema version 2 | Completed |
| CLI (`python -m app.features`) | Completed |
| Verified on the real model and dataset | Completed |

**Deliberately excluded:** clustering, explanation, recommendations, mask-based
crops. See [DECISIONS.md](DECISIONS.md) D-023.

---

## Milestone 4 — Visual explanation · **Planned**

Make individual failures visually interpretable. Method selection deferred.

Depends on: Milestone 3.

---

## Milestone 5 — Failure grouping · **Planned**

Group failures sharing a probable cause, so an engineer addresses patterns
rather than individual images.

Depends on: Milestones 3, 4.

---

## Milestone 6 — Recommendations · **Planned**

Translate grouped failures into concrete suggested actions.

Depends on: Milestone 5.

---

## Milestone 7 — Developer dashboard · **Planned**

Present analysis through a developer-facing interface.

Depends on: Milestones 3–6.

---

## Milestone 8.5 — Mask-level diagnosis · **Planned (deferred)**

Extend failure diagnosis from bounding boxes to segmentation masks. Deferred
deliberately, not dropped — see [DECISIONS.md](DECISIONS.md) D-022 for the
measured gap, the schema options, and the reasoning.

**Why it is needed.** On a segmentation dataset the current engine reports thin
structures as failing no worse than solid ones, because their bounding boxes
are fine and only their outlines are poor. That failure mode is invisible to
box IoU. Any report covering a segmentation dataset should state this
limitation until the milestone lands.

Scope:

- Extract prediction outlines from model output — currently
  `Detection.polygon` is always `None`, and this is the substantive work.
- A mask IoU function, passed to the existing matcher as its similarity
  argument.
- A `mask_findings` table keyed on `finding_id`, plus a schema version bump.
  `runs`, `images` and `findings` stay untouched, so every query written
  against the published contract keeps working (D-020).

**Not blocked by architecture.** The matcher already takes the comparison as a
parameter (D-018), the diagnosis engine forwards it, and ground-truth outlines
are already parsed and persisted. This is an addition, not a rewrite.

Depends on: Milestone 3. Sequenced after the core reasoning modules so it does
not widen the surface each of them must handle.

---

## Milestone 8 — Additional detector support · **Planned**

Extend beyond the initial detector family. The architecture already isolates
detector-specific code behind the `Detection` type (see
[DECISIONS.md](DECISIONS.md) D-005); this milestone exercises that boundary.

Depends on: Milestone 3 at minimum.

---

## Blocked pending resources

Not milestones, but they gate progress:

Nothing is currently blocked. A model and dataset are both available, and every
implemented capability has been exercised against them.
