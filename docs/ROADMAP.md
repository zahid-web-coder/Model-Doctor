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
