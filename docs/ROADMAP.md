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

## Milestone 3 — Error analysis · **Planned**

The first genuinely diagnostic capability, and the reason the inference layer
was shaped as it is.

Scope:

- Box IoU computation between predictions and ground truth.
- A matching strategy pairing predictions with labels.
- Classification of each outcome into: false positive, false negative, wrong
  class, poor localisation.
- Per-image and per-class failure summaries.
- Persisted failure records for later stages.

Open design questions to resolve **before** implementation:

1. When a prediction overlaps a label below the match threshold with the
   correct class, is that one poor-localisation failure, or a false positive
   *and* a false negative? This choice shapes every downstream count.
2. Greedy confidence-ordered matching, or optimal assignment? Greedy is simpler
   and conventional; optimal is more correct in crowded scenes.
3. Should match IoU be a single threshold or a sweep?

The annotation model already carries polygons (D-016), so mask-based analysis
is a later addition to this milestone rather than a redesign of it.

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

| Blocked item | Requires |
| --- | --- |
| Executing batch inference | A dataset |
| Executing metrics validation | A dataset and a model |
| All of Milestone 3 | A model and a labelled dataset |
