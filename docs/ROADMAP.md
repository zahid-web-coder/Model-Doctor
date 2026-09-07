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

## Milestone 4 — Visual explanation · **Completed**

Make individual failures visually interpretable.

Delivered in two parts: the Grad-CAM backend (Week 3, schema version 3) and the
dashboard views that render its output. Method selection resolved to Grad-CAM
against raw class logits — see [DECISIONS.md](DECISIONS.md) D-024.

Depends on: Milestone 3.

---

## Milestone 5 — Failure grouping · **Completed**

Group failures sharing a probable cause, so an engineer addresses patterns
rather than individual images.

Delivered as **deterministic grouping by root-cause signature**, not
unsupervised clustering. K-means over the CLIP embeddings was implemented and
measured first; it scored a silhouette of 0.13–0.23 across every *k* from 2 to
8, below the 0.25 threshold for substantial structure, and its clusters largely
re-encoded class and outcome. Factor signatures give 22 named groups over the
136-image run, 74% of failures in groups of five or more. See
[DECISIONS.md](DECISIONS.md) D-030 for the numbers and the trade-offs accepted.

| Item | Status |
| --- | --- |
| `clusters` / `cluster_members` tables, schema version 5 | Completed |
| Factor-signature grouping, `app/clustering.py` | Completed |
| `unexplained` group for failures no factor accounts for | Completed |
| Nearest-neighbour retrieval, `app/similarity.py` | Completed |
| `method` column so another grouping method needs no schema change | Completed |
| Dashboard failure-group view | Completed |

**Naming.** Every user-facing surface says *failure group*. The tables and
module keep `cluster` names so a genuinely unsupervised method can be added
later under the same `method` seam.

Depends on: Milestones 3, 4.

---

## Milestone 5.5 — Factor base rates · **Completed**

Report every factor's rate among failures against its rate among correct
findings, so a count can be read as evidence rather than as a cause.

Added because the root-cause engine had only ever seen failures. It reported
`edge_truncation` on 71% of them, which reads as the leading cause until the
control group shows 76% of *correct* detections carry it too. The largest
failure group in the product rested on a factor with no demonstrated
association. See [DECISIONS.md](DECISIONS.md) D-031.

| Item | Status |
| --- | --- |
| `factor_rates` table, schema version 6 | Completed |
| `utils/statistics.py` — lift and Fisher's exact test | Completed |
| `measure_factor_rates()` over both groups in one pass | Completed |
| Crowding neighbours fix (49 -> 73 attributions) | Completed |
| Dashboard base-rate column | Completed |

Depends on: Milestone 3, Week 7.

---

## Milestone 5.6 — Factors that discriminate · **Completed**

D-031 found that no factor separated failures from successes. This established
that the fault lay with the factors, not the dataset.

`small_object` used the COCO convention of 0.12% of image area and fired on 2
of 278 findings on one split and none of 496 on another. Recalibrated against
the run's own size distribution it became the strongest signal in the system.
A new `thin_structure` factor names the weakness the metrics had shown since
training — `door_frame` mask mAP50-95 of 0.246 against `door` at 0.599.

| Factor | test | val |
| --- | --- | --- |
| `small_object` (calibrated) | 2.77x, p < 0.001 | 3.45x, p < 0.001 |
| `thin_structure` (new) | 2.14x, p < 0.001 | 2.32x, p < 0.001 |

| Item | Status |
| --- | --- |
| `thin_structure` factor | Completed |
| `small_object` calibrated from the run's distribution | Completed |
| `calibrate_finding_factors()`, `percentile()` | Completed |
| Replicated across two splits | Completed |
| Regroup on discriminating factors only | Deferred to Milestone 6 (D-032) |

Depends on: Milestone 5.5.

---

## Milestone 5.7 — Discriminating failure groups · **Completed**

Partition failures twice: `factor-signature` over every attributed factor, and
`discriminating-signature` over only those that replicate. Consumers show the
second by default.

Adding two genuinely discriminating factors in 5.6 fragmented the grouping to 35
and 43 groups of median size 2, because grouping on every factor means grouping
on factors that carry no information. The restricted set gives the same four
groups on both splits. See [DECISIONS.md](DECISIONS.md) D-033.

| Item | Status |
| --- | --- |
| `config.DISCRIMINATING_FACTORS`, configured not computed | Completed |
| Second grouping method under the existing `method` seam | Completed |
| `group_run(allowed_factors=...)` | Completed |
| Dashboard defaults to the discriminating partition | Completed |

Depends on: Milestones 5, 5.5, 5.6.

---

## Milestone 6 — Recommendations · **Completed**

Translate grouped failures into concrete suggested actions.

**Constrained by D-031.** Recommendations may not be built on raw factor
counts. Any advice must cite lift and significance, and must be capable of
saying "this condition is common but does not distinguish failures" — which on
the reference dataset is the honest verdict for `edge_truncation` and `blur`.

**Settled in 5.7** — the factor set is fixed once from pooled evidence
(D-033), so groups stay comparable between runs.

| Item | Status |
| --- | --- |
| `recommendations` table, schema version 7 | Completed |
| Five rules over failure groups | Completed |
| Four evidence statuses, refusals stored not omitted | Completed |
| Cross-run replication via `model_sha256` | Completed |
| Traceability: recommendation → group → findings → factor rates | Completed |
| Dashboard recommendations view | Completed |

On the reference run: two actionable recommendations, one investigation, and one
explicit refusal to act on a pattern two runs disagree about. See
[DECISIONS.md](DECISIONS.md) D-034 and D-035.

Depends on: Milestones 5, 5.5, 5.7.

---

## Milestone 7 — Developer dashboard · **Completed**

Present saved analysis through a developer-facing interface. The first slice
uses the stable Milestone 3 SQLite contract, so it can be useful before visual
explanation, clustering, and recommendations publish their own tables.

| Item | Status |
| --- | --- |
| Saved-run selector and provenance | Completed |
| Run metrics and outcome charts | Completed |
| Outcome filters and per-class statistics | Completed |
| Worst-image ranking and box/polygon overlays | Completed |
| Graceful missing database and image states | Completed |
| Grad-CAM / visual explanation views | Completed |
| Root-cause views, ranked by lift rather than count | Completed |
| Failure-group views, with drill-down to members | Completed |
| Recommendation views, refusals included | Completed |
| Outline-level comparison views | Completed |

The dashboard remains read-only. It must continue to query only documented
schema tables, including when later milestone tables are added.

---

## Milestone 8.5 — Mask-level diagnosis · **Completed**

Extend failure diagnosis from bounding boxes to segmentation masks. Deferred
deliberately, not dropped — see [DECISIONS.md](DECISIONS.md) D-022 for the
measured gap, the schema options, and the reasoning.

**Delivered.** Measured on the reference run, `door` and `door_frame` have mean
box IoU of 0.878 and 0.877 — indistinguishable — against mean mask IoU of 0.797
and 0.629. Fourteen findings are correct by box and not by outline. The
limitation this milestone was written to remove is closed; reports no longer
need to warn that thin-structure failures are invisible.

Scope:

| Item | Status |
| --- | --- |
| Predicted outlines extracted in `app/inference.py` | Completed |
| `utils/masks.py` — rasterised outline overlap | Completed |
| `mask_findings` table, schema version 8 | Completed |
| CLI (`python -m app.mask_diagnosis`) | Completed |
| Verified on two splits | Completed |
| Dashboard outline view | Completed |

Pairs are **not** re-matched on mask IoU: the box pairing is kept and the
outline measured on it, so "correct by box, not by outline" stays expressible.
See [DECISIONS.md](DECISIONS.md) D-036.

**Not blocked by architecture.** The matcher already takes the comparison as a
parameter (D-018), the diagnosis engine forwards it, and ground-truth outlines
are already parsed and persisted. This is an addition, not a rewrite.

Depends on: Milestone 3. Sequenced after the core reasoning modules so it does
not widen the surface each of them must handle.

---

## Milestone 8 — Additional detector support · **Completed**

Extend beyond the initial detector family. The architecture already isolates
detector-specific code behind the `Detection` type (see
[DECISIONS.md](DECISIONS.md) D-005); this milestone exercises that boundary.

**Delivered.** RF-DETR Nano Seg runs through the whole pipeline beside YOLO26.
The boundary held: `app/rfdetr_adapter.py` and `app/detectors.py` were added and
no analysis stage changed. A saved run's family is read back from its own
checkpoint, so no schema column was needed to record it.

Scope:

| Item | Status |
| --- | --- |
| `app/detectors.py` — family dispatch, family detected from the checkpoint | Completed |
| `app/rfdetr_adapter.py` — RF-DETR behind the `Detector` contract | Completed |
| Optional dependency (`requirements-rfdetr.txt`), YOLO still the default | Completed |
| Verified end to end on the same dataset as YOLO | Completed |
| Grad-CAM explicitly refused for RF-DETR rather than mis-applied | Completed |

**Explainability is the one stage that does not yet cover both families.**
Grad-CAM targets YOLO26's feature pyramid; pointing it at RF-DETR would produce
a picture with no defensible relationship to the prediction, so it refuses. That
gap is open work, not a closed decision.

---

## Milestone 9 — Model comparison · **Completed**

Answer the question the rest of the pipeline sets up: *I trained two models,
which do I ship and why?*

**Delivered.** `/compare` puts two runs side by side — verdict, metric table,
failure profile, root causes, and a side-by-side viewer — built entirely on
endpoints that already existed. No new API route and no schema change.

Scope:

| Item | Status |
| --- | --- |
| Alignment on `(filename, truth_box)`, never on `image_id` | Completed |
| False positives kept image-level, never paired to ground truth | Completed |
| Compatibility gate on dataset, split, and ground truth | Completed |
| Every figure labelled stored / derived / unavailable | Completed |
| Per-dimension verdict, with no composite score | Completed |

The verdict deliberately computes no single number: weighting accuracy against
compute belongs to a deployment, not to a page.

---

## Milestone 9.5 — Prediction overlays · **Completed**

Show the disagreement, not just count it.

**Delivered.** Each aligned ground-truth object is drawn on the image it was
annotated in, with each run's prediction over it — same region, same zoom on
both sides, so any on-screen difference is a difference between the models.

Scope:

| Item | Status |
| --- | --- |
| Boxes and masks drawn from stored geometry, in original-image pixels | Completed |
| Ground truth solid, predictions dashed — never confusable | Completed |
| Fit / zoom, mask and context-object toggles | Completed |
| Extra detections drawn only on the pane that produced them | Completed |

No backend change: every box and polygon was already stored and served.

---

## Milestone 10 — Evaluation and benchmark persistence · **Completed**

Replace "not available" with measurement, for accuracy and for compute.

**Delivered.** Schema version 9 adds two tables. `run_evaluations` holds COCO
mAP and AR for boxes and masks; `run_benchmarks` holds latency, throughput,
memory and checkpoint size per device. Both carry the settings that produced
them, because a figure without its protocol is not a measurement.

Scope:

| Item | Status |
| --- | --- |
| `app/evaluation.py` — one COCOeval for every detector family | Completed |
| Segmentation scored from the model's own multi-component raster | Completed |
| `scripts/evaluate_run.py`, `scripts/benchmark_run.py` | Completed |
| Schema version 9: `run_evaluations`, `run_benchmarks` | Completed |
| Checkpoint SHA re-verified before anything is recorded | Completed |
| Device recorded per benchmark; cross-device comparison refused | Completed |

**One evaluator, deliberately.** A library's own validator disagrees with
another's on matching, NMS and score handling, so the two numbers are not
comparable. `RFDetrDetector.validate` refuses for that reason and points here.

mAP is **not** derived from stored findings, which hold only detections that
already passed the run's threshold and were already matched — a fraction of the
precision/recall curve. The evaluation pass re-runs inference unthresholded.

---

## Milestone 11 — MCP tools for run comparison · **Completed (Phase 1)**

Hand the stored evidence to a reasoning model in a shape it can compare,
without re-deriving anything and without being able to change anything.

**Delivered.** Two read-only MCP tools over stdio. `list_runs` says what each
run is, what evidence it carries, and which runs share its checkpoint.
`get_analysis` returns the evidence for a set of runs in one comparison-friendly
structure: identity and configuration differences, outcome counts with derived
precision/recall and the rule behind them, COCO mAP/AR with its protocol, factor
lift against a control rate, failure groups with their outcome mix,
recommendations with their evidential status, deltas against a baseline, which
groups replicate, and what each run lacks with the command that produces it.

Scope:

| Item | Status |
| --- | --- |
| `app/comparison.py` — every comparison rule, once, pure | Completed |
| `app/recommendations.py` delegates qualification and replication to it | Completed |
| `storage.connect_read_only` shared by the HTTP API and the MCP server | Completed |
| `app/mcp_server.py` — `list_runs`, `get_analysis`, stdio only | Completed |
| Input validation: ids only, capped, unknown ids named, baseline checked | Completed |
| Sentinels normalised: COCOeval `-1` and undefined lift become `null` | Completed |
| Evidence gaps reported with the remedy command | Completed |
| Project `.mcp.json`; `requirements-mcp.txt` without torch or FastAPI | Completed |
| Run labels/notes, persisted family and classes, dataset hash, per-class outcomes, image-level flips | Phase 2 |
| `/compare` on the HTTP API from the same module; retire the browser's copy of the rules | Phase 2 |

**Read-only, by construction.** No tool deletes, mutates, trains or triggers
inference. The database is opened `mode=ro`, the transport opens no port, runs
are addressed by id and the database location comes from configuration alone.

---

## Milestone 12 — Image-level diagnosis · **Completed**

Answer "how many photographs did the model handle correctly, and what shape was
the mistake" beside the existing "which object failed and why".

**Delivered.** Schema version 11 adds `image_diagnoses` and `image_coverage`.
One verdict per image, measured on outlines, with the thresholds that produced
it and every pairwise coverage stored so it can be re-read at a different one.

Scope:

| Item | Status |
| --- | --- |
| `app/image_diagnosis.py` — the pass, with a pure classifier | Completed |
| Schema version 11: `image_diagnoses`, `image_coverage` | Completed |
| Merge and split detected on masks; no box fallback | Completed |
| `partial_coverage` for the band between the thresholds, both sides | Completed |
| `GET /runs/{id}/image-diagnoses`, `GET /runs/{id}/images/{id}/coverage` | Completed |
| MCP `get_analysis` gains an `images` block | Completed |
| Images screen, filterable by verdict and by object count | Completed |
| 29 tests, including that findings and outcomes are unchanged | Completed |

**Additive, and proven so.** Every finding, outcome count and evaluation across
all eight stored runs is byte-identical after the pass (D-017, D-040).

---

## Milestone 13 — Relationship-aware diagnosis · **Completed (foundation)**

The finding-level pass says which object failed; the image-level pass says what
shape a photograph's mistake took. Neither can say that *this* missed object is
covered by the prediction the matcher gave to *that* one. Relations are a third
lens over the same findings, and they change none of them.

Scope:

| Item | Status |
| --- | --- |
| Replication semantics: fingerprint, `reproduced` distinct from `replicated` | Completed |
| `app/relations.py` — the pass, with pure per-relation rules | Completed |
| Schema 13: `finding_relations`; schema 14: `coverage_floor` | Completed |
| `merge_candidate`, `box_mask_disagreement` — no new thresholds | Completed |
| `prediction_on_matched_object` — continuous, union measured as a bitwise OR | Completed |
| `duplicate_prediction` — provisional reading, not a stored relation | Completed |
| `GET /runs/{id}/relations`, `GET /runs/{id}/relations/summary` | Completed |
| MCP `get_analysis` gains a `relations` block; `list_runs` an availability flag | Completed |
| 124 tests across the pass, the surface and the replication rules | Completed |
| UI | Deliberately not started |

**Validated against the reference database, 2026-09-04.** The pass is
deterministic and idempotent — findings are visited in id order, rasterisation
is integer, and re-running replaces rather than appends. It produced **391 rows
across all eight runs**, and every one of the ten existing analysis tables —
`findings`, `mask_findings`, `root_causes`, `factor_rates`, `clusters`,
`cluster_members`, `recommendations`, `run_evaluations`, `image_diagnoses`,
`image_coverage` — is byte-identical afterwards. Outcome counts, mAP, factor
rates, clusters and image verdicts are unchanged (D-017, D-040).

**Nothing consumes it.** No recommendation rule reads a relation, no relation
reaches `root_causes` or `factor_rates`, and no cluster changes. Relations are
additive evidence reported beside the outcome counts and never subtracted from
them.

**No causal claim is made anywhere.** A relation is measured geometry.
`duplicate_prediction` is a provisional reading of the continuous
`prediction_on_matched_object` measurement, carried with both bounds that
produced it, and the payload states in as many words that no mechanism has been
identified — suppression settings, decoding, assignment order and a model
genuinely proposing two objects are all consistent with the measurement, and
none of them is measured.

**The evidence reaches a reasoning consumer.** A live MCP round trip
reproduced the whole run 4 versus run 5 investigation from the tool alone, with
no filesystem access: that raising inference size from 448 to 640 costs
precision (0.798 → 0.613) far more than recall (0.782 → 0.713), and that **30
of the 41 additional false positives are duplicate-labelled** while split images
go 9 → 41 and merged images stay at 5. Every conclusion matched the earlier
analysis that had required reading the database directly.

**Out-of-sample validation: run 12, the same checkpoint at 448px on `val`.**
Runs 4 and 5 were the only independent pair this checkpoint had, and they differ
in exactly the variable under suspicion, so nothing separated a resolution
effect from run-to-run variation. Run 12 fixes `image_size` and varies the data.
Duplicate-like behaviour held its share without being asked to — **55.2%, 62.9%
and 65.7% of measured false positives across the three independent
configurations**, overlapping binomial intervals, the same bimodal shape and the
same ~30% floor mass. `box_ok_mask_fails` fired for the first time, once. The
checkpoint is named `best__3_.pt` and was selected on this split, so its
accuracy figures there are optimistically biased and are not read as a
generalisation result; the *shape* of the duplicate distribution is not
something model selection optimised for, and that is what the experiment tested.

**Calibration: the original justification for `0.15` was falsified, and the rule
was kept for a different reason.** The threshold had been placed inside an empty
interval [0.0900, 0.2249]. Pooled over the three configurations that interval
holds two values and the widest gap collapses to 0.0648 — narrower than random
scatter typically produces, two thirds of simulated arrangements giving a wider
one. It is now documented as **an operational cutoff in a sparse middle region,
not an empirically established boundary**. It is frozen rather than re-chosen
because every value in [0.0905, 0.2245] selects the identical set, so no
available alternative is measurably different, and re-fitting on the data that
falsified the first choice would be fitting to the latest run.

**Visual validation of the `0.05` guard, on out-of-sample cases.** Sixteen
labelled rows covering under 25% of any one object were reviewed against their
images — nine from run 12, none previously inspected — with polygon integrity
and both stored measurements verified by independent rasterisation first. None
was diffuse or spurious; twelve read as an extra prediction on an object already
found; the case nearest the floor, at coverage 0.055, lies 97.7% inside the
accepted prediction. The guard is no longer resting on a single case, and
raising it is contraindicated.

**What the label covers is broader than its name.** Of the 106 labelled rows,
86% are redundant duplicates, 9% are complementary fragments on the same object,
and a small bounded class is annotation or matcher granularity — one row in 106
covers a truth the run recorded as missed. Splitting the label is a naming
decision with its own evidence to gather, not a threshold to move. Until then
`duplicate_prediction` stays **provisional**, gates nothing, and no mechanism is
claimed: suppression settings, decoding, assignment order and a model genuinely
proposing two objects all remain consistent with the measurement, and none of
them is measured.

---

## Blocked pending resources

Not milestones, but they gate progress:

A model and dataset are both available, and every implemented capability has
been exercised against them.

**One conclusion is gated rather than blocked.** Promoting
`duplicate_prediction` beyond provisional needs an observation this database
cannot supply: a run of the same checkpoint at a fixed inference size on data
it was not calibrated against, and ideally a second checkpoint on the staircase
dataset — all five staircase runs share `546ee2af`.
