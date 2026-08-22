# Database Schema — the backend contract

**This document is the interface between the analysis backend and anything that
consumes it.** A dashboard, a report generator, or a notebook should be built
against this document alone. Reading the Python is not required, and nothing
here depends on it.

Schema version: **8** · Default location: `db/model_doctor.db` (SQLite)

Populated databases are **not** committed — image paths and filenames carry
dataset-specific identifiers. This schema is committed; the data is not.

---

## 1. What is stored

One **run** is one execution of the diagnosis engine over one dataset split. It
produces one row per **image**, and one row per **finding** — where a finding is
a single prediction *or* a single ground-truth annotation, classified into one
of five outcomes.

```
                    ┌────────────────────────────────┐
   model + dataset  │  inference → diagnosis engine  │
                    └───────────────┬────────────────┘
                                    │  --save
                                    ▼
   ┌──────────┐ 1      n ┌──────────┐ 1      n ┌────────────┐
   │   runs   ├──────────┤  images  ├──────────┤  findings  │
   └──────────┘          └──────────┘          └────────────┘
        │                                            │
        │                                            ├─▶ embeddings  (v2)
        │                                            ├─▶ heatmaps    (v3)
        │                                            ├─▶ root_causes  (v4)
        │                                            ├─▶ mask_findings (v8)
        │                                            │
        │    ┌────────────┐ 1      n ┌─────────────────┐
        ├────┤  clusters  ├──────────┤ cluster_members ├──▶ findings   (v5)
        │    └────────────┘          └─────────────────┘
        │
        ├─▶ factor_rates (v6)   per run, not per finding: each factor's rate
        │                       among failures against its rate among correct
        │
        └── reproducibility: model SHA,   clusters ──▶ recommendations (v7)
            thresholds, split, image size              one per group, per rule
```

`findings` also carries `run_id` directly, so run-scoped queries need no join
through `images`.

---

## 2. Tables

### `runs`

One row per diagnosis execution. Holds everything needed to reproduce it — a
finding without this context is an assertion with no provenance.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Run identifier |
| `created_at` | TEXT | no | ISO-8601 UTC, e.g. `2026-08-09T12:19:01+00:00` |
| `model_path` | TEXT | no | Weights file used |
| `model_sha256` | TEXT | no | SHA-256 of the weights |
| `dataset_yaml` | TEXT | no | Dataset descriptor used |
| `split` | TEXT | no | `train` \| `val` \| `test` |
| `confidence_threshold` | REAL | no | Minimum confidence at inference |
| `match_iou_threshold` | REAL | no | IoU at which a pair counts as correct |
| `localization_iou_floor` | REAL | no | Floor for the poor-localisation pass |
| `image_size` | INTEGER | no | Inference image size |

**Why `model_sha256` and not just the path.** Weight files get overwritten in
place. Without the hash, two runs that disagree look like a regression when
they may simply be different models. Compare hashes before comparing numbers.

### `images`

One row per image attempted, including images that failed.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Image identifier |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `path` | TEXT | no | Full path as diagnosed |
| `filename` | TEXT | no | Basename, for display |
| `width` | INTEGER | **yes** | Pixels; null if the image failed |
| `height` | INTEGER | **yes** | Pixels; null if the image failed |
| `prediction_count` | INTEGER | no | Predictions considered |
| `truth_count` | INTEGER | no | Ground-truth annotations considered |
| `error` | TEXT | **yes** | Non-null if the image could not be processed |

**`error` is the important column.** An image with no findings and `error IS
NULL` was processed and genuinely had nothing. An image with `error` set never
ran. Those must not be conflated in any count.

**All box coordinates are absolute pixels** relative to `width` × `height`.
Nothing is normalised. To draw a box you need the image row.

### `findings`

The core table. One row per prediction **or** per ground-truth annotation —
every one of both is present exactly once.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Finding identifier |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `image_id` | INTEGER FK → `images.id` | no | Owning image |
| `outcome` | TEXT | no | See §3 |
| `class_id` | INTEGER | yes | Zero-based class index |
| `class_name` | TEXT | no | Human-readable class |
| `confidence` | REAL | **yes** | Model certainty; null when no prediction |
| `iou` | REAL | **yes** | Overlap; null when unpaired |
| `pred_x1`…`pred_y2` | REAL | **yes** | Prediction box; null when no prediction |
| `truth_x1`…`truth_y2` | REAL | **yes** | Ground-truth box; null when no truth |
| `truth_polygon` | TEXT | **yes** | JSON `[[x,y],…]` in pixels, or null |

**`class_name` is attributed to ground truth when one exists.** For a
`wrong_class` finding, `class_name` is the class that *should* have been found,
not the one the model guessed.

> **The predicted class is not stored anywhere, and cannot be recovered.**
> This is deliberate — per-class statistics should charge a miss to the class
> that was missed (D-028). For a `wrong_class` finding, display something like
> "predicted class not recorded". Do **not** infer it by elimination from the
> class list: that happens to work while there are exactly two classes and will
> silently produce wrong labels the moment a third is added.

What each outcome can answer, for a UI showing predicted against actual:

| Outcome | Predicted | Actual |
| --- | --- | --- |
| `correct` | `class_name` | `class_name` |
| `false_positive` | `class_name` — **is** the prediction | nothing was there |
| `false_negative` | nothing was predicted | `class_name` |
| `poor_localization` | `class_name` | `class_name`, with `iou` as the story |
| `wrong_class` | **not recorded** | `class_name` |

Four of the five are fully answerable from `findings` alone.

**`truth_polygon` is populated only for segmentation datasets.** Predicted
outlines are not stored here — they live in `mask_findings.pred_polygon`, added
at version 8, so this table kept its shape exactly as D-020 promised.

### `embeddings` — added in schema version 2

One vector per failed region, produced by an image encoder. This is what the
failure-grouping milestone will cluster.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Embedding identifier |
| `finding_id` | INTEGER FK → `findings.id` | no | The finding this describes |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run, for direct filtering |
| `model_name` | TEXT | no | Encoder identifier, e.g. `ViT-B-32/laion2b_s34b_b79k` |
| `dimensions` | INTEGER | no | Vector length |
| `vector` | BLOB | no | Raw **float32** bytes, `dimensions` of them |

Unique on `(finding_id, model_name)` — re-running the extractor replaces a
vector rather than adding a second. Embeddings are derived data, so recomputing
one is a correction, unlike a run, which is an observation.

**Reading a vector.** It is packed float32, not JSON:

```python
import array
values = array.array("f"); values.frombytes(row["vector"])
# or: numpy.frombuffer(row["vector"], dtype numpy.float32)
```

**Vectors are L2-normalised**, so cosine similarity is a plain dot product.

**Never mix encoders.** Vectors from different `model_name` values are not
comparable. Always filter by one when clustering.

**Not every failure has one.** Regions smaller than 8 px in either dimension
are skipped, as are findings whose image could not be read. Use a `LEFT JOIN`
if you need all failures regardless.

```sql
SELECT f.id, f.outcome, f.class_name, e.vector
FROM findings f
LEFT JOIN embeddings e ON e.finding_id = f.id AND e.model_name = ?
WHERE f.run_id = ? AND f.outcome != 'correct';
```

### `heatmaps` — added in schema version 3

One Grad-CAM image per explained finding, showing which regions drove the
model's response there.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Heatmap identifier |
| `finding_id` | INTEGER FK → `findings.id` | no | The finding this explains |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run, for direct filtering |
| `path` | TEXT | no | Image file on disk, ready to display |
| `method` | TEXT | no | How it was produced, currently `grad-cam` |
| `target_layers` | TEXT | no | Adapter identifier, e.g. `yolo26-seg:16,19,22` |

Unique on `(finding_id, method)` — regenerating replaces rather than duplicates.

**The image is already an overlay** at the original image's exact dimensions,
so it can be shown directly and lines up with the box coordinates in
`findings`. No further processing is needed.

**Not every finding has one.** Findings with no usable geometry, or whose image
could not be read, are skipped. Use a `LEFT JOIN`:

```sql
SELECT f.id, f.outcome, f.class_name, i.path AS image, h.path AS heatmap
FROM findings f
JOIN images i ON i.id = f.image_id
LEFT JOIN heatmaps h ON h.finding_id = f.id AND h.method = 'grad-cam'
WHERE f.run_id = ? AND f.outcome != 'correct';
```

**Reading a heatmap.** A sharply localised hotspot means the model responded
strongly to that region. A diffuse map on a false negative is meaningful rather
than broken — it is the visual signature of the model not attending there.

### `root_causes` — added in schema version 4

Conditions attributed to a failure: the region was dark or blurred, the object
was small or cut off by the frame, it sat among crowded neighbours, its class is
under-represented, or that class is repeatedly misidentified.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Attribution identifier |
| `finding_id` | INTEGER FK → `findings.id` | no | The failure this concerns |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run, for direct filtering |
| `factor` | TEXT | no | One of the identifiers below |
| `score` | REAL | no | Severity in `[0, 1]` |
| `evidence` | TEXT | no | The measurement, human-readable |

Unique on `(finding_id, factor)`. **A finding may have several factors** — most
do; they are not mutually exclusive.

#### Factor identifiers

| Factor | Meaning | Example evidence |
| --- | --- | --- |
| `blur` | Little high-frequency detail in the region | `Laplacian variance 5.1 < 100` |
| `low_light` | Region underexposed | `mean luminance 23.0/255 < 60` |
| `small_object` | Object is small **relative to this run's own objects** (25th percentile) | `4.2% of image < 23.6%` |
| `thin_structure` | Long, narrow object — aspect ratio in this run's top quartile | `6.3:1 tall, above 4.1:1` |
| `edge_truncation` | Object cut off by the frame | `touches left, top, bottom` |
| `crowding` | Heavy overlap with neighbouring annotations | `1 neighbour(s) overlapping, max IoU 0.85` |
| `class_imbalance` | Class under-represented in the run | `'handle' has 3 of 210 instances` |
| `recurring_misclassification` | Class repeatedly named wrongly | `'door_frame' misidentified 11 time(s)` |

**Two factors are calibrated per run, not fixed.** `small_object` and
`thin_structure` derive their thresholds from the run's own distribution of
object sizes and shapes, so "small" means small *for this dataset*. Their
evidence strings state the threshold actually used. A fixed constant does not
transfer: the previous COCO-derived value fired on 2 of 278 findings here. See
DECISIONS D-032, and read their `factor_rates` row before drawing conclusions.

**`score` is comparable within a factor, not across factors.** A blur score of
0.9 and a crowding score of 0.9 do not mean the same thing. Rank within a
factor; do not sum across them.

**These are correlations, not proofs.** A blurred region the model missed may
have been missed because of the blur or for an unrelated reason that co-occurs
with it. Present them as hypotheses to check, not as causes established.

**Not every failure has one.** Some have no measurable condition attached. Use
a `LEFT JOIN` when you need all failures regardless.

```sql
SELECT f.id, f.outcome, f.class_name, rc.factor, rc.score, rc.evidence
FROM findings f
LEFT JOIN root_causes rc ON rc.finding_id = f.id
WHERE f.run_id = ? AND f.outcome != 'correct'
ORDER BY rc.score DESC;
```

These factors are also what **failure groups** are built from — see `clusters`
below. That was the plan recorded in D-026, and it held: grouping needed no
change to this table.

**One documented limitation.** The roadmap asks for directed confusion pairs
("A mistaken for B"). That needs the *predicted* class, which `findings` does
not store — by design, so per-class statistics charge a miss to the class that
was missed. `recurring_misclassification` therefore reports which classes are
repeatedly misidentified, without the direction. See DECISIONS D-028.

---

### `factor_rates` — added in schema version 6

**How often each factor appears among failures, and among correct findings.**

> **Never show a factor count without its base rate.** `edge_truncation`
> describes 71% of failures on the reference run — and 76% of correct
> detections. A count alone made it look like the leading cause of failure when
> it distinguishes nothing. See DECISIONS D-031.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Row identifier |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `factor` | TEXT | no | One of the per-finding factor identifiers |
| `failure_count` | INTEGER | no | Failures carrying this factor |
| `failure_total` | INTEGER | no | Failures the factor could be evaluated on |
| `correct_count` | INTEGER | no | Correct findings carrying it |
| `correct_total` | INTEGER | no | Correct findings it could be evaluated on |
| `lift` | REAL | **yes** | Failure rate ÷ correct rate |
| `p_value` | REAL | no | Two-sided Fisher's exact test |

Unique on `(run_id, factor)`.

#### Reading these numbers

- **`lift` near 1.0 means the factor explains nothing.** It is equally common
  among successes, so it does not distinguish the failures.
- **`lift` is NULL when undefined** — no correct finding carried the factor, so
  the ratio would be infinite rather than large. Render it as "n/a", never as a
  big number.
- **`p_value` ≥ 0.05 is no evidence of association**, in either direction. It
  does not mean the factor helps.
- **Several factors are tested per run.** A single p just under 0.05 is weaker
  than it appears; a Bonferroni-corrected threshold across five factors is 0.01.

#### Only per-finding factors appear here

`class_imbalance` and `recurring_misclassification` are run-level and defined in
terms of mistakes, so no control rate exists for them. They are absent from this
table by design, not by omission.

**Correct findings have no `root_causes` rows.** That table still covers
failures only. This one carries the comparison as an aggregate, so every query
written against `root_causes` returns exactly what it always did.

```sql
SELECT factor, failure_count, failure_total, correct_count, correct_total,
       lift, p_value
FROM factor_rates
WHERE run_id = ?
ORDER BY lift IS NULL, lift DESC;
```

**Factors that actually distinguish failures** — the ones worth acting on
```sql
SELECT factor, lift, p_value
FROM factor_rates
WHERE run_id = ? AND lift > 1.0 AND p_value < 0.05
ORDER BY lift DESC;
```

---

### `mask_findings` — added in schema version 8

**The same findings, re-measured at outline level.** Box IoU and mask IoU
disagree sharply on thin structures, and this table is where that disagreement
becomes visible.

> On the reference model, mean box IoU is 0.878 for `door` and 0.877 for
> `door_frame` — indistinguishable. Mean mask IoU is 0.797 and 0.629. Box-level
> diagnosis reports the thin class as healthy; it is not. See DECISIONS D-036.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Row identifier |
| `finding_id` | INTEGER FK → `findings.id` | no | The finding re-measured |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `mask_iou` | REAL | **yes** | Outline overlap, or NULL — see below |
| `mask_outcome` | TEXT | **yes** | Verdict, or NULL when unmeasured |
| `pred_polygon` | TEXT | **yes** | Predicted outline, JSON `[[x,y],…]` in pixels |

Unique on `(finding_id)`.

#### NULL means "not measured", never "scored zero"

This distinction is the whole point of the table:

| Situation | `mask_iou` |
| --- | --- |
| Outlines compared, share no pixels | `0.0` |
| False negative — no prediction, so no predicted outline | `NULL` |
| False positive — no ground truth to compare against | `NULL` |
| Detection model with no masks at all | `NULL` |

**Never treat NULL as zero.** Averaging it in would drag every score down with
data that was never measured. On the reference run 100 of 278 findings are
NULL, all of them unpaired findings — every *paired* finding was measured.

#### Verdicts

| Value | Meaning |
| --- | --- |
| `correct` | Outline overlap at or above the run's match IoU threshold |
| `poor_localization` | Above the localisation floor, below the match threshold |
| `no_overlap` | Below the localisation floor |
| `NULL` | Not measured |

The same two thresholds the box pass uses, so `correct` means the same strength
of agreement in both and the two verdicts are directly comparable.

#### The query that matters — box says correct, outline says otherwise

```sql
SELECT f.id, f.class_name, f.iou AS box_iou, m.mask_iou, m.mask_outcome
FROM mask_findings m
JOIN findings f ON f.id = m.finding_id
WHERE m.run_id = ?
  AND f.outcome = 'correct'
  AND m.mask_outcome IS NOT NULL
  AND m.mask_outcome != 'correct'
ORDER BY m.mask_iou;
```

Those are failures box-level diagnosis cannot see.

#### Box against outline, per class

```sql
SELECT f.class_name,
       COUNT(m.mask_iou)   AS pairs,
       AVG(f.iou)          AS mean_box_iou,
       AVG(m.mask_iou)     AS mean_mask_iou
FROM mask_findings m
JOIN findings f ON f.id = m.finding_id
WHERE m.run_id = ? AND m.mask_iou IS NOT NULL AND f.iou IS NOT NULL
GROUP BY f.class_name
ORDER BY mean_mask_iou;
```

A class whose outline score falls far below its box score is one that box-level
diagnosis reports as healthier than it is.

**Optional table.** A run has these rows only if the mask pass was run, and only
a segmentation model produces them at all. Absence means "not measured", not
"no problems".

---

### `clusters` and `cluster_members` — added in schema version 5

**Failure groups.** Each row in `clusters` is one group of failures that share
a cause; `cluster_members` says which failures belong to it.

> **Call these "Failure Groups" in the UI, not "clusters" and never
> "unsupervised clustering".** The default method assigns membership
> deterministically from attributed root causes — nothing is learned or fitted.
> The table names are `clusters`/`cluster_members` only so a genuinely
> unsupervised method can be added later under the same `method` column. See
> DECISIONS D-030, which records the k-means silhouette scores that ruled that
> approach out for now.

#### `clusters`

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Group identifier |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `method` | TEXT | no | How the grouping was produced |
| `label` | TEXT | no | The group's name — **this is also its explanation** |
| `size` | INTEGER | no | Number of members, denormalised for cheap ordering |

Unique on `(run_id, method, label)`.

#### `cluster_members`

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `cluster_id` | INTEGER FK → `clusters.id` | no | The group |
| `finding_id` | INTEGER FK → `findings.id` | no | A failure in it |

Primary key `(cluster_id, finding_id)`.

#### Method values

| Method | Meaning |
| --- | --- |
| `discriminating-signature` | Grouped by only those factors shown to occur more often in failures than in correct findings, on more than one split. **Show this one by default.** |
| `factor-signature` | Grouped by *every* attributed factor. The complete descriptive record. |

**Both are always written**, over the same failures. They are two partitions of
one set, not two halves of it — each independently accounts for every failure.

Prefer `discriminating-signature` in any interface. On the reference runs the
full signature produced 35 and 43 groups with a median size of 2, built largely
from a factor whose lift is 1.00x — comprehensive, and meaningless. The
discriminating partition produces the same four groups on both splits. See
DECISIONS D-033.

#### What the label means

For `factor-signature`, the label is the failure's factors, sorted and joined
with `" + "`. **No lookup is needed to name a group — the label is the name.**

| Example label | Meaning |
| --- | --- |
| `small_object` | Only that factor was attributed |
| `small_object + thin_structure` | Both were attributed to every member |
| `unexplained` | No qualifying factor was attributed |

Under `discriminating-signature`, `unexplained` means *no factor from the
qualifying set applied* — the finding may still carry other factors in
`root_causes`. Roughly half of failures land here on the reference runs, and
that is reported rather than hidden: those failures have no known cause yet.

`unexplained` is a real group, not a null. Do not filter it out by default: it
is the set of failures no current detector accounts for, which makes it the
most interesting group on the page, not the least.

#### Guarantees you can rely on

- **Every failure is in exactly one group per method.** Group sizes sum to the
  run's failure count. If they do not, something is wrong — say so rather than
  rendering it.
- **Correct findings are never grouped.** Only `outcome != 'correct'`.
- **`size` always equals the member count.** A test enforces it.
- **Regrouping replaces.** Re-running deletes that run's groups *for that
  method* and writes fresh ones, so ids are not stable across runs of the
  grouping pass. Groups from a different `method` are untouched.
- **A run may have no groups at all** — the grouping pass is optional and may
  not have been run. Treat an empty result as "not grouped yet", not "no
  failures". `clusters` is an optional table: a database written before version
  5 does not have it, so check before querying.

---

### `recommendations` — added in schema version 7

**Suggested actions, each traceable to the evidence that produced it.**

> **Show every row, including the ones that recommend nothing.** Two of the four
> statuses are deliberate refusals. A tool that only displays its confident
> answers is the tool this project was built to replace. See DECISIONS D-035.

| Column | Type | Null | Meaning |
| --- | --- | --- | --- |
| `id` | INTEGER PK | no | Recommendation identifier |
| `run_id` | INTEGER FK → `runs.id` | no | Owning run |
| `cluster_id` | INTEGER FK → `clusters.id` | no | The failure group this concerns |
| `rule` | TEXT | no | Which rule produced it — see below |
| `action` | TEXT | no | What to do, one sentence, display as-is |
| `rationale` | TEXT | no | The evidence, display as-is |
| `status` | TEXT | no | One of the four below |
| `actionable` | INTEGER | no | `1` or `0`. Derived from `status`, stored for ordering |
| `affected` | INTEGER | no | Failures this concerns |
| `priority` | REAL | no | Ordering term — the failure count, nothing more |

Unique on `(cluster_id, rule)`. Re-running replaces rather than appends.

#### Status values

| Status | Meaning | `actionable` |
| --- | --- | --- |
| `replicated` | Holds across runs of the same model | 1 |
| `provisional` | Seen once; act with care | 1 |
| `conflicting` | Runs disagree about this pattern — **do not act** | 0 |
| `insufficient_evidence` | Nothing can be concluded, and why is in `rationale` | 0 |

#### Rule identifiers

| Rule | Produces |
| --- | --- |
| `recall_on_factor` | The model misses objects under this condition |
| `precision_on_factor` | The model invents objects under this condition |
| `unexplained_backlog` | No measured condition accounts for these failures |
| `unstable_pattern` | Runs disagree; collect another run |
| `group_too_small` | Too few failures, or no qualifying factor, or a mixed group |

#### Display order — fixed, do not re-sort

```
ORDER BY actionable DESC, priority DESC, id
```

Actionable advice first, then by how many failures it addresses, then by id so
repeated reads agree. `priority` is deliberately **not** a composite score; it
carries the failure count and nothing else, so any ordering you see can be
explained.

#### Tracing a recommendation back to its evidence

Nothing is asserted without a path back to the measurement:

```
recommendation → cluster → cluster_members → findings
                    └─ label names the factors → factor_rates (lift, p)
```

```sql
-- The group, its members, and the measured rates behind one recommendation
SELECT r.action, r.status, r.rationale, c.label, c.size
FROM recommendations r
JOIN clusters c ON c.id = r.cluster_id
WHERE r.id = ?;

SELECT f.id, f.outcome, f.class_name, i.filename
FROM recommendations r
JOIN cluster_members cm ON cm.cluster_id = r.cluster_id
JOIN findings f ON f.id = cm.finding_id
JOIN images i ON i.id = f.image_id
WHERE r.id = ?;
```

#### The list, ready to render

```sql
SELECT r.id, r.action, r.rationale, r.status, r.actionable,
       r.affected, r.priority, c.label AS group_label
FROM recommendations r
JOIN clusters c ON c.id = r.cluster_id
WHERE r.run_id = ?
ORDER BY r.actionable DESC, r.priority DESC, r.id;
```

**A run may have no recommendations at all** — the pass is optional. That means
"not generated yet", not "no problems". Offer the command:
`python -m app.recommendations --run <id>`.

---

## 3. Outcome values

`findings.outcome` is one of exactly five strings.

| Value | Meaning | `pred_*` | `truth_*` | `iou` | `confidence` |
| --- | --- | :---: | :---: | :---: | :---: |
| `correct` | Matched, right class, well localised | set | set | set | set |
| `wrong_class` | Matched an object, named it wrong | set | set | set | set |
| `poor_localization` | Found the object, outlined it badly | set | set | set | set |
| `false_positive` | Predicted where nothing is | set | **null** | **null** | set |
| `false_negative` | Missed a real object | **null** | set | **null** | **null** |

Anything not `correct` is a failure. There is no separate "is_failure" column —
use `outcome != 'correct'`.

**Accounting guarantee.** For any run:

```
predictions examined  = count(correct + wrong_class + poor_localization + false_positive)
ground truths         = count(correct + wrong_class + poor_localization + false_negative)
```

Both hold exactly. If a query breaks them, the query is wrong.

---

## 4. ⚠️ These counts are NOT COCO mAP

**Do not label these numbers as mAP, and do not compare them to mAP output.**
They answer a different question and will not agree.

Two deliberate differences:

1. **A near miss is one failure, not two.** A prediction that lands on a real
   object but overlaps it by less than `match_iou_threshold` is recorded as one
   `poor_localization`. Evaluation metrics record it as a false positive *and* a
   false negative. mAP is right to do so — it must penalise a bad box twice or a
   model could game it. But "found it, outlined it badly" explains what
   happened; "one spurious detection and one miss" describes a single object as
   two unrelated failures. This project exists to explain.

   **Consequence: false-positive and false-negative counts here are lower than
   mAP implies.**

2. **A single confidence threshold.** Diagnosis runs at
   `runs.confidence_threshold` (default 0.25). mAP integrates across all
   thresholds. Fewer predictions are considered here.

Model Doctor's own mAP figures come from the validation command, separately.
If a dashboard shows both, label them distinctly and do not compute one from
the other.

---

## 5. Example queries

**List runs, newest first**
```sql
SELECT id, created_at, split, image_size, substr(model_sha256, 1, 8) AS model
FROM runs ORDER BY id DESC;
```

**Outcome breakdown for a run**
```sql
SELECT outcome, COUNT(*) AS n
FROM findings WHERE run_id = ?
GROUP BY outcome ORDER BY n DESC;
```

**Per-class statistics**
```sql
SELECT class_name,
       SUM(outcome = 'correct')            AS correct,
       SUM(outcome = 'wrong_class')        AS wrong_class,
       SUM(outcome = 'poor_localization')  AS poor_localization,
       SUM(outcome = 'false_positive')     AS false_positive,
       SUM(outcome = 'false_negative')     AS false_negative,
       ROUND(AVG(iou), 4)                  AS mean_iou
FROM findings WHERE run_id = ?
GROUP BY class_name ORDER BY class_name;
```

**Worst images by failure count**
```sql
SELECT i.id, i.filename, i.path, COUNT(*) AS failures
FROM findings f JOIN images i ON i.id = f.image_id
WHERE f.run_id = ? AND f.outcome != 'correct'
GROUP BY i.id ORDER BY failures DESC LIMIT 20;
```

**Everything needed to render one image's boxes**
```sql
SELECT i.path, i.width, i.height,
       f.outcome, f.class_name, f.confidence, f.iou,
       f.pred_x1,  f.pred_y1,  f.pred_x2,  f.pred_y2,
       f.truth_x1, f.truth_y1, f.truth_x2, f.truth_y2,
       f.truth_polygon
FROM findings f JOIN images i ON i.id = f.image_id
WHERE f.image_id = ?;
```

**Images that fail only by missing objects** — pure recall failures
```sql
SELECT i.filename, COUNT(*) AS missed
FROM findings f JOIN images i ON i.id = f.image_id
WHERE f.run_id = ?
GROUP BY i.id
HAVING SUM(f.outcome != 'false_negative' AND f.outcome != 'correct') = 0
   AND SUM(f.outcome = 'false_negative') > 0
ORDER BY missed DESC;
```

**Low-confidence false positives** — the model inventing things it is unsure of
```sql
SELECT class_name, COUNT(*) AS n, ROUND(AVG(confidence), 3) AS mean_conf
FROM findings
WHERE run_id = ? AND outcome = 'false_positive' AND confidence < 0.5
GROUP BY class_name;
```

**Images that errored**
```sql
SELECT filename, error FROM images WHERE run_id = ? AND error IS NOT NULL;
```

**Compare two runs** — verify the model first
```sql
SELECT id, split, model_sha256 FROM runs WHERE id IN (?, ?);

SELECT run_id, outcome, COUNT(*) AS n
FROM findings WHERE run_id IN (?, ?)
GROUP BY run_id, outcome;
```

**Failure groups, largest first** — the group list for a run
```sql
SELECT id, label, size
FROM clusters
WHERE run_id = ? AND method = 'discriminating-signature'
ORDER BY size DESC, label;
```

Swap the method to `'factor-signature'` for the full descriptive partition.

**The failures inside one group**
```sql
SELECT f.id, f.outcome, f.class_name, f.confidence, f.iou, i.filename, i.path
FROM cluster_members cm
JOIN findings f ON f.id = cm.finding_id
JOIN images i ON i.id = f.image_id
WHERE cm.cluster_id = ?
ORDER BY f.id;
```

**Group composition by outcome** — what kind of failure each group contains
```sql
SELECT c.label, f.outcome, COUNT(*) AS n
FROM clusters c
JOIN cluster_members cm ON cm.cluster_id = c.id
JOIN findings f ON f.id = cm.finding_id
WHERE c.run_id = ? AND c.method = 'factor-signature'
GROUP BY c.label, f.outcome
ORDER BY c.label, n DESC;
```

**Group composition by class** — which class each group hurts most
```sql
SELECT c.label, f.class_name, COUNT(*) AS n
FROM clusters c
JOIN cluster_members cm ON cm.cluster_id = c.id
JOIN findings f ON f.id = cm.finding_id
WHERE c.run_id = ? AND c.method = 'factor-signature'
GROUP BY c.label, f.class_name
ORDER BY n DESC;
```

**Which group a given finding is in** — for the image explorer
```sql
SELECT c.label
FROM cluster_members cm
JOIN clusters c ON c.id = cm.cluster_id
WHERE cm.finding_id = ? AND c.method = 'factor-signature';
```

**Check the arithmetic** — grouped total must equal the failure count
```sql
SELECT
  (SELECT COALESCE(SUM(size), 0) FROM clusters
    WHERE run_id = ? AND method = 'factor-signature')          AS grouped,
  (SELECT COUNT(*) FROM findings
    WHERE run_id = ? AND outcome != 'correct')                 AS failures;
```

---

## 6. Stability guarantees

**`runs`, `images`, and `findings` will not change shape.** Later milestones add
tables, never columns to these:

| Milestone | Table | Keyed on | Status |
| --- | --- | --- | --- |
| Feature extraction | `embeddings` | `finding_id` | **Exists (v2)** |
| Grad-CAM explanation | `heatmaps` | `finding_id` | **Exists (v3)** |
| Root-cause analysis | `root_causes` | `finding_id` | **Exists (v4)** |
| Failure grouping | `clusters`, `cluster_members` | `run_id`, `finding_id` | **Exists (v5)** |
| Factor base rates | `factor_rates` | `run_id` | **Exists (v6)** |
| Recommendations | `recommendations` | `cluster_id` | **Exists (v7)** |
| Mask-level diagnosis | `mask_findings` | `finding_id` | **Exists (v8)** |

`embeddings` arriving in version 2 is this guarantee working as intended: a new
table was added and **`runs`, `images` and `findings` did not change**. Every
query written against version 1 still returns exactly the same rows. Versions
3 through 8 held the same line.

Opening an older database upgrades it in place — the new tables are created and
existing data is untouched.

**Check before querying the optional tables.** `embeddings`, `heatmaps`,
`root_causes`, `clusters`, `cluster_members`, `factor_rates` and
`recommendations` and `mask_findings` each arrived after version 1,
so a database saved by an earlier version will not have them. Only `runs`,
`images` and `findings` are guaranteed. A missing table should degrade the one
surface that needs it, never the whole page:

```sql
SELECT name FROM sqlite_master WHERE type = 'table';
```

A query written against this document today will keep working. If a breaking
change ever becomes unavoidable, `schema_info.version` is incremented and this
document is updated first.

```sql
SELECT version FROM schema_info;   -- currently 8
```

---

## 7. Reading the schema over HTTP

A read-only API serves every table in this document as JSON, for consumers that
cannot open the SQLite file — a browser, most obviously.

```bash
./.venv/bin/uvicorn app.api:app --reload
# interactive docs at http://localhost:8000/docs
```

| Endpoint | Returns |
| --- | --- |
| `GET /health` | Database presence and schema version |
| `GET /runs` | Every run, newest first |
| `GET /runs/{id}` | One run's provenance |
| `GET /runs/{id}/outcomes` | Finding counts by outcome |
| `GET /runs/{id}/images` | Every image attempted, including errored and empty ones |
| `GET /runs/{id}/findings?limit=&offset=` | Paginated findings with a total |
| `GET /runs/{id}/groups?method=` | Failure groups, discriminating by default |
| `GET /groups/{id}/members` | The findings in one group |
| `GET /runs/{id}/root-causes?factor=` | Attributed conditions |
| `GET /runs/{id}/factor-rates` | Lift and significance per factor |
| `GET /runs/{id}/recommendations` | Suggested actions, documented order |
| `GET /runs/{id}/mask-findings?disagreements=` | Outline-level results |
| `GET /runs/{id}/findings/{id}/neighbours?limit=` | Visually similar failures |
| `GET /images/{id}` | Source image bytes |
| `GET /findings/{id}/heatmap?method=` | Grad-CAM overlay bytes |

**Image totals come from `/runs/{id}/images`, never from the findings list.**
An image processed cleanly with nothing in it has no finding, and an errored
image is indistinguishable from one that simply found nothing. Deriving counts
from findings loses both.

**It adds no analysis.** Every endpoint is a query from this document, answered
by the same readers the CLI uses. Rows are returned as documented — the API does
not reshape them for any particular screen.

**Status codes.** `404` for an unknown run, finding or image, and for a file that
is no longer on disk. `403` for a stored path outside the directories the service
may read. `503` when the database itself is absent, carrying the command that
creates one. An optional table that does not exist yields `[]`, not an error.

**Files are addressed by id only.** No endpoint accepts a filesystem path.

**Operational note.** Start the API with the same `MD_DATASETS_DIR` the runs were
diagnosed with, or their images resolve outside the allowed roots and return
`403`. Use `MD_API_FILE_ROOTS` when images live in several places. CORS defaults
to `localhost:3000` and is set with `MD_CORS_ORIGINS`.

See DECISIONS D-037.

---

## 8. Producing data

```bash
python -m app.diagnosis --split test --save     # runs, images, findings
python -m app.diagnosis --list-runs
python -m app.features --run 1                  # embeddings (downloads CLIP once)
python -m app.explainability --run 1 --imgsz 672  # heatmaps
python -m app.root_cause --run 1                  # root_causes + factor_rates
python -m app.clustering --run 1                  # clusters, cluster_members
python -m app.recommendations --run 1             # recommendations
python -m app.mask_diagnosis --run 1              # mask_findings (segmentation only)
```

Grouping reads `root_causes`, so run `app.root_cause` first. Running it before
attribution is not an error — every failure simply lands in `unexplained`.

Foreign keys are enforced, so consumers may rely on referential integrity.
Deleting a run cascades to its images, findings, and groups.
