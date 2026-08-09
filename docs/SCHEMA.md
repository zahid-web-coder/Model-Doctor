# Database Schema — the backend contract

**This document is the interface between the analysis backend and anything that
consumes it.** A dashboard, a report generator, or a notebook should be built
against this document alone. Reading the Python is not required, and nothing
here depends on it.

Schema version: **1** · Default location: `db/model_doctor.db` (SQLite)

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
        │                                            │  future milestones
        │                                            ├─▶ clusters
        │                                            ├─▶ root_causes
        └── reproducibility: model SHA,              └─▶ recommendations
            thresholds, split, image size                (not yet created)
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
not the one the model guessed. To show the wrong guess, look it up from the
prediction side in your own UI copy — the predicted class id is not stored
separately, by design: per-class statistics should charge a miss to the class
that was missed.

**`truth_polygon` is populated only for segmentation datasets.** Prediction
outlines are **not stored** — the pipeline does not extract masks from model
output, so such a column would be permanently null. When mask extraction is
implemented, it arrives as new columns or a new table, and this document is
versioned accordingly.

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

---

## 6. Stability guarantees

**`runs`, `images`, and `findings` will not change shape.** Later milestones add
tables, never columns to these:

| Milestone | Table | Keyed on |
| --- | --- | --- |
| Failure clustering | `clusters` | `finding_id` |
| Root-cause analysis | `root_causes` | `finding_id` |
| Recommendations | `recommendations` | `cluster_id` or `finding_id` |

**None of those tables exist yet.** Do not write queries against them.

A query written against this document today will keep working. If a breaking
change ever becomes unavoidable, `schema_info.version` is incremented and this
document is updated first.

```sql
SELECT version FROM schema_info;   -- currently 1
```

---

## 7. Producing data

```bash
python -m app.diagnosis --split test --save
python -m app.diagnosis --list-runs
python -m app.diagnosis --split test --save --db /custom/path.db
```

Foreign keys are enforced, so consumers may rely on referential integrity.
Deleting a run cascades to its images and findings.
