# Model Doctor

A Vision AI **failure diagnosis** platform. Standard tools tell you a model
scored `mAP50 = 0.72`. Model Doctor is being built to answer the next question:
*which* predictions were wrong, and *why*.

YOLO is the default detector and RF-DETR is supported as an optional second
family. Detector-specific code is isolated behind `app/inference.py` and
`app/detectors.py`, so every analysis stage runs unchanged on either.

## Status

The pipeline runs end to end: inference, diagnosis, Grad-CAM explanation, CLIP
feature extraction, root-cause attribution with base rates, failure grouping,
recommendations, and mask-level diagnosis — all against real trained models and
a real labelled dataset, explorable in the dashboard and served over HTTP.

Two models can be compared directly: their failure profiles, their disagreements
drawn over the images the predictions were made on, their mAP from one shared
evaluator, and their measured compute cost per device.

Schema version 9. The backend is complete.

**Grad-CAM currently targets YOLO only.** It is refused for RF-DETR rather than
pointed at an architecture it was not written for; every other stage supports
both families.

Mask-level diagnosis is built. On the reference model, `door` and `door_frame`
have near-identical mean box IoU (0.878, 0.877) and very different mean mask IoU
(0.797, 0.629) — a failure mode box-level analysis cannot see. See
[DECISIONS.md](docs/DECISIONS.md) D-036.

## Setup

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Check project health

Run this any time. It never crashes and tells you exactly what is missing:

```bash
./.venv/bin/python -m app.inference --check
```

## Expected resources

Model Doctor does not create models or datasets. Supply them here:

```
models/
  best.pt                  # any *.pt is accepted; newest wins if several

datasets/
  data.yaml                # YOLO descriptor — the source of class names
  test/
    images/frame_001.jpg
    labels/frame_001.txt   # same stem, images/ -> labels/
```

`data.yaml` must declare a `names:` field (list or dict form both work):

```yaml
path: .
train: train/images
val: val/images
test: test/images
nc: 2
names: ['crack', 'spall']
```

Label lines are `class_id x_center y_center width height`, with the four
geometry values **normalised to 0–1** relative to image size.

## Usage

```bash
# Single image
./.venv/bin/python -m app.inference --image path/to/image.jpg

# A whole split from data.yaml
./.venv/bin/python -m app.inference --split test

# The detector's own validator — YOLO only, and not the number to compare on
./.venv/bin/python -m app.inference --validate
```

`--validate` calls the model's built-in evaluation. It is a convenience for a
single YOLO model, **not** the path to compare two models: two libraries'
validators disagree on matching, NMS and score handling, so their numbers are
not comparable. Use `scripts/evaluate_run.py` for that (below).

Annotated images are written to `results/predictions/`.

## Diagnose failures

The point of the project — explaining *why* predictions fail:

```bash
./.venv/bin/python -m app.diagnosis --split test
```

Every prediction and every ground-truth annotation is classified into one of
five outcomes — correct, wrong class, poor localisation, false positive, false
negative — and reported per class, with the worst images ranked so you know
which ones to open.

**These failure counts are not mAP's.** A prediction that finds an object but
outlines it badly is reported as one *poor localisation*, not as a false
positive plus a false negative. That is deliberate: it names a cause instead of
describing one object as two unrelated errors. See
[DECISIONS.md](docs/DECISIONS.md) D-017.

## Group failures

Fixing one image at a time does not scale. Grouping asks which failures share a
cause, so you address a pattern instead:

```bash
./.venv/bin/python -m app.root_cause --run 1     # attribute causes first
./.venv/bin/python -m app.clustering --run 1     # then group by them
```

Each group is named by the conditions its members share — `edge_truncation`,
`blur + crowding` — so a group needs no interpreting. Failures that no factor
explains form an `unexplained` group rather than being dropped; those are the
ones worth opening first.

**This is deterministic grouping, not unsupervised clustering.** K-means over
the CLIP embeddings was implemented and measured first, and rejected on the
numbers — see [DECISIONS.md](docs/DECISIONS.md) D-030.

## Get suggested actions

```bash
./.venv/bin/python -m app.recommendations --run 1
```

Each group becomes an action, or an explicit statement that there is nothing to
suggest. Every recommendation carries its evidence and one of four statuses —
`replicated`, `provisional`, `conflicting`, `insufficient_evidence`. The last
two are refusals, and they are shown rather than hidden: a pattern two runs
disagree about gets "do not act yet", not a confident fix. See D-034 and D-035.

## Use a second detector

RF-DETR is an optional second family. YOLO stays the default and nothing below
is required to use it:

```bash
./.venv/bin/python -m pip install -r requirements-rfdetr.txt
MD_DETECTOR=rfdetr ./.venv/bin/python -m app.diagnosis --split test --save
```

The family of a *saved* run is read back from its own checkpoint, so a run
diagnosed with one detector is always re-opened with the same one.

## Measure accuracy and compute cost

Diagnosis explains failures; these two passes measure the model. They are
separate commands because they are separate kinds of measurement — mAP travels
between machines, latency does not.

```bash
./.venv/bin/python scripts/evaluate_run.py  --run-id 1
./.venv/bin/python scripts/benchmark_run.py --run-id 1 --device cpu
```

`evaluate_run.py` re-runs inference unthresholded and scores it with
`pycocotools` COCOeval — **one evaluator for every detector family**, box and
mask. mAP cannot be recovered from stored findings, which hold only detections
that already passed the run's confidence threshold and were already matched to
ground truth, so this pass re-runs inference rather than reusing them.

`benchmark_run.py` measures latency, throughput, memory and checkpoint size for
one model on one device, in its own process. `--device` is required: a latency
without the device that produced it cannot be compared with anything.

Both re-hash the checkpoint and refuse to run if it no longer matches the SHA
the run recorded. Results are stored per run and shown in the comparison.

## Compare two models

```bash
./.venv/bin/uvicorn app.api:app --port 8000        # API
cd web && npm run dev                              # dashboard
```

Open `/compare`. Pick two runs and the page reports which to ship and why:
precision, recall and F1 derived from stored counts; mAP from the shared
evaluator; the failure profile; root-cause lift with p-values; and measured
compute cost, compared only within a device both runs were benchmarked on.

Every figure is labelled **stored**, **derived**, or **not available** with the
reason — a metric the schema cannot supply is never rendered as zero.

Objects the two runs disagree about are drawn over the image the predictions
were made on: ground truth solid, predictions dashed, each run's own extra
detections on its own pane. Runs over different datasets, splits, or ground
truth are blocked rather than footnoted.

## Explore saved runs

Save a diagnosis run, then open it. There are two readers, both read-only and
both rendering nothing but the tables published in `docs/SCHEMA.md`:

```bash
./.venv/bin/python -m app.diagnosis --split test --save

# The web dashboard — overview, failures, root causes, clusters, comparison
./.venv/bin/uvicorn app.api:app --port 8000
cd web && npm install && npm run dev

# Or the original Streamlit explorer, which reads the SQLite file directly
./.venv/bin/streamlit run app/dashboard.py
```

The web dashboard reads through the HTTP API and is where comparison, overlays
and measured metrics live. The Streamlit explorer predates it, opens a database
file directly without needing the API, and is kept for that reason.

Both start at `db/model_doctor.db` unless pointed elsewhere — Streamlit via its
sidebar, the API via `MD_DB_PATH`. When image paths are unavailable on the
current machine, finding details remain visible and the missing image is
reported rather than failing.

## Running a demo from a cold start

A development convenience, not a production requirement — the API and dashboard
need none of this to run normally.

```bash
# 1. Point the API at the database you want to show
MD_DB_PATH="$PWD/db/columns_eval.db" MD_API_FILE_ROOTS="$PWD/datasets" \
  ./.venv/bin/uvicorn app.api:app --port 8000

# 2. If you switched databases since the last session, clear the dashboard's
#    fetch cache first — it persists on disk across restarts
rm -rf web/.next/cache

# 3. Start the dashboard
cd web && npm run dev
```

**Step 2 matters when switching databases.** Next.js caches API responses on
disk for the revalidate window, and that cache survives a dev-server restart —
so a dashboard pointed at a new database can keep showing the previous one's
runs until it is cleared. Nothing is wrong with the data; the browser is reading
a stale copy.

## Serve the schema over HTTP

For consumers that cannot open the SQLite file — a browser, most obviously:

```bash
./.venv/bin/uvicorn app.api:app --reload
```

Interactive documentation at `http://localhost:8000/docs`. The API is strictly
read-only and adds no analysis: every endpoint is a documented query answered by
the same readers the CLI uses. Files are addressed by id, never by path. See
[SCHEMA.md](docs/SCHEMA.md) section 7 and [DECISIONS.md](docs/DECISIONS.md)
D-037.

## Configuration

Every path and threshold lives in `config.py` and is overridable by environment
variable — no code edits needed to point at different data:

| Variable | Purpose |
| --- | --- |
| `MD_MODEL_PATH` | Explicit weights file |
| `MD_MODELS_DIR` | Directory scanned for `*.pt` |
| `MD_DATA_YAML` | Dataset descriptor location |
| `MD_PREDICTIONS_DIR` | Where annotated images are written |
| `MD_CONF` | Confidence threshold (default `0.25`) |
| `MD_IMGSZ` | Inference size — **must match what the model was trained at** |
| `MD_MATCH_IOU` | IoU at which a pair counts as correctly localised (`0.50`) |
| `MD_LOC_IOU` | Floor below which a pair is not a near miss (`0.10`) |
| `MD_DEVICE` | Force `cpu`, `mps`, or `cuda` |
| `MD_DETECTOR` | Detector family for a new run — `yolo` (default) or `rfdetr` |
| `MD_RFDETR_IMGSZ` | RF-DETR input resolution (must be a multiple of 12) |
| `MD_DB_PATH` | Database the CLI and API read and write |
| `MD_LOG_LEVEL` | `DEBUG` to see library internals |
| `MD_CORS_ORIGINS` | Browser origins the API accepts (default `localhost:3000`) |
| `MD_API_FILE_ROOTS` | Extra directories the API may read images from |

## Layout

```
config.py              Single source of truth for paths and constants
app/inference.py       Model loading, prediction, metrics, CLI
app/diagnosis.py       Failure classification and reporting, CLI
app/clustering.py      Failure grouping by root-cause signature, CLI
app/recommendations.py Suggested actions with evidence status, CLI
app/mask_diagnosis.py  Outline-level re-measurement of findings, CLI
app/detectors.py       Detector family dispatch; reads the family from a checkpoint
app/rfdetr_adapter.py  Optional RF-DETR family, behind the same Detector contract
app/evaluation.py      Shared COCO evaluator — one scorer for every family
app/api.py             Read-only HTTP projection of the schema
utils/masks.py         Outline overlap by rasterisation
utils/statistics.py    Lift and Fisher's exact test, for base-rate comparison
app/similarity.py      Nearest-neighbour retrieval over stored embeddings
utils/annotations.py   Generic annotation model (box + optional polygon)
utils/geometry.py      Box geometry and IoU
utils/matching.py      Generic one-to-one annotation matching
utils/resources.py     Resource discovery and verification
utils/dataset.py       data.yaml parsing and ground-truth label reading
utils/exceptions.py    Project exception hierarchy
utils/logging_utils.py Centralised logger setup
scripts/evaluate_run.py   Score a saved run with the shared evaluator, CLI
scripts/benchmark_run.py  Measure one run on one device, CLI
web/                   Next.js dashboard, including model comparison
tests/                 Test suite (run: ./.venv/bin/python -m pytest)
```

## Tests and lint

Both must pass with zero findings before any task is considered complete:

```bash
./.venv/bin/python -m pytest
```

```bash
./.venv/bin/python -m ruff check .
```

The dashboard has its own checks, run from `web/`:

```bash
npm test          # Node's built-in runner; no extra dependency
npx tsc --noEmit
npm run build
```

## Documentation

`docs/` is maintained alongside the code, not after it.

| Document | Purpose |
| --- | --- |
| [PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | **Source of truth** — current state and status |
| [VISION.md](docs/VISION.md) | Why the project exists |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the code is organised |
| [ROADMAP.md](docs/ROADMAP.md) | Milestones, with status |
| [DECISIONS.md](docs/DECISIONS.md) | Architectural decisions and trade-offs |
| [DEVELOPMENT_RULES.md](docs/DEVELOPMENT_RULES.md) | Process rules, quality gates |
| [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) | How code should be written |
| [CHANGELOG.md](docs/CHANGELOG.md) | What changed, per milestone |
