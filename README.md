# Model Doctor

A Vision AI **failure diagnosis** platform. Standard tools tell you a model
scored `mAP50 = 0.72`. Model Doctor is being built to answer the next question:
*which* predictions were wrong, and *why*.

Version 1 targets YOLO. The detector is isolated behind `app/inference.py` so
additional architectures can be added later without touching analysis code.

## Status

The pipeline runs end to end: inference, diagnosis, Grad-CAM explanation, CLIP
feature extraction, root-cause attribution with base rates, and failure
grouping, and recommendations — all against a real trained model and a real
labelled dataset, and all explorable in the dashboard.

Schema version 7. The backend is complete through recommendations.

Mask IoU and segmentation analysis are **architecture-ready but not built**, a
documented limitation on a segmentation dataset — see
[DECISIONS.md](docs/DECISIONS.md) D-022.

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

# Metrics (precision / recall / mAP, box and mask)
./.venv/bin/python -m app.inference --validate
```

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

## Explore saved runs

Save a diagnosis run, then launch the read-only Streamlit explorer:

```bash
./.venv/bin/python -m app.diagnosis --split test --save
./.venv/bin/streamlit run app/dashboard.py
```

The dashboard starts at `db/model_doctor.db`; select another saved SQLite file
in the sidebar when needed. It renders only the tables published in
`docs/SCHEMA.md`, so it does not depend on the diagnosis implementation. When
image paths are unavailable on the current machine, the finding details remain
visible and the UI reports the missing image rather than failing.

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
| `MD_LOG_LEVEL` | `DEBUG` to see library internals |

## Layout

```
config.py              Single source of truth for paths and constants
app/inference.py       Model loading, prediction, metrics, CLI
app/diagnosis.py       Failure classification and reporting, CLI
app/clustering.py      Failure grouping by root-cause signature, CLI
app/recommendations.py Suggested actions with evidence status, CLI
utils/statistics.py    Lift and Fisher's exact test, for base-rate comparison
app/similarity.py      Nearest-neighbour retrieval over stored embeddings
utils/annotations.py   Generic annotation model (box + optional polygon)
utils/geometry.py      Box geometry and IoU
utils/matching.py      Generic one-to-one annotation matching
utils/resources.py     Resource discovery and verification
utils/dataset.py       data.yaml parsing and ground-truth label reading
utils/exceptions.py    Project exception hierarchy
utils/logging_utils.py Centralised logger setup
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
