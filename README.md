# Model Doctor

A Vision AI **failure diagnosis** platform. Standard tools tell you a model
scored `mAP50 = 0.72`. Model Doctor is being built to answer the next question:
*which* predictions were wrong, and *why*.

Version 1 targets YOLO. The detector is isolated behind `app/inference.py` so
additional architectures can be added later without touching analysis code.

## Status

Day 2 — inference foundation. The model and dataset are **not yet available**,
and the codebase is written to handle that state cleanly rather than crash.

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

# Metrics (precision / recall / mAP)
./.venv/bin/python -m app.inference --validate
```

Annotated images are written to `results/predictions/`.

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
| `MD_DEVICE` | Force `cpu`, `mps`, or `cuda` |
| `MD_LOG_LEVEL` | `DEBUG` to see library internals |

## Layout

```
config.py              Single source of truth for paths and constants
app/inference.py       Model loading, prediction, structured results, CLI
utils/resources.py     Resource discovery and verification
utils/dataset.py       data.yaml parsing and ground-truth label reading
utils/logging_utils.py Centralised logger setup
tests/                 Test suite (run: ./.venv/bin/python -m pytest tests/ -q)
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
