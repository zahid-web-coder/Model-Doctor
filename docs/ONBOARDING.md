# Running Model Doctor on a second machine

Everything the analysis runs on — models, datasets, databases — is gitignored,
so a clone gives you the code and nothing to point it at. This is how to get
from a clone to a working dashboard.

There are two levels of setup. **Most work needs only the first.**

| | Needs | Size | Can do |
|---|---|---|---|
| **Read-only** | a database + its image files | ~465 MB | Every screen, all analysis, UI work, API work |
| **Full** | the above, plus models and datasets | +711 MB | Re-run inference, diagnosis and benchmarking |

The API is read-only over SQLite and the dashboard is read-only over the API,
so nothing in the UI requires a model file or a GPU. Ask for the full setup
only when you need to produce *new* runs.

---

## 1. Prerequisites

**Node 20 or newer.** The dashboard will not build on older versions, and the
error it gives is about syntax rather than versions, which sends people the
wrong way. Check with `node -v` before anything else.

```bash
node -v      # must be >= 20
python3 -V   # 3.11 or newer
```

If you use `nvm` and your shell defaults to an older Node, either `nvm use 20`
or put the version you want on `PATH` for the session.

## 2. Clone and install

```bash
git clone https://github.com/zahid-web-coder/Model-Doctor.git
cd Model-Doctor

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd web && npm install && cd ..
```

## 3. Unpack the data bundle

You will be sent a folder — call it `model-doctor-data` — laid out like this:

```
model-doctor-data/
  db/columns_eval.db
  datasets/columns_all_1280_yolo/test/images/*.jpg
  results/heatmaps/run_2/*.png  *.preview.jpg
```

Put it anywhere. **Do not put it inside the repository** — the paths below are
absolute and the repo ignores these directories anyway.

## 4. The one thing that will otherwise waste your afternoon

`images.path` and `heatmaps.path` in the database are **absolute paths**
recorded on the machine that ran the diagnosis. On your machine they point at
nothing, so every table and chart renders correctly and **every image 404s**.

`MD_PATH_REMAP` rewrites the prefix when a file is served. The database is not
modified — it stays a true record of where the run was actually diagnosed.

```bash
export MD_DB_PATH="/path/to/model-doctor-data/db/columns_eval.db"
export MD_PATH_REMAP="/Users/user_/Desktop/model doctor=/path/to/model-doctor-data"
export MD_API_FILE_ROOTS="/path/to/model-doctor-data"
```

The left-hand side of `MD_PATH_REMAP` is the prefix stored *in the database*.
If you are unsure what it is, read it from the database itself:

```bash
sqlite3 "/path/to/model-doctor-data/db/columns_eval.db" "SELECT path FROM images LIMIT 1;"
```

`MD_API_FILE_ROOTS` is a separate, deliberate check: the API refuses to serve
any file outside the roots it was started with, so a remap alone cannot make it
read arbitrary files on your disk. Both are required.

## 5. Run it

Two processes. Backend first:

```bash
.venv/bin/python -m uvicorn app.api:create_app --factory --port 8000
```

Then the dashboard, in a second terminal:

```bash
cd web && npm run dev -- --port 3100
```

Open <http://localhost:3100>. If the API is on a non-default host or port, set
`NEXT_PUBLIC_API_BASE` before starting the dashboard.

## 6. Check it actually worked

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health     # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/images/1   # 200
```

`/health` returning 200 while `/images/1` returns **403** means the remap is
wrong or `MD_API_FILE_ROOTS` does not cover where the files actually are. A
**404** means the remap is right but the file is missing from the bundle.

In the dashboard, the Heatmaps tab is the fastest visual check: tiles render
from disk, so if they load, paths are resolving.

---

## Full setup: running inference

Only needed to produce new runs.

```
models/     *.pt checkpoints
datasets/   the full dataset, including the training split
```

Both are gitignored and must be copied across separately. Once in place:

```bash
.venv/bin/python -m app.cli diagnose --help
```

Note that `MD_MODEL_FILENAME`, `MD_CONF`, `MD_MATCH_IOU` and the other
thresholds are all environment-configurable — see `config.py`, where each
constant documents what it does and why its default is what it is.

## Tests

```bash
.venv/bin/python -m pytest          # backend
cd web && npm test                  # frontend
cd web && npx tsc --noEmit          # types
```

These need no data bundle — the suites build their own fixtures.

---

## Running a new analysis from the browser

The dashboard can run the whole pipeline itself — upload a checkpoint and a
dataset, and it performs inference, evaluation, diagnosis, clustering,
recommendations and (for YOLO) heatmaps without a terminal.

It needs a second service, the control API:

```bash
uvicorn app.control:app --host 127.0.0.1 --port 8001
```

Then open **New Analysis** in the sidebar.

> **Run one worker, and do not expose port 8001 to a network.**
>
> Start it exactly as shown above — no `--workers`. The job queue lives in the
> process, and a second worker would mark the first one's running job as failed.
> The control API has no authentication, and by design it executes the model
> you upload. Anyone who can reach it can run code in this process. Bind it to
> `127.0.0.1`, which is the default above. This is a local tool.

If the dashboard runs on a different host or port from the default, set
`NEXT_PUBLIC_CONTROL_BASE` before starting it.

### What it accepts

| | |
|---|---|
| Model | `.pt` / `.pth`, up to `MD_MAX_MODEL_MB` (2048 by default) |
| Dataset | `.zip` / `.tar` / `.tar.gz`, up to `MD_MAX_DATASET_MB` (8192 by default) |
| Layout | YOLO: a `data.yaml` naming the classes, beside splits holding `images/` and `labels/` |
| Detectors | Whatever `app.detectors.SUPPORTED_FAMILIES` lists — currently YOLO and RF-DETR |

The family is read from the checkpoint itself. A file that matches no
supported family is refused by name rather than half-run.

### What it does not do

* **No compute benchmarking.** Latency and memory are only meaningful under
  controlled conditions, which a service running beside other work cannot
  promise. Use `scripts/benchmark_run.py` on a quiet machine.
* **No heatmaps for RF-DETR.** No attribution method has been validated for it
  in this project, so the stage is skipped and the UI says so rather than
  producing a map that would not correspond to the prediction.
* **No cancellation.** A queued or running analysis runs to completion or
  failure. Restarting the service closes out anything in flight.

### Housekeeping

Uploads are kept in `MD_WORKSPACE_DIR` (default `~/.model-doctor/workspace`),
outside the repository. Each analysis keeps its own copy of the dataset, so the
newest `MD_WORKSPACE_KEEP` workspaces (5 by default) are retained and older
ones are removed when a new analysis starts.

## Letting Claude compare runs (MCP)

Model Doctor's stored analysis is available to a reasoning model through two
read-only MCP tools, `list_runs` and `get_analysis`. Nothing here can write:
the database is opened `mode=ro`, the transport is stdio, and there is no tool
that deletes, mutates, trains, or runs inference.

The repository ships a project-scoped `.mcp.json` that registers the server
as `model-doctor` against `db/manual.db`. In Claude Code, open the project and
approve the server once when prompted, or register it yourself:

```bash
claude mcp add -s project model-doctor -e MD_DB_PATH="$PWD/db/manual.db" -- .venv/bin/python -m app.mcp_server
```

Then ask, for example:

> Compare runs 4, 5 and 7 and tell me what I should investigate next. Cite
> the evidence from the Model Doctor analysis and clearly distinguish evidence
> from inference.

The server needs only `requirements-mcp.txt` — no torch, no FastAPI — so it
can run on a machine that has the database and nothing else. Point
`MD_DB_PATH` at a different database to serve different runs; there is no
tool argument for it, deliberately.
