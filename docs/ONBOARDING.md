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
