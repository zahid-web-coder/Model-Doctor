# Deploying the Model Doctor API

The API is read-only. It serves the schema documented in `docs/SCHEMA.md` as
JSON, plus image and heatmap bytes. It runs no model and needs no GPU.

## What the image contains, and what it does not

**Contains:** `config.py`, `app/`, `utils/`, and the two dependencies in
`requirements-api.txt` — FastAPI and Uvicorn. About 46 MB of packages.

**Does not contain:** the model, the dataset, the heatmaps, or the database.
Those are mounted at runtime. Where they should live is a storage decision that
has not been taken yet, and baking them into the image would pre-empt it.

## Build

```bash
docker build -f deploy/Dockerfile -t model-doctor-api .
```

## Run

The API refuses to serve any file outside its configured roots (D-037), so the
mounted paths **must match the paths recorded in the database**. Check what a
run actually stored before mounting:

```sql
SELECT path FROM images LIMIT 1;
```

```bash
docker run --rm -p 8000:8000 \
  -v /host/path/to/db:/data/db:ro \
  -v /host/path/to/images:/data/images:ro \
  -v /host/path/to/results:/data/results:ro \
  -e MD_CORS_ORIGINS=https://your-frontend.example \
  model-doctor-api
```

Mounts are read-only (`:ro`) because the service never writes.

## The path-matching problem

`images.path` holds an absolute path from whichever machine ran the diagnosis —
for example `/Users/someone/Downloads/dataset/test/images/x.jpg`. Inside a
container that path does not exist, so every image returns **404**, or **403**
if it resolves outside the configured roots.

Three ways to handle it, in order of preference:

1. **Mount to matching paths.** If the database records
   `/data/images/test/...`, mount the dataset at `/data/images`. Cleanest, and
   requires re-running the diagnosis inside the deployment layout.
2. **Re-run the diagnosis with deployment paths.** Set `MD_DATASETS_DIR` to the
   final location before `--save`, so the recorded paths are the deployed ones.
3. **Add the host's real root to `MD_API_FILE_ROOTS`.** Works when the
   directory structure is reproduced but rooted elsewhere.

Rewriting stored paths is not offered: `images.path` is part of a frozen
contract and the run recorded where it actually read from.

## Health

```bash
curl http://localhost:8000/health
```

Returns the schema version and whether the database is present. A missing
database yields **503** with the command that produces one, not a crash.

## Not yet decided

- **Hosting provider.** Needs a persistent filesystem for the mounts, so
  serverless platforms are unsuitable for the API. The Next.js frontend has no
  such constraint.
- **Where the dataset and database live** in the deployed environment, which
  determines which of the three path strategies above applies.
