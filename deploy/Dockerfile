# Model Doctor — read-only API, serving the public demo dataset.
#
# Carries the API dependencies (see requirements-api.txt) and one prepared
# demo run. No model weights: the API never loads the ML stack, which is what
# keeps this image small enough to be worth deploying.
#
# The demo data under demo/ is a sanitised copy — neutral filenames, no
# absolute paths, no customer identifiers, no real model digest. See
# demo/reports/sanitization.md, and demo/manifest.json for its publication
# status.

FROM python:3.11-slim

# Fail fast and log immediately; both matter when the only view of a container
# is its stdout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Dependencies first, so a code change does not invalidate the layer.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# The engine package. No tests, no docs, no notebooks — see .dockerignore.
#
# One directory rather than three: `config.py`, `app/` and `utils/` moved under
# `model_doctor/` when the package became installable, so the older per-directory
# copies named paths that no longer exist and the build failed here. Modules the
# read API never imports (the control service, the dashboard, the MCP server)
# come along, and cannot run in this image anyway: their dependencies are not
# installed, and the command below serves the read API only.
COPY model_doctor/ ./model_doctor/

# The demo dataset, baked in rather than mounted.
#   /data/db      the SQLite database
#   /data/images  the images referenced by images.path
#   /data/results heatmap PNGs and previews referenced by heatmaps.path
#
# Baked in because the free hosting tier this image targets has no persistent
# disk, so there is nothing to mount from. An image layer is also strictly
# better here: the data cannot go missing across a restart, a spin-down or a
# redeploy, and there is no mounting step to get wrong.
#
# There is deliberately no VOLUME. A volume at /data would shadow these
# directories at run time and the API would start against an empty tree.
#
# The database records these exact paths, so nothing below has to be remapped.
COPY demo/db/      /data/db/
COPY demo/images/  /data/images/
COPY demo/results/ /data/results/

ENV MD_DB_PATH=/data/db/model_doctor.db \
    MD_DATASETS_DIR=/data/images \
    MD_RESULTS_DIR=/data/results

# The API refuses to serve a file outside these roots, so they must match the
# paths recorded in the database. See D-037.
ENV MD_API_FILE_ROOTS=/data/images,/data/results

# Browser origins permitted to call this service. Override per environment;
# never widen to "*", because the API serves local file contents.
ENV MD_CORS_ORIGINS=http://localhost:3000

EXPOSE 8000

# A non-root user: the API only ever reads, so it needs no write capability.
RUN useradd --create-home --uid 10001 api && chown -R api /srv
USER api

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "model_doctor.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
