# Model Doctor — read-only API.
#
# Deliberately does NOT contain the model, the dataset, or the database. Those
# are mounted at runtime, because where the data lives is a storage decision
# that has not been made yet and baking it in would pre-empt it.
#
# The image carries only the API dependencies (see requirements-api.txt), which
# is possible because the API never loads the ML stack.

FROM python:3.11-slim

# Fail fast and log immediately; both matter when the only view of a container
# is its stdout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Dependencies first, so a code change does not invalidate the layer.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API imports. No tests, no docs, no notebooks.
COPY config.py ./
COPY app/ ./app/
COPY utils/ ./utils/

# Mount points. Nothing is copied into them at build time.
#   /data/db      the SQLite database
#   /data/images  the dataset images referenced by images.path
#   /data/results heatmap PNGs referenced by heatmaps.path
VOLUME ["/data"]

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

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
