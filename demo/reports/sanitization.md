# Sanitization / leak-scan report

Scope: `bundle/db/demo.db` and every HTTP response the read API produces from it.

## Rewritten in the demo copy

| Field | Private value (shape) | Demo value |
|---|---|---|
| `images.path` | `<private-root>/datasets/<private>/test/images/<customer-id>…jpg` | `/data/images/demo-NNNN.jpg` |
| `images.filename` | `<subject>_<customer-id>_<tag>_<date>_<export-hash>.jpg` | `demo-NNNN.jpg` |
| `images.error` | (unused) | `NULL` |
| `heatmaps.path` | `<private-root>/results/heatmaps/run_2/finding_N.png` | `/data/results/heatmaps/run_2/finding_N.png` |
| `runs.model_path` | `<private-root>/models/<private>.pt` | `models/demo/yolo-seg.pt` |
| `runs.model_sha256` | real digest of a proprietary checkpoint | 64 zeros — a documented placeholder, not a digest |
| `runs.dataset_yaml` | `<private-root>/datasets/<private>/data.yaml` | `/data/images/demo.yaml` |
| `runs.name` | `NULL` | `Structural columns · YOLO segmentation @672` |
| `runs.inference_library_version` | — | `NULL` |
| `run_evaluations.mask_source` | wording naming an internal record type | `model raster (all classes), …` |

Not copied at all: **`jobs`** (upload tokens, absolute build paths, log tails)
and **`run_benchmarks`** (host platform, latency, checkpoint byte size).

## Scan results

Raw database file bytes were scanned for sixteen patterns: absolute home and
workspace path prefixes, the customer-id prefix, the three internal record
tags, the customer-reference prefix, the dataset-export infix, the company
name, the real model digest, a database connection scheme, an authorization
header scheme, and the two private dataset names. **0 occurrences each.**

HTTP responses — 19 data routes, 2,454 KB of body scanned for the same
sixteen patterns plus tracebacks, private IP ranges and home directories:
**0 occurrences**.
Error bodies (`/runs/999`, `/images/999999`, `/findings/999999/heatmap`) return
plain messages with no path or traceback.

`/openapi.json` matches one internal-tag pattern 22 times, entirely inside
OpenAPI's own schema-reference vocabulary. Not a data leak.

## Accepted, not removed

`heatmaps.target_layers` = `yolo26-seg:16,19,22` on all 265 rows. This names a
**public** architecture and the Grad-CAM layer indices; it identifies no private
artefact, and the heatmap view uses it as context. Flagged rather than changed.

`/health` echoes the configured `MD_DB_PATH`. Harmless when that variable is
`/data/db/model_doctor.db`, as the deployment image sets it — but the variable
must never be given a path that reveals anything.
