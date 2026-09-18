# Model Doctor — public demo bundle

One complete Model Doctor run, prepared for public deployment.

## What this is

A **physically separate copy** of a single run. The private source database was
opened read-only (`immutable=1`) and never modified.

| | |
|---|---|
| Run | id 2 — "Structural columns · YOLO segmentation @672", test split |
| Images | 151 (all distinct by content SHA-256) |
| Findings | 265, with 265 mask findings |
| Heatmaps | 265 — every finding has one; full-resolution **and** preview shipped |
| Root causes | 126 across 6 factors |
| Clusters | 22, with 170 members |
| Recommendations | 4 |
| Relations | 45 |
| Embeddings | 85 (back the "similar failures" route) |

## Layout

    db/demo.db                              schema 15, journal_mode=DELETE
    images/demo-0001.jpg …                  960 px long side, q80, metadata stripped
    results/heatmaps/run_2/finding_N.png            960 px, PNG-8 256 colours
    results/heatmaps/run_2/finding_N.preview.jpg    as produced by the engine
    manifest.json                           content hash -> demo key

Paths inside the database are `/data/...` only. Mount or copy this bundle at
`/data` and the API resolves everything with no code change.

## What was sanitised

Original filenames, absolute paths, workspace UUIDs, customer identifiers, the
dataset name and the real model SHA-256 are all gone. `jobs` and
`run_benchmarks` were not copied at all. `runs.model_sha256` carries a
documented placeholder of 64 zeros — it is **not** a digest of anything.

`heatmaps.target_layers` still reads `yolo26-seg:16,19,22`. That is the public
architecture name and the Grad-CAM layer indices, kept deliberately: it is
technical context the heatmap view uses, and it names no private artefact.

## PUBLICATION_REVIEW = PENDING

**These photographs have not been reviewed or cleared for publication.** They
are known to contain workers, third-party signage carrying phone numbers and
street addresses, QR codes, and neighbouring properties. Sanitising metadata
does not sanitise pixels. Nothing in this bundle is approved for public
release; see `manifest.json`.
