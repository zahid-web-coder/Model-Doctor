# Model Doctor — public demo bundle

Two complete Model Doctor runs over the same test split, prepared for public deployment.

## What this is

A **physically separate copy** of two runs. The private source database was
opened read-only and never modified.

| | Run 2 — YOLO segmentation @672 | Run 1 — RF-DETR segmentation @480 |
|---|---|---|
| Split | test | test (the same 151 photographs) |
| Findings | 265, with 265 mask findings | 276, with 276 mask findings |
| Heatmaps | 265 — full-resolution **and** preview shipped | none — Grad-CAM is not produced for RF-DETR |
| Root causes | 126 across 6 factors | 124 across 6 factors |
| Clusters | 22, with 170 members | 24, with 162 members |
| Recommendations | 4 | 4 |
| Relations | 45 | 53 |
| Embeddings | 85 | 81 |

Run 1 was added so the comparison screen has two runs over the same data. It
introduces **no new photographs**: every one of its image rows points at a
`demo-NNNN.jpg` already in this bundle. It was sanitised by the same rules as
run 2 (`reports/sanitization.md`), and the finished file was scanned — every
text value and the raw bytes — for the private paths, filenames, filename
fragments and model digests: 0 occurrences.

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
