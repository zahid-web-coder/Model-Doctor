# Validation report

## API — 23/23 routes returned 200, no 500s

health · runs · run detail · outcomes · images · findings · mask-findings ·
image-diagnoses · image coverage · factor-rates · root-causes · groups ·
group members · recommendations · relations · relations summary · evaluation ·
benchmarks (200, empty — table intentionally absent) · heatmaps · neighbours ·
image bytes (`image/jpeg`, 115 KB) · heatmap preview (`image/jpeg`, 50 KB) ·
heatmap full-res (`image/png`, 227 KB).

## Frontend — 11/11 routes 200, rendered and inspected

Overview, Root Causes, Clusters, Failures, Runs, Compare, Images, Heatmaps,
Recommendations, Reports, landing. Verified by screenshot:

- Overview shows 85 failures of 265 findings, the failure distribution
  (41 false negative / 37 false positive / 7 poor localization / 0 wrong class)
  and the ranked factor table with lift and p-values.
- Images lists neutral filenames (`demo-0138.jpg`) with thumbnails loading.
- Heatmaps paginates 265 tiles from previews.
- Clicking a tile opens the full-resolution Grad-CAM overlay with mask IoU,
  box IoU and confidence.
- Sidebar omits "New Analysis" and "Settings" — the read-only build behaving
  as intended.
- Browser console: no errors.

**Compare is not exercisable.** It needs two runs; this bundle is one run by
design.

## Test suites

Backend `pytest`: **997 passed**, 0 failed.
Frontend: **99 passed**, 0 failed.

## Private source integrity

`db/manual.db` SHA-256 `64e81782…ac467b` and mtime `2026-09-07T17:24:52`
identical before and after. Source images unchanged (151 files, same newest
mtime). Heatmap source directory unchanged. The source was opened
`immutable=1` throughout, which does not touch `-wal`/`-shm`.
