# Model Doctor — Comprehensive Bug Report & Audit

**Date**: 2026-08-29  
**Scope**: Full codebase audit covering `config.py`, `app/`, `utils/`, `scripts/`, `tests/`, and `web/`.

---

## Executive Summary

A comprehensive code audit of the Model Doctor repository identified **13 distinct issues** ranging from high-severity logic and platform compatibility errors to data flow, API performance, and frontend-backend synchronization gaps.

### Findings Matrix

| # | Severity | Category | Affected File(s) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **High** | CLI / Matching | `app/diagnosis.py` | `--match-iou` CLI argument is completely ignored in `main()`. |
| **2** | **High** | Platform Compatibility | `scripts/benchmark_run.py`, `scripts/benchmark_cpu.py` | Top-level `import resource` crashes on Windows (`ModuleNotFoundError`). |
| **3** | **High** | Evaluation / Ground Truth | `app/evaluation.py` | `build_ground_truth()` discards all 5-token YOLO bounding box annotations. |
| **4** | **High** | Path Portability | `config.py` | `remap_path()` separator matching fails on Windows backslash paths (`\`). |
| **5** | **Medium** | CLI / Inference | `app/inference.py` | `detector.validate()` is called without `split=args.split`, ignoring `--split`. |
| **6** | **Medium** | Data Flow / I/O | `app/features.py`, `app/root_cause.py` | Batch jobs open raw image paths without `config.remap_path(...)`. |
| **7** | **Medium** | API & Previews | `app/api.py` | `get_heatmap` checks file existence on unremapped path before calling `verified_file`. |
| **8** | **Medium** | API Performance | `app/api.py` | `/runs/{id}/findings` loads entire table into memory before Python slicing. |
| **9** | **Medium** | Explainability | `app/explainability.py` | `_anchor_centres()` hardcodes candidate image sizes; fails on arbitrary resolutions. |
| **10** | **Medium** | Error Handling | `app/mask_diagnosis.py` | Catches wrong exception for image load failures; reports misleading model error. |
| **11** | **Low** | API / Frontend Sync | `app/storage.py`, `web/src/lib/api/rows.ts` | `load_cluster_members` query omits `image_id`, preventing frontend thumbnails. |
| **12** | **Low** | CAM Geometry | `utils/cam.py` | Discrepancy between `int()` in `prepare_image` and `int(round())` in `render_overlay`. |
| **13** | **Low** | Standalone Scripts | `scripts/analyse_failures.py`, `scripts/compare_detectors.py` | Hardcodes dataset category `"column"` and class ID `0`. |

---

## Detailed Bug Reports

---

### 1. CLI Argument `--match-iou` Silently Ignored in Diagnosis
- **Severity**: High
- **Category**: Logic / CLI
- **Location**: `app/diagnosis.py` (Lines 640, 675–734)
- **Description**:
  In `build_parser()`, the CLI option `--match-iou` is defined:
  ```python
  parser.add_argument(
      "--match-iou",
      type=float,
      default=config.MATCH_IOU_THRESHOLD,
      help=f"Minimum IoU for a match. Defaults to {config.MATCH_IOU_THRESHOLD}.",
  )
  ```
  However, in `main()`, `args.match_iou` is never passed to `diagnose_split()` and is omitted when constructing `RunContext`:
  ```python
  context = RunContext(
      model_path=str(model_path),
      model_sha256=file_sha256(model_path),
      dataset_yaml=str(dataset.source_path),
      split=args.split,
      confidence_threshold=args.confidence,
      match_iou_threshold=config.MATCH_IOU_THRESHOLD,  # <--- HARDCODED DEFAULT
      localization_iou_floor=config.LOCALIZATION_IOU_FLOOR,
      image_size=args.imgsz or config.IMAGE_SIZE,
  )
  ```
- **Impact**: Any user running `python -m app.diagnosis --match-iou 0.70` will silently have their flag ignored. The diagnosis matching and persisted metadata will always use the default `0.50`.
- **Recommended Fix**: Pass `match_iou=args.match_iou` to `diagnose_split()` and set `match_iou_threshold=args.match_iou` in `RunContext`.

---

### 2. Unix-Only `import resource` Crashes on Windows
- **Severity**: High
- **Category**: Platform Compatibility
- **Location**: `scripts/benchmark_run.py` (Line 59), `scripts/benchmark_cpu.py` (Line 29)
- **Description**:
  Both benchmarking scripts execute `import resource` at top-level. The Python standard library `resource` module is Unix-only. On Windows, executing `python scripts/benchmark_run.py` or importing the module results in:
  ```
  ModuleNotFoundError: No module named 'resource'
  ```
- **Impact**: The benchmark suite cannot be imported or executed on Windows environments.
- **Recommended Fix**: Guard the import conditionally and provide a cross-platform memory tracking fallback (using `psutil` or Windows API `GetProcessMemoryInfo`):
  ```python
  try:
      import resource
  except ImportError:
      resource = None
  ```

---

### 3. Bounding Box Ground Truth Skipped in COCO Evaluation
- **Severity**: High
- **Category**: Model Evaluation
- **Location**: `app/evaluation.py` (Lines 140–141)
- **Description**:
  In `build_ground_truth()`:
  ```python
  for line in label_path.read_text().splitlines():
      parts = line.split()
      if len(parts) <= 5:
          continue  # a box-only line; segmentation needs a polygon
  ```
- **Impact**: Standard YOLO detection datasets contain 5 tokens per line (`<class_id> <x_center> <y_center> <width> <height>`). `build_ground_truth()` completely skips all 5-field lines. When running evaluation on standard bounding-box detection tasks (`TASK_BBOX`), the ground truth annotations dictionary is completely empty, yielding 0 detections and invalid evaluations.
- **Recommended Fix**: Parse 5-field box lines for `TASK_BBOX` by converting normalized `(x_c, y_c, w, h)` to pixel bounding box coordinates `[x1, y1, width, height]`.

---

### 4. Windows Path Separator Mismatch in Path Remapping
- **Severity**: High
- **Category**: Path Portability
- **Location**: `config.py` (Lines 353–365)
- **Description**:
  `remap_path()` performs string prefix checking using POSIX forward slashes:
  ```python
  old_norm = old.rstrip("/")
  if stored.startswith(old_norm + "/"):
      remainder = stored[len(old_norm) + 1:]
      return f"{new}/{remainder}"
  ```
- **Impact**: On Windows systems or with mixed path representations (where paths contain `\` or `C:\...`), `stored.startswith(old_norm + "/")` evaluates to `False`, silently failing to remap paths when loading databases created on other machines.
- **Recommended Fix**: Normalize both `stored` and `old` with `Path(...).as_posix()` or `os.path.normpath()` before prefix checking and replacement.

---

### 5. Detector Validation Discards `--split` Argument
- **Severity**: Medium
- **Category**: CLI / Validation
- **Location**: `app/inference.py` (Lines 992–999)
- **Description**:
  In `app/inference.py` `main()`:
  ```python
  if args.validate:
      logger.info("Running validation on split...")
      metrics = detector.validate()  # <--- split=args.split NOT PASSED
      print(json.dumps(metrics.to_dict(), indent=2))
  ```
  `Detector.validate` has the signature `def validate(self, split: str = "val") -> ValidationMetrics:`.
- **Impact**: If a user runs `python -m app.inference --validate --split test`, the validation will still run against `"val"` because `args.split` is never passed.
- **Recommended Fix**: Call `detector.validate(split=args.split)`.

---

### 6. Missing Path Remapping in Batch Analysis Modules
- **Severity**: Medium
- **Category**: Data Flow / I/O
- **Location**: `app/features.py` (Lines 351–360), `app/root_cause.py` (Lines 744–754)
- **Description**:
  In `extract_run_embeddings` and `analyse_run`:
  ```python
  # app/features.py
  with Image.open(row["path"]) as handle:
      rgb_image = handle.convert("RGB")

  # app/root_cause.py
  with Image.open(path) as handle:
      grey_image = np.asarray(handle.convert("L"))
  ```
  The raw database `path` is opened directly without `config.remap_path(...)`.
- **Impact**: If a database is generated on one computer and features/root-cause analysis is run on another (using `MODEL_DOCTOR_PATH_MAP`), `Image.open` throws `FileNotFoundError`.
- **Recommended Fix**: Wrap image paths with `config.remap_path(Path(path))` before opening.

---

### 7. Heatmap Preview Path Verification 404 Bug
- **Severity**: Medium
- **Category**: API & Previews
- **Location**: `app/api.py` (Lines 526–532)
- **Description**:
  In `get_heatmap(finding_id, preview=False)`:
  ```python
  stored = Path(row["path"])
  if preview:
      companion = preview_path(stored)
      if companion.is_file():  # <--- CHECKED BEFORE REMAPPING
          return verified_file(str(companion))
  return verified_file(str(stored))
  ```
- **Impact**: `companion.is_file()` checks the raw path stored in SQLite *before* `verified_file()` has the chance to apply `config.remap_path()`. On any remapped database, preview heatmaps will never be detected as existing and will fall back to full-resolution images or 404.
- **Recommended Fix**: Remap `stored` with `config.remap_path(stored)` before calling `preview_path()`.

---

### 8. Full Findings Table Loaded into Memory Without SQL Pagination
- **Severity**: Medium
- **Category**: API Performance
- **Location**: `app/api.py` (Lines 290–308)
- **Description**:
  Endpoint docstring explicitly states:
  > *"Paging is required: returning all findings at once for a run with 5,000 images would blow the memory budget."*
  
  However, the implementation does:
  ```python
  findings = storage.load_findings(connection, run_id)
  total = len(findings)
  items = findings[offset : offset + limit]
  ```
- **Impact**: `load_findings()` performs `SELECT * FROM findings WHERE run_id = ?` and loads all tens of thousands of findings into Python objects on every single page request, defeating the memory protection.
- **Recommended Fix**: Implement SQL `LIMIT ? OFFSET ?` and `SELECT COUNT(*)` in `storage.load_findings`.

---

### 9. Grad-CAM Anchor Resolution Hardcoding
- **Severity**: Medium
- **Category**: Explainability
- **Location**: `app/explainability.py` (Lines 148–193)
- **Description**:
  `_anchor_centres()` computes anchor grids by testing only a static list of sizes:
  ```python
  for candidate in (config.IMAGE_SIZE, 640, 672, 1280):
      ...
  raise ExplainabilityError(f"Cannot derive anchor grid for {num_anchors} anchors...")
  ```
- **Impact**: If a run is evaluated or explained at resolutions such as 384, 480, 512, 576, 768, or 1024, Grad-CAM raises `ExplainabilityError` and fails to generate heatmaps.
- **Recommended Fix**: Dynamically calculate grid strides based on the model's actual feature map layers or accept the `image_size` parameter.

---

### 10. Mask Diagnosis Exception Handling Logic Error
- **Severity**: Medium
- **Category**: Error Handling
- **Location**: `app/mask_diagnosis.py` (Lines 225–235, 275–280)
- **Description**:
  `diagnose_masks()` catches `ModelDoctorError`:
  ```python
  try:
      prediction = engine.predict_image(Path(path), save_annotated=False)
  except ModelDoctorError as exc:
      unreadable += 1
  ```
  However, `engine.predict_image()` is designed *never* to raise on per-image errors; it sets `prediction.error` instead.
- **Impact**: `unreadable` is never incremented. If images fail to load, `prediction.detections` is empty, causing `outlines_seen` to remain `False`. Then lines 275–280 raise:
  `MaskDiagnosisError("The model produced no outlines, so there is nothing to measure at mask level. This is expected for a detection model...")`, hiding the actual file load failure behind an incorrect diagnostic message.
- **Recommended Fix**: Check `if prediction.error is not None:` and increment `unreadable` accordingly.

---

### 11. `image_id` Missing from `load_cluster_members` SQL Query
- **Severity**: Low
- **Category**: API / Frontend Sync
- **Location**: `app/storage.py` (Lines 1233–1246), `web/src/lib/api/rows.ts` (Lines 145–155)
- **Description**:
  In `load_cluster_members()`:
  ```sql
  SELECT f.id AS finding_id, f.outcome, f.class_name, f.confidence, f.iou,
         i.filename, i.path
  FROM cluster_members cm ...
  ```
  `i.id AS image_id` is omitted from the SELECT statement.
- **Impact**: The web UI needs `image_id` to construct `/images/{image_id}` thumbnail URLs. Because `image_id` is absent, the frontend cannot request thumbnails for cluster members.
- **Recommended Fix**: Add `i.id AS image_id` to the SQL query in `load_cluster_members()`.

---

### 12. 1-Pixel Sub-Pixel CAM Overlay Shift
- **Severity**: Low
- **Category**: CAM Geometry
- **Location**: `utils/cam.py` (Lines 232 vs 260–264)
- **Description**:
  - In `prepare_image()`: `canvas.paste(resized, (int(pad_x), int(pad_y)))` truncates towards zero.
  - In `render_overlay()`: `top = int(round(letterbox.pad_y))`, `left = int(round(letterbox.pad_x))` rounds to nearest integer.
- **Impact**: When `pad_x` or `pad_y` has a half-pixel value (e.g. `12.5`), the image is pasted at pixel 12, but the heatmap is cropped from pixel 13, causing a 1-pixel spatial alignment offset.
- **Recommended Fix**: Use consistent integer truncation or rounding in both functions (e.g., `int(pad_x)` in both).

---

### 13. Hardcoded Dataset Properties in Standalone Scripts
- **Severity**: Low
- **Category**: Standalone Scripts
- **Location**: `scripts/analyse_failures.py` (Line 85), `scripts/compare_detectors.py` (Line 152)
- **Description**:
  Both scripts hardcode `class_name="column"` and category ID `0`:
  ```python
  "categories": [{"id": 0, "name": "column", "supercategory": "none"}]
  ```
- **Impact**: Running these scripts on any dataset other than the sample `columns` dataset assigns all annotations to `"column"`, distorting multi-class failure breakdowns.
- **Recommended Fix**: Read class names dynamically from `DatasetConfig.class_names` via `load_dataset_config()`.
