"""Score any detector against ground truth with one shared COCO evaluator.

**This module exists because the obvious comparison is invalid.** Running
Ultralytics' ``model.val()`` for YOLO and RF-DETR's own evaluator for RF-DETR
produces two numbers that look alike and cannot be compared: different matching
rules, different NMS, different area bands, different score handling. Any
difference between them would be partly the evaluators and partly the models,
with no way to separate the two. ``RFDetrDetector.validate`` refuses to run for
exactly this reason and points here.

Everything is therefore symmetric by construction:

* **One ground truth.** Built from the YOLO labels the dataset ships, for every
  family. Categories come from ``data.yaml``, never from a constant.
* **One evaluator.** ``pycocotools`` COCOeval, for both families, box and mask,
  at COCO's own IoU sweep of 0.50:0.05:0.95.
* **One confidence sweep.** Predictions are collected at
  :data:`SWEEP_CONFIDENCE`, far below any operating point, so the
  precision/recall curve is essentially complete. Thresholding before mAP
  truncates the curve and flatters whichever model is more confident.
* **One detection cap.** :data:`MAX_DETECTIONS` per image, recorded alongside
  every result, because a model that emits an order of magnitude more
  low-confidence boxes than another is affected by this cap and the reader
  needs to know where it sat.

**Masks are encoded from the model's own raster, never from the stored
polygon.** Model Doctor's annotation model carries one polygon per object, so
both families reduce a mask to its largest component on the way in — RF-DETR in
``_mask_to_polygon``, Ultralytics in ``masks2segments``. That reduction is fine
for display and storage, and wrong for scoring: it would charge a model for a
representation choice rather than a prediction. Segmentation is therefore
evaluated from the full multi-component raster each model actually produced.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from model_doctor.utils.dataset import label_path_for_image
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Low enough that the PR curve is essentially complete. mAP integrates over
# every operating point, so discarding low-confidence predictions before
# evaluation silently caps recall.
SWEEP_CONFIDENCE: float = 0.01

# COCO's own default. Recorded with every result rather than assumed, because a
# model that emits more than this per image is being truncated by it.
MAX_DETECTIONS: int = 100

# COCO's standard sweep. Stated as data so it can be persisted and shown.
IOU_THRESHOLDS: str = "0.50:0.05:0.95"

# The two things COCOeval can score. Model Doctor names them as COCO does.
TASK_BBOX = "bbox"
TASK_SEGM = "segm"
TASKS: tuple[str, ...] = (TASK_BBOX, TASK_SEGM)

# Every extension present in a split. Globbing "*.jpg" alone silently dropped
# five .jpeg files the first time this was run — the quiet kind of truncation
# that makes a comparison wrong without making it look wrong.
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# The order COCOeval reports its summary statistics in.
_STAT_KEYS: tuple[str, ...] = (
    "map50_95", "map50", "map75", "map_small", "map_medium", "map_large",
    "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large",
)


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot be run or trusted."""


def list_images(images_dir: Path) -> list[Path]:
    """Return every image in the directory, sorted, regardless of extension."""
    return sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------
# Ground truth — one canonical set, from the dataset's own labels
# ---------------------------------------------------------------------------
def build_ground_truth(
    images_dir: Path, class_names: dict[int, str]
) -> dict[str, Any]:
    """Build a COCO-format ground truth from a split's YOLO labels.

    The YOLO export is canonical for every family, so the evaluator sees one
    set of annotations rather than one per model. Categories are taken from the
    dataset's ``names``, so a dataset with different or more classes needs no
    change here.

    Args:
        images_dir: A split's ``images`` directory.
        class_names: ``{id: name}`` from :class:`~utils.dataset.DatasetConfig`.

    Returns:
        A COCO ``dict`` with ``images``, ``annotations`` and ``categories``.

    Raises:
        EvaluationError: If the directory holds no images.
    """
    from PIL import Image

    image_paths = list_images(images_dir)
    if not image_paths:
        raise EvaluationError(f"No images found in {images_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as handle:
            width, height = handle.size
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = label_path_for_image(image_path)
        if not label_path.is_file():
            continue

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])

            if len(parts) == 5:
                # Detection box: class_id x_center y_center w h (normalized)
                x_c, y_c, w, h = (float(v) for v in parts[1:])
                x1 = (x_c - w / 2) * width
                y1 = (y_c - h / 2) * height
                box_w = w * width
                box_h = h * height
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                annotation_id += 1
            elif len(parts) > 5:
                coords = [float(value) for value in parts[1:]]
                xs = [coords[i] * width for i in range(0, len(coords), 2)]
                ys = [coords[i + 1] * height for i in range(0, len(coords), 2)]
                polygon: list[float] = []
                for x, y in zip(xs, ys, strict=True):
                    polygon.extend([x, y])

                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                    "segmentation": [polygon],
                })
                annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        # Derived from the dataset, never a constant: a two-class dataset must
        # evaluate as two classes without editing this file.
        "categories": [
            {"id": class_id, "name": name, "supercategory": "none"}
            for class_id, name in sorted(class_names.items())
        ],
    }


# ---------------------------------------------------------------------------
# Mask encoding — the model's own raster, all components
# ---------------------------------------------------------------------------
def encode_mask(mask: np.ndarray) -> dict[str, Any] | None:
    """Encode a boolean instance mask as COCO RLE.

    The mask must already be in original-image pixels — the same space as every
    box and every ground-truth polygon (D-006). Nothing is rescaled here; a
    caller holding a mask at model-input resolution must map it back first,
    because silently resizing would hide the mismatch rather than surface it.

    Every component is preserved. This is the whole point: reducing a
    multi-part instance to its largest blob is a storage convenience, and
    scoring against it would penalise a model for Model Doctor's
    representation rather than for its prediction.

    Args:
        mask: 2-D boolean array, ``True`` inside the object.

    Returns:
        A COCO RLE with an ASCII ``counts`` string, or ``None`` if the mask is
        empty — an empty mask is not a prediction and must not become one.
    """
    from pycocotools import mask as mask_utils

    if mask.ndim != 2:
        raise EvaluationError(f"Expected a 2-D mask, got shape {mask.shape}")
    if not mask.any():
        return None

    # Fortran order is what pycocotools expects; asfortranarray is a no-op when
    # the array already has it.
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ---------------------------------------------------------------------------
# Predictions -> COCO
# ---------------------------------------------------------------------------
@dataclass
class PredictionRecord:
    """Every prediction one model made on one image, for the evaluator.

    ``mask_rles`` runs parallel to ``detections`` by index. It is populated
    only when the detector was asked to keep rasters, and an entry is ``None``
    when that particular instance had no mask.
    """

    file_name: str
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    mask_rles: list[dict[str, Any] | None] = field(default_factory=list)
    inference_ms: float = 0.0
    error: str | None = None


def to_coco_detections(
    records: list[PredictionRecord], ground_truth: dict[str, Any], task: str
) -> list[dict[str, Any]]:
    """Convert prediction records into COCO detection entries for one task.

    Box and mask entries are built from the *same* predictions, so the two
    scores describe one set of outputs rather than two differently-derived
    ones. A record whose image is absent from the ground truth is skipped
    rather than guessed at.
    """
    if task not in TASKS:
        raise EvaluationError(f"Unknown task '{task}'; expected one of {TASKS}")

    ids = {image["file_name"]: image["id"] for image in ground_truth["images"]}

    entries: list[dict[str, Any]] = []
    for record in records:
        image_id = ids.get(record.file_name)
        if image_id is None:
            continue
        for index, (box, score, class_id) in enumerate(
            zip(record.boxes, record.scores, record.class_ids, strict=True)
        ):
            entry: dict[str, Any] = {
                "image_id": image_id,
                "category_id": int(class_id),
                "score": float(score),
            }
            if task == TASK_BBOX:
                x1, y1, x2, y2 = box
                entry["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                rle = (
                    record.mask_rles[index]
                    if index < len(record.mask_rles)
                    else None
                )
                if rle is None:
                    continue  # this instance produced no mask; not a prediction
                entry["segmentation"] = rle
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------
def coco_evaluate(
    ground_truth: dict[str, Any], detections: list[dict[str, Any]], task: str
) -> dict[str, float]:
    """Run COCOeval and return its twelve summary statistics.

    Raises:
        EvaluationError: If ``pycocotools`` is missing.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise EvaluationError(
            "pycocotools is required to evaluate. Install the core "
            "requirements: pip install -r requirements.txt"
        ) from exc

    if not detections:
        return {}

    # COCOeval prints unconditionally; capture it so logs stay readable.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(detections))
        evaluator = COCOeval(coco_gt, coco_dt, task)
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        key: float(value)
        for key, value in zip(_STAT_KEYS, evaluator.stats, strict=False)
    }


@dataclass
class TaskEvaluation:
    """One task's metrics, with the settings that produced them."""

    task: str
    metrics: dict[str, float]
    prediction_count: int


def evaluate_predictions(
    records: list[PredictionRecord],
    ground_truth: dict[str, Any],
) -> list[TaskEvaluation]:
    """Score one model's predictions on both tasks with the shared evaluator."""
    results: list[TaskEvaluation] = []
    for task in TASKS:
        detections = to_coco_detections(records, ground_truth, task)
        metrics = coco_evaluate(ground_truth, detections, task)
        if not metrics:
            logger.warning("No %s predictions to evaluate; skipping", task)
            continue
        results.append(
            TaskEvaluation(
                task=task, metrics=metrics, prediction_count=len(detections)
            )
        )
    return results


__all__ = [
    "IOU_THRESHOLDS",
    "MAX_DETECTIONS",
    "SWEEP_CONFIDENCE",
    "TASKS",
    "TASK_BBOX",
    "TASK_SEGM",
    "EvaluationError",
    "PredictionRecord",
    "TaskEvaluation",
    "build_ground_truth",
    "coco_evaluate",
    "encode_mask",
    "evaluate_predictions",
    "list_images",
    "to_coco_detections",
]
