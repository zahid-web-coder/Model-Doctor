"""Score two detectors against one ground truth with one evaluator.

**This script exists because the obvious comparison is invalid.** Running
Ultralytics' ``model.val()`` for YOLO and RF-DETR's own evaluator for RF-DETR
produces two numbers that look alike and cannot be compared: different matching
rules, different NMS, different area bands, different score handling. Any
difference between them would be partly the evaluators and partly the models,
with no way to separate the two.

So everything here is deliberately symmetric:

* **One ground truth.** The YOLO-format test labels, for both models. The COCO
  export of the same dataset disagrees with it on 42 of 228 boxes — the two
  exports differ wherever an instance is multi-part, because YOLO's format
  allows one polygon per object and the conversion merged the parts. Scoring
  each model against its own export would measure that conversion, not the
  models.
* **One evaluator.** ``pycocotools`` COCOeval, for both, box and mask.
* **One inference threshold.** Predictions are collected once at a very low
  threshold so the precision/recall curve is complete, then filtered to the
  operating point for the failure analysis. Thresholding before mAP would
  truncate the curve and flatter whichever model is more confident.

Stage 1 writes raw predictions to disk so the analysis can be re-run without
paying for inference again.

Usage::

    python scripts/compare_detectors.py --stage predict
    python scripts/compare_detectors.py --stage evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Low enough that the PR curve is essentially complete. mAP integrates over
# every operating point, so discarding low-confidence predictions before
# evaluation silently caps recall.
SWEEP_CONFIDENCE = 0.01
# The operating point the failure analysis reports at, matching the project's
# existing default so these findings are comparable to earlier runs.
OPERATING_CONFIDENCE = 0.25

# Every extension present in the split. Globbing "*.jpg" alone silently dropped
# five .jpeg files on the first run — 146 of 151 images — which is the quiet
# kind of truncation that makes a comparison wrong without making it look wrong.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(images_dir: Path) -> list[Path]:
    """Return every image in the directory, sorted, regardless of extension."""
    return sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


@dataclass
class ModelSpec:
    """One model under test."""

    key: str
    label: str
    weights: Path
    family: str  # "yolo" | "rfdetr"
    native_imgsz: int
    notes: str = ""


@dataclass
class ImageRecord:
    """Predictions for one image from one model."""

    file_name: str
    width: int
    height: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    inference_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Ground truth — one canonical set, built from the YOLO labels
# ---------------------------------------------------------------------------
def load_ground_truth(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    """Build a COCO-format ground truth from the YOLO test labels.

    The YOLO export is the canonical ground truth for both models (see the
    module docstring). Converting it into COCO's structure here means the
    evaluator sees one set of annotations, not two.
    """
    from PIL import Image

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for image_id, image_path in enumerate(list_images(images_dir), start=1):
        with Image.open(image_path) as handle:
            width, height = handle.size
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) <= 5:
                continue  # a box-only line; this dataset is fully segmented
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            xs = [coords[i] * width for i in range(0, len(coords), 2)]
            ys = [coords[i + 1] * height for i in range(0, len(coords), 2)]
            polygon: list[float] = []
            for x, y in zip(xs, ys, strict=True):
                polygon.extend([x, y])

            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "segmentation": [polygon],
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "column", "supercategory": "none"}],
    }


# ---------------------------------------------------------------------------
# Stage 1 — inference
# ---------------------------------------------------------------------------
def build_detector(spec: ModelSpec, confidence: float) -> Any:
    """Return a loaded detector for the spec, by family."""
    if spec.family == "rfdetr":
        from app.rfdetr_adapter import RFDetrDetector

        detector = RFDetrDetector(
            model_path=str(spec.weights),
            confidence=confidence,
            image_size=spec.native_imgsz,
        )
    else:
        from app.inference import Detector

        detector = Detector(
            model_path=str(spec.weights),
            confidence=confidence,
            image_size=spec.native_imgsz,
        )
    detector.load()
    return detector


def run_inference(spec: ModelSpec, images_dir: Path, out_path: Path) -> None:
    """Run one model over every test image and write raw predictions."""
    detector = build_detector(spec, SWEEP_CONFIDENCE)
    image_paths = list_images(images_dir)

    # A warm-up pass, excluded from timing. The first call pays for lazy
    # kernel compilation and weight paging; including it would make whichever
    # model is measured first look slower for reasons that have nothing to do
    # with the model.
    if image_paths:
        detector.predict_image(image_paths[0], save_annotated=False)

    records: list[ImageRecord] = []
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths):
        prediction = detector.predict_image(image_path, save_annotated=False)
        records.append(ImageRecord(
            file_name=image_path.name,
            width=prediction.image_width,
            height=prediction.image_height,
            inference_ms=prediction.inference_ms,
            error=prediction.error,
            detections=[{
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": [d.x1, d.y1, d.x2, d.y2],
                "polygon": d.polygon,
            } for d in prediction.detections],
        ))
        if (index + 1) % 25 == 0:
            logger.info("%s: %d/%d", spec.key, index + 1, len(image_paths))

    wall = time.perf_counter() - started
    payload = {
        "model": {
            "key": spec.key,
            "label": spec.label,
            "weights": str(spec.weights),
            "family": spec.family,
            "native_imgsz": spec.native_imgsz,
            "checkpoint_bytes": spec.weights.stat().st_size,
            "notes": spec.notes,
        },
        "sweep_confidence": SWEEP_CONFIDENCE,
        "wall_seconds": wall,
        "images": [vars(r) for r in records],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    logger.info("%s: wrote %s (%.1fs wall)", spec.key, out_path, wall)


# ---------------------------------------------------------------------------
# Stage 2 — one evaluator, both models
# ---------------------------------------------------------------------------
def to_coco_predictions(
    payload: dict[str, Any], gt: dict[str, Any], kind: str
) -> list[dict[str, Any]]:
    """Convert stored predictions into COCO detection records.

    ``kind`` is ``"bbox"`` or ``"segm"``. Masks are encoded from the polygon
    Model Doctor already carries, so box and mask scoring use the same
    predictions rather than two differently-derived sets.
    """
    from pycocotools import mask as mask_utils

    ids = {image["file_name"]: image["id"] for image in gt["images"]}
    sizes = {
        image["file_name"]: (image["height"], image["width"])
        for image in gt["images"]
    }

    results: list[dict[str, Any]] = []
    for record in payload["images"]:
        image_id = ids.get(record["file_name"])
        if image_id is None:
            continue
        height, width = sizes[record["file_name"]]
        for det in record["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            entry = {
                "image_id": image_id,
                "category_id": det["class_id"],
                "score": det["confidence"],
            }
            if kind == "bbox":
                entry["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            else:
                polygon = det.get("polygon")
                if not polygon or len(polygon) < 3:
                    continue
                flat = [float(v) for point in polygon for v in point]
                rles = mask_utils.frPyObjects([flat], height, width)
                rle = mask_utils.merge(rles)
                rle["counts"] = rle["counts"].decode("ascii")
                entry["segmentation"] = rle
            results.append(entry)
    return results


def coco_evaluate(
    gt: dict[str, Any], predictions: list[dict[str, Any]], kind: str
) -> dict[str, float]:
    """Run COCOeval and return the standard summary metrics."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {}

    # COCOeval prints unconditionally; capture it so the report stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(predictions))
        ev = COCOeval(coco_gt, coco_dt, kind)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    keys = [
        "map50_95", "map50", "map75", "map_small", "map_medium", "map_large",
        "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large",
    ]
    return {k: float(v) for k, v in zip(keys, ev.stats, strict=False)}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["predict", "evaluate"], required=True)
    parser.add_argument("--dataset", default="datasets/columns_all_1280_yolo")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/comparison")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested stage and return an exit code."""
    args = build_parser().parse_args(argv)
    root = Path(args.dataset)
    images_dir = root / args.split / "images"
    labels_dir = root / args.split / "labels"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The 2x2. The first comparison ran each model at its own training
    # resolution, which confounds architecture with input size: RF-DETR's
    # recall advantage could have been the architecture, or simply that 312
    # and 672 are different experiments. Running both models at both
    # resolutions separates the two, and is the only way to answer whether
    # RF-DETR is better or merely differently configured.
    yolo = Path("models/columns/yolo26_nano_seg.pt")
    rfdetr = Path("models/columns/rfdetr_nano_seg.pt")

    # YOLO at its native resolution is the reference line. It is not swept:
    # the question here is where RF-DETR should run, not how to retune YOLO.
    specs = [
        ModelSpec("yolo26", "YOLO26 @672 (ref)", yolo, "yolo", 672, "reference"),
    ]
    # RF-DETR accepts any resolution that is a multiple of 12, interpolating
    # its positional encodings. 768 is past the range asked for and is included
    # only to show whether accuracy is still climbing at 672 or has levelled
    # off — an operating point chosen at the edge of a sweep is a guess.
    for res in (312, 384, 480, 576, 672, 768):
        specs.append(ModelSpec(
            f"rfdetr_{res}", f"RF-DETR @{res}", rfdetr, "rfdetr", res,
            "native" if res == 312 else "",
        ))

    if args.stage == "predict":
        for spec in specs:
            if not spec.weights.is_file():
                logger.error("missing weights: %s", spec.weights)
                return 1
            run_inference(spec, images_dir, out_dir / f"{spec.key}.json")
        return 0

    gt = load_ground_truth(images_dir, labels_dir)
    (out_dir / "ground_truth.json").write_text(json.dumps(gt))
    logger.info("ground truth: %d images, %d annotations",
                len(gt["images"]), len(gt["annotations"]))

    summary: dict[str, Any] = {"ground_truth": {
        "images": len(gt["images"]), "annotations": len(gt["annotations"])}}
    for spec in specs:
        path = out_dir / f"{spec.key}.json"
        if not path.is_file():
            logger.error("missing predictions: %s — run --stage predict", path)
            return 1
        payload = json.loads(path.read_text())
        entry: dict[str, Any] = {"model": payload["model"]}
        for kind in ("bbox", "segm"):
            preds = to_coco_predictions(payload, gt, kind)
            entry[kind] = coco_evaluate(gt, preds, kind)
            entry[f"{kind}_count"] = len(preds)
        times = [i["inference_ms"] for i in payload["images"] if not i["error"]]
        times.sort()
        entry["latency"] = {
            "mean_ms": sum(times) / len(times) if times else None,
            "median_ms": times[len(times) // 2] if times else None,
            "p90_ms": times[int(len(times) * 0.9)] if times else None,
            "fps": 1000.0 / (sum(times) / len(times)) if times else None,
        }
        summary[spec.key] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
