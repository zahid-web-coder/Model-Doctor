"""Evaluate a saved Model Doctor run and persist its mAP and AR.

Answers *how good is this model, really* for a run that already exists, using
the shared COCO evaluator rather than whichever validator the model's own
library ships. Both detector families take exactly this path — the point of the
exercise is that neither gets its own.

**mAP cannot be recovered from stored findings.** A run persists detections
that already passed its confidence threshold and were already matched to
ground truth; on this project's own data that is 252 of 5,454 predictions for
one model and 224 of 581 for the other. mAP integrates over the whole
precision/recall curve, so scoring the survivors would measure the operating
point, not the model, and would flatter whichever model is more confident. This
script therefore re-runs inference at
:data:`model_doctor.app.evaluation.SWEEP_CONFIDENCE`.

**Provenance is enforced, not assumed.** The checkpoint on disk is re-hashed
and compared with the SHA the run recorded. A mismatch aborts: attributing a
score to weights that have since changed is worse than having no score, because
it is indistinguishable from a real one.

Usage::

    python -m model_doctor.app.evaluate_run --run-id 1
    python -m model_doctor.app.evaluate_run --run-id 1 --db db/columns_eval.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_doctor import config
from model_doctor.app import evaluation, storage
from model_doctor.app.detectors import build_detector, detect_family
from model_doctor.app.storage import file_sha256
from model_doctor.utils.dataset import load_dataset_config
from model_doctor.utils.ground_truth import open_ground_truth
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

EVALUATOR_NAME = "pycocotools COCOeval"

# What the segmentation score was computed from. Recorded because it is a real
# methodological choice: the model's own multi-component raster, not the
# single-component polygon Model Doctor stores for display.
MASK_SOURCE = "model raster (all components), RLE-encoded"


def evaluator_version() -> str | None:
    """Return the installed pycocotools version, or ``None`` if unknown."""
    try:
        from importlib.metadata import version

        return version("pycocotools")
    except Exception:  # pragma: no cover - best-effort provenance
        return None


def verify_checkpoint(run: storage.RunRecord) -> Path:
    """Return the run's checkpoint path, or abort if it is not the same file.

    Raises:
        SystemExit: If the checkpoint is missing or its SHA-256 no longer
            matches what the run recorded.
    """
    path = Path(run.model_path)
    if not path.is_file():
        raise SystemExit(
            f"Run {run.id} names a checkpoint that is not on this machine:\n"
            f"  {path}\n"
            "Evaluation must run against the exact weights the run used."
        )

    actual = file_sha256(path)
    if actual != run.model_sha256:
        raise SystemExit(
            f"Checkpoint SHA-256 does not match run {run.id}.\n"
            f"  recorded: {run.model_sha256}\n"
            f"  on disk : {actual}\n"
            "Refusing to attribute an evaluation to weights that have changed."
        )
    logger.info("Checkpoint verified against run %d (%s)", run.id, actual[:12])
    return path


def collect_predictions(
    run: storage.RunRecord, weights: Path, images_dir: Path
) -> list[evaluation.PredictionRecord]:
    """Run the run's own model over its own split at the sweep confidence.

    The family is read from the checkpoint rather than taken from
    ``config.DETECTOR_FAMILY``: the environment says what a *new* run would
    use, and this is an old one. Building an RF-DETR run with the YOLO loader
    because the shell happened to be configured for YOLO would fail loudly at
    best and mis-score at worst.
    """
    family = detect_family(weights)
    logger.info("Run %d used the %s family", run.id, family)
    detector = build_detector(
        family,
        model_path=str(weights),
        confidence=evaluation.SWEEP_CONFIDENCE,
        image_size=run.image_size,
        keep_raw_masks=True,
    )
    detector.load()

    images = evaluation.list_images(images_dir)
    logger.info(
        "Evaluating %d image(s) at confidence %.3f, image size %d",
        len(images),
        evaluation.SWEEP_CONFIDENCE,
        run.image_size,
    )

    records: list[evaluation.PredictionRecord] = []
    for index, image_path in enumerate(images, start=1):
        prediction = detector.predict_image(image_path, save_annotated=False)
        record = evaluation.PredictionRecord(
            file_name=image_path.name,
            width=prediction.image_width,
            height=prediction.image_height,
            inference_ms=prediction.inference_ms,
            error=prediction.error,
        )
        for detection in prediction.detections:
            record.boxes.append(
                (detection.x1, detection.y1, detection.x2, detection.y2)
            )
            record.scores.append(float(detection.confidence))
            record.class_ids.append(int(detection.class_id))
            record.mask_rles.append(detection.mask_rle)
        records.append(record)
        if index % 25 == 0 or index == len(images):
            logger.info("  [%d/%d] predicted", index, len(images))
    return records


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument(
        "--db", type=Path, default=None, help="Database. Defaults to config.DB_PATH."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one run and persist the result."""
    args = build_parser().parse_args(argv)
    db_path = args.db or config.DB_PATH

    with storage.connect(db_path) as connection:
        run = next(
            (r for r in storage.list_runs(connection) if r.id == args.run_id), None
        )
        if run is None:
            raise SystemExit(f"No run {args.run_id} in {db_path}")

        weights = verify_checkpoint(run)
        dataset = load_dataset_config(Path(run.dataset_yaml))
        images_dir = dataset.splits.get(run.split)
        if images_dir is None:
            raise SystemExit(
                f"Split '{run.split}' is not present in {run.dataset_yaml}. "
                f"Available: {', '.join(dataset.splits) or 'none'}"
            )

        # Built by whichever reader the split's format calls for. A COCO split
        # hands back its own annotations — boxes, areas and segmentation exactly
        # as annotated — rather than a conversion of them, so the evaluator
        # scores what the dataset says rather than a round trip through another
        # format.
        source = open_ground_truth(images_dir, dataset.class_names)
        ground_truth = source.coco_ground_truth(images_dir)
        logger.info(
            "Ground truth (%s): %d image(s), %d annotation(s), %d class(es)",
            source.format,
            len(ground_truth["images"]),
            len(ground_truth["annotations"]),
            len(ground_truth["categories"]),
        )

        records = collect_predictions(run, weights, images_dir)
        results = evaluation.evaluate_predictions(records, ground_truth)
        if not results:
            raise SystemExit("The model produced no predictions to evaluate.")

        version = evaluator_version()
        for result in results:
            storage.save_evaluation(
                connection,
                run_id=run.id,
                task=result.task,
                evaluator=EVALUATOR_NAME,
                evaluator_version=version,
                sweep_confidence=evaluation.SWEEP_CONFIDENCE,
                iou_thresholds=evaluation.IOU_THRESHOLDS,
                max_detections=evaluation.MAX_DETECTIONS,
                ground_truth=(
                    f"{source.format.upper()} annotations, "
                    f"split '{run.split}'"
                ),
                gt_images=len(ground_truth["images"]),
                gt_annotations=len(ground_truth["annotations"]),
                prediction_count=result.prediction_count,
                metrics=result.metrics,
                mask_source=(
                    MASK_SOURCE if result.task == evaluation.TASK_SEGM else None
                ),
            )
            logger.info(
                "%s: mAP@50 %.4f  mAP@50:95 %.4f  (%d predictions)",
                result.task,
                result.metrics.get("map50", float("nan")),
                result.metrics.get("map50_95", float("nan")),
                result.prediction_count,
            )

    logger.info("Saved evaluation for run %d to %s", args.run_id, db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
