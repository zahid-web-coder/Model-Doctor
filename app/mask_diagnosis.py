"""Re-examine a saved run's findings at outline level rather than box level.

**The gap this closes.** On the reference model, box-level diagnosis reports
`door` and `door_frame` failing almost identically — mean box IoU 0.878 against
0.877 — while the model's own mask metric says otherwise: mAP50-95 of 0.599
against 0.246. A door frame is a thin rectangular annulus. Its bounding box is
easy and its outline is not, so a few pixels of boundary error destroy mask IoU
while barely moving box IoU. Measuring only boxes reports the dominant failure
mode of thin-structure classes as no failure at all (D-022).

**Why this pass re-runs inference.** Every other analysis module reads a saved
run and writes a new table. This one cannot: `findings` stores ground-truth
outlines but not predicted ones, and `findings` is frozen (D-020). So the model
is run again over the same images, at the same size and threshold the run
recorded, and its outlines are matched back to the findings already stored.

**Why pairs are not re-matched on mask IoU.** The box-level pairing is left
exactly as it was, and the outline is measured on that same pair. Re-pairing
would produce a second, different set of findings and lose the comparison that
makes this useful — *this* finding, which the box called correct, has an
outline that is not. That disagreement is the output (D-036).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from app import storage
from app.detectors import SUPPORTED_FAMILIES, build_detector, detect_family
from app.inference import Detector
from utils.exceptions import ModelDoctorError
from utils.geometry import BoxGeometryMixin, box_iou
from utils.logging_utils import get_logger
from utils.masks import mask_iou

logger = get_logger(__name__)


class MaskDiagnosisError(ModelDoctorError):
    """Mask-level diagnosis could not run, with a reason the caller can act on."""


# Outline-level verdicts. Deliberately a smaller vocabulary than the box-level
# `Outcome`: this pass re-measures an existing pair, so it can only say how
# well the outlines agree — not whether the object was found, which the box
# pass already decided.
MASK_CORRECT: str = "correct"
MASK_POOR: str = "poor_localization"
MASK_NO_OVERLAP: str = "no_overlap"

# A re-run prediction is the same detection as a stored one when their boxes
# coincide almost exactly. Inference is deterministic given the same weights,
# image size and threshold — all recorded on the run — so a true match scores
# far above this. The bar is high on purpose: attributing an outline to the
# wrong finding would corrupt the comparison silently.
_IDENTITY_IOU: float = 0.98


@dataclass(frozen=True)
class MaskDiagnosisReport:
    """What one outline-level pass measured."""

    run_id: int
    findings_examined: int
    measured: int
    unmeasurable: int
    disagreements: int
    images_unreadable: int
    mean_mask_iou: float | None
    mean_box_iou: float | None

    def describe(self) -> str:
        """Render the report, leading with the disagreement that motivates it."""
        width = 70
        lines = [
            "",
            "Mask-level diagnosis",
            "=" * width,
            f"  Run                    : {self.run_id}",
            f"  Findings examined      : {self.findings_examined}",
            f"  Outlines measured      : {self.measured}",
            f"  Not measurable         : {self.unmeasurable}",
        ]
        if self.images_unreadable:
            lines.append(f"  Images unreadable      : {self.images_unreadable}")
        if self.mean_box_iou is not None and self.mean_mask_iou is not None:
            lines.append("")
            lines.append(f"  Mean box IoU           : {self.mean_box_iou:.3f}")
            lines.append(f"  Mean mask IoU          : {self.mean_mask_iou:.3f}")
        lines.append("")
        lines.append(
            f"  Correct by box, not by outline : {self.disagreements}"
        )
        lines.append(
            "  Those are failures box-level diagnosis cannot see."
        )
        lines.append("=" * width)
        lines.append("")
        return "\n".join(lines)


def classify_mask(score: float | None) -> str | None:
    """Turn an outline overlap into a verdict, using the run's own thresholds.

    The same two thresholds the box pass uses, so "correct" means the same
    strength of agreement in both and the two verdicts are comparable.

    Args:
        score: Outline overlap, or ``None`` when it could not be measured.

    Returns:
        A verdict, or ``None`` when nothing was measured. ``None`` is not a
        failure — it is the absence of a measurement.
    """
    if score is None:
        return None
    if score >= config.MATCH_IOU_THRESHOLD:
        return MASK_CORRECT
    if score >= config.LOCALIZATION_IOU_FLOOR:
        return MASK_POOR
    return MASK_NO_OVERLAP


def _stored_box(row: Any) -> tuple[float, float, float, float] | None:
    """Return a finding's predicted box, or ``None`` when it has no prediction."""
    values = [row[f"pred_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
    return tuple(values) if values[0] is not None else None  # type: ignore[return-value]


@dataclass(frozen=True)
class _Box(BoxGeometryMixin):
    """Adapter giving raw corners the interface ``box_iou`` expects.

    Uses the shared mixin rather than re-deriving `area`, so this cannot drift
    from the geometry every other comparison in the project uses (D-007).
    """

    x1: float
    y1: float
    x2: float
    y2: float


def diagnose_masks(
    connection: Any,
    run_id: int,
    detector: Detector | None = None,
    image_size: int | None = None,
) -> MaskDiagnosisReport:
    """Measure outline agreement for a saved run's findings, and store it.

    Args:
        connection: An open database connection.
        run_id: Run to re-examine. Its findings must already be saved.
        detector: A loaded detector. One is created and loaded if omitted.
        image_size: Expected inference size. Defaults to the size the run
            recorded. Used to verify this process can reproduce the run's
            predictions, not to change them — see below.

    Returns:
        A description of what was measured.

    Raises:
        MaskDiagnosisError: If the run has no findings, if this process is
            configured for a different image size than the run used, or if the
            model produces no outlines at all — the last meaning it is not a
            segmentation model and there is nothing here to measure.
    """
    run = storage.load_run(connection, run_id)
    if run is None:
        raise MaskDiagnosisError(f"Run {run_id} does not exist.")

    rows = storage.load_findings_for_embedding(connection, run_id, failures_only=False)
    if not rows:
        raise MaskDiagnosisError(
            f"Run {run_id} has no findings. Diagnose and save it first."
        )

    engine = detector or Detector()
    engine.load()  # idempotent: returns immediately if already loaded

    # Re-running at a different size would produce different boxes and
    # different outlines, and the comparison against stored findings would be
    # against predictions the run never made. Say so rather than quietly
    # measuring the wrong thing.
    #
    # The comparison is against the engine that is about to run, not against
    # the global MD_IMGSZ. Those are the same thing only when the caller took
    # the default YOLO detector; a second detector family has its own
    # configured size, and checking the global would reject a perfectly
    # consistent RF-DETR run while advising a fix that mis-sizes YOLO.
    expected = image_size or run.image_size
    if expected != engine.image_size:
        raise MaskDiagnosisError(
            f"Run {run_id} was diagnosed at image size {expected}, but this "
            f"detector is configured for {engine.image_size}. Re-running at a "
            "different size would not reproduce its predictions. Configure the "
            f"detector for {expected} and try again."
        )

    by_image: dict[str, list[Any]] = {}
    for row in rows:
        by_image.setdefault(str(row["path"]), []).append(row)

    entries: list[tuple[int, float | None, str | None, object]] = []
    measured = unmeasurable = unreadable = disagreements = 0
    outlines_seen = False

    for path, findings in by_image.items():
        try:
            prediction = engine.predict_image(Path(path), save_annotated=False)
        except ModelDoctorError as exc:
            unreadable += 1
            logger.warning("%s", exc)
            for row in findings:
                entries.append((int(row["finding_id"]), None, None, None))
                unmeasurable += 1
            continue

        outlines_seen = outlines_seen or any(
            d.polygon is not None for d in prediction.detections
        )

        for row in findings:
            finding_id = int(row["finding_id"])
            stored = _stored_box(row)
            truth_polygon = row["truth_polygon"]

            predicted_outline = None
            if stored is not None:
                # Re-attach the outline to the finding whose box it reproduces.
                candidate = max(
                    prediction.detections,
                    key=lambda d, s=stored: box_iou(d, _Box(*s)),
                    default=None,
                )
                if candidate is not None and box_iou(
                    candidate, _Box(*stored)
                ) >= _IDENTITY_IOU:
                    predicted_outline = candidate.polygon

            truth_points = None
            if truth_polygon:
                import json as _json

                truth_points = _json.loads(truth_polygon)

            score = mask_iou(predicted_outline, truth_points)
            verdict = classify_mask(score)
            entries.append((finding_id, score, verdict, predicted_outline))

            if score is None:
                unmeasurable += 1
            else:
                measured += 1
                if row["outcome"] == "correct" and verdict != MASK_CORRECT:
                    disagreements += 1

    if not outlines_seen:
        raise MaskDiagnosisError(
            "The model produced no outlines, so there is nothing to measure at "
            "mask level. This is expected for a detection model; mask "
            "diagnosis applies to segmentation models only."
        )

    written = storage.save_mask_findings(connection, run_id, entries)
    logger.info("Measured %d outline(s) for run %d", written, run_id)

    # Both means come from one query over the pairs where *both* measurements
    # exist, so the two numbers describe the same findings and can be compared.
    # Averaging separately populated sets would contrast different objects.
    averages = connection.execute(
        """
        SELECT AVG(f.iou) AS mean_box, AVG(m.mask_iou) AS mean_mask
        FROM mask_findings m JOIN findings f ON f.id = m.finding_id
        WHERE m.run_id = ? AND m.mask_iou IS NOT NULL AND f.iou IS NOT NULL
        """,
        (run_id,),
    ).fetchone()

    return MaskDiagnosisReport(
        run_id=run_id,
        findings_examined=len(rows),
        measured=measured,
        unmeasurable=unmeasurable,
        disagreements=disagreements,
        images_unreadable=unreadable,
        mean_mask_iou=averages["mean_mask"],
        mean_box_iou=averages["mean_box"],
    )


def format_class_comparison(connection: Any, run_id: int) -> str:
    """Contrast box and outline agreement per class — the D-022 table, measured."""
    rows = connection.execute(
        """
        SELECT f.class_name,
               COUNT(m.mask_iou) AS measured,
               ROUND(AVG(f.iou), 3) AS mean_box,
               ROUND(AVG(m.mask_iou), 3) AS mean_mask
        FROM mask_findings m JOIN findings f ON f.id = m.finding_id
        WHERE m.run_id = ? AND m.mask_iou IS NOT NULL AND f.iou IS NOT NULL
        GROUP BY f.class_name
        ORDER BY mean_mask
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return "\n  No paired findings to compare.\n"

    lines = ["", "Box against outline, per class", "-" * 62]
    lines.append(f"  {'CLASS':<20}{'PAIRS':>8}{'BOX IoU':>12}{'MASK IoU':>12}")
    for row in rows:
        lines.append(
            f"  {row['class_name']:<20}{row['measured']:>8}"
            f"{row['mean_box']:>12.3f}{row['mean_mask']:>12.3f}"
        )
    lines.append("-" * 62)
    lines.append(
        "  A class whose outline score falls far below its box score is one\n"
        "  box-level diagnosis reports as healthier than it is."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the mask-diagnosis command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-masks",
        description="Re-examine a saved run's findings at outline level.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=str, help="Database path override.")
    parser.add_argument(
        "--imgsz", type=int, help="Inference size. Defaults to the run's own."
    )
    parser.add_argument(
        "--model", help="Weights override. Defaults to the run's own checkpoint."
    )
    parser.add_argument(
        "--detector",
        choices=SUPPORTED_FAMILIES,
        help=(
            "Detector family. Defaults to whatever the run's own checkpoint "
            "says it is, so this normally needs no flag."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to measure.
    """
    args = build_parser().parse_args(argv)

    with storage.connect(Path(args.db) if args.db else None) as connection:
        run_id = args.run
        if run_id is None:
            runs = storage.list_runs(connection)
            if not runs:
                logger.error(
                    "No runs found. Run: python -m app.diagnosis --split test --save"
                )
                return 1
            run_id = runs[0].id
            logger.info("Using newest run %d", run_id)

        try:
            # Everything about the detector comes from the run itself unless
            # overridden: the weights, the family and the size.
            #
            # This pass re-runs inference and matches the outlines back to
            # findings that are already stored, so it has to reproduce the
            # run's own predictions. Taking the weights from the run rather
            # than from discovery is what makes that true — discovery scans the
            # models directory and would happily pick a different checkpoint,
            # producing outlines for predictions this run never made.
            #
            # The family is read from the checkpoint rather than assumed, so
            # the right loader is chosen without a flag and without a new
            # column in the schema.
            saved = storage.load_run(connection, run_id)
            weights = args.model or (saved.model_path if saved else None)
            family = args.detector or (detect_family(weights) if weights else None)
            engine = build_detector(
                family,
                model_path=str(weights) if weights else None,
                image_size=args.imgsz or (saved.image_size if saved else None),
            )
            report = diagnose_masks(
                connection, run_id, image_size=args.imgsz, detector=engine
            )
        except MaskDiagnosisError as exc:
            logger.error("%s", exc)
            return 1

        comparison = format_class_comparison(connection, run_id)

    print(report.describe())
    print(comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
