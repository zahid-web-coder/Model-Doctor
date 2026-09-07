"""The diagnosis engine — why a model's predictions failed.

Metrics tell you *how much* a model fails. This module answers *which* objects
it got wrong and *in what way*, which is the whole reason Model Doctor exists.

Every prediction and every ground-truth annotation on an image ends up in
exactly one of five outcomes:

============================  ==============================================
Correct                      Matched, right class, well localised.
Wrong class                  Matched the right object, named it wrong.
Poor localisation            Found the object, outlined it badly.
False positive               Predicted something that is not there.
False negative               Missed something that is there.
============================  ==============================================

**The design choice that defines this module.** A prediction overlapping a
label by less than the match threshold could be reported two ways: as one
*poor localisation*, or as a false positive plus a false negative. Evaluation
metrics choose the latter — below threshold means no match, so both sides count
as errors. This module chooses the former.

The reason is the project's purpose. "Found it, outlined it badly" explains
what happened. "One spurious detection and one miss" describes a single object
as two unrelated failures and hides the cause. Model Doctor exists to explain,
not to reproduce a benchmark.

**Consequence, stated plainly: the false-positive and false-negative counts
here will not equal those implied by mAP.** That is deliberate. Do not compare
them directly. See DECISIONS D-017.

Mask support is architecture-ready but not implemented: the matcher accepts a
similarity function, so switching to mask overlap is a one-line change at the
call site. Nothing here computes mask IoU.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app.detectors import SUPPORTED_FAMILIES, build_detector
from app.inference import Detector
from utils.annotations import ObjectAnnotation
from utils.dataset import DatasetConfig, load_dataset_config, load_ground_truth
from utils.exceptions import DatasetConfigError, ModelLoadError, ResourceNotFoundError
from utils.geometry import box_iou
from utils.logging_utils import get_logger
from utils.matching import SimilarityFn, match_annotations
from utils.resources import find_images, format_report, verify_all

logger = get_logger(__name__)


#: Which distribution actually runs inference, per detector family. An
#: RF-DETR run is not executed by Ultralytics, and recording Ultralytics'
#: version against one would be precisely the plausible-looking substitute
#: this column exists to avoid.
_INFERENCE_LIBRARIES: dict[str, str] = {"yolo": "ultralytics", "rfdetr": "rfdetr"}


def _inference_library_version(family: str | None) -> str | None:
    """Return the version of the library that ran inference, or ``None``.

    Deliberately not the version stored inside a checkpoint: that is the
    version that *trained* the weights and is routinely older than the one
    running them. Confusing the two would make a run look reproducible on a
    stack it never touched.

    The version is read from installed distribution metadata rather than a
    module attribute, because not every one of these packages defines
    ``__version__``.

    Args:
        family: Detector family as the caller gave it, resolved here the same
            way :func:`app.detectors.build_detector` resolves it so the two
            cannot disagree about which library a run used.

    Returns:
        ``"<distribution> <version>"``, or ``None`` when the family is
        unknown or the version cannot be read.
    """
    resolved = (family or config.DETECTOR_FAMILY).strip().lower()
    package = _INFERENCE_LIBRARIES.get(resolved)
    if package is None:
        return None
    try:
        return f"{package} {metadata.version(package)}"
    except Exception:  # pragma: no cover - depends on the environment
        return None


class Outcome(Enum):
    """What happened to one prediction or one ground-truth annotation.

    ``CORRECT`` is included so that every annotation on an image is accounted
    for. A diagnosis that only listed failures could not report a rate, and
    "3 failures" means nothing without knowing whether there were 4 objects or
    400.
    """

    CORRECT = "correct"
    WRONG_CLASS = "wrong_class"
    POOR_LOCALIZATION = "poor_localization"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"

    @property
    def is_failure(self) -> bool:
        """Return whether this outcome represents a mistake."""
        return self is not Outcome.CORRECT


@dataclass(frozen=True)
class Finding:
    """One outcome, with the evidence that produced it.

    Both annotations are kept, not just the failing one. Explaining a wrong
    class means showing what was predicted *and* what was there; explaining
    poor localisation means showing both boxes and the overlap between them.
    A record that dropped either side would state a conclusion without its
    evidence.

    Attributes:
        outcome: The classification.
        prediction: The predicted annotation, or ``None`` for a false negative.
        truth: The ground-truth annotation, or ``None`` for a false positive.
        iou: Overlap between the pair, or ``None`` when there is no pair.
    """

    outcome: Outcome
    prediction: ObjectAnnotation | None = None
    truth: ObjectAnnotation | None = None
    iou: float | None = None

    @property
    def is_failure(self) -> bool:
        """Return whether this finding is a mistake."""
        return self.outcome.is_failure

    @property
    def class_name(self) -> str:
        """Return the class this finding concerns.

        Ground truth is preferred when present, because per-class statistics
        should attribute a failure to the class that *should* have been
        found — otherwise a wrong-class prediction would be counted against the
        class the model wrongly guessed rather than the one it actually missed.
        """
        if self.truth is not None:
            return self.truth.class_name
        if self.prediction is not None:
            return self.prediction.class_name
        return "unknown"

    def describe(self) -> str:
        """Return a one-line human explanation of this finding."""
        if self.outcome is Outcome.CORRECT:
            return f"correct '{self.class_name}' (IoU {self.iou:.2f})"
        if self.outcome is Outcome.WRONG_CLASS:
            predicted = self.prediction.class_name if self.prediction else "?"
            return (
                f"predicted '{predicted}' where '{self.class_name}' is "
                f"(IoU {self.iou:.2f})"
            )
        if self.outcome is Outcome.POOR_LOCALIZATION:
            return (
                f"found '{self.class_name}' but IoU is only {self.iou:.2f}, "
                f"below the {config.MATCH_IOU_THRESHOLD:.2f} match threshold"
            )
        if self.outcome is Outcome.FALSE_POSITIVE:
            confidence = self.prediction.confidence if self.prediction else None
            suffix = f" at {confidence:.2f} confidence" if confidence else ""
            return f"predicted '{self.class_name}' where nothing is{suffix}"
        return f"missed a '{self.class_name}' entirely"


@dataclass(frozen=True)
class ImageDiagnosis:
    """The complete diagnosis for one image.

    Attributes:
        image_path: The image this concerns.
        predictions: Every prediction considered.
        truths: Every ground-truth annotation considered.
        findings: One entry per prediction and per ground truth, so nothing is
            unaccounted for.
        image_width: Source width in pixels, ``0`` when unknown.
        image_height: Source height in pixels, ``0`` when unknown.
        error: Set when the image could not be diagnosed, leaving a batch
            otherwise intact.
    """

    image_path: Path
    predictions: Sequence[ObjectAnnotation] = field(default_factory=list)
    truths: Sequence[ObjectAnnotation] = field(default_factory=list)
    findings: Sequence[Finding] = field(default_factory=list)
    # Dimensions are carried so that persisted findings can be interpreted
    # without re-opening the image: a pixel box means nothing without the frame
    # it sits in. Defaulted, so callers that do not know them still work.
    image_width: int = 0
    image_height: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether this image was diagnosed without error."""
        return self.error is None

    @property
    def failures(self) -> list[Finding]:
        """Return only the findings that represent mistakes."""
        return [f for f in self.findings if f.is_failure]

    @property
    def matches(self) -> list[Finding]:
        """Return findings that paired a prediction with a ground truth."""
        return [
            f for f in self.findings if f.prediction is not None and f.truth is not None
        ]

    @property
    def counts(self) -> dict[Outcome, int]:
        """Return how many findings fall into each outcome."""
        tally = dict.fromkeys(Outcome, 0)
        for finding in self.findings:
            tally[finding.outcome] += 1
        return tally

    @property
    def mean_iou(self) -> float | None:
        """Return average overlap across paired findings, or ``None``."""
        pairs = [f.iou for f in self.matches if f.iou is not None]
        return sum(pairs) / len(pairs) if pairs else None

    @property
    def is_clean(self) -> bool:
        """Return whether this image produced no failures at all."""
        return self.succeeded and not self.failures


@dataclass(frozen=True)
class ClassStatistics:
    """Per-class outcome totals.

    This is where a dataset-wide diagnosis becomes actionable: an aggregate
    failure count says the model is imperfect, whereas a per-class breakdown
    says *which* class is dragging it down and *how* it fails.
    """

    class_name: str
    counts: dict[Outcome, int] = field(default_factory=dict)
    iou_total: float = 0.0
    iou_samples: int = 0

    @property
    def instances(self) -> int:
        """Return ground-truth instances seen for this class."""
        return (
            self.counts.get(Outcome.CORRECT, 0)
            + self.counts.get(Outcome.WRONG_CLASS, 0)
            + self.counts.get(Outcome.POOR_LOCALIZATION, 0)
            + self.counts.get(Outcome.FALSE_NEGATIVE, 0)
        )

    @property
    def failures(self) -> int:
        """Return the number of failing findings for this class."""
        return sum(n for outcome, n in self.counts.items() if outcome.is_failure)

    @property
    def mean_iou(self) -> float | None:
        """Return mean overlap across this class's paired findings."""
        return self.iou_total / self.iou_samples if self.iou_samples else None


@dataclass(frozen=True)
class DatasetDiagnosis:
    """Aggregate diagnosis across many images.

    Attributes:
        diagnoses: Per-image results, kept so any headline number can be traced
            back to the specific images that produced it.
    """

    diagnoses: Sequence[ImageDiagnosis] = field(default_factory=list)

    @property
    def succeeded(self) -> list[ImageDiagnosis]:
        """Return the image diagnoses that completed."""
        return [d for d in self.diagnoses if d.succeeded]

    @property
    def failed(self) -> list[ImageDiagnosis]:
        """Return the image diagnoses that errored."""
        return [d for d in self.diagnoses if not d.succeeded]

    @property
    def counts(self) -> dict[Outcome, int]:
        """Return outcome totals across every diagnosed image."""
        tally = dict.fromkeys(Outcome, 0)
        for diagnosis in self.succeeded:
            for outcome, number in diagnosis.counts.items():
                tally[outcome] += number
        return tally

    @property
    def total_predictions(self) -> int:
        """Return how many predictions were examined."""
        return sum(len(d.predictions) for d in self.succeeded)

    @property
    def total_truths(self) -> int:
        """Return how many ground-truth annotations were examined."""
        return sum(len(d.truths) for d in self.succeeded)

    @property
    def mean_iou(self) -> float | None:
        """Return mean overlap across every paired finding."""
        values = [
            f.iou for d in self.succeeded for f in d.matches if f.iou is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def clean_images(self) -> int:
        """Return how many images produced no failures."""
        return sum(1 for d in self.succeeded if d.is_clean)

    @property
    def per_class(self) -> dict[str, ClassStatistics]:
        """Return per-class statistics, keyed by class name."""
        tallies: dict[str, Counter] = defaultdict(Counter)
        iou_totals: dict[str, float] = defaultdict(float)
        iou_counts: dict[str, int] = defaultdict(int)

        for diagnosis in self.succeeded:
            for finding in diagnosis.findings:
                name = finding.class_name
                tallies[name][finding.outcome] += 1
                if finding.iou is not None:
                    iou_totals[name] += finding.iou
                    iou_counts[name] += 1

        return {
            name: ClassStatistics(
                class_name=name,
                counts=dict(counter),
                iou_total=iou_totals[name],
                iou_samples=iou_counts[name],
            )
            for name, counter in sorted(tallies.items())
        }

    def worst_images(self, limit: int = 10) -> list[ImageDiagnosis]:
        """Return the images with the most failures, worst first.

        The point of the whole pipeline: turning "recall is low" into a list of
        specific images to go and look at.
        """
        ranked = sorted(self.succeeded, key=lambda d: len(d.failures), reverse=True)
        return [d for d in ranked if d.failures][:limit]


def diagnose_image(
    image_path: Path,
    predictions: Sequence[ObjectAnnotation],
    truths: Sequence[ObjectAnnotation],
    match_threshold: float | None = None,
    localization_floor: float | None = None,
    similarity: SimilarityFn = box_iou,
    image_width: int = 0,
    image_height: int = 0,
) -> ImageDiagnosis:
    """Classify every prediction and ground truth for one image.

    Runs two matching passes. The first accepts pairs at the match threshold —
    these are hits, correct or wrong-class. The second re-examines whatever is
    left at a much lower floor, catching predictions that landed on an object
    but outlined it poorly. Anything still unpaired is a genuine invention or a
    genuine miss.

    The second pass is what separates this from a metrics calculation, and it
    is why the counts here differ from mAP's.

    Args:
        image_path: The image being diagnosed, carried through for traceability.
        predictions: Predicted annotations.
        truths: Ground-truth annotations.
        match_threshold: Overlap at which a pair counts as correctly localised.
            Defaults to :data:`config.MATCH_IOU_THRESHOLD`.
        localization_floor: Overlap below which a pair is not credible at all.
            Defaults to :data:`config.LOCALIZATION_IOU_FLOOR`.
        similarity: Comparison function, defaulting to box IoU. Passing a
            mask-based function is the extension point for segmentation
            analysis; nothing here assumes boxes.
        image_width: Source width in pixels, carried into the result so stored
            findings can be interpreted without re-opening the image.
        image_height: Source height in pixels, carried for the same reason.

    Returns:
        An :class:`ImageDiagnosis` in which every prediction and every ground
        truth appears exactly once.
    """
    threshold = (
        match_threshold if match_threshold is not None else config.MATCH_IOU_THRESHOLD
    )
    floor = (
        localization_floor
        if localization_floor is not None
        else config.LOCALIZATION_IOU_FLOOR
    )

    findings: list[Finding] = []

    # Pass 1 — confident pairings. ignore_class is on so that a prediction
    # landing squarely on an object but naming it wrong is caught as a wrong
    # class, rather than being lost as an unrelated false positive plus miss.
    primary = match_annotations(
        predictions, truths, threshold=threshold, similarity=similarity
    )
    for match in primary.matches:
        findings.append(
            Finding(
                outcome=Outcome.CORRECT if match.same_class else Outcome.WRONG_CLASS,
                prediction=match.prediction,
                truth=match.truth,
                iou=match.similarity,
            )
        )

    # Pass 2 — the leftovers, at a much lower bar. Re-indexed against only the
    # still-unpaired annotations so the second pass cannot reuse anything.
    leftover_predictions = [
        predictions[i] for i in primary.unmatched_prediction_indices
    ]
    leftover_truths = [truths[i] for i in primary.unmatched_truth_indices]

    secondary = match_annotations(
        leftover_predictions, leftover_truths, threshold=floor, similarity=similarity
    )
    for match in secondary.matches:
        # A weak overlap with the wrong class is reported as a wrong class:
        # naming the object incorrectly is the more consequential error, and
        # reporting it as poor localisation would hide it.
        findings.append(
            Finding(
                outcome=(
                    Outcome.POOR_LOCALIZATION
                    if match.same_class
                    else Outcome.WRONG_CLASS
                ),
                prediction=match.prediction,
                truth=match.truth,
                iou=match.similarity,
            )
        )

    for index in secondary.unmatched_prediction_indices:
        findings.append(
            Finding(
                outcome=Outcome.FALSE_POSITIVE,
                prediction=leftover_predictions[index],
            )
        )
    for index in secondary.unmatched_truth_indices:
        findings.append(
            Finding(outcome=Outcome.FALSE_NEGATIVE, truth=leftover_truths[index])
        )

    return ImageDiagnosis(
        image_path=image_path,
        predictions=list(predictions),
        truths=list(truths),
        findings=findings,
        image_width=image_width,
        image_height=image_height,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
_OUTCOME_LABELS: dict[Outcome, str] = {
    Outcome.CORRECT: "Correct",
    Outcome.WRONG_CLASS: "Wrong class",
    Outcome.POOR_LOCALIZATION: "Poor localisation",
    Outcome.FALSE_POSITIVE: "False positive",
    Outcome.FALSE_NEGATIVE: "False negative",
}


def format_image_diagnosis(diagnosis: ImageDiagnosis) -> str:
    """Render one image's diagnosis, listing each finding with its evidence."""
    header = f"\n{diagnosis.image_path.name}"
    if not diagnosis.succeeded:
        return f"{header}\n  ERROR: {diagnosis.error}\n"

    lines = [
        header,
        f"  {len(diagnosis.predictions)} prediction(s), "
        f"{len(diagnosis.truths)} ground truth(s), "
        f"{len(diagnosis.failures)} failure(s)",
        "-" * 78,
    ]
    if not diagnosis.findings:
        lines.append("  Nothing predicted and nothing to find.")
    for finding in diagnosis.findings:
        lines.append(
            f"  {_OUTCOME_LABELS[finding.outcome]:<18} {finding.describe()}"
        )
    lines.append("-" * 78)
    return "\n".join(lines) + "\n"


def format_dataset_diagnosis(summary: DatasetDiagnosis, worst: int = 10) -> str:
    """Render the dataset-wide diagnosis report."""
    counts = summary.counts
    total_findings = sum(counts.values())
    failures = sum(n for outcome, n in counts.items() if outcome.is_failure)

    lines = ["", "Model Doctor — diagnosis", "=" * 72]
    lines.append(f"  Images diagnosed   : {len(summary.succeeded)}")
    if summary.failed:
        lines.append(f"  Images failed      : {len(summary.failed)}")
    lines.append(f"  Images with no fault: {summary.clean_images}")
    lines.append(f"  Predictions        : {summary.total_predictions}")
    lines.append(f"  Ground truths      : {summary.total_truths}")
    mean_iou = summary.mean_iou
    lines.append(
        f"  Mean IoU (matched) : {mean_iou:.4f}" if mean_iou is not None else
        "  Mean IoU (matched) : n/a"
    )
    lines.append("")
    lines.append("  OUTCOME              COUNT     SHARE")
    for outcome in Outcome:
        number = counts[outcome]
        share = (100.0 * number / total_findings) if total_findings else 0.0
        lines.append(f"  {_OUTCOME_LABELS[outcome]:<18} {number:>7} {share:>8.1f}%")
    lines.append(f"  {'TOTAL FAILURES':<18} {failures:>7}")

    lines.append("")
    lines.append("  PER CLASS")
    lines.append(
        f"  {'CLASS':<16}{'INST':>6}{'OK':>6}{'WRONG':>7}{'POORLOC':>9}"
        f"{'FP':>6}{'FN':>6}{'MEAN IoU':>10}"
    )
    for name, stats in summary.per_class.items():
        iou = stats.mean_iou
        lines.append(
            f"  {name:<16}{stats.instances:>6}"
            f"{stats.counts.get(Outcome.CORRECT, 0):>6}"
            f"{stats.counts.get(Outcome.WRONG_CLASS, 0):>7}"
            f"{stats.counts.get(Outcome.POOR_LOCALIZATION, 0):>9}"
            f"{stats.counts.get(Outcome.FALSE_POSITIVE, 0):>6}"
            f"{stats.counts.get(Outcome.FALSE_NEGATIVE, 0):>6}"
            + (f"{iou:>10.3f}" if iou is not None else f"{'n/a':>10}")
        )

    worst_images = summary.worst_images(worst)
    if worst_images:
        lines.append("")
        lines.append(f"  WORST IMAGES (top {len(worst_images)} by failure count)")
        for diagnosis in worst_images:
            breakdown = Counter(
                _OUTCOME_LABELS[f.outcome] for f in diagnosis.failures
            )
            detail = ", ".join(f"{n} {label.lower()}" for label, n in breakdown.items())
            lines.append(
                f"    {len(diagnosis.failures):>3} — {diagnosis.image_path.name[:54]}"
            )
            lines.append(f"        {detail}")

    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def diagnose_split(
    detector: Detector,
    dataset: DatasetConfig,
    split: str,
    limit: int | None = None,
    match_iou: float | None = None,
) -> DatasetDiagnosis:
    """Run inference and diagnosis across every image in a dataset split.

    Ground truth is read per image using that image's own dimensions, because
    label files are normalised and only become pixel coordinates once the
    image size is known.

    A failure on one image is recorded and the run continues. A single
    unreadable file must not discard the diagnosis of a thousand others.

    Args:
        detector: A loaded (or loadable) detector.
        dataset: Parsed dataset descriptor, supplying class names and splits.
        split: Which split to diagnose.
        limit: Process at most this many images.
        match_iou: IoU at or above which a prediction counts as landing on a
            ground-truth box. ``None`` uses :data:`config.MATCH_IOU_THRESHOLD`.
            Threaded through so the ``--match-iou`` flag reaches the matcher
            rather than being accepted and discarded.

    Returns:
        A :class:`DatasetDiagnosis` covering every image attempted.
    """
    split_dir = dataset.splits.get(split)
    if split_dir is None:
        raise ResourceNotFoundError(
            f"Split '{split}' is not available. Declared: "
            f"{', '.join(dataset.splits) or 'none'}"
        )

    images = find_images(split_dir)
    if limit:
        images = images[:limit]
    logger.info("Diagnosing %d image(s) from split '%s'", len(images), split)

    diagnoses: list[ImageDiagnosis] = []
    for index, image_path in enumerate(images, start=1):
        prediction = detector.predict_image(image_path, save_annotated=False)
        if not prediction.succeeded:
            diagnoses.append(
                ImageDiagnosis(image_path=image_path, error=prediction.error)
            )
            continue

        truths = load_ground_truth(
            image_path,
            prediction.image_width,
            prediction.image_height,
            dataset.class_names,
        )
        diagnosis = diagnose_image(
            image_path,
            prediction.detections,
            truths,
            match_threshold=match_iou,
            image_width=prediction.image_width,
            image_height=prediction.image_height,
        )
        diagnoses.append(diagnosis)

        if index % 25 == 0 or index == len(images):
            logger.info("  [%d/%d] processed", index, len(images))

    return DatasetDiagnosis(diagnoses=diagnoses)


def build_parser() -> argparse.ArgumentParser:
    """Construct the diagnosis command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-diagnose",
        description="Explain why a detector's predictions fail.",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split to diagnose (default: test)."
    )
    parser.add_argument("--model", help="Path to specific .pt weights.")
    parser.add_argument(
        "--detector",
        choices=SUPPORTED_FAMILIES,
        help=(
            "Detector family. Defaults to MD_DETECTOR (currently "
            f"'{config.DETECTOR_FAMILY}'). Both families produce the same "
            "findings, so runs from either are directly comparable."
        ),
    )
    parser.add_argument("--conf", type=float, help="Confidence threshold override.")
    parser.add_argument("--imgsz", type=int, help="Inference image size override.")
    parser.add_argument(
        "--match-iou", type=float, help="IoU at which a pair counts as correct."
    )
    parser.add_argument("--limit", type=int, help="Diagnose at most N images.")
    parser.add_argument(
        "--worst", type=int, default=10, help="How many worst images to list."
    )
    parser.add_argument(
        "--show", type=int, default=0, help="Print per-image detail for the first N."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist this run to the SQLite diagnosis database.",
    )
    parser.add_argument(
        "--db", type=str, help="Database path override (default: db/model_doctor.db)."
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List stored runs and exit."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the diagnosis engine.

    Returns:
        ``0`` on success, ``1`` when a required resource is missing.
    """
    args = build_parser().parse_args(argv)
    config.ensure_output_dirs()

    # Imported here rather than at module scope: app.diagnosis must not depend
    # on app.storage, or the one-way layering documented in storage.py becomes
    # a cycle. The CLI orchestrates; the engine stays unaware of persistence.
    from app import storage

    db_path = Path(args.db) if args.db else None

    if args.list_runs:
        with storage.connect(db_path) as connection:
            print(storage.format_run_summary(storage.list_runs(connection)))
        return 0

    model_override = Path(args.model) if args.model else None
    statuses = verify_all(model_override)
    print(format_report(statuses))
    if not all(s.available for s in statuses if s.required):
        logger.error("Cannot diagnose until every required resource is present.")
        return 1

    try:
        dataset = load_dataset_config()
    except DatasetConfigError as exc:
        logger.error("%s", exc)
        return 1

    # The only line in the run pipeline that had to change. Everything after
    # this point — diagnose_split, the outcome taxonomy, the run context, the
    # findings — already works on any Detector, because it consumes Detection
    # objects and has never known which library produced them.
    detector = build_detector(
        args.detector,
        model_path=str(model_override) if model_override else None,
        confidence=args.conf,
        image_size=args.imgsz,
    )
    try:
        summary = diagnose_split(
            detector, dataset, args.split, limit=args.limit, match_iou=args.match_iou
        )
    except (ResourceNotFoundError, ModelLoadError) as exc:
        logger.error("%s", exc)
        return 1

    for diagnosis in summary.succeeded[: args.show]:
        print(format_image_diagnosis(diagnosis))
    print(format_dataset_diagnosis(summary, worst=args.worst))

    if args.save:
        weights = detector.model_path or config.MODEL_PATH
        context = storage.build_run_context(
            model_path=weights,
            dataset_yaml=dataset.source_path,
            split=args.split,
            confidence=detector.confidence,
            match_iou=(
                args.match_iou
                if args.match_iou is not None
                else config.MATCH_IOU_THRESHOLD
            ),
            localization_floor=config.LOCALIZATION_IOU_FLOOR,
            image_size=detector.image_size,
            # Schema 15. Recorded because nothing else can recover them, and
            # because a run that cannot say what it passed cannot afterwards
            # be asked whether varying it would have mattered.
            nms_iou=detector.iou,
            max_detections=config.MAX_DETECTIONS,
            library_version=_inference_library_version(args.detector),
        )
        with storage.connect(db_path) as connection:
            run_id = storage.save_dataset_diagnosis(connection, context, summary)
        print(f"  Saved as run {run_id} in {db_path or config.DB_PATH}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
