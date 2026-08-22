"""Root-cause analysis — attributing failures to conditions.

The diagnosis engine says *what* went wrong. This module proposes *why*: the
object was small, the region was dark or blurred, it was cut off by the frame,
it sat among crowded neighbours, its class is under-represented, or that class
is repeatedly misidentified.

These are **correlations, not proofs**. A blurred region that the model missed
may have been missed because of the blur, or for an unrelated reason that
happens to co-occur with it. The engine reports evidence and a severity; the
causal claim belongs to the engineer reading it. Naming a factor is useful
precisely because it turns "recall is low" into a hypothesis that can be
checked.

**Where it attaches, and why that matters.** A factor is recorded against a
*finding*. Once failures are grouped, a per-cluster summary is a ``GROUP BY``
over the same rows — no new pipeline, no schema change, no rework:

.. code-block:: sql

    SELECT c.cluster_id, rc.factor, COUNT(*)
    FROM root_causes rc JOIN clusters c ON c.finding_id = rc.finding_id
    GROUP BY c.cluster_id, rc.factor;

Choosing the attachment point correctly today is what makes that free later.

**Two scopes, one output.** Some conditions are visible in a single finding
(blur, size); others only exist across a whole run (class imbalance, repeated
misidentification). Both produce the same :class:`FactorEvidence` and land in
the same table, so a consumer never has to know which kind it is looking at.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app import storage
from utils.exceptions import ModelDoctorError
from utils.geometry import box_iou
from utils.logging_utils import get_logger
from utils.statistics import fisher_exact_two_sided, lift, percentile

logger = get_logger(__name__)


class RootCauseError(ModelDoctorError):
    """Root-cause analysis could not run."""


# Factor identifiers. Stored verbatim, so they are part of what a consumer
# reads; changing one is a contract change and belongs in SCHEMA.md.
BLUR = "blur"
LOW_LIGHT = "low_light"
SMALL_OBJECT = "small_object"
EDGE_TRUNCATION = "edge_truncation"
CROWDING = "crowding"
THIN_STRUCTURE = "thin_structure"
CLASS_IMBALANCE = "class_imbalance"
RECURRING_MISCLASSIFICATION = "recurring_misclassification"


@dataclass(frozen=True)
class FactorEvidence:
    """One condition proposed as contributing to a failure.

    Attributes:
        factor: Identifier, one of the module-level constants.
        score: Severity in ``[0, 1]``. Comparable *within* a factor, so the
            worst instances can be ranked; not comparable *across* factors,
            since a blur score and a crowding score measure different things.
        evidence: The measurement that produced it, in human-readable form.
            Kept because a factor name alone is an assertion — the number
            behind it is what lets someone agree or disagree.
    """

    factor: str
    score: float
    evidence: str


@dataclass
class FindingContext:
    """One finding, with everything a per-finding detector may need.

    Attributes:
        finding_id: Database identifier.
        outcome: The failure classification.
        class_name: Class the finding concerns.
        box: Region in absolute pixels, or ``None`` when the finding has none.
        image_width: Source width in pixels.
        image_height: Source height in pixels.
        region: Cropped pixels as a greyscale array, or ``None`` when the image
            could not be read. Detectors that need pixels check for this.
        neighbours: Boxes of other annotations on the same image, for
            occlusion and crowding.
    """

    finding_id: int
    outcome: str
    class_name: str
    box: tuple[float, float, float, float] | None
    image_width: int | None
    image_height: int | None
    region: Any = None
    neighbours: Sequence[tuple[float, float, float, float]] = ()


class FindingFactor(Protocol):
    """A condition detectable from a single finding."""

    @property
    def name(self) -> str:
        """Return the factor identifier."""
        ...

    @property
    def needs_pixels(self) -> bool:
        """Return whether this detector requires the image to be read."""
        ...

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence if the condition holds, otherwise ``None``."""
        ...


class RunFactor(Protocol):
    """A condition that only exists across a whole run."""

    @property
    def name(self) -> str:
        """Return the factor identifier."""
        ...

    def detect(self, contexts: Sequence[FindingContext]) -> dict[int, FactorEvidence]:
        """Return evidence keyed by finding id."""
        ...


# ---------------------------------------------------------------------------
# Per-finding factors
# ---------------------------------------------------------------------------
class BlurFactor:
    """Flags regions with little high-frequency detail.

    Sharpness is measured as the variance of the Laplacian: a crisp region has
    strong second derivatives that vary a lot, a blurred one does not. It is
    the standard measure, cheap, and needs no reference image.
    """

    name = BLUR
    needs_pixels = True

    def __init__(self, threshold: float | None = None) -> None:
        """Store the variance below which a region counts as blurred."""
        self.threshold = (
            threshold if threshold is not None else config.BLUR_VARIANCE_THRESHOLD
        )

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when the region's Laplacian variance is low."""
        if context.region is None or context.region.size == 0:
            return None
        import cv2

        variance = float(cv2.Laplacian(context.region, cv2.CV_64F).var())
        if variance >= self.threshold:
            return None
        # Severity rises as variance falls, reaching 1.0 at a perfectly flat
        # region. Scaled against the threshold so the score means "how far past
        # the line", which is comparable between images.
        score = max(0.0, min(1.0, 1.0 - variance / self.threshold))
        return FactorEvidence(
            self.name,
            score,
            f"Laplacian variance {variance:.1f} < {self.threshold:.0f}",
        )


class LowLightFactor:
    """Flags underexposed regions by mean luminance."""

    name = LOW_LIGHT
    needs_pixels = True

    def __init__(self, threshold: float | None = None) -> None:
        """Store the mean luminance below which a region counts as dark."""
        self.threshold = (
            threshold if threshold is not None else config.LOW_LIGHT_THRESHOLD
        )

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when the region's mean brightness is low."""
        if context.region is None or context.region.size == 0:
            return None
        mean = float(context.region.mean())
        if mean >= self.threshold:
            return None
        score = max(0.0, min(1.0, 1.0 - mean / self.threshold))
        return FactorEvidence(
            self.name, score, f"mean luminance {mean:.1f}/255 < {self.threshold:.0f}"
        )


class SmallObjectFactor:
    """Flags objects occupying a very small fraction of the frame.

    Expressed as a fraction of image area rather than in pixels, so the same
    threshold is meaningful across resolutions.
    """

    name = SMALL_OBJECT
    needs_pixels = False

    def __init__(self, fraction: float | None = None) -> None:
        """Store the area fraction below which an object counts as small."""
        self.fraction = (
            fraction if fraction is not None else config.SMALL_OBJECT_AREA_FRACTION
        )

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when the box covers very little of the image."""
        if context.box is None or not context.image_width or not context.image_height:
            return None
        x1, y1, x2, y2 = context.box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        total = float(context.image_width * context.image_height)
        if total <= 0:
            return None
        share = area / total
        if share >= self.fraction:
            return None
        score = max(0.0, min(1.0, 1.0 - share / self.fraction))
        return FactorEvidence(
            self.name,
            score,
            f"{share * 100:.3f}% of image < {self.fraction * 100:.2f}%",
        )


class ThinStructureFactor:
    """Flags long, narrow objects by the aspect ratio of their bounding box.

    Thin structures are what detectors handle worst, and nothing else in the
    factor set names the condition. It is not a proxy for size: measured within
    a single class, and with area held at its median, thinness still separates
    failures from successes (D-032).

    A door frame is the canonical case here — a thin rectangle around an
    opening. Its mask mAP50-95 on this project's model is 0.246 against 0.599
    for the solid door, a gap invisible to box-level analysis and unnamed by
    any factor until now.
    """

    name = THIN_STRUCTURE
    needs_pixels = False

    def __init__(self, ratio: float | None = None) -> None:
        """Store the aspect ratio above which an object counts as thin."""
        self.ratio = ratio if ratio is not None else config.THIN_STRUCTURE_RATIO

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when the box is much longer than it is wide."""
        if context.box is None:
            return None
        x1, y1, x2, y2 = context.box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return None

        longer, shorter = max(width, height), min(width, height)
        aspect = longer / shorter
        if aspect < self.ratio:
            return None

        # Severity saturates at twice the threshold: past that the object is
        # emphatically thin and finer gradations say nothing useful.
        score = max(0.0, min(1.0, (aspect - self.ratio) / self.ratio))
        orientation = "tall" if height > width else "wide"
        return FactorEvidence(
            self.name,
            score,
            f"{aspect:.1f}:1 {orientation}, above {self.ratio:.1f}:1",
        )


class EdgeTruncationFactor:
    """Flags objects cut off by the frame.

    A truncated object is only partly visible, so the model sees less evidence
    than the annotation implies. On datasets photographed close up this is
    common rather than exceptional, which is worth knowing before concluding
    the model is simply weak.
    """

    name = EDGE_TRUNCATION
    needs_pixels = False

    def __init__(self, margin: float | None = None) -> None:
        """Store the edge margin, as a fraction of image size."""
        self.margin = margin if margin is not None else config.EDGE_TRUNCATION_MARGIN

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when the box touches or nearly touches the frame."""
        if context.box is None or not context.image_width or not context.image_height:
            return None
        x1, y1, x2, y2 = context.box
        width, height = float(context.image_width), float(context.image_height)
        margin_x, margin_y = width * self.margin, height * self.margin

        touching = [
            name
            for name, hit in (
                ("left", x1 <= margin_x),
                ("top", y1 <= margin_y),
                ("right", x2 >= width - margin_x),
                ("bottom", y2 >= height - margin_y),
            )
            if hit
        ]
        if not touching:
            return None
        # More truncated sides means less of the object is visible, so severity
        # scales with how many edges it runs into.
        return FactorEvidence(
            self.name,
            min(1.0, len(touching) / 4.0 + 0.25),
            f"touches {', '.join(touching)}",
        )


class CrowdingFactor:
    """Flags regions overlapping other annotations.

    A proxy for occlusion, and labelled as such. True occlusion is not
    observable from annotations alone — nothing records what is in front of
    what — but heavy overlap between annotated objects is the condition under
    which occlusion occurs, and it is measurable.
    """

    name = CROWDING
    needs_pixels = False

    def __init__(self, threshold: float | None = None) -> None:
        """Store the IoU above which neighbours count as crowding."""
        self.threshold = (
            threshold if threshold is not None else config.CROWDING_IOU_THRESHOLD
        )

    def detect(self, context: FindingContext) -> FactorEvidence | None:
        """Return evidence when a neighbouring annotation overlaps heavily."""
        if context.box is None or not context.neighbours:
            return None

        overlaps = [
            box_iou(_Box(*context.box), _Box(*other)) for other in context.neighbours
        ]
        worst = max(overlaps, default=0.0)
        if worst < self.threshold:
            return None
        crowded = sum(1 for value in overlaps if value >= self.threshold)
        return FactorEvidence(
            self.name,
            min(1.0, worst),
            f"{crowded} neighbour(s) overlapping, max IoU {worst:.2f}",
        )


@dataclass(frozen=True)
class _Box:
    """Adapter giving raw corners the geometry interface ``box_iou`` expects."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        """Return the box area in square pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


# ---------------------------------------------------------------------------
# Run-level factors
# ---------------------------------------------------------------------------
class ClassImbalanceFactor:
    """Flags failures belonging to under-represented classes.

    Measured against an even split rather than against the largest class, so
    the threshold means the same thing whatever the number of classes.
    """

    name = CLASS_IMBALANCE

    def __init__(self, ratio: float | None = None) -> None:
        """Store the share of an even split below which a class is flagged."""
        self.ratio = ratio if ratio is not None else config.CLASS_IMBALANCE_RATIO

    def detect(self, contexts: Sequence[FindingContext]) -> dict[int, FactorEvidence]:
        """Return evidence for every finding of an under-represented class."""
        counts = Counter(context.class_name for context in contexts)
        if len(counts) < 2:
            return {}

        total = sum(counts.values())
        even_share = total / len(counts)
        threshold = even_share * self.ratio

        flagged = {name for name, n in counts.items() if n < threshold}
        if not flagged:
            return {}

        results: dict[int, FactorEvidence] = {}
        for context in contexts:
            if context.class_name not in flagged:
                continue
            count = counts[context.class_name]
            results[context.finding_id] = FactorEvidence(
                self.name,
                max(0.0, min(1.0, 1.0 - count / even_share)),
                f"'{context.class_name}' has {count} of {total} instances "
                f"({count / total * 100:.1f}%, even split would be "
                f"{100 / len(counts):.1f}%)",
            )
        return results


class RecurringMisclassificationFactor:
    """Flags classes that are repeatedly named wrongly.

    **Scope limitation, stated rather than hidden.** The stored contract records
    the class that *should* have been found, not the one the model guessed
    instead — a deliberate choice so per-class statistics charge a miss to the
    class that was missed. Directed confusion pairs therefore cannot be derived
    from stored findings. What can be, and what this reports, is which classes
    are repeatedly misidentified. That is the actionable half: it names the
    class whose labelling needs attention.
    """

    name = RECURRING_MISCLASSIFICATION

    def __init__(self, minimum: int = 3) -> None:
        """Store how many occurrences constitute a recurring pattern."""
        self.minimum = minimum

    def detect(self, contexts: Sequence[FindingContext]) -> dict[int, FactorEvidence]:
        """Return evidence for findings of frequently-misidentified classes."""
        wrong = Counter(
            context.class_name
            for context in contexts
            if context.outcome == "wrong_class"
        )
        totals = Counter(context.class_name for context in contexts)

        results: dict[int, FactorEvidence] = {}
        for context in contexts:
            if context.outcome != "wrong_class":
                continue
            count = wrong[context.class_name]
            if count < self.minimum:
                continue
            share = count / max(1, totals[context.class_name])
            results[context.finding_id] = FactorEvidence(
                self.name,
                max(0.0, min(1.0, share)),
                f"'{context.class_name}' misidentified {count} time(s), "
                f"{share * 100:.0f}% of its findings",
            )
        return results


def default_finding_factors() -> list[FindingFactor]:
    """Return the per-finding detectors used unless a caller overrides them.

    Thresholds come from configuration. Prefer :func:`calibrate_finding_factors`
    when the run's own contexts are available — two of these detectors measure
    "unusual for this dataset", which a fixed constant cannot express.
    """
    return [
        BlurFactor(),
        LowLightFactor(),
        SmallObjectFactor(),
        ThinStructureFactor(),
        EdgeTruncationFactor(),
        CrowdingFactor(),
    ]


def calibrate_finding_factors(
    contexts: Sequence[FindingContext],
) -> list[FindingFactor]:
    """Return the per-finding detectors with thresholds drawn from the data.

    Size and shape are relative properties. "Small" means small *for this
    dataset*, and a constant cannot say that: the configured 0.12% of image
    area comes from the COCO convention and fired on 2 of 278 findings here,
    while "smaller than three quarters of the others" is among the strongest
    predictors of failure this project has measured (D-032).

    Both thresholds are computed over **every** finding, correct ones included.
    Deriving them from failures alone would make the reference distribution the
    thing being measured, and the factor would then fire on a fixed share of
    failures by construction.

    Detectors whose thresholds are absolute physical quantities — blur, light,
    edge proximity, overlap — are unchanged. A dark region is dark regardless
    of how dark the rest of the dataset is.

    Args:
        contexts: Every finding in the run. An empty or geometry-free sequence
            leaves all detectors on their configured defaults.

    Returns:
        The standard detector set, with size and thinness calibrated where the
        data allowed it.
    """
    areas: list[float] = []
    aspects: list[float] = []
    for context in contexts:
        if context.box is None:
            continue
        x1, y1, x2, y2 = context.box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        if context.image_width and context.image_height:
            total = float(context.image_width * context.image_height)
            if total > 0:
                areas.append((width * height) / total)
        aspects.append(max(width, height) / min(width, height))

    area_threshold = percentile(areas, config.SMALL_OBJECT_PERCENTILE)
    aspect_threshold = percentile(aspects, config.THIN_STRUCTURE_PERCENTILE)

    if area_threshold is not None:
        logger.debug(
            "Calibrated small_object to %.4f of image area (p%d of %d objects)",
            area_threshold,
            round(config.SMALL_OBJECT_PERCENTILE * 100),
            len(areas),
        )
    if aspect_threshold is not None:
        logger.debug(
            "Calibrated thin_structure to %.2f:1 (p%d of %d objects)",
            aspect_threshold,
            round(config.THIN_STRUCTURE_PERCENTILE * 100),
            len(aspects),
        )

    return [
        BlurFactor(),
        LowLightFactor(),
        SmallObjectFactor(area_threshold),
        ThinStructureFactor(aspect_threshold),
        EdgeTruncationFactor(),
        CrowdingFactor(),
    ]


def default_run_factors() -> list[RunFactor]:
    """Return the run-level detectors used unless a caller overrides them."""
    return [ClassImbalanceFactor(), RecurringMisclassificationFactor()]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisReport:
    """What one attribution pass did."""

    run_id: int
    considered: int = 0
    attributed: int = 0
    unexplained: int = 0
    images_unreadable: int = 0
    counts: dict[str, int] | None = None

    def describe(self) -> str:
        """Render the report for console output."""
        lines = ["", "Root-cause analysis", "=" * 66]
        lines.append(f"  Run                : {self.run_id}")
        lines.append(f"  Failures considered: {self.considered}")
        lines.append(f"  Factors attributed : {self.attributed}")
        share = (
            100.0 * (self.considered - self.unexplained) / self.considered
            if self.considered
            else 0.0
        )
        lines.append(
            f"  Failures explained : {self.considered - self.unexplained} "
            f"({share:.0f}%)"
        )
        if self.unexplained:
            lines.append(f"  No factor found    : {self.unexplained}")
        if self.images_unreadable:
            lines.append(f"  Images unreadable  : {self.images_unreadable}")

        if self.counts:
            lines.append("")
            lines.append(f"  {'FACTOR':<30}{'FAILURES':>10}{'SHARE':>10}")
            for factor, number in sorted(self.counts.items(), key=lambda kv: -kv[1]):
                pct = 100.0 * number / self.considered if self.considered else 0.0
                lines.append(f"  {factor:<30}{number:>10}{pct:>9.1f}%")
        lines.append("=" * 66)
        lines.append("")
        return "\n".join(lines)


def _box_from_row(row: Any) -> tuple[float, float, float, float] | None:
    """Return the region a finding concerns.

    Ground truth first, prediction otherwise — matching how a finding is
    attributed to a class. A missed object is asked about where it *is*; an
    invented one about where the model *put* it.
    """
    for prefix in ("truth", "pred"):
        values = [row[f"{prefix}_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
        if all(value is not None for value in values):
            return (values[0], values[1], values[2], values[3])
    return None


def build_contexts(
    rows: Sequence[Any], load_pixels: bool = True
) -> tuple[list[FindingContext], int]:
    """Turn database rows into contexts the detectors can examine.

    Rows are grouped by image so each file is opened once however many findings
    it carries, and so every finding knows its neighbours — which is what makes
    crowding measurable at all.

    Args:
        rows: Finding rows joined to their image.
        load_pixels: Whether to read image data. Skipping it disables the
            factors that need pixels and makes the pass much faster.

    Returns:
        ``(contexts, unreadable_image_count)``.
    """
    by_image: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_image[row["path"]].append(row)

    contexts: list[FindingContext] = []
    unreadable = 0

    for path, group in by_image.items():
        greyscale = None
        if load_pixels:
            try:
                import cv2
                import numpy as np
                from PIL import Image

                with Image.open(path) as handle:
                    greyscale = cv2.cvtColor(
                        np.array(handle.convert("RGB")), cv2.COLOR_RGB2GRAY
                    )
            except Exception as exc:
                # One unreadable image disables only the pixel-based factors
                # for its findings; the geometric ones still apply.
                unreadable += 1
                logger.warning("Could not read %s: %s", path, exc)

        boxes = [(_box_from_row(row), row) for row in group]
        present = [box for box, _ in boxes if box is not None]

        for box, row in boxes:
            region = None
            if greyscale is not None and box is not None:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(greyscale.shape[1], x2)
                y2 = min(greyscale.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    region = greyscale[y1:y2, x1:x2]

            contexts.append(
                FindingContext(
                    finding_id=row["finding_id"],
                    outcome=row["outcome"],
                    class_name=row["class_name"],
                    box=box,
                    image_width=row["width"],
                    image_height=row["height"],
                    region=region,
                    neighbours=[other for other in present if other is not box],
                )
            )

    return contexts, unreadable


def analyse_run(
    connection: Any,
    run_id: int,
    finding_factors: Sequence[FindingFactor] | None = None,
    run_factors: Sequence[RunFactor] | None = None,
    load_pixels: bool = True,
) -> AnalysisReport:
    """Attribute factors to a saved run's failures and store them.

    Reads findings back from the database, as feature extraction and
    explanation do, so the write path is untouched and a run saved earlier can
    be analysed without re-running inference.

    Args:
        connection: An open database connection.
        run_id: Run to analyse. Its findings must already be saved.
        finding_factors: Per-finding detectors. Defaults to the standard set.
        run_factors: Run-level detectors. Defaults to the standard set.
        load_pixels: Whether to read images. Disabling it skips blur and
            low-light detection.

    Returns:
        A description of what was attributed.

    Raises:
        RootCauseError: If the run has no failures to analyse.
    """
    per_run = list(run_factors) if run_factors is not None else default_run_factors()

    # Every finding is loaded, not just the failures, so that each failure's
    # `neighbours` include correctly-detected objects. Crowding is a property
    # of what is physically nearby; a failure beside a correct detection is
    # just as crowded as one beside another failure. Loading failures alone
    # made those neighbours invisible and under-reported crowding by a third
    # on the reference run (D-031).
    rows = storage.load_findings_for_embedding(connection, run_id, failures_only=False)
    contexts, unreadable = build_contexts(rows, load_pixels=load_pixels)
    failure_contexts = [c for c in contexts if c.outcome != "correct"]
    if not failure_contexts:
        raise RootCauseError(
            f"Run {run_id} has no failures to analyse. Diagnose and save it first."
        )

    # Calibrated against every finding, so "small" and "thin" mean unusual for
    # this dataset rather than unusual against a constant from another one.
    per_finding = (
        list(finding_factors)
        if finding_factors is not None
        else calibrate_finding_factors(contexts)
    )

    entries: list[tuple[int, str, float, str]] = []
    explained: set[int] = set()

    for context in failure_contexts:
        for factor in per_finding:
            if factor.needs_pixels and context.region is None:
                continue
            evidence = factor.detect(context)
            if evidence is None:
                continue
            entries.append(
                (context.finding_id, evidence.factor, evidence.score, evidence.evidence)
            )
            explained.add(context.finding_id)

    # Run-level factors receive failures only, exactly as before. Both derive a
    # denominator from the list they are given — "3 of 210 instances", "17% of
    # its findings" — so widening it would silently redefine what those numbers
    # mean. That is a separate decision from fixing crowding's neighbours.
    for run_factor in per_run:
        for finding_id, evidence in run_factor.detect(failure_contexts).items():
            entries.append(
                (finding_id, evidence.factor, evidence.score, evidence.evidence)
            )
            explained.add(finding_id)

    storage.save_root_causes(connection, run_id, entries)
    counts = storage.factor_counts(connection, run_id)
    logger.info("Attributed %d factor(s) across run %d", len(entries), run_id)

    return AnalysisReport(
        run_id=run_id,
        considered=len(failure_contexts),
        attributed=len(entries),
        unexplained=len(failure_contexts) - len(explained),
        images_unreadable=unreadable,
        counts=counts,
    )


@dataclass(frozen=True)
class FactorRateReport:
    """What one base-rate pass measured, ready to print."""

    run_id: int
    failure_total: int
    correct_total: int
    rates: Sequence[Any] = ()
    images_unreadable: int = 0

    def describe(self) -> str:
        """Render the comparison, leading with what distinguishes failures."""
        lines = ["", "Factor base rates", "=" * 78]
        lines.append(f"  Run                : {self.run_id}")
        lines.append(f"  Failures           : {self.failure_total}")
        lines.append(f"  Correct (control)  : {self.correct_total}")
        if self.images_unreadable:
            lines.append(f"  Images unreadable  : {self.images_unreadable}")
        lines.append("")
        lines.append(
            f"  {'FACTOR':<26}{'FAILURES':>12}{'CORRECT':>12}{'LIFT':>8}{'p':>8}"
        )
        for rate in self.rates:
            lift_text = "  n/a" if rate.lift is None else f"{rate.lift:.2f}x"
            lines.append(
                f"  {rate.factor:<26}"
                f"{rate.failure_count:>5}/{rate.failure_total:<3}"
                f"{100 * rate.failure_rate:>4.0f}%"
                f"{rate.correct_count:>5}/{rate.correct_total:<3}"
                f"{100 * rate.correct_rate:>4.0f}%"
                f"{lift_text:>8}{rate.p_value:>8.3f}"
            )
        lines.append("")
        lines.append(
            "  Lift is how many times more common a factor is among failures."
        )
        lines.append(
            "  1.00x means it describes successes just as often and therefore"
        )
        lines.append(
            "  explains nothing. Treat p >= 0.05 as no evidence of association,"
        )
        lines.append(
            "  and note that testing several factors makes a single p near 0.05"
        )
        lines.append("  weaker than it looks.")
        lines.append("=" * 78)
        lines.append("")
        return "\n".join(lines)


def measure_factor_rates(
    connection: Any,
    run_id: int,
    finding_factors: Sequence[FindingFactor] | None = None,
    load_pixels: bool = True,
) -> FactorRateReport:
    """Measure each factor's rate among failures against correct findings.

    A factor's count means nothing on its own. ``edge_truncation`` described
    71% of failures on the reference run, which reads as a cause until the
    control group shows it described 76% of correct detections. This pass
    supplies that denominator (D-031).

    Both groups are measured in a single pass over the same contexts, so the
    two rates cannot drift apart through differing preparation.

    Only per-finding factors are measured. Run-level factors such as
    ``recurring_misclassification`` are defined in terms of mistakes and have
    no meaningful analogue among correct findings, so giving them a base rate
    would invent a comparison rather than report one.

    Correct findings get **no** ``root_causes`` rows — only these aggregates.
    That table is documented as covering failures, and every query written
    against it keeps returning exactly what it did before.

    Args:
        connection: An open database connection.
        run_id: Run to measure. Its findings must already be saved.
        finding_factors: Per-finding detectors. Defaults to the standard set.
        load_pixels: Whether to read images. Disabling it skips the factors
            that need pixel data, which are then absent from the result rather
            than reported as never firing.

    Returns:
        The measured rates, ordered by lift descending.

    Raises:
        RootCauseError: If the run has no findings at all.
    """
    rows = storage.load_findings_for_embedding(connection, run_id, failures_only=False)
    if not rows:
        raise RootCauseError(
            f"Run {run_id} has no findings to measure. Diagnose and save it first."
        )

    contexts, unreadable = build_contexts(rows, load_pixels=load_pixels)
    failures = [c for c in contexts if c.outcome != "correct"]
    correct = [c for c in contexts if c.outcome == "correct"]

    # The same calibration `analyse_run` uses, over the same contexts, so a
    # stored attribution and its base rate cannot disagree about the threshold.
    per_finding = (
        list(finding_factors)
        if finding_factors is not None
        else calibrate_finding_factors(contexts)
    )

    if not failures or not correct:
        logger.warning(
            "Run %d has %d failure(s) and %d correct finding(s). A base rate "
            "needs both, so no rates were stored.",
            run_id,
            len(failures),
            len(correct),
        )
        return FactorRateReport(
            run_id=run_id,
            failure_total=len(failures),
            correct_total=len(correct),
            images_unreadable=unreadable,
        )

    entries: list[tuple[str, int, int, int, int, float | None, float]] = []
    for factor in per_finding:

        def tally(
            group: Sequence[Any], detector: FindingFactor = factor
        ) -> tuple[int, int]:
            """Count detections, and how many contexts could be evaluated.

            A factor needing pixels cannot fire on a context without a region.
            Counting those in the denominator would understate its rate, and
            would do so unevenly between the two groups.
            """
            hits = evaluable = 0
            for context in group:
                if detector.needs_pixels and context.region is None:
                    continue
                evaluable += 1
                if detector.detect(context) is not None:
                    hits += 1
            return hits, evaluable

        failure_hits, failure_total = tally(failures)
        correct_hits, correct_total = tally(correct)
        if not failure_total or not correct_total:
            continue

        entries.append(
            (
                factor.name,
                failure_hits,
                failure_total,
                correct_hits,
                correct_total,
                lift(failure_hits, failure_total, correct_hits, correct_total),
                fisher_exact_two_sided(
                    failure_hits, failure_total, correct_hits, correct_total
                ),
            )
        )

    storage.save_factor_rates(connection, run_id, entries)
    logger.info("Measured base rates for %d factor(s) in run %d", len(entries), run_id)

    return FactorRateReport(
        run_id=run_id,
        failure_total=len(failures),
        correct_total=len(correct),
        rates=storage.load_factor_rates(connection, run_id),
        images_unreadable=unreadable,
    )


def format_factor_detail(rows: Sequence[Any], limit: int = 5) -> str:
    """Render the strongest evidence per factor, as worked examples.

    A count says a factor is common; an example says what it looked like. Both
    are needed before anyone will act on it.
    """
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[row.factor].append(row)

    lines = ["", "Strongest evidence per factor", "-" * 66]
    for factor in sorted(grouped, key=lambda f: -len(grouped[f])):
        entries = grouped[factor][:limit]
        lines.append(f"  {factor}  ({len(grouped[factor])} failure(s))")
        for row in entries:
            lines.append(
                f"      finding {row.finding_id:<6} {row.outcome:<18} "
                f"{row.class_name:<12} {row.evidence}"
            )
        lines.append("")
    lines.append("-" * 66)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the root-cause command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-root-cause",
        description="Attribute a run's failures to measurable conditions.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=str, help="Database path override.")
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="Skip factors that need image data (blur, low light).",
    )
    parser.add_argument(
        "--detail", type=int, default=3, help="Examples to show per factor."
    )
    parser.add_argument(
        "--no-base-rates",
        action="store_true",
        help=(
            "Skip measuring how often each factor appears among correct "
            "findings. Without that control group a factor's count cannot "
            "support a claim about cause."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to analyse.
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
            report = analyse_run(
                connection, run_id, load_pixels=not args.no_pixels
            )
            rate_report = (
                None
                if args.no_base_rates
                else measure_factor_rates(
                    connection, run_id, load_pixels=not args.no_pixels
                )
            )
        except RootCauseError as exc:
            logger.error("%s", exc)
            return 1

        stored = storage.load_root_causes(connection, run_id)

    print(report.describe())
    if rate_report is not None:
        print(rate_report.describe())
    if args.detail:
        print(format_factor_detail(stored, limit=args.detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
