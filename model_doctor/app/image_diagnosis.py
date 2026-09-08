"""Image-level diagnosis — what shape the model's mistake took, per photograph.

**A second lens, not a replacement.** :mod:`model_doctor.app.diagnosis` answers *which
object failed and in what way*; this answers *how many photographs the model
handled correctly, and when it did not, what the mistake looked like*. Both are
true and they answer different questions. Nothing here changes a finding, an
outcome, or a count — D-017 and every existing query are untouched (D-040).

**Why it was needed.** One prediction stretched across two annotated
staircases produced a `poor_localization` *and* a `false_negative`: two
findings for one model error, and the second read as "the model never saw this
object" when the model had in fact covered 80% of it. Counting that as a miss
points at the wrong fix — more training data for small objects would not help
a merge. The finding-level view cannot see this, because a finding knows only
its own pairing. An image can see all of them at once.

**Geometry is masks, never boxes.** An axis-aligned box around a diagonal
staircase sweeps across its neighbour, so box overlap manufactures
relationships the objects do not have. Measured on this project's data, boxes
overstate object area by 1.2x to 2.9x, and two flights whose masks share 14%
had boxes sharing 92%. A run without stored outlines is therefore reported as
**not measured** rather than analysed from boxes (D-040).

**Thresholds are stored, and the raw pairs are kept.** The coverage
distribution is strongly bimodal — on 430 objects, 72% above 0.90 and 15%
below 0.05 — so the "untouched" threshold sits in a wide empty band and is
insensitive. The "found it" threshold is *not*: merge and split counts move
with it. So it travels with every verdict, and every pair above a floor is
persisted, letting a consumer re-threshold without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_doctor import config
from model_doctor.app import storage
from model_doctor.utils.exceptions import ModelDoctorError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImageDiagnosisError(ModelDoctorError):
    """Image-level diagnosis could not run, with a reason the caller can act on."""


#: Every verdict this pass can assign. Stored verbatim, so changing one is a
#: contract change and belongs in SCHEMA.md.
IMAGE_VERDICTS: tuple[str, ...] = (
    "empty",
    "zero_prediction",
    "merged",
    "split",
    "merged_and_split",
    "partly_missed",
    "spurious",
    "partly_missed_and_spurious",
    "partial_coverage",
    "clean",
)

#: How geometry was compared. Only masks are supported, deliberately.
METHOD_MASK: str = "mask"


@dataclass(frozen=True)
class ImageEvidence:
    """One image's findings, reduced to what a verdict needs.

    Assembled by the caller so :func:`classify` stays pure and testable on
    constructed input, with no database and no image decoding.

    Attributes:
        truth_ids: Findings carrying an annotated object.
        pred_ids: Findings carrying a prediction.
        coverage: ``(truth_id, pred_id) -> share of the truth mask covered``.
            Absent pairs are zero.
        outcomes: Finding-level outcome counts, carried through unchanged.
    """

    image_id: int
    truth_ids: tuple[int, ...]
    pred_ids: tuple[int, ...]
    coverage: Mapping[tuple[int, int], float]
    outcomes: Mapping[str, int] = field(default_factory=dict)

    def covered(self, truth_id: int, hit: float) -> int:
        """How many predictions cover this object at or above ``hit``."""
        return sum(
            1 for p in self.pred_ids if self.coverage.get((truth_id, p), 0.0) >= hit
        )

    def spans(self, pred_id: int, hit: float) -> int:
        """How many objects this prediction covers at or above ``hit``."""
        return sum(
            1 for t in self.truth_ids if self.coverage.get((t, pred_id), 0.0) >= hit
        )

    def best_coverage(self, truth_id: int) -> float:
        """The strongest coverage any prediction achieves on this object."""
        return max(
            (self.coverage.get((truth_id, p), 0.0) for p in self.pred_ids), default=0.0
        )

    def best_reach(self, pred_id: int) -> float:
        """The strongest coverage this prediction achieves on any object."""
        return max(
            (self.coverage.get((t, pred_id), 0.0) for t in self.truth_ids), default=0.0
        )


def classify(
    evidence: ImageEvidence,
    cover_hit: float | None = None,
    cover_miss: float | None = None,
) -> dict[str, Any]:
    """Return one verdict for one image, with the quantities behind it.

    **Precedence matters and is deliberate.** ``merged`` and ``split`` outrank
    ``partly_missed``: when one prediction spans two objects, the object the
    one-to-one matcher could not pair is a *consequence* of the merge, and
    reporting it as a miss is exactly the error this pass exists to correct.

    ``partial_coverage`` names the band between the two thresholds, on either
    side: an object the model reached but did not take, or a prediction that
    reached an object without taking it. Without it an image could hold an
    unexplained false negative — or false positive — and still be called clean.

    **A ``poor_localization`` finding can sit in a ``clean`` image, and that is
    not a contradiction.** This pass measures mask coverage; the finding-level
    outcome measures box overlap. An outline covering 99% of an object whose
    box scored 0.28 IoU is exactly that: the mask was right and the box was
    loose. Reporting both is the point of a second lens (D-036).

    ``zero_prediction`` is separate from ``partly_missed`` for the same kind of
    reason — nothing was attempted, which is a different failure from
    attempting and missing — and ``empty`` is separate from ``clean``, because
    an image with nothing to find and nothing found is correct behaviour that
    should not inflate a clean rate.

    Args:
        evidence: The image's objects, predictions and pairwise coverage.
        cover_hit: Coverage at which a prediction counts as having found an
            object. Defaults to :data:`config.IMAGE_COVER_HIT`.
        cover_miss: Coverage below which an object counts as untouched.
            Defaults to :data:`config.IMAGE_COVER_MISS`.

    Returns:
        The verdict and the counts that produced it, ready to persist.
    """
    hit = config.IMAGE_COVER_HIT if cover_hit is None else cover_hit
    miss = config.IMAGE_COVER_MISS if cover_miss is None else cover_miss

    merged = any(evidence.spans(p, hit) > 1 for p in evidence.pred_ids)
    split = any(evidence.covered(t, hit) > 1 for t in evidence.truth_ids)
    untouched = sum(1 for t in evidence.truth_ids if evidence.best_coverage(t) < miss)
    on_nothing = sum(1 for p in evidence.pred_ids if evidence.best_reach(p) < miss)
    # Objects the model reached but did not take: above the "untouched" bar and
    # below the "found it" one. Without a name for this band an image could
    # contain an unexplained false negative and still read as clean, which is
    # how image 629 — one object covered 0.47 — first slipped through.
    partial = sum(
        1 for t in evidence.truth_ids if miss <= evidence.best_coverage(t) < hit
    )
    # The same band on the prediction side: a prediction that reaches an object
    # without taking it is neither a second hit (no split) nor aimed at nothing
    # (not spurious). Left unnamed it produced the mirror of the same bug — an
    # image holding a false positive reported clean.
    partial_preds = sum(
        1 for p in evidence.pred_ids if miss <= evidence.best_reach(p) < hit
    )

    if not evidence.truth_ids and not evidence.pred_ids:
        verdict = "empty"
    elif evidence.truth_ids and not evidence.pred_ids:
        verdict = "zero_prediction"
    elif merged and split:
        verdict = "merged_and_split"
    elif merged:
        verdict = "merged"
    elif split:
        verdict = "split"
    elif untouched and on_nothing:
        verdict = "partly_missed_and_spurious"
    elif untouched:
        verdict = "partly_missed"
    elif on_nothing:
        verdict = "spurious"
    elif partial or partial_preds:
        verdict = "partial_coverage"
    else:
        verdict = "clean"

    return {
        "verdict": verdict,
        "merged": merged,
        "split": split,
        "objects_untouched": untouched,
        "objects_partial": partial,
        "predictions_partial": partial_preds,
        "predictions_on_nothing": on_nothing,
        "gt_count": len(evidence.truth_ids),
        "pred_count": len(evidence.pred_ids),
        **{
            outcome: evidence.outcomes.get(outcome, 0)
            for outcome in (
                "correct",
                "false_negative",
                "false_positive",
                "poor_localization",
                "wrong_class",
            )
        },
    }


def rasterise(polygon: Sequence[Sequence[float]], width: int, height: int):
    """Rasterise one stored polygon. Imported lazily; the readers need no OpenCV.

    **Public because a second pass must measure the same way.** Relations
    compare their geometry against the coverage this module already stores, and
    two rasterisers — or a rasteriser against vector areas — would disagree in
    the third decimal and make the two sets of numbers uncomparable. Whatever
    ``fillPoly`` does with an awkward ring, everything measuring this run does
    identically.

    Stored polygons are a single ring of at least three points: both detector
    families reduce a mask to its largest component on the way in, so holes and
    multiple parts cannot reach here.
    """
    import cv2
    import numpy as np

    canvas = np.zeros((height, width), np.uint8)
    cv2.fillPoly(canvas, [np.array(polygon, np.int32).reshape(-1, 1, 2)], 1)
    return canvas


#: Retained so nothing that imported the private name breaks. Same function.
_mask = rasterise


def build_evidence(
    rows: Sequence[Any], image_id: int, width: int, height: int, floor: float
) -> tuple[ImageEvidence, list[storage.ImageCoverageRow]]:
    """Measure every object against every prediction on one image.

    Only pairs at or above ``floor`` are returned for storage: a row saying two
    shapes do not overlap carries no more than its own absence.

    A finding contributes an object when it has a stored ground-truth polygon
    and a prediction when it has a predicted one. A ``wrong_class`` finding has
    both, and is treated as a covered object here — the class error is already
    recorded at finding level and is not an image-level failure mode.
    """
    truths = [(r["id"], r["truth_polygon"]) for r in rows if r["truth_polygon"]]
    preds = [(r["id"], r["pred_polygon"]) for r in rows if r["pred_polygon"]]

    coverage: dict[tuple[int, int], float] = {}
    stored: list[storage.ImageCoverageRow] = []
    if truths and preds and width and height:
        truth_masks = {i: _mask(json.loads(p), width, height) for i, p in truths}
        pred_masks = {i: _mask(json.loads(p), width, height) for i, p in preds}
        for t, tm in truth_masks.items():
            area = int(tm.sum())
            if not area:
                continue
            for p, pm in pred_masks.items():
                share = int((tm & pm).sum()) / area
                if share <= 0:
                    continue
                coverage[(t, p)] = share
                if share >= floor:
                    stored.append(
                        storage.ImageCoverageRow(
                            truth_finding_id=t, pred_finding_id=p, coverage=share
                        )
                    )

    outcomes: dict[str, int] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1

    evidence = ImageEvidence(
        image_id=image_id,
        truth_ids=tuple(i for i, _ in truths),
        pred_ids=tuple(i for i, _ in preds),
        coverage=coverage,
        outcomes=outcomes,
    )
    stored.sort(key=lambda c: -c.coverage)
    return evidence, stored


def analyse_run(
    connection: Any,
    run_id: int,
    cover_hit: float | None = None,
    cover_miss: float | None = None,
) -> dict[str, Any]:
    """Diagnose every image in a run and persist the verdicts.

    Raises:
        ImageDiagnosisError: If the run has no stored outlines. Boxes are not a
            fallback: they overstate diagonal objects and manufacture overlap,
            and a wrong measurement is worse than a missing one (D-040).
    """
    hit = config.IMAGE_COVER_HIT if cover_hit is None else cover_hit
    miss = config.IMAGE_COVER_MISS if cover_miss is None else cover_miss

    if storage.load_run(connection, run_id) is None:
        raise ImageDiagnosisError(f"No run with id {run_id}.")
    if not storage.has_table(connection, "mask_findings"):
        raise ImageDiagnosisError(
            f"Run {run_id} has no outline measurements, and image-level "
            "diagnosis is measured on masks. Produce them with: "
            f"python -m model_doctor.app.mask_diagnosis --run {run_id}"
        )

    images = connection.execute(
        "SELECT id, width, height FROM images WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    if not images:
        raise ImageDiagnosisError(f"Run {run_id} has no images.")

    entries = []
    tally: dict[str, int] = {}
    for image in images:
        rows = connection.execute(
            """
            SELECT f.id, f.outcome, f.truth_polygon, m.pred_polygon
            FROM findings f
            LEFT JOIN mask_findings m ON m.finding_id = f.id
            WHERE f.image_id = ? AND f.run_id = ?
            """,
            (image["id"], run_id),
        ).fetchall()
        evidence, coverage = build_evidence(
            rows, image["id"], image["width"], image["height"],
            config.IMAGE_COVERAGE_FLOOR,
        )
        fields = classify(evidence, hit, miss)
        tally[fields["verdict"]] = tally.get(fields["verdict"], 0) + 1
        entries.append((image["id"], fields, coverage))

    written = storage.save_image_diagnoses(
        connection, run_id, entries, cover_hit=hit, cover_miss=miss, method=METHOD_MASK
    )
    logger.info("Diagnosed %d image(s) for run %d", written, run_id)
    return {"run_id": run_id, "images": written, "verdicts": tally,
            "cover_hit": hit, "cover_miss": miss}


def format_report(summary: Mapping[str, Any]) -> str:
    """Render the pass's own summary."""
    total = summary["images"]
    lines = ["", "Model Doctor — image-level diagnosis", "=" * 62]
    lines.append(f"  Run              : {summary['run_id']}")
    lines.append(f"  Images diagnosed : {total}")
    lines.append(
        f"  Thresholds       : found >= {summary['cover_hit']:.2f}, "
        f"untouched < {summary['cover_miss']:.2f} (mask coverage)"
    )
    lines.append("")
    lines.append("  VERDICT                          COUNT     SHARE")
    for verdict in IMAGE_VERDICTS:
        n = summary["verdicts"].get(verdict, 0)
        share = (100.0 * n / total) if total else 0.0
        lines.append(f"  {verdict:<30} {n:>7} {share:>8.1f}%")
    lines.append("=" * 62)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-image-diagnosis",
        description="Diagnose each image as a whole: clean, merged, split, or missed.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=Path, help="Database path override.")
    parser.add_argument(
        "--cover-hit", type=float,
        help=f"Coverage at which a prediction found an object "
             f"(default {config.IMAGE_COVER_HIT}).",
    )
    parser.add_argument(
        "--cover-miss", type=float,
        help=f"Coverage below which an object is untouched "
             f"(default {config.IMAGE_COVER_MISS}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when the run cannot be diagnosed.
    """
    args = build_parser().parse_args(argv)
    with storage.connect(args.db) as connection:
        run_id = args.run
        if run_id is None:
            runs = storage.list_runs(connection)
            if not runs:
                logger.error("No runs in the database.")
                return 1
            run_id = runs[0].id
        try:
            summary = analyse_run(connection, run_id, args.cover_hit, args.cover_miss)
        except ImageDiagnosisError as exc:
            logger.error("%s", exc)
            return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
