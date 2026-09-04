"""Relationships between findings, measured on masks.

The finding-level pass classifies each object-prediction pairing on its own, and
the image-level pass says what shape an image's mistake took. Neither can say
that *this* missed object is covered by the prediction the matcher gave to *that*
one — the first sees a pairing, the second sees a photograph, and the sentence
that connects two findings has no home in either.

**A relation is additive evidence, never a reclassification.** Every finding
keeps the outcome the matcher gave it. A false negative with a
``merge_candidate`` is still a false negative in every count, every rate and
every mAP figure; what the relation adds is that a prediction assigned to
another object covers it, which is a different problem from never having seen
it (D-017, D-036).

**A relation is not a factor, and cannot be made into one.** A factor is a
property of one object, measured on failures *and* on correct detections, and
admitted only when its lift over that control rate is significant — the check
that stops "common among failures" being read as "causes failures". A relation
attaches only to a failure by construction, so it has no control rate, its lift
is undefined, and :func:`app.comparison.factor_qualifies` would reject it. It is
stored apart and never enters that pipeline.

**Two relations here, deliberately.** Both reuse thresholds the run already
stored, so nothing in this module involves a judgement call about where a
boundary sits. ``duplicate_prediction`` needs a threshold of its own and is not
implemented yet: its continuous distribution has been measured but the boundary
rests on four configurations, and a number chosen now would be one nobody could
later argue with.

Nothing here writes to any table but ``finding_relations``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import storage
from app.image_diagnosis import rasterise
from utils.exceptions import ModelDoctorError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class RelationError(ModelDoctorError):
    """Relations could not be measured, with a reason the caller can act on."""


#: A ground-truth object the matcher paired with a prediction. These are the
#: outcomes that carry both a truth polygon and a predicted one.
MATCHED_OUTCOMES: tuple[str, ...] = ("correct", "poor_localization", "wrong_class")

MERGE_CANDIDATE: str = "merge_candidate"
BOX_MASK_DISAGREEMENT: str = "box_mask_disagreement"

#: Every relation this module can produce. ``duplicate_prediction`` is named in
#: the design and deliberately absent here.
RELATIONS: tuple[str, ...] = (MERGE_CANDIDATE, BOX_MASK_DISAGREEMENT)

#: The mask covers the object; the box did not match it.
MASK_OK_BOX_FAILS: str = "mask_ok_box_fails"
#: The box matched; the mask is barely on the object.
BOX_OK_MASK_FAILS: str = "box_ok_mask_fails"


@dataclass(frozen=True)
class ImageGeometry:
    """One image's findings reduced to what a relation needs.

    Attributes:
        truth: Rasterised ground truth by finding id.
        predicted: Rasterised predictions by finding id.
        outcome: Each finding's stored outcome.
        skipped: Findings whose polygon could not be rasterised. Reported so a
            caller can say "not measured" rather than record a zero — the
            distinction ``lift`` and ``images.error`` also preserve.
    """

    truth: dict[int, Any]
    predicted: dict[int, Any]
    outcome: dict[int, str]
    skipped: tuple[int, ...] = ()


def _coverage(truth_mask: Any, pred_mask: Any) -> float | None:
    """Share of a ground-truth mask a prediction intersects, or None if unmeasurable.

    The same quantity ``image_coverage`` stores, computed the same way, so a
    relation and an image verdict can never disagree about one pair. ``None``
    for a zero-area object: an undefined ratio is not a zero one.
    """
    area = int(truth_mask.sum())
    if not area:
        return None
    return int((truth_mask & pred_mask).sum()) / area


def build_geometry(rows: Sequence[Any], width: int, height: int) -> ImageGeometry:
    """Rasterise every polygon on one image.

    A polygon that cannot be rasterised is skipped rather than treated as
    empty. Stored polygons are single rings of at least three points, so this
    should never fire; if a future ingest changes that, the run reports fewer
    measurements instead of silently reporting zeroes.
    """
    truth: dict[int, Any] = {}
    predicted: dict[int, Any] = {}
    outcome: dict[int, str] = {}
    skipped: list[int] = []

    if not width or not height:
        return ImageGeometry({}, {}, {}, tuple(int(r["id"]) for r in rows))

    for row in rows:
        finding_id = int(row["id"])
        outcome[finding_id] = str(row["outcome"])
        # `sqlite3.Row` supports membership over values, not keys, so the
        # column list is asked for explicitly.
        columns = set(row.keys())
        for column, target in (
            ("truth_polygon", truth),
            ("pred_polygon", predicted),
        ):
            raw = row[column] if column in columns else None
            if not raw:
                continue
            try:
                target[finding_id] = rasterise(json.loads(raw), width, height)
            except Exception:  # noqa: BLE001 - a bad polygon must not fail a run
                logger.warning(
                    "Could not rasterise %s for finding %d; not measured",
                    column,
                    finding_id,
                )
                skipped.append(finding_id)

    return ImageGeometry(truth, predicted, outcome, tuple(sorted(set(skipped))))


def merge_candidates(
    geometry: ImageGeometry, cover_hit: float
) -> list[storage.FindingRelationRow]:
    """Missed objects that a prediction assigned elsewhere already covers.

    **Attributed to the false negative, never to the prediction.** One
    relationship produces one row; the prediction's own finding is named as the
    partner. Attributing to both ends would let a single merge be counted twice
    in any report that summed relations.

    A candidate needs the covering prediction to belong to a *different*
    finding that is itself matched to some other object. A prediction covering
    the object it was already paired with is not a merge, it is the pairing.

    Uses the run's own stored ``cover_hit``, so this introduces no threshold.
    On run 5 it identifies findings #1173, #1228, #1255 and #1293 — the four
    the manual investigation found, at coverages 0.761 to 1.000.
    """
    found: list[storage.FindingRelationRow] = []
    for finding_id, truth_mask in sorted(geometry.truth.items()):
        if geometry.outcome.get(finding_id) != "false_negative":
            continue
        best: tuple[float, int] | None = None
        for other_id, pred_mask in sorted(geometry.predicted.items()):
            if other_id == finding_id:
                continue
            # The covering prediction must itself be paired with an object.
            # An unmatched prediction covering a missed object is not a merge;
            # it is two failures on one image, and calling it a merge would
            # claim the matcher made a choice it never made.
            if geometry.outcome.get(other_id) not in MATCHED_OUTCOMES:
                continue
            share = _coverage(truth_mask, pred_mask)
            if share is None or share < cover_hit:
                continue
            if best is None or share > best[0]:
                best = (share, other_id)
        if best is None:
            continue
        coverage, partner = best
        found.append(
            storage.FindingRelationRow(
                finding_id=finding_id,
                relation=MERGE_CANDIDATE,
                value=coverage,
                qualifies=True,
                cover_hit=cover_hit,
                partner_finding_id=partner,
                best_coverage=coverage,
                # How many predictions cover this object at the threshold. More
                # than one means the object is both merged into a neighbour and
                # reached by something else, which is worth seeing without
                # producing a second row.
                span=sum(
                    1
                    for other_id, pred_mask in geometry.predicted.items()
                    if other_id != finding_id
                    and geometry.outcome.get(other_id) in MATCHED_OUTCOMES
                    and (_coverage(truth_mask, pred_mask) or 0.0) >= cover_hit
                ),
                threshold=cover_hit,
            )
        )
    return found


def box_mask_disagreements(
    geometry: ImageGeometry, cover_hit: float, cover_miss: float
) -> list[storage.FindingRelationRow]:
    """Findings whose box verdict and mask geometry tell different stories.

    Two directions, both attributed to the finding itself — the relation is one
    a finding has with its own geometry, so there is no partner and no second
    row to be counted.

    ``mask_ok_box_fails``: the matcher scored the box below its IoU threshold
    and called this a poor localisation, while the predicted mask covers the
    object at or above ``cover_hit``. A box around a diagonal object overstates
    its area by 1.2x to 2.9x, which is exactly when this happens. On run 5 it is
    8 of the 11 poor localisations.

    ``box_ok_mask_fails``: the reverse, and measured to be empty across every
    run in the reference database. The detector exists so that zero is a
    measurement rather than an assumption.

    Neither changes the finding's outcome. The taxonomy is the matcher's, and
    this pass reports a disagreement rather than resolving it (D-036).
    """
    found: list[storage.FindingRelationRow] = []
    for finding_id, truth_mask in sorted(geometry.truth.items()):
        pred_mask = geometry.predicted.get(finding_id)
        if pred_mask is None:
            continue
        outcome = geometry.outcome.get(finding_id)
        share = _coverage(truth_mask, pred_mask)
        if share is None:
            continue

        direction: str | None = None
        if outcome == "poor_localization" and share >= cover_hit:
            direction = MASK_OK_BOX_FAILS
        elif outcome == "correct" and share < cover_miss:
            direction = BOX_OK_MASK_FAILS
        if direction is None:
            continue

        found.append(
            storage.FindingRelationRow(
                finding_id=finding_id,
                relation=BOX_MASK_DISAGREEMENT,
                direction=direction,
                value=share,
                qualifies=True,
                cover_hit=cover_hit,
                best_coverage=share,
                threshold=cover_hit if direction == MASK_OK_BOX_FAILS else cover_miss,
            )
        )
    return found


def measure_image(
    rows: Sequence[Any], width: int, height: int, cover_hit: float, cover_miss: float
) -> list[storage.FindingRelationRow]:
    """Every relation on one image.

    Deterministic: findings are visited in id order and the rasterisation is
    integer, so two passes over one image produce identical rows in identical
    order.
    """
    geometry = build_geometry(rows, width, height)
    return merge_candidates(geometry, cover_hit) + box_mask_disagreements(
        geometry, cover_hit, cover_miss
    )


def analyse_run(connection: Any, run_id: int) -> dict[str, Any]:
    """Measure every relation in a run and persist them.

    Thresholds come from the run's own stored image diagnosis, never from
    :mod:`config`: a run analysed at one coverage threshold must not be
    described at another, and a run with no image diagnosis has no threshold to
    borrow.

    Writes to ``finding_relations`` alone. Findings, outcomes, mask findings,
    factor rates, clusters, recommendations, evaluations, image diagnoses and
    image coverage are read-only to this pass, and the golden invariant tests
    assert their contents are unchanged by it.

    Raises:
        RelationError: If the run does not exist, has no stored outlines, or
            has no image diagnosis to take thresholds from.
    """
    if storage.load_run(connection, run_id) is None:
        raise RelationError(f"No run with id {run_id}.")
    if not storage.has_table(connection, "mask_findings"):
        raise RelationError(
            f"Run {run_id} has no outline measurements, and relations are "
            f"measured on masks. Run: python -m app.mask_diagnosis --run {run_id}"
        )

    thresholds = connection.execute(
        "SELECT DISTINCT cover_hit, cover_miss FROM image_diagnoses WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    if not thresholds:
        raise RelationError(
            f"Run {run_id} has no image diagnosis, so there is no stored "
            f"coverage threshold to measure against. Run: "
            f"python -m app.image_diagnosis --run {run_id}"
        )
    if len(thresholds) > 1:
        raise RelationError(
            f"Run {run_id} stores {len(thresholds)} different coverage "
            "thresholds; re-run the image diagnosis so one applies throughout."
        )
    cover_hit = float(thresholds[0]["cover_hit"])
    cover_miss = float(thresholds[0]["cover_miss"])

    images = connection.execute(
        """
        SELECT DISTINCT f.image_id AS image_id, i.width AS width, i.height AS height
        FROM findings f JOIN images i ON i.id = f.image_id
        WHERE f.run_id = ?
        ORDER BY f.image_id
        """,
        (run_id,),
    ).fetchall()

    produced: list[storage.FindingRelationRow] = []
    for image in images:
        rows = connection.execute(
            """
            SELECT f.id, f.outcome, f.truth_polygon, m.pred_polygon
            FROM findings f LEFT JOIN mask_findings m ON m.finding_id = f.id
            WHERE f.run_id = ? AND f.image_id = ?
            ORDER BY f.id
            """,
            (run_id, image["image_id"]),
        ).fetchall()
        produced.extend(
            measure_image(
                rows, image["width"], image["height"], cover_hit, cover_miss
            )
        )

    written = storage.save_finding_relations(connection, run_id, produced)
    counts: dict[str, int] = {}
    for row in produced:
        key = (
            row.relation
            if row.direction is None
            else f"{row.relation}:{row.direction}"
        )
        counts[key] = counts.get(key, 0) + 1
    logger.info(
        "Measured %d relation(s) for run %d across %d image(s)",
        written,
        run_id,
        len(images),
    )
    return {
        "run_id": run_id,
        "relations": written,
        "counts": counts,
        "cover_hit": cover_hit,
        "cover_miss": cover_miss,
        "images": len(images),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command line for measuring one run's relations."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure relationships between findings. Additive evidence: no "
            "outcome, metric, factor rate, cluster or image verdict changes."
        )
    )
    parser.add_argument("--run", type=int, required=True, help="Run id to measure.")
    parser.add_argument(
        "--database", type=Path, default=None, help="Database, defaults to config."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Measure one run's relations and report what was found."""
    args = build_parser().parse_args(argv)
    try:
        with storage.connect(args.database) as connection:
            summary = analyse_run(connection, args.run)
    except (RelationError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"run {summary['run_id']}: {summary['relations']} relation(s) "
        f"over {summary['images']} image(s) "
        f"at cover_hit {summary['cover_hit']}, cover_miss {summary['cover_miss']}"
    )
    for key in sorted(summary["counts"]):
        print(f"  {key}: {summary['counts'][key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
