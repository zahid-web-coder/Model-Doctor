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
is undefined, and :func:`model_doctor.app.comparison.factor_qualifies` would reject it.
It is
stored apart and never enters that pipeline.

**Two of the three relations invent no threshold.** ``merge_candidate`` and
``box_mask_disagreement`` reuse bounds the run already stored, so neither
involves a judgement call about where a boundary sits.
``prediction_on_matched_object`` is the exception: it is stored as a continuous
measurement, and ``duplicate_prediction`` — a *reading* of that measurement, not
a row of its own — is the one place in this module where a number had to be
chosen. It is `provisional` for that reason, gates nothing, and carries both of
its bounds on every row it labels. See :data:`DUPLICATE_MIN_ONOBJECT` for what
that choice does and does not rest on.

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

from model_doctor.app import storage
from model_doctor.app.image_diagnosis import rasterise
from model_doctor.utils.exceptions import ModelDoctorError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RelationError(ModelDoctorError):
    """Relations could not be measured, with a reason the caller can act on."""


#: A ground-truth object the matcher paired with a prediction. These are the
#: outcomes that carry both a truth polygon and a predicted one.
MATCHED_OUTCOMES: tuple[str, ...] = ("correct", "poor_localization", "wrong_class")

MERGE_CANDIDATE: str = "merge_candidate"
BOX_MASK_DISAGREEMENT: str = "box_mask_disagreement"

#: How much of an unmatched prediction lies on objects the model already found.
#:
#: **Named for what it measures, not for what it might mean.** The obvious name
#: would be ``duplicate_prediction``, and the obvious name would be a verdict:
#: it would assert that a prediction overlapping a found object is a redundant
#: copy of it, on a boundary nobody has agreed. This measures the quantity and
#: stops. Whether some share of it constitutes a duplicate is a later decision,
#: made against a threshold that will have to be argued for.
#:
#: Every row this produces carries ``qualifies = False``. Nothing is classified.
PREDICTION_ON_MATCHED_OBJECT: str = "prediction_on_matched_object"

#: Every relation this module stores. ``duplicate_prediction`` is deliberately
#: absent: it is a *reading* of :data:`PREDICTION_ON_MATCHED_OBJECT`, not a row
#: of its own, so one relationship still produces exactly one row.
RELATIONS: tuple[str, ...] = (
    MERGE_CANDIDATE,
    BOX_MASK_DISAGREEMENT,
    PREDICTION_ON_MATCHED_OBJECT,
)

#: The name for a :data:`PREDICTION_ON_MATCHED_OBJECT` measurement that meets
#: the rule below. A label a reader applies, never an outcome and never a cause.
#:
#: **The name is narrower than the phenomenon, and the taxonomy stays
#: provisional for that reason.** What the rule actually selects is *an extra
#: prediction associated with an object the model had already found*. Measured
#: over the 106 labelled rows in runs 4, 5 and 12, that population is not one
#: thing:
#:
#: * **redundant duplicates — 91 of 106 (86%).** The prediction lies at least
#:   half inside the prediction the matcher already accepted. "Duplicate" is the
#:   right word for these.
#: * **complementary fragments — 10 of 106 (9%).** The prediction sits on the
#:   same annotated object but barely touches the accepted one, covering a part
#:   it left out: the bottom steps of a flight whose top was found, the strip
#:   below where the first mask stopped. Still an extra prediction for one
#:   object, but nothing is duplicated.
#: * **annotation and matcher granularity — a bounded error class.** The
#:   prediction is a defensible reading of the scene that the annotation or the
#:   assignment settled differently: a dogleg staircase annotated as one object
#:   and detected as two flights, or a prediction that fully covers a truth this
#:   run recorded as missed. Calling these duplicates describes the annotation,
#:   not the prediction. Two turned up among sixteen reviewed images; exactly
#:   one of the 106 covers an unmatched truth at or above ``cover_hit``, so the
#:   class is real and small, not systemic.
#:
#: Splitting the label along those lines is a naming decision with its own
#: evidence to gather, not a threshold to move. Until it is made, every row
#: carries the single provisional label and the caveat travels with it.
DUPLICATE_PREDICTION: str = "duplicate_prediction"

#: How much weight that label can bear. `provisional` in the sense
#: :mod:`model_doctor.app.recommendations` uses: worth reporting, not yet
#: worth acting
# on.
DUPLICATE_STATUS: str = "provisional"

#: How much of the prediction must lie on ground the model already found.
#:
#: **An operational cutoff in a sparse middle region, not an empirically
#: established boundary.** It was originally placed inside an empty interval
#: [0.0900, 0.2249], and that justification did not survive. Pooling the three
#: independent configurations of one checkpoint (runs 4, 5 and 12; 169
#: measurements) put two values inside the interval and collapsed the widest gap
#: to 0.0648 — *narrower* than random scatter typically produces over the same
#: range, with two thirds of simulated arrangements giving a wider one. On the
#: two configurations it was chosen from, the interval was already only
#: suggestive, never significant. Read plainly: there is no evidence of a
#: natural boundary here, at 0.15 or anywhere else in the middle.
#:
#: The number is nonetheless close to inert, for a different reason than the one
#: first claimed. The distribution is strongly bimodal — 30% of measurements at
#: the floor, 57% above 0.50, only 14% in between — and every value in
#: [0.0905, 0.2245] selects the identical set, because
#: :data:`DUPLICATE_COVERAGE_FLOOR` removes what lies near the cutoff on both
#: sides. Across the whole plausible range 0.01 to 0.50 the selection moves by
#: 14 rows in 169, roughly half the sensitivity of the guard.
#:
#: So it is frozen at 0.15 not because it is right but because no value
#: available is measurably different, and re-choosing one on the same data that
#: falsified the first choice would be fitting to the latest run.
DUPLICATE_MIN_ONOBJECT: float = 0.15

#: How much of *some one* found object it must cover.
#:
#: **A guard, not a second threshold.** To duplicate an object a prediction has
#: to substantially overlap that object; a prediction spread thinly across
#: several objects' edges duplicates none of them. Set far below the run's
#: ``cover_hit`` on purpose: it excludes "touches nothing", never "covers a
#: little". Raising it to ``cover_hit`` was measured and is *worse* — it drops
#: fragments and half-bands that are visibly second detections of an object
#: already found.
#:
#: **This was the weakest part of the rule and is no longer.** It once rested on
#: a single observed case. Sixteen labelled rows covering less than 25% of any
#: one object were then reviewed against their images, nine of them from a split
#: the rule was never derived on: **none was diffuse or spurious**, and twelve
#: read as an extra prediction on an object already found. The case nearest the
#: floor, at coverage 0.055, lies 97.7% inside the prediction the matcher had
#: already accepted — so raising the floor even to 0.06 would discard a real
#: one. Low coverage among these rows means "detected a piece of it", not
#: "detected nothing": they are single steps of multi-step flights and end
#: slices of foreshortened slabs.
DUPLICATE_COVERAGE_FLOOR: float = 0.05

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


def is_duplicate_like(
    row: storage.FindingRelationRow,
    min_onobject: float = DUPLICATE_MIN_ONOBJECT,
    coverage_floor: float = DUPLICATE_COVERAGE_FLOOR,
) -> bool:
    """Whether one measurement reads as a duplicate under the provisional rule.

    **An interpretation, not a measurement.** The stored row is the same either
    way; this says how to read it, and a caller passing different bounds gets a
    different reading of the identical number. That is why no threshold is
    baked into what is written.

    Says nothing about *why* the prediction exists. Suppression settings,
    decoding, assignment order and a model genuinely proposing two objects are
    all consistent with a measurement like this, and none of them is measured
    here.
    """
    if row.relation != PREDICTION_ON_MATCHED_OBJECT:
        return False
    return row.value >= min_onobject and (row.best_coverage or 0.0) >= coverage_floor


def matched_union(geometry: ImageGeometry) -> Any | None:
    """The ground truth on this image that some prediction is already assigned to.

    **A bitwise OR, never a sum of pairwise intersections.** Two annotations may
    overlap each other — adjacent flights of one staircase share a boundary —
    and adding their intersections with a prediction would count the shared
    pixels twice, producing a fraction above 1.0 for a prediction that lies
    entirely on annotated ground. The union is exact and cannot do that.

    Only matched objects contribute. An object the matcher could not pair is a
    failure of its own, and a prediction landing on it is not evidence that the
    model already found it — which is the whole claim this measurement exists to
    support. Returns ``None`` when nothing on the image is matched.
    """
    union = None
    for finding_id, mask in sorted(geometry.truth.items()):
        if geometry.outcome.get(finding_id) not in MATCHED_OUTCOMES:
            continue
        union = mask if union is None else (union | mask)
    return union


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


def prediction_on_matched_objects(
    geometry: ImageGeometry, cover_hit: float
) -> list[storage.FindingRelationRow]:
    """Measure how much of each unmatched prediction lies on found objects.

    ``onobject_matched(p) = |p AND union(matched truths)| / |p|``, with the
    union taken by :func:`matched_union` and the denominator the prediction's
    own area — the one quantity ``image_coverage`` does not store, since its
    coverage divides by the *object's* area instead.

    **A row for every measurable false positive, including the zeros.** A
    prediction touching nothing matched measures 0.0, and that is a
    measurement: dropping it would leave the stored distribution unable to say
    how many predictions land nowhere near a found object, which is most of
    them. Only a prediction whose mask has no area at all is skipped — an
    undefined ratio is not a zero one.

    ``qualifies`` records whether the measurement meets
    :func:`is_duplicate_like`'s provisional rule, with both bounds stored beside
    it so a later reader can see what produced the flag and re-read the value
    against different ones. The measurement itself is unaffected by the rule.

    ``span`` stays a separate qualifier rather than part of the rule: a
    prediction covering several already-found objects raises a question about
    annotation granularity, and there are three such measurements in the whole
    reference database — far too few to put inside a threshold.
    """
    union = matched_union(geometry)
    found: list[storage.FindingRelationRow] = []

    for finding_id, pred_mask in sorted(geometry.predicted.items()):
        if geometry.outcome.get(finding_id) != "false_positive":
            continue
        area = int(pred_mask.sum())
        if not area:
            continue

        on_object = 0.0 if union is None else int((pred_mask & union).sum()) / area

        # Which found object this prediction sits on most, and how many it
        # covers substantially. Both are measured against matched truths only,
        # for the same reason the union is.
        best, partner, span = 0.0, None, 0
        for other_id, truth_mask in sorted(geometry.truth.items()):
            if geometry.outcome.get(other_id) not in MATCHED_OUTCOMES:
                continue
            share = _coverage(truth_mask, pred_mask)
            if share is None:
                continue
            if share > best:
                best, partner = share, other_id
            if share >= cover_hit:
                span += 1

        found.append(
            storage.FindingRelationRow(
                finding_id=finding_id,
                relation=PREDICTION_ON_MATCHED_OBJECT,
                value=on_object,
                qualifies=(
                    on_object >= DUPLICATE_MIN_ONOBJECT
                    and best >= DUPLICATE_COVERAGE_FLOOR
                ),
                cover_hit=cover_hit,
                partner_finding_id=partner,
                best_coverage=best,
                span=span,
                threshold=DUPLICATE_MIN_ONOBJECT,
                coverage_floor=DUPLICATE_COVERAGE_FLOOR,
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
    return (
        merge_candidates(geometry, cover_hit)
        + box_mask_disagreements(geometry, cover_hit, cover_miss)
        + prediction_on_matched_objects(geometry, cover_hit)
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
            "measured on masks. Run: python -m model_doctor.app.mask_diagnosis "
            f"--run {run_id}"
        )

    thresholds = connection.execute(
        "SELECT DISTINCT cover_hit, cover_miss FROM image_diagnoses WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    if not thresholds:
        raise RelationError(
            f"Run {run_id} has no image diagnosis, so there is no stored "
            f"coverage threshold to measure against. Run: "
            f"python -m model_doctor.app.image_diagnosis --run {run_id}"
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
