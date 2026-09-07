"""Relationships between findings: the rules, and the promise to disturb nothing.

Two things are tested here. The rules themselves, on constructed geometry, so
each relation and its attribution side is pinned without a database. And the
guarantee the whole step rests on: measuring relations leaves every finding,
outcome, metric, factor rate, cluster, recommendation and image verdict exactly
as it was.

That second guarantee is why the golden test compares **full table contents**
rather than counts. A pass that rewrote a finding's confidence while keeping the
outcome would satisfy a count check and still have destroyed the thing this step
promised not to touch.

``duplicate_prediction`` is deliberately absent. Its threshold rests on four
configurations, and a number chosen now would be one nobody could later argue
with — so the two relations that reuse thresholds the run already stored are
implemented first, and this file tests only those.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app import relations, storage
from app.relations import (
    BOX_MASK_DISAGREEMENT,
    BOX_OK_MASK_FAILS,
    DUPLICATE_COVERAGE_FLOOR,
    DUPLICATE_MIN_ONOBJECT,
    DUPLICATE_PREDICTION,
    DUPLICATE_STATUS,
    MASK_OK_BOX_FAILS,
    MERGE_CANDIDATE,
    PREDICTION_ON_MATCHED_OBJECT,
    RelationError,
    box_mask_disagreements,
    build_geometry,
    is_duplicate_like,
    matched_union,
    merge_candidates,
    prediction_on_matched_objects,
)
from app.storage import RunContext

HIT, MISS = 0.50, 0.25
SIZE = 100

#: Every table this pass must leave untouched. Compared by full contents.
UNTOUCHED: tuple[str, ...] = (
    "findings",
    "mask_findings",
    "root_causes",
    "factor_rates",
    "clusters",
    "cluster_members",
    "recommendations",
    "run_evaluations",
    "image_diagnoses",
    "image_coverage",
)


def square(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def row(
    finding_id: int,
    outcome: str,
    truth: list[list[int]] | None = None,
    pred: list[list[int]] | None = None,
) -> dict[str, Any]:
    """One finding as the pass reads it, without a database."""

    class Row(dict):
        def keys(self):  # noqa: D102 - mimics sqlite3.Row
            return list(super().keys())

    return Row(
        id=finding_id,
        outcome=outcome,
        truth_polygon=json.dumps(truth) if truth else None,
        pred_polygon=json.dumps(pred) if pred else None,
    )


def geometry(rows: list[dict[str, Any]]):
    return build_geometry(rows, SIZE, SIZE)


class TestMergeCandidate:
    """A missed object that a prediction assigned elsewhere already covers."""

    def test_the_reference_case(self) -> None:
        """Image 737's shape: one prediction across two annotated objects.

        The matcher pairs it with the first and reports the second as never
        seen. It was seen; it was attributed elsewhere.
        """
        found = merge_candidates(
            geometry(
                [
                    row(
                        1,
                        "poor_localization",
                        square(10, 10, 40, 40),
                        square(5, 5, 90, 90),
                    ),
                    row(2, "false_negative", square(50, 50, 80, 80)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].finding_id == 2, "attributed to the missed object"
        assert found[0].partner_finding_id == 1, "names the covering prediction"
        assert found[0].relation == MERGE_CANDIDATE
        assert found[0].value == pytest.approx(1.0)

    def test_the_prediction_is_never_the_one_labelled(self) -> None:
        """One relationship, one row. Labelling both ends would double-count."""
        found = merge_candidates(
            geometry(
                [
                    row(1, "correct", square(10, 10, 40, 40), square(5, 5, 90, 90)),
                    row(2, "false_negative", square(50, 50, 80, 80)),
                ]
            ),
            HIT,
        )
        assert {f.finding_id for f in found} == {2}

    def test_an_unmatched_prediction_covering_a_miss_is_not_a_merge(self) -> None:
        """The matcher must have made a choice for there to be a merge.

        A false positive covering a missed object is two failures on one image.
        Calling it a merge would claim an attribution that never happened.
        """
        found = merge_candidates(
            geometry(
                [
                    row(1, "false_positive", None, square(5, 5, 90, 90)),
                    row(2, "false_negative", square(50, 50, 80, 80)),
                ]
            ),
            HIT,
        )
        assert found == []

    def test_coverage_below_the_threshold_is_not_a_merge(self) -> None:
        """The object must be substantially covered, not merely touched."""
        found = merge_candidates(
            geometry(
                [
                    row(1, "correct", square(0, 0, 20, 20), square(0, 0, 50, 20)),
                    # 10 of 40 columns overlap: coverage 0.25, below hit.
                    row(2, "false_negative", square(40, 0, 80, 20)),
                ]
            ),
            HIT,
        )
        assert found == []

    def test_only_false_negatives_are_considered(self) -> None:
        """A correct or poorly localised finding is already paired."""
        found = merge_candidates(
            geometry(
                [
                    row(1, "correct", square(10, 10, 40, 40), square(5, 5, 90, 90)),
                    row(2, "correct", square(50, 50, 80, 80), square(50, 50, 80, 80)),
                ]
            ),
            HIT,
        )
        assert found == []

    def test_the_strongest_coverer_is_the_partner(self) -> None:
        """When two predictions cover it, the row names the one that covers most."""
        found = merge_candidates(
            geometry(
                [
                    row(1, "correct", square(0, 0, 10, 10), square(20, 0, 60, 40)),
                    row(2, "correct", square(90, 90, 99, 99), square(20, 0, 99, 40)),
                    row(3, "false_negative", square(20, 0, 60, 40)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].partner_finding_id == 1, "covers it fully; #2 covers it too"
        assert found[0].span == 2, "multiplicity is a column, not a second row"

    def test_the_stored_threshold_travels_with_the_row(self) -> None:
        """A run measured at one threshold must not be read at another."""
        found = merge_candidates(
            geometry(
                [
                    row(1, "correct", square(10, 10, 40, 40), square(5, 5, 90, 90)),
                    row(2, "false_negative", square(50, 50, 80, 80)),
                ]
            ),
            0.9,
        )
        assert found[0].cover_hit == 0.9
        assert found[0].threshold == 0.9


class TestBoxMaskDisagreement:
    """A finding whose box verdict and mask geometry tell different stories."""

    def test_mask_covers_while_the_box_did_not_match(self) -> None:
        """The 8-of-11 case on run 5: the outline was right, the box was loose."""
        found = box_mask_disagreements(
            geometry(
                [
                    row(
                        1,
                        "poor_localization",
                        square(10, 10, 90, 90),
                        square(10, 10, 90, 90),
                    )
                ]
            ),
            HIT,
            MISS,
        )
        assert len(found) == 1
        assert found[0].relation == BOX_MASK_DISAGREEMENT
        assert found[0].direction == MASK_OK_BOX_FAILS
        assert found[0].finding_id == 1
        assert found[0].partner_finding_id is None, "it relates to its own geometry"

    def test_box_matched_while_the_mask_is_barely_on_the_object(self) -> None:
        """The other direction, which measures zero on every stored run.

        Implemented so that zero is a measurement rather than an assumption.
        """
        found = box_mask_disagreements(
            geometry(
                [row(1, "correct", square(0, 0, 100, 100), square(0, 0, 100, 10))]
            ),
            HIT,
            MISS,
        )
        assert len(found) == 1
        assert found[0].direction == BOX_OK_MASK_FAILS

    def test_agreement_produces_nothing(self) -> None:
        """A poor localisation whose mask is also poor is not a disagreement."""
        found = box_mask_disagreements(
            geometry(
                [
                    row(
                        1,
                        "poor_localization",
                        square(0, 0, 100, 100),
                        square(0, 0, 100, 5),
                    )
                ]
            ),
            HIT,
            MISS,
        )
        assert found == []

    def test_a_correct_finding_with_a_good_mask_produces_nothing(self) -> None:
        """Agreement in the other direction produces nothing either."""
        found = box_mask_disagreements(
            geometry(
                [row(1, "correct", square(10, 10, 90, 90), square(10, 10, 90, 90))]
            ),
            HIT,
            MISS,
        )
        assert found == []

    def test_a_false_negative_has_no_prediction_to_disagree_with(self) -> None:
        """With no prediction there is no mask to contradict the box."""
        found = box_mask_disagreements(
            geometry([row(1, "false_negative", square(10, 10, 90, 90))]), HIT, MISS
        )
        assert found == []


class TestPredictionOnMatchedObject:
    """The continuous measurement, and what it deliberately does not do."""

    def test_a_prediction_entirely_on_a_found_object_measures_one(self) -> None:
        """All of it lies on an object the model already found."""
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 100, 100), square(0, 0, 50, 50)),
                    row(2, "false_positive", None, square(10, 10, 40, 40)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].finding_id == 2
        assert found[0].relation == PREDICTION_ON_MATCHED_OBJECT
        assert found[0].value == pytest.approx(1.0)

    def test_a_prediction_touching_nothing_measures_zero_and_is_still_recorded(
        self,
    ) -> None:
        """Zero is a measurement.

        Dropping these rows would leave the stored distribution unable to say
        how many predictions land nowhere near a found object — which is most
        of them, and the reason the distribution has a shape at all.
        """
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 20, 20), square(0, 0, 20, 20)),
                    row(2, "false_positive", None, square(60, 60, 90, 90)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].value == 0.0
        assert found[0].partner_finding_id is None

    def test_an_unmatched_object_does_not_count_as_already_found(self) -> None:
        """The semantic case this measurement turns on.

        A prediction landing on a *missed* object is not evidence that the
        model already found it — nothing found it. Including false negatives in
        the union would make every spurious prediction on a missed object look
        like a redundant copy of a detection that never happened. Run 5's
        #1191 is exactly this: its only overlap is with the unmatched #1193,
        and it must measure 0.000.
        """
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "false_negative", square(0, 0, 100, 100)),
                    row(2, "false_positive", None, square(10, 10, 40, 40)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].value == 0.0, "an unmatched object is not an already-found one"

    def test_a_partly_overlapping_prediction_measures_the_share(self) -> None:
        """The measurement is continuous, not a yes or no."""
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 100, 50), square(0, 0, 100, 50)),
                    # Half of it lies on the object, half below it.
                    row(2, "false_positive", None, square(0, 25, 100, 75)),
                ]
            ),
            HIT,
        )
        assert found[0].value == pytest.approx(0.5, abs=0.02)

    def test_overlapping_annotations_cannot_push_it_above_one(self) -> None:
        """The union-versus-sum case.

        Two annotations sharing pixels would, under a sum of pairwise
        intersections, count the shared area twice and report more than all of
        a prediction lying on annotated ground.
        """
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 60, 100), square(0, 0, 60, 100)),
                    row(2, "correct", square(40, 0, 100, 100), square(40, 0, 100, 100)),
                    row(3, "false_positive", None, square(0, 0, 100, 100)),
                ]
            ),
            HIT,
        )
        assert len(found) == 1
        assert found[0].value <= 1.0
        assert found[0].value == pytest.approx(1.0)

    def test_only_false_positives_are_measured(self) -> None:
        """A matched prediction is not an unmatched one."""
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 50, 50), square(0, 0, 50, 50)),
                    row(
                        2,
                        "poor_localization",
                        square(60, 60, 90, 90),
                        square(60, 60, 90, 90),
                    ),
                ]
            ),
            HIT,
        )
        assert found == []

    def test_a_zero_area_prediction_produces_no_row(self) -> None:
        """No denominator, so no measurement — not a zero."""
        off_canvas = square(2 * SIZE, 2 * SIZE, 3 * SIZE, 3 * SIZE)
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 50, 50), square(0, 0, 50, 50)),
                    row(2, "false_positive", None, off_canvas),
                ]
            ),
            HIT,
        )
        assert found == []

    def test_the_measurement_does_not_depend_on_the_interpretation(self) -> None:
        """The value is the geometry; `qualifies` is a reading of it.

        Step 5 stored these rows with nothing classified. Step 6 added a
        provisional rule, and the measured value must be identical either way —
        which is what makes re-reading at other bounds a query rather than a
        re-run.
        """
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 100, 100), square(0, 0, 50, 50)),
                    row(2, "false_positive", None, square(10, 10, 40, 40)),
                ]
            ),
            HIT,
        )
        assert found[0].value == pytest.approx(1.0)
        assert found[0].threshold is not None, "the bound that decided it is stored"

    def test_the_best_covered_object_is_named(self) -> None:
        """The partner is the object this prediction sits on most.

        Object 1 extends below the prediction and is half covered; object 2 is
        covered entirely. Both clear the hit threshold, so the span is two and
        the partner is the stronger of them.
        """
        found = prediction_on_matched_objects(
            geometry(
                [
                    row(1, "correct", square(0, 0, 10, 80), square(0, 0, 10, 80)),
                    row(2, "correct", square(0, 0, 100, 40), square(0, 0, 100, 40)),
                    row(3, "false_positive", None, square(0, 0, 100, 40)),
                ]
            ),
            HIT,
        )
        assert found[0].partner_finding_id == 2
        assert found[0].best_coverage == pytest.approx(1.0)
        assert found[0].span == 2, "it substantially covers both"


class TestTheProvisionalDuplicateRule:
    """`duplicate_prediction` as a reading of the stored measurement.

    Every case here was chosen from the reference database and inspected as an
    image before the rule was written, so the fixtures reproduce shapes that
    were judged by looking rather than by the number being tested.
    """

    def measure(self, rows):
        """Measure one constructed image at the standard hit threshold."""
        return prediction_on_matched_objects(geometry(rows), HIT)

    def test_an_obvious_duplicate_qualifies(self) -> None:
        """Two outlines on one found object — run 5 #1165's shape."""
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 80, 80), square(0, 0, 80, 80)),
                row(2, "false_positive", None, square(5, 5, 75, 75)),
            ]
        )
        assert len(got) == 1
        assert got[0].qualifies is True
        assert is_duplicate_like(got[0])

    def test_a_diffuse_prediction_covering_nothing_does_not(self) -> None:
        """Run 4 #830: onobject 0.265 but it covers no object above 2.5%.

        The guard exists for exactly this. Without it a prediction spread over
        several objects' edges would read as a duplicate of all of them.
        """
        # A large found object, and a small prediction clipping its edge: half
        # the prediction is on found ground, yet it covers 2% of the object.
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 90, 100), square(0, 0, 90, 100)),
                row(2, "false_positive", None, square(80, 0, 100, 20)),
            ]
        )
        assert len(got) == 1
        assert got[0].value >= DUPLICATE_MIN_ONOBJECT, "it is on found ground"
        assert got[0].best_coverage < DUPLICATE_COVERAGE_FLOOR, "but covers nothing"
        assert got[0].qualifies is False
        assert not is_duplicate_like(got[0])

    def test_a_fragment_inside_a_found_object_qualifies(self) -> None:
        """A band or sliver inside a staircase that was already found.

        Run 5 #1212 and #1091 cover 16% and 7% of their objects. A fragment
        is still a second detection of something already detected,
        which is why the guard sits far below `cover_hit`. Requiring
        `best_coverage >= cover_hit` would drop these.
        """
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 100, 100), square(0, 0, 100, 100)),
                row(2, "false_positive", None, square(0, 0, 100, 12)),
            ]
        )
        assert len(got) == 1
        assert got[0].value == pytest.approx(1.0), "wholly inside the object"
        assert got[0].best_coverage < 0.25, "yet covers little of it"
        assert got[0].qualifies is True
        assert not (got[0].best_coverage >= HIT), (
            "a cover_hit rule would have excluded this fragment"
        )

    def test_the_1022_versus_1223_distinction(self) -> None:
        """The pair that decided the rule, reproduced as geometry.

        #1022 is a prediction sprawling across the frame that happens to
        contain a small object: high coverage of it, but almost none of the
        prediction is on found ground. #1223 is a second outline on a small
        object, four times its area. Coverage alone calls both duplicates;
        looking at the images says only the second is one.
        """
        sprawl = self.measure(
            [
                row(1, "correct", square(0, 0, 9, 100), square(0, 0, 9, 100)),
                row(2, "false_positive", None, square(0, 0, 100, 100)),
            ]
        )[0]
        second_outline = self.measure(
            [
                row(1, "correct", square(0, 0, 50, 50), square(0, 0, 50, 50)),
                row(2, "false_positive", None, square(0, 0, 100, 50)),
            ]
        )[0]
        assert sprawl.best_coverage >= 0.75, "the small object is well covered"
        assert sprawl.qualifies is False, "#1022: a sprawl, not a duplicate"
        assert second_outline.best_coverage >= 0.75
        assert second_outline.qualifies is True, "#1223: a second outline"

    def test_the_threshold_boundary(self) -> None:
        """Inclusive at the bound, and nothing qualifies just below it."""
        r = storage.FindingRelationRow(
            finding_id=1,
            relation=PREDICTION_ON_MATCHED_OBJECT,
            value=DUPLICATE_MIN_ONOBJECT,
            qualifies=True,
            cover_hit=HIT,
            best_coverage=DUPLICATE_COVERAGE_FLOOR,
        )
        assert is_duplicate_like(r)
        from dataclasses import replace

        assert not is_duplicate_like(replace(r, value=DUPLICATE_MIN_ONOBJECT - 1e-9))
        assert not is_duplicate_like(
            replace(r, best_coverage=DUPLICATE_COVERAGE_FLOOR - 1e-9)
        )

    def test_an_unmatched_object_cannot_make_a_duplicate(self) -> None:
        """Run 5 #1191: its only overlap is with an object nothing found.

        Landing on a *missed* object is not a second detection of anything.
        """
        got = self.measure(
            [
                row(1, "false_negative", square(0, 0, 100, 100)),
                row(2, "false_positive", None, square(10, 10, 90, 90)),
            ]
        )
        assert got[0].value == 0.0
        assert got[0].qualifies is False

    def test_span_stays_a_separate_qualifier(self) -> None:
        """Run 5 #1287: one prediction over two flights found individually.

        `span` is recorded and is not part of the rule. Three measurements in
        the whole reference database carry it, which is far too few to gate on,
        and what it raises is a question about annotation granularity rather
        than about duplication.
        """
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 45, 100), square(0, 0, 45, 100)),
                row(2, "correct", square(55, 0, 100, 100), square(55, 0, 100, 100)),
                row(3, "false_positive", None, square(0, 0, 100, 100)),
            ]
        )
        assert got[0].span == 2
        assert got[0].qualifies is True
        # The rule reaches the same verdict with span ignored entirely.
        assert is_duplicate_like(got[0])

    def test_confidence_is_never_consulted(self) -> None:
        """It carries no discriminating power and is not a condition.

        Measured across 184 false positives: mean 0.522 among duplicate-like
        and 0.520 among the rest, with a best achievable accuracy equal to the
        base rate.
        """
        import inspect

        source = inspect.getsource(relations.prediction_on_matched_objects)
        source += inspect.getsource(is_duplicate_like)
        assert "confidence" not in source

    def test_the_rule_is_reported_as_provisional(self) -> None:
        """It describes; it does not yet license anything."""
        assert DUPLICATE_STATUS == "provisional"
        assert DUPLICATE_PREDICTION == "duplicate_prediction"

    def test_the_label_is_not_a_stored_relation(self) -> None:
        """One relationship, one row. The label is a reading of that row."""
        assert DUPLICATE_PREDICTION not in relations.RELATIONS

    def test_both_bounds_travel_with_the_row(self) -> None:
        """A reader must be able to see what produced the flag."""
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 80, 80), square(0, 0, 80, 80)),
                row(2, "false_positive", None, square(5, 5, 75, 75)),
            ]
        )
        assert got[0].threshold == DUPLICATE_MIN_ONOBJECT
        assert got[0].coverage_floor == DUPLICATE_COVERAGE_FLOOR

    def test_a_caller_may_read_it_at_different_bounds(self) -> None:
        """The stored measurement is unchanged by how it is interpreted."""
        got = self.measure(
            [
                row(1, "correct", square(0, 0, 100, 100), square(0, 0, 100, 100)),
                row(2, "false_positive", None, square(0, 0, 100, 12)),
            ]
        )[0]
        assert is_duplicate_like(got)
        assert not is_duplicate_like(got, coverage_floor=HIT), (
            "the same row reads differently under a stricter guard"
        )


class TestMatchedUnion:
    """The union itself, since everything above rests on it."""

    def test_only_matched_objects_are_in_it(self) -> None:
        """A missed object contributes no pixels to the union."""
        union = matched_union(
            geometry(
                [
                    row(1, "correct", square(0, 0, 10, 10), square(0, 0, 10, 10)),
                    row(2, "false_negative", square(50, 50, 90, 90)),
                ]
            )
        )
        assert union is not None
        assert int(union.sum()) == pytest.approx(int(union.sum()))
        # The missed object's area must not appear in the union.
        assert union[70, 70] == 0
        assert union[5, 5] == 1

    def test_an_image_with_nothing_matched_has_no_union(self) -> None:
        """Absent, not empty: there is nothing to measure against."""
        union = matched_union(
            geometry([row(1, "false_negative", square(0, 0, 50, 50))])
        )
        assert union is None


class TestGeometrySafety:
    """What the pass does when geometry cannot be measured."""

    def test_a_zero_area_object_is_not_measured_rather_than_zero(self) -> None:
        """An undefined ratio is not a zero one.

        A polygon entirely outside the canvas rasterises to nothing. A
        degenerate ring inside it does not — `fillPoly` fills a repeated point
        as one pixel — so this uses the case that genuinely has no area.
        """
        off_canvas = square(2 * SIZE, 2 * SIZE, 3 * SIZE, 3 * SIZE)
        found = box_mask_disagreements(
            geometry([row(1, "poor_localization", off_canvas, square(0, 0, 90, 90))]),
            HIT,
            MISS,
        )
        assert found == []

    def test_an_unrasterisable_polygon_is_skipped_and_reported(self) -> None:
        """A bad polygon must not fail the run, nor be recorded as an absence."""
        broken = row(1, "poor_localization", None, None)
        broken["truth_polygon"] = json.dumps([[[1, 2], [3, 4]], [[5, 6]]])
        got = geometry([broken])
        assert 1 in got.skipped
        assert 1 not in got.truth

    def test_a_zero_area_object_never_becomes_a_disagreement(self) -> None:
        """The guard matters in exactly one direction, and this is it.

        With no guard, a zero-area object divides by a substituted 1 and yields
        coverage 0.0 — which is below `cover_miss`, so a *correct* finding
        would be reported as a box/mask disagreement on the strength of an
        object that has no area to disagree about.
        """
        off_canvas = square(2 * SIZE, 2 * SIZE, 3 * SIZE, 3 * SIZE)
        found = box_mask_disagreements(
            geometry([row(1, "correct", off_canvas, square(0, 0, 90, 90))]),
            HIT,
            MISS,
        )
        assert found == []

    def test_an_image_without_dimensions_measures_nothing(self) -> None:
        """No canvas means nothing can be measured, not that nothing is there."""
        got = build_geometry([row(1, "correct", square(0, 0, 10, 10))], 0, 0)
        assert got.truth == {} and got.predicted == {}
        assert got.skipped == (1,)


@pytest.fixture()
def analysed(tmp_path: Path) -> tuple[Path, int]:
    """A run with an image diagnosis, so relations have a threshold to use."""
    from app.image_diagnosis import analyse_run as diagnose

    database = tmp_path / "relations.db"
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="r" * 64,
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?,'/i.jpg','i.jpg',?,?,1,2)",
            (run_id, SIZE, SIZE),
        )
        image_id = connection.execute(
            "SELECT id FROM images WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        ids = []
        for outcome, truth in (
            ("poor_localization", square(10, 10, 40, 40)),
            ("false_negative", square(50, 50, 80, 80)),
        ):
            cursor = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, "
                "truth_polygon) VALUES (?,?,?,'a',?)",
                (run_id, image_id, outcome, json.dumps(truth)),
            )
            ids.append(int(cursor.lastrowid))
        storage.save_mask_findings(
            connection,
            run_id,
            [(ids[0], 0.2, "poor_localization", square(5, 5, 90, 90))],
        )
        diagnose(connection, run_id)
    return database, run_id


def dump(connection: sqlite3.Connection, table: str) -> list[tuple]:
    """Every row of a table, ordered, as comparable tuples."""
    if not storage.has_table(connection, table):
        return []
    return [
        tuple(r) for r in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
    ]


class TestGoldenInvariants:
    """Measuring relations must change nothing that already existed."""

    def test_every_other_table_is_byte_identical(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Full contents, not counts.

        A pass that rewrote a confidence while preserving an outcome would
        satisfy a count check and still have destroyed what this step promised
        to leave alone.
        """
        database, run_id = analysed
        with storage.connect(database) as connection:
            before = {table: dump(connection, table) for table in UNTOUCHED}
            relations.analyse_run(connection, run_id)
            after = {table: dump(connection, table) for table in UNTOUCHED}
        for table in UNTOUCHED:
            assert after[table] == before[table], f"{table} changed"

    def test_outcome_counts_are_unchanged(self, analysed: tuple[Path, int]) -> None:
        """The five-outcome tally is exactly as the matcher left it."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            query = (
                "SELECT outcome, COUNT(*) FROM findings WHERE run_id=? GROUP BY outcome"
            )
            before = connection.execute(query, (run_id,)).fetchall()
            relations.analyse_run(connection, run_id)
            assert connection.execute(query, (run_id,)).fetchall() == before

    def test_image_verdicts_are_unchanged(self, analysed: tuple[Path, int]) -> None:
        """The image pass owns verdicts; this one may not move them."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            before = [
                (d.image_id, d.verdict, d.merged, d.split)
                for d in storage.load_image_diagnoses(connection, run_id)
            ]
            relations.analyse_run(connection, run_id)
            after = [
                (d.image_id, d.verdict, d.merged, d.split)
                for d in storage.load_image_diagnoses(connection, run_id)
            ]
        assert after == before

    def test_only_the_relation_table_gains_rows(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Scoped writes, asserted over every table in the database."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            tables = [
                r[0]
                for r in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            ]
            before = {t: dump(connection, t) for t in tables}
            relations.analyse_run(connection, run_id)
            after = {t: dump(connection, t) for t in tables}
        changed = {t for t in tables if before[t] != after[t]}
        assert changed == {"finding_relations"}, f"unexpected writes to {changed}"

    def test_the_schema_records_the_relation_work(self) -> None:
        """13 added the table; 14 widened it for the second bound.

        Pinned as the presence of the relation work rather than as the current
        version number, which later steps move for reasons of their own —
        schema 15 recorded inference provenance, and that says nothing about
        whether *this* step's schema is intact.
        """
        assert storage.SCHEMA_VERSION >= 14
        assert ("finding_relations", "coverage_floor", "REAL") in storage._ADDED_COLUMNS

    def test_the_outcome_taxonomy_is_untouched(self) -> None:
        """Relations are additive evidence; the five outcomes are the matcher's."""
        from app.diagnosis import Outcome

        assert {o.value for o in Outcome} == {
            "correct",
            "wrong_class",
            "poor_localization",
            "false_positive",
            "false_negative",
        }

    def test_relations_never_reach_the_factor_pipeline(
        self, analysed: tuple[Path, int]
    ) -> None:
        """A relation has no control rate, so a factor row for one is a bug."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            factors = {
                r["factor"]
                for r in connection.execute(
                    "SELECT factor FROM root_causes WHERE run_id=?", (run_id,)
                )
            } | {r.factor for r in storage.load_factor_rates(connection, run_id)}
        assert not factors & set(relations.RELATIONS)


class TestPassBehaviour:
    """Determinism, idempotence, and refusing to guess a threshold."""

    def test_two_passes_produce_identical_rows(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Deterministic: integer rasterisation, findings visited in id order."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            first = storage.load_finding_relations(connection, run_id)
            relations.analyse_run(connection, run_id)
            second = storage.load_finding_relations(connection, run_id)
        assert first == second

    def test_rerunning_replaces_rather_than_appends(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Idempotent, like every other pass over an already-saved run."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            relations.analyse_run(connection, run_id)
            count = connection.execute(
                "SELECT COUNT(*) FROM finding_relations WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            loaded = len(storage.load_finding_relations(connection, run_id))
        assert count == loaded

    def test_a_run_without_an_image_diagnosis_is_refused(
        self, tmp_path: Path
    ) -> None:
        """There is no threshold to borrow, and inventing one is worse."""
        database = tmp_path / "bare.db"
        with storage.connect(database) as connection:
            run_id = storage.save_run(
                connection,
                RunContext(
                    model_path="m.pt",
                    model_sha256="s" * 64,
                    dataset_yaml="d.yaml",
                    split="test",
                    confidence_threshold=0.25,
                    match_iou_threshold=0.5,
                    localization_iou_floor=0.1,
                    image_size=640,
                ),
            )
            with pytest.raises(RelationError, match="image_diagnosis"):
                relations.analyse_run(connection, run_id)

    def test_the_runs_own_threshold_decides_not_a_default(
        self, tmp_path: Path
    ) -> None:
        """A run analysed at one coverage threshold is measured at that one.

        The fixture's merge sits at coverage 0.60: it is a merge at the default
        0.50 and is not one at 0.90. Re-running the image diagnosis moves the
        stored threshold, and the relation pass must follow it rather than any
        constant of its own.
        """
        from app.image_diagnosis import analyse_run as diagnose

        database = tmp_path / "threshold.db"
        with storage.connect(database) as connection:
            run_id = storage.save_run(
                connection,
                RunContext(
                    model_path="m.pt",
                    model_sha256="t" * 64,
                    dataset_yaml="d.yaml",
                    split="test",
                    confidence_threshold=0.25,
                    match_iou_threshold=0.5,
                    localization_iou_floor=0.1,
                    image_size=640,
                ),
            )
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?,'/i.jpg','i.jpg',?,?,1,2)",
                (run_id, SIZE, SIZE),
            )
            image_id = connection.execute(
                "SELECT id FROM images WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            ids = []
            for outcome, truth in (
                ("correct", square(0, 0, 10, 10)),
                # 60 of its 100 columns fall inside the prediction below.
                ("false_negative", square(0, 20, 100, 40)),
            ):
                ids.append(
                    int(
                        connection.execute(
                            "INSERT INTO findings (run_id, image_id, outcome, "
                            "class_name, truth_polygon) VALUES (?,?,?,'a',?)",
                            (run_id, image_id, outcome, json.dumps(truth)),
                        ).lastrowid
                    )
                )
            storage.save_mask_findings(
                connection, run_id, [(ids[0], 0.9, "correct", square(0, 0, 60, 60))]
            )

            diagnose(connection, run_id, cover_hit=0.5, cover_miss=0.25)
            relations.analyse_run(connection, run_id)
            at_half = storage.load_finding_relations(
                connection, run_id, MERGE_CANDIDATE
            )
            assert len(at_half) == 1, "0.60 coverage is a merge at hit 0.50"
            assert at_half[0].cover_hit == 0.5

            diagnose(connection, run_id, cover_hit=0.9, cover_miss=0.25)
            relations.analyse_run(connection, run_id)
            at_nine = storage.load_finding_relations(
                connection, run_id, MERGE_CANDIDATE
            )
        assert at_nine == [], "0.60 coverage is not a merge at hit 0.90"

    def test_an_unknown_run_is_refused(self, analysed: tuple[Path, int]) -> None:
        """A run that does not exist is an error, not an empty measurement."""
        database, _ = analysed
        with (
            storage.connect(database) as connection,
            pytest.raises(RelationError, match="No run"),
        ):
            relations.analyse_run(connection, 999)

    def test_duplicate_prediction_is_not_produced_yet(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Its threshold is not settled, so the pass must not pretend it is."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            produced = {
                r.relation for r in storage.load_finding_relations(connection, run_id)
            }
        assert "duplicate_prediction" not in produced
        assert "duplicate_prediction" not in relations.RELATIONS


class TestPartnerIntegrity:
    """A relation must point at something real."""

    def test_every_partner_exists_in_the_same_run_and_image(
        self, analysed: tuple[Path, int]
    ) -> None:
        """A relation must point at a real, matched finding beside it."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            rows = connection.execute(
                "SELECT finding_id, partner_finding_id FROM finding_relations "
                "WHERE run_id=? AND partner_finding_id IS NOT NULL",
                (run_id,),
            ).fetchall()
            assert rows, "the fixture should produce at least one partnered relation"
            for r in rows:
                pair = connection.execute(
                    "SELECT a.run_id, a.image_id, b.run_id, b.image_id, b.outcome "
                    "FROM findings a JOIN findings b ON b.id=? WHERE a.id=?",
                    (r["partner_finding_id"], r["finding_id"]),
                ).fetchone()
                assert pair is not None
                assert pair[0] == pair[2] == run_id
                assert pair[1] == pair[3], "partner must be on the same image"
                assert pair[4] in relations.MATCHED_OUTCOMES

    def test_a_finding_carries_at_most_one_relation(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Outcome makes the two mutually exclusive; this asserts it holds."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            worst = connection.execute(
                "SELECT MAX(n) FROM (SELECT COUNT(*) n FROM finding_relations "
                "WHERE run_id=? GROUP BY finding_id)",
                (run_id,),
            ).fetchone()[0]
        assert worst == 1

    def test_the_unique_constraint_refuses_a_second_row(
        self, analysed: tuple[Path, int]
    ) -> None:
        """The anti-double-counting rule is a constraint, not a convention."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            existing = connection.execute(
                "SELECT finding_id, relation FROM finding_relations WHERE run_id=? "
                "LIMIT 1",
                (run_id,),
            ).fetchone()
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO finding_relations (run_id, finding_id, relation, "
                    "value, qualifies, cover_hit, created_at) "
                    "VALUES (?,?,?,1.0,1,0.5,'now')",
                    (run_id, existing["finding_id"], existing["relation"]),
                )


class TestDeleteAndFootprint:
    """Derived data must not outlive the run it describes."""

    def test_the_footprint_counts_relations(
        self, analysed: tuple[Path, int]
    ) -> None:
        """What a delete would destroy has to include this table."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            counts = storage.run_footprint(connection, run_id)
        assert counts["finding_relations"] > 0

    def test_deleting_a_run_removes_its_relations(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Derived data must not outlive the run it describes."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            removed = storage.delete_run(connection, run_id)
            left = connection.execute(
                "SELECT COUNT(*) FROM finding_relations WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        assert removed["finding_relations"] > 0
        assert left == 0

    def test_relations_of_another_run_survive(self, analysed: tuple[Path, int]) -> None:
        """Deleting one run must not reach into another's relations.

        The second run gets its own findings rather than borrowing the first's:
        a relation pointing at a deleted finding *should* cascade away, and a
        test that shared findings would be asserting the opposite.
        """
        database, run_id = analysed
        with storage.connect(database) as connection:
            other = storage.save_run(
                connection,
                RunContext(
                    model_path="m.pt",
                    model_sha256="z" * 64,
                    dataset_yaml="d.yaml",
                    split="test",
                    confidence_threshold=0.25,
                    match_iou_threshold=0.5,
                    localization_iou_floor=0.1,
                    image_size=640,
                ),
            )
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?,'/o.jpg','o.jpg',?,?,1,1)",
                (other, SIZE, SIZE),
            )
            other_image = connection.execute(
                "SELECT id FROM images WHERE run_id=?", (other,)
            ).fetchone()[0]
            other_finding = int(
                connection.execute(
                    "INSERT INTO findings (run_id, image_id, outcome, class_name) "
                    "VALUES (?,?,'false_negative','a')",
                    (other, other_image),
                ).lastrowid
            )
            storage.save_finding_relations(
                connection,
                other,
                [
                    storage.FindingRelationRow(
                        finding_id=other_finding,
                        relation=MERGE_CANDIDATE,
                        value=0.9,
                        qualifies=True,
                        cover_hit=HIT,
                    )
                ],
            )
            relations.analyse_run(connection, run_id)
            storage.delete_run(connection, run_id)
            left = connection.execute(
                "SELECT COUNT(*) FROM finding_relations WHERE run_id=?", (other,)
            ).fetchone()[0]
        assert left == 1


class TestLoading:
    """What a reader gets back."""

    def test_a_run_measured_before_the_pass_existed_reads_as_empty(
        self, analysed: tuple[Path, int]
    ) -> None:
        """Empty means not measured.

        A caller must say so rather than report "no relationships found".
        """
        database, run_id = analysed
        with storage.connect(database) as connection:
            assert storage.load_finding_relations(connection, run_id) == []

    def test_filtering_by_relation(self, analysed: tuple[Path, int]) -> None:
        """A caller can ask for one relation without reading them all."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            relations.analyse_run(connection, run_id)
            merges = storage.load_finding_relations(
                connection, run_id, MERGE_CANDIDATE
            )
        assert merges and all(r.relation == MERGE_CANDIDATE for r in merges)

    def test_a_database_without_the_table_degrades_to_empty(
        self, analysed: tuple[Path, int]
    ) -> None:
        """A partially migrated database must not raise (SCHEMA.md §6)."""
        database, run_id = analysed
        with storage.connect(database) as connection:
            connection.execute("DROP TABLE IF EXISTS finding_relations")
            assert storage.load_finding_relations(connection, run_id) == []
