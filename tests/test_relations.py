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
    MASK_OK_BOX_FAILS,
    MERGE_CANDIDATE,
    RelationError,
    box_mask_disagreements,
    build_geometry,
    merge_candidates,
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

    def test_the_schema_reports_thirteen(self) -> None:
        """The new table is a schema change, and it is recorded."""
        assert storage.SCHEMA_VERSION == 13

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
