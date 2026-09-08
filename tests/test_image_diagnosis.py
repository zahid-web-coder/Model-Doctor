"""Image-level diagnosis: the verdict rules, and the promise not to disturb anything.

Two things are tested here. The rules themselves, on constructed coverage
matrices, so each verdict and its precedence is pinned without a database or an
image. And the guarantee this milestone rests on: running the pass leaves every
finding, outcome count and evaluation exactly as it was (D-017, D-040).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.image_diagnosis import (
    IMAGE_VERDICTS,
    ImageDiagnosisError,
    ImageEvidence,
    analyse_run,
    classify,
)
from model_doctor.app.storage import RunContext

HIT, MISS = 0.50, 0.25


def evidence(truths, preds, coverage, outcomes=None) -> ImageEvidence:
    """An image reduced to what a verdict needs."""
    return ImageEvidence(
        image_id=1,
        truth_ids=tuple(truths),
        pred_ids=tuple(preds),
        coverage=coverage,
        outcomes=outcomes or {},
    )


class TestVerdictRules:
    """Each verdict, on constructed geometry."""

    def test_nothing_to_find_and_nothing_found_is_empty(self) -> None:
        """Nothing to find and nothing found is empty."""
        assert classify(evidence([], [], {}), HIT, MISS)["verdict"] == "empty"

    def test_an_empty_image_is_never_clean(self) -> None:
        """A blank photograph is correct behaviour, not a success to count."""
        assert classify(evidence([], [], {}), HIT, MISS)["verdict"] != "clean"

    def test_one_object_one_prediction_is_clean(self) -> None:
        """One object one prediction is clean."""
        got = classify(evidence([1], [2], {(1, 2): 0.95}), HIT, MISS)
        assert got["verdict"] == "clean"
        assert got["merged"] is False and got["split"] is False

    def test_an_object_with_no_prediction_at_all_is_zero_prediction(self) -> None:
        """An object with no prediction at all is zero prediction."""
        got = classify(evidence([1], [], {}), HIT, MISS)
        assert got["verdict"] == "zero_prediction"

    def test_one_prediction_over_two_objects_is_merged(self) -> None:
        """The image-737 shape: one mask spanning two annotated objects."""
        got = classify(evidence([1, 2], [3], {(1, 3): 0.98, (2, 3): 0.80}), HIT, MISS)
        assert got["verdict"] == "merged"
        assert got["merged"] is True

    def test_two_predictions_on_one_object_is_split(self) -> None:
        """Two predictions on one object is split."""
        got = classify(evidence([1], [2, 3], {(1, 2): 0.60, (1, 3): 0.55}), HIT, MISS)
        assert got["verdict"] == "split"
        assert got["split"] is True

    def test_both_at_once_is_reported_as_both(self) -> None:
        """Both at once is reported as both."""
        cov = {(1, 3): 0.9, (2, 3): 0.9, (1, 4): 0.7}
        assert classify(evidence([1, 2], [3, 4], cov), HIT, MISS)["verdict"] == (
            "merged_and_split"
        )

    def test_an_untouched_object_beside_a_found_one_is_partly_missed(self) -> None:
        """An untouched object beside a found one is partly missed."""
        got = classify(evidence([1, 2], [3], {(1, 3): 0.95}), HIT, MISS)
        assert got["verdict"] == "partly_missed"
        assert got["objects_untouched"] == 1

    def test_a_prediction_touching_nothing_is_spurious(self) -> None:
        """A prediction touching nothing is spurious."""
        got = classify(evidence([1], [2, 3], {(1, 2): 0.95}), HIT, MISS)
        assert got["verdict"] == "spurious"
        assert got["predictions_on_nothing"] == 1

    def test_missed_and_spurious_together_are_named_together(self) -> None:
        """Missed and spurious together are named together."""
        got = classify(evidence([1, 2], [3, 4], {(1, 3): 0.95}), HIT, MISS)
        assert got["verdict"] == "partly_missed_and_spurious"

    def test_an_object_reached_but_not_taken_is_not_clean(self) -> None:
        """The band between the thresholds must be named, not dropped.

        Image 629 in the reference data: one object covered 0.47 by the
        prediction matched to its neighbour. Above the untouched bar and below
        the found-it bar, it fell through every branch and the image — which
        holds a real false negative — was reported clean.
        """
        got = classify(
            evidence([1, 2], [3], {(1, 3): 0.97, (2, 3): 0.47}), HIT, MISS
        )
        assert got["verdict"] == "partial_coverage"
        assert got["objects_partial"] == 1
        assert got["objects_untouched"] == 0, "0.47 is reached, not untouched"
        assert got["merged"] is False, "0.47 is below the found-it bar"

    def test_a_prediction_that_reaches_without_taking_is_not_clean(self) -> None:
        """The mirror of the same band, on the prediction side.

        A second prediction covers the object at 0.34: not a second hit, so no
        split, and not aimed at nothing, so not spurious. Left unnamed, an
        image holding a false positive was reported clean.
        """
        got = classify(
            evidence([1], [2, 3], {(1, 2): 0.99, (1, 3): 0.34}), HIT, MISS
        )
        assert got["verdict"] == "partial_coverage"
        assert got["predictions_partial"] == 1
        assert got["predictions_on_nothing"] == 0
        assert got["split"] is False

    def test_a_poor_localisation_may_sit_in_a_clean_image(self) -> None:
        """Not a contradiction: two lenses measuring different things.

        The finding-level outcome judges box overlap; this pass judges mask
        coverage. An outline covering 99% of an object whose box scored 0.28
        IoU is exactly that, and reporting both is the point of a second lens
        (D-036).
        """
        got = classify(
            evidence([1], [1], {(1, 1): 0.99}, {"poor_localization": 1}), HIT, MISS
        )
        assert got["verdict"] == "clean"
        assert got["poor_localization"] == 1

    def test_a_clean_image_has_every_object_accounted_for(self) -> None:
        """Clean means found, not merely not-missed."""
        ev = evidence([1, 2], [3, 4], {(1, 3): 0.95, (2, 4): 0.92})
        got = classify(ev, HIT, MISS)
        assert got["verdict"] == "clean"
        assert got["objects_partial"] == 0

    def test_every_verdict_is_declared(self) -> None:
        """Nothing can be produced that the schema has not published."""
        cases = [
            evidence([], [], {}),
            evidence([1], [], {}),
            evidence([1], [2], {(1, 2): 0.95}),
            evidence([1, 2], [3], {(1, 3): 0.98, (2, 3): 0.80}),
            evidence([1], [2, 3], {(1, 2): 0.6, (1, 3): 0.55}),
            evidence([1, 2], [3, 4], {(1, 3): 0.9, (2, 3): 0.9, (1, 4): 0.7}),
            evidence([1, 2], [3], {(1, 3): 0.95}),
            evidence([1], [2, 3], {(1, 2): 0.95}),
            evidence([1, 2], [3, 4], {(1, 3): 0.95}),
            evidence([1, 2], [3], {(1, 3): 0.97, (2, 3): 0.47}),
            evidence([1], [2, 3], {(1, 2): 0.99, (1, 3): 0.34}),
        ]
        for case in cases:
            assert classify(case, HIT, MISS)["verdict"] in IMAGE_VERDICTS


class TestPrecedence:
    """Which verdict wins when several apply, and why it matters."""

    def test_a_merge_outranks_the_miss_it_causes(self) -> None:
        """The whole point of this pass.

        One prediction covers two objects. The matcher can pair it only once,
        so the second object is a false negative at finding level — but the
        model did cover it, and calling the image "partly missed" would repeat
        the error this milestone exists to correct.
        """
        got = classify(evidence([1, 2], [3], {(1, 3): 0.98, (2, 3): 0.80}), HIT, MISS)
        assert got["verdict"] == "merged"
        assert got["objects_untouched"] == 0, "0.80 is covered, not untouched"

    def test_zero_prediction_never_becomes_partly_missed(self) -> None:
        """Attempting nothing is a different failure from attempting and missing."""
        assert classify(evidence([1, 2], [], {}), HIT, MISS)["verdict"] == (
            "zero_prediction"
        )


class TestThresholds:
    """Thresholds are reported, not hidden."""

    def test_the_hit_threshold_decides_merged(self) -> None:
        """A pair at 0.60 is a merge at hit=0.5 and is not at hit=0.7."""
        ev = evidence([1, 2], [3], {(1, 3): 0.98, (2, 3): 0.60})
        assert classify(ev, 0.50, MISS)["verdict"] == "merged"
        assert classify(ev, 0.70, MISS)["verdict"] != "merged"

    def test_the_miss_threshold_decides_untouched(self) -> None:
        """The miss threshold decides untouched."""
        ev = evidence([1, 2], [3], {(1, 3): 0.95, (2, 3): 0.20})
        assert classify(ev, HIT, 0.25)["objects_untouched"] == 1
        assert classify(ev, HIT, 0.10)["objects_untouched"] == 0

    def test_defaults_come_from_configuration(self) -> None:
        """Defaults come from configuration."""
        ev = evidence([1, 2], [3], {(1, 3): 0.98, (2, 3): 0.60})
        assert classify(ev)["verdict"] == classify(
            ev, config.IMAGE_COVER_HIT, config.IMAGE_COVER_MISS
        )["verdict"]

    def test_wrong_class_is_not_an_image_verdict(self) -> None:
        """A class error is a finding-level fact, carried through untouched.

        The object was found; naming the wrong class is not a shape of
        image-level mistake, so the image is clean and the count is preserved.
        """
        got = classify(
            evidence([1], [1], {(1, 1): 0.95}, {"wrong_class": 1}), HIT, MISS
        )
        assert got["verdict"] == "clean"
        assert got["wrong_class"] == 1


@pytest.fixture()
def run_with_masks(tmp_path: Path) -> tuple[Path, int]:
    """A run whose single image has two objects and one prediction spanning both."""
    database = tmp_path / "img.db"

    def square(x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt", model_sha256="sha-img", dataset_yaml="d.yaml",
                split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
                localization_iou_floor=0.1, image_size=640,
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?,'/i.jpg','i.jpg',100,100,1,2)",
            (run_id,),
        )
        image_id = connection.execute(
            "SELECT id FROM images WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        ids = []
        for outcome, poly in (
            ("poor_localization", square(10, 10, 40, 40)),
            ("false_negative", square(50, 50, 80, 80)),
        ):
            cur = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, "
                "truth_polygon) VALUES (?,?,?,'a',?)",
                (run_id, image_id, outcome, __import__("json").dumps(poly)),
            )
            ids.append(cur.lastrowid)
        # one predicted mask spanning both annotated squares
        storage.save_mask_findings(
            connection, run_id,
            [(ids[0], 0.2, "poor_localization", square(5, 5, 90, 90))],
        )
    return database, run_id


class TestPassAgainstADatabase:
    """The pass end to end, and what it must not disturb."""

    def test_a_merge_is_recorded_with_the_pair_that_proves_it(
        self, run_with_masks: tuple[Path, int]
    ) -> None:
        """A merge is recorded with the pair that proves it."""
        database, run_id = run_with_masks
        with storage.connect(database) as connection:
            analyse_run(connection, run_id)
            rows = storage.load_image_diagnoses(connection, run_id, with_coverage=True)
        assert len(rows) == 1
        row = rows[0]
        assert row.verdict == "merged"
        assert row.merged is True
        assert row.cover_hit == config.IMAGE_COVER_HIT
        assert row.method == "mask"
        covered = {c.truth_finding_id for c in row.coverage}
        assert len(covered) == 2, "both objects traced to the one prediction"

    def test_rerunning_replaces_rather_than_duplicates(
        self, run_with_masks: tuple[Path, int]
    ) -> None:
        """A pass over a saved run must not leave two verdicts for one image."""
        database, run_id = run_with_masks
        with storage.connect(database) as connection:
            analyse_run(connection, run_id)
            analyse_run(connection, run_id, cover_hit=0.9)
            rows = storage.load_image_diagnoses(connection, run_id)
            pairs = connection.execute(
                "SELECT COUNT(*) FROM image_coverage"
            ).fetchone()[0]
        assert len(rows) == 1
        assert rows[0].cover_hit == 0.9, "the newer threshold is the stored one"
        assert pairs > 0

    def test_findings_and_outcomes_are_untouched(
        self, run_with_masks: tuple[Path, int]
    ) -> None:
        """D-017 and every existing count survive the pass unchanged."""
        database, run_id = run_with_masks
        with storage.connect(database) as connection:
            sql = (
                "SELECT id, outcome, class_name, truth_polygon "
                "FROM findings ORDER BY id"
            )
            before = connection.execute(sql).fetchall()
            counts_before = storage.outcome_counts(connection, run_id)
            analyse_run(connection, run_id)
            after = connection.execute(sql).fetchall()
            counts_after = storage.outcome_counts(connection, run_id)
        assert [tuple(r) for r in before] == [tuple(r) for r in after]
        assert counts_before == counts_after

    def test_a_run_without_outlines_is_refused_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """No box fallback: a wrong measurement is worse than a missing one."""
        database = tmp_path / "nomask.db"
        with storage.connect(database) as connection:
            run_id = storage.save_run(
                connection,
                RunContext(
                    model_path="m.pt", model_sha256="s", dataset_yaml="d.yaml",
                    split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
                    localization_iou_floor=0.1, image_size=640,
                ),
            )
            connection.execute("DROP TABLE IF EXISTS mask_findings")
            with pytest.raises(ImageDiagnosisError, match="mask"):
                analyse_run(connection, run_id)

    def test_an_unknown_run_is_refused(self, run_with_masks: tuple[Path, int]) -> None:
        """An unknown run is refused."""
        database, _ = run_with_masks
        with (
            storage.connect(database) as connection,
            pytest.raises(ImageDiagnosisError, match="No run"),
        ):
            analyse_run(connection, 999)

    def test_the_optional_table_degrades_to_empty(self, tmp_path: Path) -> None:
        """A database saved before version 11 reports nothing, and does not fail."""
        database = tmp_path / "old.db"
        with storage.connect(database) as connection:
            connection.execute("DROP TABLE IF EXISTS image_diagnoses")
            assert storage.load_image_diagnoses(connection, 1) == []
            assert storage.load_image_coverage(connection, 1, 1) == []


class TestSummary:
    """The aggregate an API or MCP consumer receives."""

    def test_empty_images_are_reported_separately_from_clean(
        self, run_with_masks: tuple[Path, int]
    ) -> None:
        """Empty images are reported separately from clean."""
        from model_doctor.app.comparison import image_summary

        database, run_id = run_with_masks
        with storage.connect(database) as connection:
            analyse_run(connection, run_id)
            rows = storage.load_image_diagnoses(connection, run_id)
        summary = image_summary(rows)
        assert summary is not None
        assert "empty" in summary
        assert summary["clean"] + summary["affected"] == summary["scored"]
        assert summary["thresholds"]["cover_hit"] == config.IMAGE_COVER_HIT

    def test_no_diagnoses_summarise_to_none_not_zero(self) -> None:
        """Not measured must not read as measured-and-perfect."""
        from model_doctor.app.comparison import image_summary

        assert image_summary([]) is None
