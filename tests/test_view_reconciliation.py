"""The Failures view and the Images view must never disagree about a run.

Two screens describe the same analysis through different tables. Failures pages
``findings``; Images reads the per-image tallies in ``image_diagnoses``. Nothing
recomputes one from the other at read time, so a change to either write path can
let them drift — and drift here is silent. Both screens keep rendering, each
confident, and a reader has no way to tell which one is lying.

**These are reconciliation invariants, not classifier rules.** The verdict rules
live in ``test_image_diagnosis.py`` and are not restated here. What is pinned is
the relationship between the two views: per image, not only in aggregate, since
totals can agree while individual images are wrong in opposite directions.

**The mask-vs-box distinction is preserved, not flattened.** A ``clean`` image
may hold a ``poor_localization`` finding: this pass measures mask coverage and
the finding-level outcome measures box overlap, so an outline covering 99% of an
object whose box scored 0.28 IoU is both (D-036). Asserting that away would
convert a documented design decision into a test failure. What must never happen
is a ``clean`` image holding a *structural* failure — a false negative or false
positive — because those are exactly what the pass claims to have ruled out.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from model_doctor.app import storage
from model_doctor.app.image_diagnosis import analyse_run
from model_doctor.app.storage import RunContext

#: Outcomes that reference an annotated object.
ANNOTATED = ("correct", "wrong_class", "poor_localization", "false_negative")
#: Outcomes that reference something the model predicted.
PREDICTED = ("correct", "wrong_class", "poor_localization", "false_positive")
#: Failures the image pass claims to have ruled out when it says "clean".
STRUCTURAL = ("false_negative", "false_positive", "wrong_class")


def square(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@pytest.fixture()
def mixed_run(tmp_path: Path) -> tuple[Path, int, dict[str, int]]:
    """A run whose images cover every shape the two views can disagree about.

    Built to include the case that motivated the image pass at all: image 737 of
    the reference run, where one prediction stretched across two annotated
    staircases. The one-to-one matcher paired it with the first object and
    reported the second as a false negative — "never saw it" — when the same
    prediction covered 80% of it. Failures reports two findings, Images reports
    one merge, and both are correct. Nothing in this file may make that a
    conflict.
    """
    database = tmp_path / "reconcile.db"
    images: dict[str, int] = {}

    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="sha-reconcile",
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )

        def add_image(name: str, truths: int, preds: int) -> int:
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?,?,?,200,200,?,?)",
                (run_id, f"/{name}.jpg", f"{name}.jpg", preds, truths),
            )
            image_id = connection.execute(
                "SELECT id FROM images WHERE run_id=? AND filename=?",
                (run_id, f"{name}.jpg"),
            ).fetchone()[0]
            images[name] = image_id
            return image_id

        def add_finding(
            image_id: int,
            outcome: str,
            truth: list[list[int]] | None,
            iou: float | None,
        ) -> int:
            cursor = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, "
                "truth_polygon, iou) VALUES (?,?,?,'staircase',?,?)",
                (
                    run_id,
                    image_id,
                    outcome,
                    json.dumps(truth) if truth else None,
                    iou,
                ),
            )
            return int(cursor.lastrowid)

        masks: list[tuple[int, float, str, list[list[int]]]] = []

        # The image-737 shape: two annotated objects, one prediction spanning
        # both. Paired with the first, so the second reads as a false negative.
        merged_image = add_image("merged", truths=2, preds=1)
        paired = add_finding(
            merged_image, "poor_localization", square(10, 10, 40, 40), 0.20
        )
        add_finding(merged_image, "false_negative", square(50, 50, 80, 80), None)
        masks.append((paired, 0.20, "poor_localization", square(5, 5, 90, 90)))

        # Clean: one object, one prediction, well covered and well matched.
        clean_image = add_image("clean", truths=1, preds=1)
        clean_finding = add_finding(
            clean_image, "correct", square(10, 10, 90, 90), 0.93
        )
        masks.append((clean_finding, 0.93, "correct", square(12, 12, 88, 88)))

        # Mask right, box loose — the 26-image case (D-036). The outline covers
        # the object almost entirely while the box scores below the match
        # threshold, so Images says clean and Failures says poor localization.
        loose_image = add_image("loose", truths=1, preds=1)
        loose_finding = add_finding(
            loose_image, "poor_localization", square(20, 20, 80, 80), 0.28
        )
        masks.append((loose_finding, 0.28, "poor_localization", square(18, 18, 82, 82)))

        # A prediction aimed at nothing.
        spurious_image = add_image("spurious", truths=1, preds=2)
        kept = add_finding(spurious_image, "correct", square(10, 10, 60, 60), 0.88)
        extra = add_finding(spurious_image, "false_positive", None, None)
        masks.append((kept, 0.88, "correct", square(12, 12, 58, 58)))
        masks.append((extra, 0.40, "false_positive", square(150, 150, 190, 190)))

        # An object nothing reached at all.
        missed_image = add_image("missed", truths=1, preds=0)
        add_finding(missed_image, "false_negative", square(30, 30, 70, 70), None)

        # Partly missed: one object found, a second nowhere near a prediction.
        # **This image is what makes the clean invariant bite.** Without it
        # every missed object in this fixture sits on an image with no
        # predictions at all, which is classified `zero_prediction` long before
        # the untouched branch is reached — so a bug that called a partly
        # missed image clean would pass unnoticed. That is image 629's bug
        # exactly: a real false negative inside an image reported clean.
        partly_image = add_image("partly", truths=2, preds=1)
        found = add_finding(partly_image, "correct", square(10, 10, 60, 60), 0.91)
        add_finding(partly_image, "false_negative", square(120, 120, 180, 180), None)
        masks.append((found, 0.91, "correct", square(12, 12, 58, 58)))

        # Image 629's shape: an object reached but not taken. Its single object
        # was covered 0.472 — above the "untouched" bar, below the "found it"
        # one — so before that band was named, an image holding a real false
        # negative was reported clean. Reproduced here because a reconciliation
        # test that cannot see the band cannot guard against its return.
        # 47 of the object's 100x100 pixels are covered: coverage 0.47.
        partial_image = add_image("partial", truths=1, preds=1)
        add_finding(partial_image, "false_negative", square(20, 20, 120, 120), None)
        reached = add_finding(partial_image, "false_positive", None, None)
        masks.append((reached, 0.30, "false_positive", square(20, 20, 67, 120)))

        # Nothing to find and nothing found — correct behaviour, and separate
        # from clean so it cannot inflate a clean rate.
        add_image("empty", truths=0, preds=0)

        storage.save_mask_findings(connection, run_id, masks)

    return database, run_id, images


def read_both(database: Path, run_id: int):
    """The two views, as each screen loads them."""
    with storage.connect(database) as connection:
        analyse_run(connection, run_id)
        diagnoses = storage.load_image_diagnoses(connection, run_id)
        findings = storage.load_findings_page(
            connection, run_id, limit=10_000, offset=0
        )
    per_image: dict[int, Counter] = {}
    for finding in findings:
        per_image.setdefault(finding.image_id, Counter())[finding.outcome] += 1
    return {d.image_id: d for d in diagnoses}, per_image, findings


class TestPerImageReconciliation:
    """Image by image, because totals can agree while images do not."""

    def test_every_outcome_tally_matches_the_findings_on_that_image(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """Per image and per outcome, the two views count the same findings."""
        database, run_id, _ = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)

        for image_id, counts in per_image.items():
            row = diagnoses[image_id]
            for outcome, expected in row.outcomes.items():
                assert counts.get(outcome, 0) == expected, (
                    f"image {image_id}: Failures counts {counts.get(outcome, 0)} "
                    f"{outcome}, Images claims {expected}"
                )

    def test_totals_agree_whichever_view_they_are_summed_from(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """The run-level counts a reader compares between screens."""
        database, run_id, _ = mixed_run
        diagnoses, _, findings = read_both(database, run_id)

        from_failures = Counter(f.outcome for f in findings)
        from_images = Counter()
        for row in diagnoses.values():
            from_images.update(row.outcomes)
        for outcome in set(from_failures) | set(from_images):
            assert from_failures[outcome] == from_images[outcome], (
                f"{outcome}: Failures totals {from_failures[outcome]}, "
                f"Images totals {from_images[outcome]}"
            )

    def test_every_image_holding_a_finding_is_diagnosed(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """A finding on an undiagnosed image is invisible to the Images view."""
        database, run_id, _ = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)
        assert set(per_image) <= set(diagnoses)

    def test_a_diagnosed_image_with_no_findings_is_empty(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """The only image that may hold nothing is the one with nothing to hold."""
        database, run_id, _ = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)
        for image_id, row in diagnoses.items():
            if image_id not in per_image:
                assert row.verdict == "empty", (
                    f"image {image_id} has no findings but is called {row.verdict}"
                )

    def test_counts_cover_the_findings_that_reference_them(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """An image cannot hold more findings than it has objects or predictions."""
        database, run_id, _ = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)
        for image_id, counts in per_image.items():
            row = diagnoses[image_id]
            assert row.gt_count >= sum(counts.get(o, 0) for o in ANNOTATED), (
                f"image {image_id} reports {row.gt_count} objects but "
                "Failures references more"
            )
            assert row.pred_count >= sum(counts.get(o, 0) for o in PREDICTED), (
                f"image {image_id} reports {row.pred_count} predictions but "
                "Failures references more"
            )


class TestCleanMeansClean:
    """What the word must and must not be allowed to hide."""

    def test_a_clean_image_never_holds_a_structural_failure(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """No false negative or false positive may hide inside a clean verdict.

        This is the invariant that would have caught image 629, where an object
        covered 0.472 fell between the two thresholds and an image holding a
        real false negative reported clean.
        """
        database, run_id, _ = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)
        for image_id, row in diagnoses.items():
            if row.verdict != "clean":
                continue
            held = {o: per_image.get(image_id, Counter()).get(o, 0) for o in STRUCTURAL}
            assert not any(held.values()), (
                f"image {image_id} is called clean but Failures reports {held}"
            )

    def test_mask_clean_with_a_loose_box_stays_representable(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """The documented mask-vs-box case must survive these invariants.

        A `clean` image holding a `poor_localization` finding is D-036 working
        as designed — the outline was right and the box was loose — and the two
        views disagreeing about *that* is the point of having both. The UI words
        it "Mask clean · box loose" rather than plain "Clean"; this pins the data
        shape that wording depends on.
        """
        database, run_id, images = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)

        row = diagnoses[images["loose"]]
        assert row.verdict == "clean", "coverage was above the hit threshold"
        assert row.outcomes["poor_localization"] == 1, "the loose box is still reported"
        assert per_image[images["loose"]]["poor_localization"] == 1
        # And it is a qualified clean, not a structural failure in disguise.
        assert row.outcomes["false_negative"] == 0
        assert row.outcomes["false_positive"] == 0

    def test_an_empty_image_is_not_counted_as_clean(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """Nothing to find and nothing found must not inflate the clean rate."""
        database, run_id, images = mixed_run
        diagnoses, _, _ = read_both(database, run_id)
        row = diagnoses[images["empty"]]
        assert row.verdict == "empty"
        assert row.gt_count == 0 and row.pred_count == 0


class TestTheMergedCase:
    """Image 737: one prediction across two objects, reported by both views."""

    def test_both_views_describe_the_merge_without_contradicting_each_other(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """Two findings and one merge are the same event seen twice.

        Failures sees a poor localization and a false negative, because the
        matcher pairs one prediction with one object. Images sees one prediction
        covering both. Neither is wrong, and the reconciliation must hold
        anyway — this is the case that motivated the second view.
        """
        database, run_id, images = mixed_run
        diagnoses, per_image, _ = read_both(database, run_id)

        image_id = images["merged"]
        row = diagnoses[image_id]
        assert row.verdict == "merged"
        assert row.merged is True
        # Failures still reports both findings, unchanged by the merge verdict.
        assert per_image[image_id]["poor_localization"] == 1
        assert per_image[image_id]["false_negative"] == 1
        # And the tallies agree with them.
        assert row.outcomes["poor_localization"] == 1
        assert row.outcomes["false_negative"] == 1

    def test_the_false_negative_is_traceable_to_the_prediction_that_caused_it(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """The merge is evidenced, not asserted.

        Without the stored coverage pair the Images view would claim a merge a
        reader could not check against the Failures row it explains.
        """
        database, run_id, images = mixed_run
        with storage.connect(database) as connection:
            analyse_run(connection, run_id)
            rows = storage.load_image_diagnoses(connection, run_id, with_coverage=True)
        row = next(r for r in rows if r.image_id == images["merged"])
        covered = {c.truth_finding_id for c in row.coverage}
        assert len(covered) == 2, "both objects trace to the one prediction"

    def test_reconciliation_survives_a_threshold_change(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """Verdicts move with the coverage threshold; the tallies must not.

        The counts come from the findings, so a reader who changes a threshold
        should see the *shape* of the mistake re-described without the two views
        starting to disagree about how many findings exist.
        """
        database, run_id, _ = mixed_run
        with storage.connect(database) as connection:
            analyse_run(connection, run_id, cover_hit=0.95, cover_miss=0.90)
            diagnoses = storage.load_image_diagnoses(connection, run_id)
            findings = storage.load_findings_page(
                connection, run_id, limit=10_000, offset=0
            )
        per_image: dict[int, Counter] = {}
        for finding in findings:
            per_image.setdefault(finding.image_id, Counter())[finding.outcome] += 1

        for row in diagnoses:
            counts = per_image.get(row.image_id, Counter())
            for outcome, tally in row.outcomes.items():
                assert tally == counts.get(outcome, 0), (
                    f"image {row.image_id}: {outcome} drifted when the threshold moved"
                )


class TestTheApiServesTheSameNumbers:
    """The screens read HTTP, so the reconciliation has to hold there too."""

    def test_the_per_image_filter_returns_what_the_tallies_claim(
        self, mixed_run: tuple[Path, int, dict[str, int]]
    ) -> None:
        """`?image_id=` is what the Images lightbox draws from.

        A filter that returned a different set from the tallies beside it would
        put the contradiction inside one screen rather than between two.
        """
        database, run_id, images = mixed_run
        with storage.connect(database) as connection:
            analyse_run(connection, run_id)
            diagnoses = {
                d.image_id: d for d in storage.load_image_diagnoses(connection, run_id)
            }
            for name, image_id in images.items():
                page = storage.load_findings_page(
                    connection, run_id, limit=100, offset=0, image_id=image_id
                )
                counts = Counter(f.outcome for f in page)
                row = diagnoses[image_id]
                for outcome, tally in row.outcomes.items():
                    assert counts.get(outcome, 0) == tally, f"{name}: {outcome}"
                assert all(f.image_id == image_id for f in page), (
                    f"{name}: the filter leaked findings from another image"
                )
