"""Tests for the recommendation engine.

The rules decide what an engineer is told to do. A wrong recommendation does
not raise — it produces confident, specific, unfounded advice — so the decision
each rule makes is tested as a pure function on constructed evidence, and the
refusals are tested as carefully as the actions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import config
from app import storage
from app.clustering import DISCRIMINATING_METHOD, UNEXPLAINED_LABEL, group_run
from app.recommendations import (
    BACKLOG_RULE,
    CONFLICTING,
    INSUFFICIENT,
    PRECISION_RULE,
    PROVISIONAL,
    RECALL_RULE,
    REPLICATED,
    TOO_SMALL_RULE,
    UNSTABLE_RULE,
    GroupEvidence,
    RecommendationError,
    comparable_runs,
    evaluate,
    recommend_run,
)
from app.storage import RunContext


def _evidence(**overrides) -> GroupEvidence:
    """Build a group whose pattern is strong, replicated, and actionable."""
    base = {
        "cluster_id": 1,
        "label": "small_object",
        "size": 20,
        "factors": ("small_object",),
        "false_negatives": 16,
        "false_positives": 2,
        "dominant_class": "door_frame",
        "dominant_class_share": 0.8,
        "qualified_factors": ("small_object",),
        "excluded_factors": (),
        "comparison_runs": (3,),
        "outcome_agrees": True,
    }
    return GroupEvidence(**{**base, **overrides})


# ---------------------------------------------------------------------------
# The rules, with no database involved
# ---------------------------------------------------------------------------
def test_a_replicated_miss_pattern_recommends_recall() -> None:
    """The case the whole milestone exists to produce."""
    proposals = evaluate(_evidence())

    assert len(proposals) == 1
    rule, action, _, status, actionable = proposals[0]
    assert rule == RECALL_RULE
    assert status == REPLICATED
    assert actionable is True
    assert "recall" in action.lower()


def test_a_replicated_invention_pattern_recommends_precision() -> None:
    """The mirror case: the model is over-predicting, not under-detecting."""
    proposals = evaluate(_evidence(false_negatives=2, false_positives=16))

    rule, action, _, status, actionable = proposals[0]
    assert rule == PRECISION_RULE
    assert status == REPLICATED
    assert actionable is True
    assert "precision" in action.lower()


def test_a_group_seen_only_once_is_provisional_not_replicated() -> None:
    """One observation is not a replication, and must not be presented as one."""
    proposals = evaluate(_evidence(comparison_runs=(), outcome_agrees=None))

    _, _, _, status, actionable = proposals[0]
    assert status == PROVISIONAL
    assert actionable is True


def test_runs_that_disagree_produce_no_action() -> None:
    """The thin_structure case: same group, opposite meaning, so do not act."""
    proposals = evaluate(_evidence(outcome_agrees=False))

    assert len(proposals) == 1
    rule, action, _, status, actionable = proposals[0]
    assert rule == UNSTABLE_RULE
    assert status == CONFLICTING
    assert actionable is False
    assert "do not act" in action.lower()


def test_a_small_group_is_reported_rather_than_acted_on() -> None:
    """Below the threshold an outcome split of three to two is not a pattern."""
    proposals = evaluate(
        _evidence(size=config.MIN_RECOMMENDATION_GROUP_SIZE - 1, false_negatives=5)
    )

    rule, _, _, status, actionable = proposals[0]
    assert rule == TOO_SMALL_RULE
    assert status == INSUFFICIENT
    assert actionable is False


def test_the_unexplained_group_becomes_an_investigation_not_a_fix() -> None:
    """No measured condition applies, so there is nothing to recommend fixing."""
    proposals = evaluate(
        _evidence(
            label=UNEXPLAINED_LABEL, factors=(), qualified_factors=(), size=64
        )
    )

    rule, action, _, status, actionable = proposals[0]
    assert rule == BACKLOG_RULE
    assert status == INSUFFICIENT
    assert actionable is False
    assert "investigate" in action.lower()


def test_a_group_whose_factors_do_not_qualify_yields_no_action() -> None:
    """A factor's frequency alone must never justify advice (D-031)."""
    proposals = evaluate(
        _evidence(
            label="edge_truncation",
            factors=("edge_truncation",),
            qualified_factors=(),
            excluded_factors=(("edge_truncation", 0.93, 0.34),),
        )
    )

    _, _, rationale, status, actionable = proposals[0]
    assert status == INSUFFICIENT
    assert actionable is False
    assert "edge_truncation" in rationale


def test_a_mixed_group_refuses_to_pick_one_action() -> None:
    """Neither outcome dominates, so it is two problems rather than one."""
    proposals = evaluate(_evidence(false_negatives=10, false_positives=10))

    _, action, _, status, actionable = proposals[0]
    assert status == INSUFFICIENT
    assert actionable is False
    assert "mixes" in action.lower()


def test_every_group_yields_at_least_one_row() -> None:
    """Silence is indistinguishable from 'the pass never ran'."""
    for evidence in (
        _evidence(),
        _evidence(size=2),
        _evidence(outcome_agrees=False),
        _evidence(label=UNEXPLAINED_LABEL, factors=(), qualified_factors=()),
        _evidence(false_negatives=10, false_positives=10),
    ):
        assert evaluate(evidence), "a group with nothing to say still gets a row"


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------
def test_the_rationale_names_the_factors_that_were_ruled_out() -> None:
    """Explaining beats pattern-matching: say what was excluded and why."""
    proposals = evaluate(
        _evidence(
            label="small_object + edge_truncation",
            factors=("small_object", "edge_truncation"),
            qualified_factors=("small_object",),
            excluded_factors=(("edge_truncation", 0.93, 0.34),),
        )
    )

    _, _, rationale, _, _ = proposals[0]
    assert "Ruled out" in rationale
    assert "edge_truncation" in rationale
    assert "0.93x" in rationale


def test_the_rationale_names_the_runs_it_compared_against() -> None:
    """A claim of replication must say what it replicated against."""
    _, _, rationale, _, _ = evaluate(_evidence(comparison_runs=(3, 5)))[0]

    assert "run(s) 3, 5" in rationale


def test_a_single_observation_says_so_in_its_rationale() -> None:
    """The absence of a comparison is stated, not left to be inferred."""
    _, _, rationale, _, _ = evaluate(
        _evidence(comparison_runs=(), outcome_agrees=None)
    )[0]

    assert "single observation" in rationale


def test_an_undefined_lift_is_rendered_as_not_available() -> None:
    """A NULL lift must never be printed as though it were a measurement."""
    _, _, rationale, _, _ = evaluate(
        _evidence(
            qualified_factors=(),
            excluded_factors=(("small_object", None, 0.2),),
        )
    )[0]

    assert "n/a" in rationale


# ---------------------------------------------------------------------------
# Persistence and the full pass
# ---------------------------------------------------------------------------
def _seed(database: Path, *, with_rates: bool = True) -> int:
    """Create a run with one large, clearly-missing failure group."""
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="sha-for-tests",
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
            "prediction_count, truth_count) VALUES (?, 'a.jpg', 'a.jpg', 99, 99, 1, 1)",
            (run_id,),
        )
        image_id = connection.execute(
            "SELECT id FROM images WHERE run_id = ?", (run_id,)
        ).fetchone()["id"]

        finding_ids = []
        for index in range(12):
            cursor = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name) "
                "VALUES (?, ?, ?, 'door_frame')",
                (
                    run_id,
                    image_id,
                    "false_negative" if index < 10 else "false_positive",
                ),
            )
            finding_ids.append(int(cursor.lastrowid))

        storage.save_root_causes(
            connection,
            run_id,
            [(fid, "small_object", 0.8, "tiny") for fid in finding_ids],
        )
        if with_rates:
            storage.save_factor_rates(
                connection,
                run_id,
                [("small_object", 10, 12, 2, 40, 2.5, 0.001)],
            )
        group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"small_object"},
        )
    return run_id


def test_a_full_pass_stores_advice_with_its_evidence(tmp_path: Path) -> None:
    """End to end: groups in, traceable recommendations out."""
    database = tmp_path / "recommend.db"
    run_id = _seed(database)

    with storage.connect(database) as connection:
        report = recommend_run(connection, run_id)
        stored = storage.load_recommendations(connection, run_id)

    assert report.written == len(stored)
    assert stored
    top = stored[0]
    assert top.status == PROVISIONAL
    assert top.actionable is True
    assert top.cluster_label == "small_object"
    assert top.rationale


def test_recommendations_are_ordered_actionable_first(tmp_path: Path) -> None:
    """The documented order: actionable, then by failures addressed, then id."""
    database = tmp_path / "ordered.db"
    run_id = _seed(database)

    with storage.connect(database) as connection:
        recommend_run(connection, run_id)
        # A large non-actionable group must still sort below a small actionable one.
        cluster_id = connection.execute(
            "SELECT id FROM clusters WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone()["id"]
        storage.save_recommendations(
            connection,
            run_id,
            [
                (
                    cluster_id, "manual", "Investigate", "because",
                    INSUFFICIENT, False, 999, 999.0,
                )
            ],
        )
        stored = storage.load_recommendations(connection, run_id)

    assert [row.actionable for row in stored] == sorted(
        [row.actionable for row in stored], reverse=True
    )
    assert stored[0].actionable is True


def test_rerunning_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    """Advice is derived data: recomputing corrects, it does not add a second."""
    database = tmp_path / "rerun.db"
    run_id = _seed(database)

    with storage.connect(database) as connection:
        recommend_run(connection, run_id)
        first = storage.load_recommendations(connection, run_id)
        recommend_run(connection, run_id)
        second = storage.load_recommendations(connection, run_id)

    assert len(first) == len(second)


def test_regrouping_discards_advice_derived_from_the_old_partition(
    tmp_path: Path,
) -> None:
    """Cluster ids are unstable, and stale advice is worse than none (D-034)."""
    database = tmp_path / "cascade.db"
    run_id = _seed(database)

    with storage.connect(database) as connection:
        recommend_run(connection, run_id)
        assert storage.load_recommendations(connection, run_id)

        group_run(
            connection,
            run_id,
            method=DISCRIMINATING_METHOD,
            allowed_factors={"small_object"},
        )

        assert storage.load_recommendations(connection, run_id) == []


def test_a_run_without_groups_explains_what_to_do(tmp_path: Path) -> None:
    """An actionable error beats an empty result that looks like 'no advice'."""
    database = tmp_path / "ungrouped.db"
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="x",
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        with pytest.raises(RecommendationError, match="app.clustering"):
            recommend_run(connection, run_id)


def test_a_run_without_factor_rates_cannot_confirm_a_pattern(
    tmp_path: Path,
) -> None:
    """Missing data is not agreement. It must not manufacture replication."""
    database = tmp_path / "no_rates.db"
    _seed(database, with_rates=True)
    bare_run = _seed(database, with_rates=False)

    with storage.connect(database) as connection:
        candidates = storage.load_runs_for_model(connection, "sha-for-tests")
        independent, reproductions = comparable_runs(
            connection, bare_run, candidates
        )

        assert bare_run not in independent
        assert bare_run not in reproductions
        for run in (*independent, *reproductions):
            assert storage.load_factor_rates(connection, run)


def test_recommendations_cascade_when_the_run_is_deleted(tmp_path: Path) -> None:
    """Derived data must not outlive the run it describes."""
    database = tmp_path / "delete.db"
    run_id = _seed(database)

    with storage.connect(database) as connection:
        recommend_run(connection, run_id)
        connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

        assert storage.load_recommendations(connection, run_id) == []


def test_existing_tables_are_untouched_by_this_milestone(tmp_path: Path) -> None:
    """D-020: every published table keeps its exact shape."""
    database = tmp_path / "frozen.db"
    _seed(database)

    expected = {
        "findings": 17,
        "root_causes": 6,
        "clusters": 5,
        "cluster_members": 2,
        "factor_rates": 9,
        "embeddings": 6,
        "heatmaps": 6,
    }
    with storage.connect(database) as connection:
        for table, columns in expected.items():
            actual = connection.execute(f"PRAGMA table_info({table})").fetchall()
            assert len(actual) == columns, f"{table} changed shape"


def test_evidence_is_frozen_so_rules_cannot_disagree() -> None:
    """Each rule must read the same facts about a group."""
    evidence = _evidence()
    with pytest.raises(AttributeError):
        evidence.size = 5  # type: ignore[misc]
    assert replace(evidence, size=5).size == 5
