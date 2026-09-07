"""Reproduction is not replication.

Runs 5, 6, 7 and 10 of the reference database share a checkpoint, a dataset, a
split and every threshold. They return byte-identical outcome counts, mAP,
factor lifts, group sizes and image verdicts — deltas of exactly zero on all of
them. Inference is deterministic, so that is what re-running one command looks
like, and counting it as three confirmations turned one observation into four.

**A run can only replicate a pattern if it could have contradicted it.** That is
what the fingerprint decides: same checkpoint, dataset, split and thresholds
means same numbers by construction, so agreement carries no information. These
tests pin that distinction at every layer that reports it — the fingerprint
itself, the comparison module, the recommendation status, and the cross-run
evidence an MCP consumer reads.

Nothing here asserts anything about *what* a pattern is. Findings, metrics,
factors, clusters and the taxonomy are untouched by this change, and the last
class in this file exists to prove exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app import comparison, storage
from app.clustering import DISCRIMINATING_METHOD
from app.recommendations import (
    ACTIONABLE_STATUSES,
    CONFLICTING,
    PROVISIONAL,
    REPLICATED,
    REPRODUCED,
    GroupEvidence,
    _status_for,
    comparable_runs,
    gather_evidence,
)
from app.storage import RunContext

BASE = dict(
    model_path="/weights/best.pt",
    model_sha256="f" * 64,
    dataset_yaml="/data/staircase/data.yaml",
    split="test",
    confidence_threshold=0.25,
    match_iou_threshold=0.5,
    localization_iou_floor=0.1,
    image_size=448,
)


def context(**overrides: Any) -> RunContext:
    return RunContext(**{**BASE, **overrides})


def record(run_id: int, **overrides: Any) -> storage.RunRecord:
    fields = {**BASE, **overrides}
    return storage.RunRecord(
        id=run_id,
        created_at="2026-09-02T00:00:00+00:00",
        model_path=fields["model_path"],
        model_sha256=fields["model_sha256"],
        dataset_yaml=fields["dataset_yaml"],
        split=fields["split"],
        confidence_threshold=fields["confidence_threshold"],
        match_iou_threshold=fields["match_iou_threshold"],
        localization_iou_floor=fields["localization_iou_floor"],
        image_size=fields["image_size"],
    )


class TestTheFingerprint:
    """What makes two runs the same experiment."""

    def test_a_re_execution_shares_its_fingerprint(self) -> None:
        """The run-5/7/10 case: nothing differs, so nothing can disagree."""
        assert comparison.run_fingerprint(record(5)) == comparison.run_fingerprint(
            record(7)
        )

    def test_a_different_inference_size_is_a_different_experiment(self) -> None:
        """Run 4 against run 5: the only real contrast this checkpoint has."""
        assert comparison.run_fingerprint(record(5)) != comparison.run_fingerprint(
            record(4, image_size=640)
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_sha256", "a" * 64),
            ("split", "val"),
            ("image_size", 640),
            ("confidence_threshold", 0.35),
            ("match_iou_threshold", 0.6),
            ("localization_iou_floor", 0.2),
            ("dataset_yaml", "/data/columns/data.yaml"),
        ],
    )
    def test_every_field_that_changes_inference_changes_the_fingerprint(
        self, field: str, value: Any
    ) -> None:
        """A field left out would let a real difference read as a repeat."""
        assert comparison.run_fingerprint(record(1)) != comparison.run_fingerprint(
            record(2, **{field: value})
        )

    def test_a_rename_does_not_make_a_new_experiment(self) -> None:
        """Run 6 is run 5's checkpoint under a different filename.

        `model_path` is deliberately excluded: the weights are identified by
        their hash, and Ultralytics writes every checkpoint to `best.pt`.
        """
        assert comparison.run_fingerprint(record(5)) == comparison.run_fingerprint(
            record(6, model_path="/weights/yolo-26n-staircase.pt")
        )

    def test_the_same_dataset_by_another_path_is_not_a_second_observation(
        self,
    ) -> None:
        """The conservative direction, chosen deliberately.

        Identifying the dataset by directory rather than by absolute path means
        the same data reached two ways cannot read as two observations. The
        opposite error understates independence, which costs a weaker claim
        rather than a false one.
        """
        assert comparison.run_fingerprint(record(5)) == comparison.run_fingerprint(
            record(7, dataset_yaml="/elsewhere/staircase/data.yaml")
        )


class TestPartitioning:
    """Splitting candidates into those that can test a pattern and those that repeat."""

    def test_the_reference_shape(self) -> None:
        """Run 5 against 4, 6, 7, 10: one contrast, three repeats."""
        subject = record(5)
        others = [
            record(4, image_size=640),
            record(6, model_path="/weights/renamed.pt"),
            record(7),
            record(10),
        ]
        independent, reproductions = comparison.partition_by_fingerprint(
            subject, others
        )
        assert independent == [4]
        assert reproductions == [6, 7, 10]

    def test_the_subject_never_compares_against_itself(self) -> None:
        """A run is not evidence for itself, in either list."""
        independent, reproductions = comparison.partition_by_fingerprint(
            record(5), [record(5), record(7)]
        )
        assert 5 not in independent and 5 not in reproductions

    def test_copies_of_a_contrast_count_once(self) -> None:
        """Run 4's view: 5, 6, 7 and 10 are one configuration, not four.

        Excluding re-executions of the subject is only half the problem. Runs
        5, 6, 7 and 10 are also copies of each other, so run 4 comparing
        against all four would count one observation four times at one remove.
        """
        independent, reproductions = comparison.partition_by_fingerprint(
            record(4, image_size=640),
            [record(5), record(6), record(7), record(10)],
        )
        assert independent == [5], "one representative per distinct configuration"
        assert reproductions == [6, 7, 10]

    def test_the_representative_is_stable(self) -> None:
        """Whichever order they arrive in, the same run represents them."""
        runs = [record(10), record(7), record(5), record(6)]
        first, _ = comparison.partition_by_fingerprint(
            record(4, image_size=640), runs
        )
        second, _ = comparison.partition_by_fingerprint(
            record(4, image_size=640), list(reversed(runs))
        )
        assert first == second == [5]

    def test_two_distinct_configurations_both_count(self) -> None:
        """Collapsing copies must not collapse genuine contrasts."""
        independent, reproductions = comparison.partition_by_fingerprint(
            record(5),
            [record(4, image_size=640), record(3, split="val"), record(7)],
        )
        assert independent == [3, 4]
        assert reproductions == [7]

    def test_no_candidates_yields_nothing_rather_than_agreement(self) -> None:
        """Nothing to compare against must produce nothing, not silent assent."""
        assert comparison.partition_by_fingerprint(record(5), []) == ([], [])


class TestTheStatusDecision:
    """Four cases, and only one of them is replication."""

    def evidence(self, **overrides: Any) -> GroupEvidence:
        """One group's facts, with only the replication fields varied."""
        fields: dict[str, Any] = dict(
            cluster_id=1,
            label="small_object",
            size=22,
            factors=("small_object",),
            false_negatives=14,
            false_positives=5,
            dominant_class="staircase",
            dominant_class_share=1.0,
            qualified_factors=("small_object",),
            excluded_factors=(),
            comparison_runs=(),
            outcome_agrees=None,
        )
        fields.update(overrides)
        return GroupEvidence(**fields)

    def test_an_agreeing_independent_run_replicates(self) -> None:
        """The only path to `replicated`."""
        got = _status_for(
            self.evidence(comparison_runs=(4,), outcome_agrees=True)
        )
        assert got == REPLICATED

    def test_agreeing_re_executions_alone_only_reproduce(self) -> None:
        """The bug this change exists to fix.

        Before the fingerprint existed, runs 7 and 10 agreeing with run 5 made
        its pattern `replicated`. They cannot disagree, so they cannot confirm.
        """
        got = _status_for(
            self.evidence(reproduction_runs=(7, 10), reproductions_agree=True)
        )
        assert got == REPRODUCED
        assert got != REPLICATED

    def test_an_independent_run_outranks_any_number_of_repeats(self) -> None:
        """Three repeats plus one real contrast is one observation confirmed."""
        got = _status_for(
            self.evidence(
                comparison_runs=(4,),
                outcome_agrees=True,
                reproduction_runs=(6, 7, 10),
                reproductions_agree=True,
            )
        )
        assert got == REPLICATED

    def test_a_disagreeing_independent_run_conflicts(self) -> None:
        """A real contradiction outranks any amount of reproduction."""
        got = _status_for(
            self.evidence(
                comparison_runs=(4,),
                outcome_agrees=False,
                reproduction_runs=(7,),
                reproductions_agree=True,
            )
        )
        assert got == CONFLICTING

    def test_re_executions_that_differ_are_conflicting_not_replicating(self) -> None:
        """Two identical configurations disagreeing is a data problem.

        It is not a failed replication — there is nothing to replicate — and it
        must not be actionable until somebody has looked at why.
        """
        got = _status_for(
            self.evidence(reproduction_runs=(7,), reproductions_agree=False)
        )
        assert got == CONFLICTING

    def test_nothing_to_compare_against_stays_provisional(self) -> None:
        """One observation with nothing beside it is what provisional means."""
        assert _status_for(self.evidence()) == PROVISIONAL

    def test_reproduced_is_actionable_at_the_same_weight_as_provisional(self) -> None:
        """It is one observation, which is what `provisional` already means.

        Named separately so a reader can see that agreement exists and why it
        does not count, rather than wondering whether the other runs were
        overlooked.
        """
        assert REPRODUCED in ACTIONABLE_STATUSES
        assert PROVISIONAL in ACTIONABLE_STATUSES
        assert CONFLICTING not in ACTIONABLE_STATUSES


class TestCrossRunEvidence:
    """What an MCP consumer reads."""

    def group(self, size: int, misses: int) -> comparison.GroupRow:
        """One failure group with a given miss share."""
        return comparison.GroupRow(
            cluster_id=size,
            label="small_object",
            size=size,
            outcomes={"false_negative": misses, "false_positive": size - misses},
        )

    def test_repeats_are_named_separately_from_contrasts(self) -> None:
        """The runs 5/7/10 shape, as an MCP consumer receives it."""
        groups = {
            5: [self.group(22, 14)],
            7: [self.group(22, 14)],
            10: [self.group(22, 14)],
        }
        prints = {rid: comparison.run_fingerprint(record(rid)) for rid in (5, 7, 10)}
        out = comparison.cross_run(5, {}, {}, {}, groups, {}, fingerprints=prints)
        block = out["group_replication"]["small_object"]
        assert block["independent_runs"] == []
        assert block["reproductions"] == [7, 10]
        assert block["outcome_agrees"] is None, (
            "three re-executions must not read as agreement"
        )
        assert block["present_in"] == [5, 7, 10], "the runs are still listed"

    def test_an_independent_run_decides_agreement(self) -> None:
        """Run 4 against run 5: a differently configured run can confirm."""
        groups = {5: [self.group(22, 14)], 4: [self.group(26, 12)]}
        prints = {
            5: comparison.run_fingerprint(record(5)),
            4: comparison.run_fingerprint(record(4, image_size=640)),
        }
        out = comparison.cross_run(4, {}, {}, {}, groups, {}, fingerprints=prints)
        block = out["group_replication"]["small_object"]
        assert block["independent_runs"] == [5]
        assert block["reproductions"] == []
        assert block["outcome_agrees"] is not None

    def test_omitting_fingerprints_preserves_the_previous_behaviour(self) -> None:
        """Callers that have not been updated must not silently change verdict."""
        groups = {5: [self.group(22, 14)], 7: [self.group(22, 14)]}
        out = comparison.cross_run(5, {}, {}, {}, groups, {})
        block = out["group_replication"]["small_object"]
        assert block["reproductions"] == []
        assert block["outcome_agrees"] is True

    def test_comparability_names_the_re_executions(self) -> None:
        """A reader seeing zero deltas should be told why they are zero."""
        runs = [record(5), record(7), record(10), record(4, image_size=640)]
        out = comparison.comparability(runs, {}, {})
        assert out["independent_configurations"] == 2
        assert out["reproduction_groups"] == [{"fingerprint_of": 5, "runs": [5, 7, 10]}]
        assert any("re-execute one configuration" in w for w in out["warnings"])

    def test_comparability_says_nothing_when_every_run_is_distinct(self) -> None:
        """No warning where there is nothing to warn about."""
        out = comparison.comparability(
            [record(4, image_size=640), record(5)], {}, {}
        )
        assert out["reproduction_groups"] == []
        assert not any("re-execute" in w for w in out["warnings"])

    def test_the_caveat_states_the_distinction(self) -> None:
        """A consumer reading only the caveats still learns the rule."""
        joined = " ".join(comparison.CAVEATS)
        assert "fingerprint" in joined
        assert "reproduction" in joined


def seed(database: Path, **overrides: Any) -> int:
    """One run with factor rates and a group, enough to be compared."""
    with storage.connect(database) as connection:
        run_id = storage.save_run(connection, context(**overrides))
        connection.execute(
            "INSERT INTO images (run_id, path, filename, prediction_count, "
            "truth_count) VALUES (?,'/i.jpg','i.jpg',1,1)",
            (run_id,),
        )
        image_id = connection.execute(
            "SELECT id FROM images WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        ids = []
        for outcome in ("false_negative",) * 14 + ("false_positive",) * 8:
            cur = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name) "
                "VALUES (?,?,?,'staircase')",
                (run_id, image_id, outcome),
            )
            ids.append(cur.lastrowid)
        connection.execute(
            "INSERT INTO factor_rates (run_id, factor, failure_count, "
            "failure_total, correct_count, correct_total, lift, p_value) "
            "VALUES (?,'small_object',42,73,16,158,5.682,0.0)",
            (run_id,),
        )
        cur = connection.execute(
            "INSERT INTO clusters (run_id, method, label, size) VALUES (?,?,?,?)",
            (run_id, DISCRIMINATING_METHOD, "small_object", len(ids)),
        )
        cluster_id = cur.lastrowid
        for finding_id in ids:
            connection.execute(
                "INSERT INTO cluster_members (cluster_id, finding_id) VALUES (?,?)",
                (cluster_id, finding_id),
            )
        for finding_id in ids:
            connection.execute(
                "INSERT INTO root_causes (finding_id, run_id, factor, score, "
                "evidence) VALUES (?,?,'small_object',1.0,'{}')",
                (finding_id, run_id),
            )
    return run_id


class TestAgainstADatabase:
    """The whole path, on stored rows."""

    def test_a_run_and_its_re_execution_produce_no_replication(
        self, tmp_path: Path
    ) -> None:
        """Seeding the same configuration twice must not confirm anything."""
        database = tmp_path / "repeat.db"
        first = seed(database)
        second = seed(database)
        with storage.connect(database) as connection:
            candidates = storage.load_runs_for_model(connection, BASE["model_sha256"])
            independent, reproductions = comparable_runs(
                connection, first, candidates
            )
            assert independent == []
            assert reproductions == [second]

            evidence = gather_evidence(connection, first, independent, reproductions)
            assert evidence, "the seeded group should be present"
            group = evidence[0]
            assert group.comparison_runs == ()
            assert group.outcome_agrees is None
            assert group.reproduction_runs == (second,)
            assert _status_for(group) == REPRODUCED

    def test_a_differently_configured_run_can_replicate(self, tmp_path: Path) -> None:
        """The run-4-against-run-5 shape."""
        database = tmp_path / "contrast.db"
        first = seed(database)
        other = seed(database, image_size=640)
        with storage.connect(database) as connection:
            candidates = storage.load_runs_for_model(connection, BASE["model_sha256"])
            independent, reproductions = comparable_runs(
                connection, first, candidates
            )
            assert independent == [other]
            assert reproductions == []

            group = gather_evidence(connection, first, independent, reproductions)[0]
            assert group.outcome_agrees is True
            assert _status_for(group) == REPLICATED

    def test_repeats_never_outvote_a_contrast(self, tmp_path: Path) -> None:
        """One real contrast plus three repeats is still one contrast."""
        database = tmp_path / "mixed.db"
        first = seed(database)
        contrast = seed(database, image_size=640)
        repeats = [seed(database), seed(database), seed(database)]
        with storage.connect(database) as connection:
            candidates = storage.load_runs_for_model(connection, BASE["model_sha256"])
            independent, reproductions = comparable_runs(
                connection, first, candidates
            )
        assert independent == [contrast]
        assert reproductions == sorted(repeats)

    def test_the_rationale_distinguishes_the_two(self, tmp_path: Path) -> None:
        """A reader must be told the other runs exist and why they do not count."""
        from app.recommendations import _rationale

        database = tmp_path / "wording.db"
        first = seed(database)
        second = seed(database)
        with storage.connect(database) as connection:
            independent, reproductions = comparable_runs(
                connection,
                first,
                storage.load_runs_for_model(connection, BASE["model_sha256"]),
            )
            group = gather_evidence(connection, first, independent, reproductions)[0]
        text = _rationale(group, "headline")
        assert str(second) in text
        assert "repeats the observation rather than confirming it" in text
        assert "rests on a single observation" in text


class TestNothingElseMoved:
    """The change is to replication semantics and to nothing else."""

    def test_stored_analysis_values_are_untouched(self, tmp_path: Path) -> None:
        """Findings, factor rates, clusters and memberships as seeded.

        The fingerprint is derived from `runs` at read time and writes nothing.
        A change that altered any of these would be out of scope by definition.
        """
        database = tmp_path / "intact.db"
        run_id = seed(database)
        with storage.connect(database) as connection:
            before = {
                "findings": connection.execute(
                    "SELECT outcome, COUNT(*) FROM findings WHERE run_id=? "
                    "GROUP BY outcome",
                    (run_id,),
                ).fetchall(),
                "rates": connection.execute(
                    "SELECT factor, lift, p_value FROM factor_rates WHERE run_id=?",
                    (run_id,),
                ).fetchall(),
                "clusters": connection.execute(
                    "SELECT label, size FROM clusters WHERE run_id=?", (run_id,)
                ).fetchall(),
                "members": connection.execute(
                    "SELECT COUNT(*) FROM cluster_members cm JOIN clusters c "
                    "ON c.id=cm.cluster_id WHERE c.run_id=?",
                    (run_id,),
                ).fetchone(),
            }
            candidates = storage.load_runs_for_model(connection, BASE["model_sha256"])
            independent, reproductions = comparable_runs(
                connection, run_id, candidates
            )
            gather_evidence(connection, run_id, independent, reproductions)

            after = {
                "findings": connection.execute(
                    "SELECT outcome, COUNT(*) FROM findings WHERE run_id=? "
                    "GROUP BY outcome",
                    (run_id,),
                ).fetchall(),
                "rates": connection.execute(
                    "SELECT factor, lift, p_value FROM factor_rates WHERE run_id=?",
                    (run_id,),
                ).fetchall(),
                "clusters": connection.execute(
                    "SELECT label, size FROM clusters WHERE run_id=?", (run_id,)
                ).fetchall(),
                "members": connection.execute(
                    "SELECT COUNT(*) FROM cluster_members cm JOIN clusters c "
                    "ON c.id=cm.cluster_id WHERE c.run_id=?",
                    (run_id,),
                ).fetchone(),
            }
        assert after == before

    def test_replication_semantics_store_nothing(self) -> None:
        """The fingerprint is derived at read time; it has no table of its own.

        Pinned as an absence rather than a version number, which later steps
        move for their own reasons — schema 13 added `finding_relations`, and
        that says nothing about whether *this* step began storing something.
        """
        statements = " ".join(storage.SCHEMA_STATEMENTS).lower()
        assert "fingerprint" not in statements
        assert "reproduction" not in statements
        assert not any(
            "fingerprint" in column for _, column, _ in storage._ADDED_COLUMNS
        )
        # Pinned by meaning rather than by position in the tuple, which later
        # steps append to: schema 15 added inference provenance to `runs`, and
        # a positional slice would have read that as this step storing
        # something. What matters is that no column on `runs` records
        # replication state — the fingerprint is still derived at read time.
        run_columns = {
            column for table, column, _ in storage._ADDED_COLUMNS if table == "runs"
        }
        assert not any(
            word in column
            for column in run_columns
            for word in ("fingerprint", "reproduc", "replicat")
        )

    def test_the_outcome_taxonomy_is_unchanged(self) -> None:
        """The five outcomes are out of scope for this step, and stay put."""
        from app.diagnosis import Outcome

        assert {o.value for o in Outcome} == {
            "correct",
            "wrong_class",
            "poor_localization",
            "false_positive",
            "false_negative",
        }

    def test_cross_run_still_reports_deltas_and_lifts_the_same_way(self) -> None:
        """Only the replication block gained fields."""
        out = comparison.cross_run(5, {5: {}, 7: {}}, {}, {}, {}, {})
        assert set(out) == {
            "baseline",
            "outcome_deltas",
            "metric_deltas",
            "factor_lift",
            "group_replication",
            "shared_actionable_recommendations",
        }

    def test_json_serialisable(self) -> None:
        """Everything added must survive the MCP transport."""
        prints = {rid: comparison.run_fingerprint(record(rid)) for rid in (5, 7)}
        payload = {
            "comparability": comparison.comparability([record(5), record(7)], {}, {}),
            "cross_run": comparison.cross_run(
                5, {}, {}, {}, {}, {}, fingerprints=prints
            ),
        }
        assert json.loads(json.dumps(payload)) == payload
