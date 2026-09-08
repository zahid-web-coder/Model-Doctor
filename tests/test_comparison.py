"""The comparison rules, tested where they now live.

These rules used to be scattered — evaluation comparability in the browser,
factor qualification in the recommendations engine, replication behind a
database connection. Each test here pins a rule the way a second consumer will
rely on it, so a change is a deliberate contract change rather than drift.
"""

from __future__ import annotations

import pytest

from model_doctor.app import comparison as c
from model_doctor.app.storage import (
    BenchmarkRow,
    EvaluationRow,
    FactorRateRow,
    RecommendationRow,
    RunRecord,
)


def run(rid: int = 1, **overrides) -> RunRecord:
    """A run record with sensible defaults."""
    values = dict(
        id=rid,
        created_at="2026-09-01T00:00:00+00:00",
        model_path="/x/models/best.pt",
        model_sha256="a" * 64,
        dataset_yaml="/x/datasets/stairs_1280_yolo/data.yaml",
        split="test",
        confidence_threshold=0.25,
        match_iou_threshold=0.5,
        localization_iou_floor=0.1,
        image_size=640,
    )
    values.update(overrides)
    return RunRecord(**values)


def evaluation(rid: int, task: str = "bbox", **overrides) -> EvaluationRow:
    """An evaluation row under the project's standard protocol."""
    values = dict(
        id=rid * 10,
        run_id=rid,
        task=task,
        evaluator="pycocotools COCOeval",
        evaluator_version="2.0.7",
        sweep_confidence=0.01,
        iou_thresholds="0.50:0.95",
        max_detections=100,
        ground_truth="yolo-labels",
        gt_images=174,
        gt_annotations=202,
        prediction_count=500,
        mask_source=None,
        metrics={"map50": 0.8, "map50_95": 0.56, "map75": 0.6, "map_small": -1.0},
        created_at="2026-09-01T00:00:00+00:00",
    )
    values.update(overrides)
    return EvaluationRow(**values)


def factor(rid: int, name: str, lift: float | None, p: float) -> FactorRateRow:
    """A factor-rate row with counts consistent with the given lift."""
    return FactorRateRow(
        run_id=rid, factor=name,
        failure_count=40, failure_total=73, correct_count=16, correct_total=158,
        lift=lift, p_value=p,
    )


def rec(
    rid: int, rule: str, label: str, status: str, actionable: bool
) -> RecommendationRow:
    """A recommendation row."""
    return RecommendationRow(
        id=1, run_id=rid, cluster_id=1, cluster_label=label, rule=rule,
        action="do", rationale="because", status=status, actionable=actionable,
        affected=10, priority=10.0,
    )


def bench(rid: int, device: str, size: int = 640) -> BenchmarkRow:
    """A benchmark row."""
    return BenchmarkRow(
        id=1, run_id=rid, device=device, image_size=size, warmup_images=5,
        measured_images=50, created_at="now", measurements={"fps": 20.0},
    )


class TestFactorQualification:
    """The bar a factor must clear before any action may cite it."""

    def test_more_common_and_significant_qualifies(self) -> None:
        """More common and significant qualifies."""
        assert c.factor_qualifies(5.68, 0.0) is True

    def test_a_factor_more_common_in_successes_never_qualifies(self) -> None:
        """The edge_truncation case: 71% of failures, 76% of successes."""
        assert c.factor_qualifies(0.87, 0.023) is False

    def test_an_insignificant_lift_does_not_qualify(self) -> None:
        """An insignificant lift does not qualify."""
        assert c.factor_qualifies(2.47, 0.083) is False

    def test_an_undefined_lift_is_not_infinite(self) -> None:
        """No correct finding carried the factor: undefined, not overwhelming."""
        assert c.factor_qualifies(None, 0.0) is False


class TestReplication:
    """Whether two groups with the same label do the same thing."""

    def test_identical_miss_shares_agree(self) -> None:
        """Identical miss shares agree."""
        assert c.outcomes_agree(14, 22, 14, 22) is True

    def test_a_significantly_different_miss_share_disagrees(self) -> None:
        """85% misses on one run and none on the other is not replication."""
        assert c.outcomes_agree(17, 20, 0, 20) is False

    def test_matches_the_recommendations_engine(self) -> None:
        """The engine delegates here; a change breaks both or neither."""
        from model_doctor.app import recommendations
        assert recommendations.outcomes_agree is c.outcomes_agree
        assert recommendations.factor_qualifies is c.factor_qualifies


class TestEvaluationComparability:
    """Two mAP figures are comparable only under one protocol."""

    def test_same_protocol_is_comparable(self) -> None:
        """Same protocol is comparable."""
        assert c.comparable_evaluations(evaluation(1), evaluation(2)) is True

    @pytest.mark.parametrize(
        "field, value",
        [
            ("sweep_confidence", 0.25),
            ("iou_thresholds", "0.50"),
            ("max_detections", 300),
            ("evaluator", "ultralytics val"),
        ],
    )
    def test_any_protocol_difference_is_not_comparable(
        self, field: str, value
    ) -> None:
        """Any protocol difference is not comparable."""
        other = evaluation(2, **{field: value})
        assert c.comparable_evaluations(evaluation(1), other) is False

    def test_benchmarks_on_different_devices_are_not_comparable(self) -> None:
        """They measure different machines, not different models."""
        assert c.comparable_benchmarks(bench(1, "cpu"), bench(2, "mps")) is False
        assert c.comparable_benchmarks(bench(1, "cpu"), bench(2, "cpu")) is True


class TestNormalisation:
    """Absence is reported, never filled in."""

    def test_missing_outcomes_become_zero_so_columns_align(self) -> None:
        """Missing outcomes become zero so columns align."""
        out = c.normalise_outcomes({"correct": 3})
        assert out == {
            "correct": 3, "false_positive": 0, "false_negative": 0,
            "poor_localization": 0, "wrong_class": 0,
        }

    def test_cocoeval_sentinel_becomes_none(self) -> None:
        """-1 means no objects in that band, and must never be averaged in."""
        assert c.measured(-1.0) is None
        assert c.measured(0.0) == 0.0
        assert c.measured(0.5) == 0.5
        assert c.measured(None) is None

    def test_evaluation_summary_translates_the_sentinel(self) -> None:
        """Evaluation summary translates the sentinel."""
        out = c.evaluation_summary([evaluation(1)])
        assert out["bbox"]["metrics"]["map_small"] is None
        assert out["bbox"]["metrics"]["map50"] == 0.8
        assert out["bbox"]["settings"]["max_detections"] == 100

    def test_undefined_lift_sorts_last_and_stays_none(self) -> None:
        """Undefined lift sorts last and stays none."""
        rows = [
            factor(1, "a", None, 0.0),
            factor(1, "b", 2.0, 0.01),
            factor(1, "c", 5.0, 0.0),
        ]
        out = c.factor_summary(rows)
        assert [r["factor"] for r in out] == ["c", "b", "a"]
        assert out[-1]["lift"] is None
        assert out[-1]["qualifies"] is False

    def test_labels_mirror_the_dashboard(self) -> None:
        """Model is the file name, dataset is its directory."""
        r = run()
        assert c.model_name(r) == "best.pt"
        assert c.dataset_name(r) == "stairs_1280_yolo"


class TestDerivedMetrics:
    """Precision and recall from the finer taxonomy, with the rule attached."""

    def test_poor_localisation_counts_against_both_axes(self) -> None:
        """One near miss is a false positive and a false negative here."""
        out = c.derived_metrics({"correct": 8, "poor_localization": 2})
        # TP=8, FP=2, FN=2 -> 0.8 both ways
        assert out["precision"] == 0.8
        assert out["recall"] == 0.8
        assert out["f1"] == 0.8
        assert out["rule"] == c.CONFUSION_RULE

    def test_no_predictions_means_no_precision_not_zero(self) -> None:
        """No predictions means no precision not zero."""
        out = c.derived_metrics({"false_negative": 5})
        assert out["precision"] is None
        assert out["recall"] == 0.0
        assert out["f1"] is None

    def test_matches_the_staircase_run(self) -> None:
        """Run 5's real counts, so the arithmetic is pinned to known data."""
        out = c.outcome_summary(
            {"correct": 158, "false_positive": 29, "false_negative": 33,
             "poor_localization": 11}
        )
        assert out["total"] == 231
        assert out["failures"] == 73
        assert out["failure_rate"] == 0.316
        assert out["derived"]["precision"] == round(158 / (158 + 40), 4)
        assert out["derived"]["recall"] == round(158 / (158 + 44), 4)


class TestCrossRun:
    """Every run measured against the baseline, and what replicates."""

    def test_config_differences_report_only_what_differs(self) -> None:
        """Config differences report only what differs."""
        diffs = c.config_differences([run(4, image_size=640), run(5, image_size=448)])
        assert [d["field"] for d in diffs] == ["image_size"]
        assert diffs[0]["values"] == {4: 640, 5: 448}

    def test_two_paths_to_one_dataset_are_not_a_difference(self) -> None:
        """Runs 2 and 3: same dataset, different workspace path."""
        a = run(2, dataset_yaml="/a/columns_all_1280_yolo/data.yaml")
        b = run(3, dataset_yaml="/b/ws/columns_all_1280_yolo/data.yaml")
        assert c.config_differences([a, b]) == []
        assert c.comparability([a, b], {}, {})["same_dataset"]["by_name"] is True

    def test_dataset_hash_is_reported_as_unknown_not_true(self) -> None:
        """No hash is stored, and the response must say so rather than assume."""
        out = c.comparability([run(1), run(2)], {}, {})
        assert out["same_dataset"]["by_hash"] is None

    def test_different_checkpoints_warn_about_replication(self) -> None:
        """Different checkpoints warn about replication."""
        out = c.comparability([run(1), run(2, model_sha256="b" * 64)], {}, {})
        assert out["same_model"] is False
        assert any("different checkpoints" in w for w in out["warnings"])

    def test_mismatched_evaluation_protocol_is_flagged(self) -> None:
        """Mismatched evaluation protocol is flagged."""
        evals = {1: [evaluation(1)], 2: [evaluation(2, max_detections=300)]}
        out = c.comparability([run(1), run(2)], evals, {})
        assert out["evaluation_settings_match"] is False
        assert any("not comparable" in w for w in out["warnings"])

    def test_deltas_are_other_minus_baseline(self) -> None:
        """A negative false-positive delta is an improvement."""
        outcomes = {
            4: {"correct": 144, "false_positive": 70},
            5: {"correct": 158, "false_positive": 29},
        }
        evals = {
            4: [evaluation(4, metrics={"map50": 0.691, "map50_95": 0.405})],
            5: [evaluation(5, metrics={"map50": 0.802, "map50_95": 0.563})],
        }
        out = c.cross_run(4, outcomes, evals, {}, {}, {})
        assert out["outcome_deltas"][5]["false_positive"] == -41
        assert out["outcome_deltas"][5]["correct"] == 14
        assert out["metric_deltas"][5]["map50_bbox"] == 0.111
        assert 4 not in out["outcome_deltas"]

    def test_a_metric_missing_on_either_side_yields_no_delta(self) -> None:
        """A metric missing on either side yields no delta."""
        evals = {
            4: [evaluation(4, metrics={"map50": 0.7})],
            5: [evaluation(5, metrics={})],
        }
        out = c.cross_run(4, {4: {}, 5: {}}, evals, {}, {}, {})
        assert out["metric_deltas"][5]["map50_bbox"] is None

    def test_group_replication_uses_the_shared_verdict(self) -> None:
        """Group replication uses the shared verdict."""
        groups = {
            4: [c.GroupRow(1, "small_object", 22, {"false_negative": 14})],
            5: [c.GroupRow(2, "small_object", 22, {"false_negative": 14})],
            7: [c.GroupRow(3, "thin_structure", 6, {"false_negative": 1})],
        }
        out = c.cross_run(4, {}, {}, {}, groups, {})
        assert out["group_replication"]["small_object"]["present_in"] == [4, 5]
        assert out["group_replication"]["small_object"]["outcome_agrees"] is True
        assert out["group_replication"]["thin_structure"]["outcome_agrees"] is None

    def test_shared_actionable_recommendations_are_the_intersection(self) -> None:
        """Shared actionable recommendations are the intersection."""
        recs = {
            4: [
                rec(4, "recall_on_factor", "small_object", "replicated", True),
                rec(
                    4, "group_too_small", "thin_structure",
                    "insufficient_evidence", False,
                ),
            ],
            5: [rec(5, "recall_on_factor", "small_object", "replicated", True)],
        }
        out = c.cross_run(4, {}, {}, {}, {}, recs)
        shared = out["shared_actionable_recommendations"]
        assert shared == ["recall_on_factor:small_object"]

    def test_evidence_gaps_carry_the_remedy(self) -> None:
        """Evidence gaps carry the remedy."""
        gaps = c.evidence_gaps([4], {4: {"evaluation": True}})
        kinds = {g["missing"] for g in gaps}
        assert "evaluation" not in kinds
        assert "benchmarks" in kinds
        bench_gap = next(g for g in gaps if g["missing"] == "benchmarks")
        assert "--run-id 4" in bench_gap["how"]


class TestGroupSummary:
    """Groups largest first, with what they do and who is in them."""

    def test_unexplained_has_no_factors_and_is_kept(self) -> None:
        """The unexplained group is a named state, not an omission."""
        group = c.GroupRow(9, "unexplained", 25, {"false_positive": 25})
        out = c.group_summary([group])
        assert out[0]["factors"] == []
        assert out[0]["size"] == 25

    def test_label_splits_into_factors_and_miss_share_is_computed(self) -> None:
        """Label splits into factors and miss share is computed."""
        out = c.group_summary([
            c.GroupRow(1, "small_object + thin_structure", 20,
                       {"false_negative": 8, "false_positive": 12}, {"staircase": 20}),
        ])
        assert out[0]["factors"] == ["small_object", "thin_structure"]
        assert out[0]["miss_share"] == 0.4
        assert out[0]["dominant_class"] == "staircase"
        assert out[0]["dominant_class_share"] == 1.0


class TestRunIdentity:
    """Paths stay on the operator's machine unless asked for."""

    def test_paths_are_omitted_by_default(self) -> None:
        """Paths are omitted by default."""
        out = c.run_identity(run())
        assert "path" not in out["model"]
        assert "yaml_path" not in out["dataset"]
        assert out["model"]["sha256_short"] == "a" * 8

    def test_paths_are_included_only_on_request(self) -> None:
        """Paths are included only on request."""
        out = c.run_identity(run(), include_paths=True)
        assert out["model"]["path"] == "/x/models/best.pt"
