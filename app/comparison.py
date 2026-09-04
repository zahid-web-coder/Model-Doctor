"""Comparison semantics — the one place that decides what may be compared.

**This module exists so the rules are written once.** Before it, the answer to
"can these two evaluations be compared?" lived in the browser
(``web/src/lib/compare/metrics.ts``), the answer to "does this factor
distinguish failures?" lived in :mod:`app.recommendations`, and the answer to
"do two runs agree about a failure group?" lived in a function that could only
run with a database connection in hand. A second consumer — the MCP server —
would have had to copy all three, and copies drift. Everything here is pure:
it takes rows :mod:`app.storage` has already loaded and returns plain
dictionaries. No SQL, no HTTP, no MCP.

**It adds no analysis.** Every number is either a stored measurement or an
arithmetic combination of stored counts whose rule is stated beside it. The
project's taxonomy is deliberately finer than the two-bucket one precision and
recall come from, and collapsing it involves a choice (SCHEMA.md §4); that
choice is recorded in :data:`CONFUSION_RULE` and travels with every figure it
produces.

**Absence is reported, never filled in.** COCOeval writes ``-1.0`` for an area
band with no objects; ``lift`` is ``NULL`` when no correct finding carried the
factor. Both become ``None`` here, and a caller must decide how to present
"not measured" rather than receive a number that will be read as one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from app.storage import (
    BenchmarkRow,
    EvaluationRow,
    FactorRateRow,
    RecommendationRow,
    RunRecord,
)
from utils.statistics import fisher_exact_two_sided

# ---------------------------------------------------------------------------
# Rules, stated once
# ---------------------------------------------------------------------------

#: The five outcomes, in the order they are reported.
OUTCOMES: tuple[str, ...] = (
    "correct",
    "false_positive",
    "false_negative",
    "poor_localization",
    "wrong_class",
)

#: How the five outcomes collapse to true/false positives and negatives.
#:
#: ``poor_localization`` and ``wrong_class`` count as **both** a false positive
#: and a false negative: the model produced a detection that should not stand
#: and left a ground-truth object unaccounted for. Scoring either one alone
#: would flatter the model on one axis. This is why these figures do not equal
#: the ones mAP implies — see SCHEMA.md §4 — and why the rule is shipped with
#: every figure derived from it.
CONFUSION_RULE: str = (
    "poor_localization and wrong_class each count as both a false positive "
    "and a false negative"
)

#: Significance threshold shared by factor qualification and replication.
SIGNIFICANCE: float = 0.05

#: Evaluation settings two runs must share before their mAP may be compared.
#:
#: mAP is only meaningful against a stated protocol. Two runs scored at
#: different confidence sweeps, IoU ranges or detection caps produce numbers
#: that look alike and measure different things.
EVALUATION_PROTOCOL_FIELDS: tuple[str, ...] = (
    "evaluator",
    "sweep_confidence",
    "iou_thresholds",
    "max_detections",
)

#: Run settings whose difference is reported when runs are compared.
#:
#: ``dataset`` and ``model`` compare by derived identity — dataset name and
#: checkpoint hash — because two paths to the same file are not a difference,
#: and one path to two different files is.
CONFIG_FIELDS: tuple[str, ...] = (
    "model_sha256",
    "dataset",
    "split",
    "image_size",
    "confidence_threshold",
    "match_iou_threshold",
    "localization_iou_floor",
)

#: Caveats every comparison carries. Fixed text, so a consumer sees the same
#: warning however it reached the numbers.
CAVEATS: tuple[str, ...] = (
    "Outcome counts are not COCO mAP and will not agree with it: a near miss "
    "is one poor_localization here and both a false positive and a false "
    "negative there (SCHEMA.md §4).",
    "Factors are correlations measured against a control rate, not proven "
    "causes; the engine reports evidence and the causal claim belongs to the "
    "reader (D-031).",
    "lift is null when undefined, never infinity; map_small, map_medium and "
    "map_large are null when the dataset has no objects in that band.",
    "Recommendations carry an evidence status; only replicated, reproduced "
    "and provisional are actionable (D-035).",
    "Runs sharing a fingerprint are re-executions of one configuration, not "
    "independent observations: their agreement is reproduction, and only a "
    "run with a different fingerprint can replicate a pattern.",
)

#: How to fill each kind of missing evidence. The command is the remedy.
REMEDIES: Mapping[str, str] = {
    "evaluation": "python scripts/evaluate_run.py --run-id {run_id}",
    "benchmarks": "python scripts/benchmark_run.py --run-id {run_id} --device <device>",
    "factor_rates": "python -m app.root_cause --run {run_id}",
    "groups": "python -m app.clustering --run {run_id}",
    "recommendations": "python -m app.recommendations --run {run_id}",
    "mask_findings": "python -m app.mask_diagnosis --run {run_id}",
    "image_diagnoses": "python -m app.image_diagnosis --run {run_id}",
}


def factor_qualifies(lift: float | None, p_value: float) -> bool:
    """Return whether a factor is measurably more common in failures.

    The bar a factor must clear before any action may cite it: more frequent
    among failures than among correct findings, and distinguishable from
    chance. ``lift`` of ``None`` means no correct finding carried the factor,
    which is undefined rather than infinite, and does not qualify.
    """
    return lift is not None and lift > 1.0 and p_value < SIGNIFICANCE


def outcomes_agree(
    here_misses: int, here_size: int, there_misses: int, there_size: int
) -> bool:
    """Return whether two groups with the same label do the same thing.

    A group can be stable while its meaning is not: the same share of failures
    on two runs, yet 85% misses on one and evenly divided on the other.
    Membership replicating is not the same as the pattern replicating, and only
    the second justifies an action. Agreement is the absence of a significant
    difference in the miss share, by Fisher's exact test.
    """
    return (
        fisher_exact_two_sided(here_misses, here_size, there_misses, there_size)
        >= SIGNIFICANCE
    )


#: The fields that make two runs the same experiment.
#:
#: Everything an operator could change that would make the inference different.
#: Two runs agreeing on all of them cannot disagree about anything, because
#: inference is deterministic — so their agreement is not evidence.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "model_sha256",
    "dataset",
    "split",
    "image_size",
    "confidence_threshold",
    "match_iou_threshold",
    "localization_iou_floor",
)


def run_fingerprint(run: RunRecord) -> str:
    """Return what makes this run's configuration distinct.

    **Two runs with the same fingerprint are re-executions, not observations.**
    Runs 5, 6, 7 and 10 of the reference database share a checkpoint, a
    dataset, a split and every threshold, and return byte-identical outcome
    counts, mAP, factor lifts, group sizes and image verdicts — deltas of
    exactly zero on all of them. Treating that as three confirmations of run 5
    manufactures confidence out of a repeated command.

    The dataset is identified by directory name rather than by full path, which
    is deliberately the conservative direction: the same data reached by two
    paths must not read as two observations. The reverse error — two different
    datasets sharing a directory name — understates independence, which costs a
    weaker claim rather than a false one. No dataset content hash is stored;
    ``comparability`` says so for the same reason.
    """
    return "|".join(
        [
            run.model_sha256,
            dataset_name(run),
            run.split,
            str(run.image_size),
            f"{run.confidence_threshold:.6g}",
            f"{run.match_iou_threshold:.6g}",
            f"{run.localization_iou_floor:.6g}",
        ]
    )


def partition_by_fingerprint(
    subject: RunRecord, others: Sequence[RunRecord]
) -> tuple[list[int], list[int]]:
    """Split candidate runs into those that can replicate and those that repeat.

    Args:
        subject: The run whose pattern is being tested.
        others: Candidate comparison runs, which may include the subject.

    **One representative per distinct configuration.** Excluding re-executions
    of the *subject* is only half of it: runs 5, 6, 7 and 10 are also copies of
    each other, so comparing run 4 against all four would count one observation
    four times at one remove. Each distinct fingerprint contributes exactly one
    independent run — the lowest id, so the choice is stable across calls — and
    every further copy joins the reproductions.

    Returns:
        ``(independent, reproductions)`` as sorted id lists. Each entry in
        ``independent`` has a fingerprint differing from the subject's and from
        every other entry, so it could confirm or contradict. ``reproductions``
        re-execute a configuration already accounted for — the subject's own, or
        one an independent run already represents — and can do neither.
    """
    mine = run_fingerprint(subject)
    seen: dict[str, int] = {}
    reproductions: list[int] = []
    for other in sorted(others, key=lambda r: r.id):
        if other.id == subject.id:
            continue
        fingerprint = run_fingerprint(other)
        if fingerprint == mine or fingerprint in seen:
            reproductions.append(other.id)
            continue
        seen[fingerprint] = other.id
    return sorted(seen.values()), sorted(reproductions)


def comparable_evaluations(a: EvaluationRow, b: EvaluationRow) -> bool:
    """Return whether two evaluations were produced under the same protocol."""
    return all(getattr(a, f) == getattr(b, f) for f in EVALUATION_PROTOCOL_FIELDS)


def comparable_benchmarks(a: BenchmarkRow, b: BenchmarkRow) -> bool:
    """Return whether two benchmarks measured the same kind of machine.

    Latency and memory are properties of a model *on a device*. Figures from
    different devices measure different machines rather than different models.
    """
    return a.device == b.device and a.image_size == b.image_size


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_outcomes(counts: Mapping[str, int]) -> dict[str, int]:
    """Return every outcome with a count, missing ones as zero.

    A run with no wrong-class findings has no ``wrong_class`` row to count; a
    consumer comparing runs needs the key present so columns align.
    """
    return {outcome: int(counts.get(outcome, 0)) for outcome in OUTCOMES}


def measured(value: float | None) -> float | None:
    """Return a stored metric, or ``None`` where COCOeval recorded ``-1``.

    ``-1.0`` is the evaluator's sentinel for an area band with no objects. It
    is stored verbatim so the database is a true record of what the evaluator
    said; it is translated here so nothing downstream averages it in.
    """
    if value is None or value < 0:
        return None
    return float(value)


def model_name(run: RunRecord) -> str:
    """Return the checkpoint's file name, the way the dashboard shows it."""
    return PurePosixPath(run.model_path).name or run.model_path


def dataset_name(run: RunRecord) -> str:
    """Return the dataset directory's name, taken from its ``data.yaml`` path."""
    parts = PurePosixPath(run.dataset_yaml).parts
    return parts[-2] if len(parts) >= 2 else run.dataset_yaml


# ---------------------------------------------------------------------------
# Per-run summaries
# ---------------------------------------------------------------------------


def derived_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    """Return precision, recall and F1 from outcome counts, with the rule.

    ``None`` rather than zero when a denominator is empty: a run with no
    predictions has no precision, not a precision of nothing.
    """
    outcomes = normalise_outcomes(counts)
    tp = outcomes["correct"]
    fp = (
        outcomes["false_positive"]
        + outcomes["poor_localization"]
        + outcomes["wrong_class"]
    )
    fn = (
        outcomes["false_negative"]
        + outcomes["poor_localization"]
        + outcomes["wrong_class"]
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rule": CONFUSION_RULE,
    }


def outcome_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    """Return counts, the failure share, and the derived figures."""
    outcomes = normalise_outcomes(counts)
    total = sum(outcomes.values())
    failures = total - outcomes["correct"]
    return {
        "counts": outcomes,
        "total": total,
        "failures": failures,
        "failure_rate": round(failures / total, 4) if total else None,
        "derived": derived_metrics(outcomes),
    }


def evaluation_summary(rows: Sequence[EvaluationRow]) -> dict[str, Any]:
    """Return stored mAP/AR per task, each with the protocol behind it."""
    out: dict[str, Any] = {}
    for row in rows:
        out[row.task] = {
            "metrics": {k: measured(v) for k, v in row.metrics.items()},
            "settings": {
                "evaluator": row.evaluator,
                "sweep_confidence": row.sweep_confidence,
                "iou_thresholds": row.iou_thresholds,
                "max_detections": row.max_detections,
                "ground_truth": row.ground_truth,
                "gt_images": row.gt_images,
                "gt_annotations": row.gt_annotations,
                "prediction_count": row.prediction_count,
            },
        }
    return out


def factor_summary(rows: Sequence[FactorRateRow]) -> list[dict[str, Any]]:
    """Return each factor's rate on failures against its rate on successes.

    Sorted strongest lift first, undefined last. A count is never reported
    without its control rate (D-031).
    """

    def rate(count: int, total: int) -> float | None:
        return round(count / total, 4) if total else None

    entries = [
        {
            "factor": r.factor,
            "failure_rate": rate(r.failure_count, r.failure_total),
            "correct_rate": rate(r.correct_count, r.correct_total),
            "failure_count": r.failure_count,
            "failure_total": r.failure_total,
            "correct_count": r.correct_count,
            "correct_total": r.correct_total,
            "lift": None if r.lift is None else round(r.lift, 3),
            "p_value": round(r.p_value, 4),
            "qualifies": factor_qualifies(r.lift, r.p_value),
        }
        for r in rows
    ]
    return sorted(entries, key=lambda e: (e["lift"] is None, -(e["lift"] or 0.0)))


@dataclass(frozen=True)
class GroupRow:
    """One failure group with the outcome mix of its members.

    Assembled by the caller from a :class:`ClusterRow` and its members, so this
    module never has to open a connection to know what a group *does*.
    """

    cluster_id: int
    label: str
    size: int
    outcomes: Mapping[str, int]
    classes: Mapping[str, int] = field(default_factory=dict)


def group_summary(groups: Sequence[GroupRow]) -> list[dict[str, Any]]:
    """Return failure groups largest first, each with its outcome mix."""
    out = []
    for g in sorted(groups, key=lambda g: -g.size):
        outcomes = normalise_outcomes(g.outcomes)
        top = max(g.classes.items(), key=lambda kv: kv[1], default=(None, 0))
        out.append(
            {
                "cluster_id": g.cluster_id,
                "label": g.label,
                "size": g.size,
                "factors": [] if g.label == "unexplained" else [
                    part.strip() for part in g.label.split("+")
                ],
                "outcomes": {k: v for k, v in outcomes.items() if v},
                "miss_share": (
                    round(outcomes["false_negative"] / g.size, 4) if g.size else None
                ),
                "dominant_class": top[0],
                "dominant_class_share": round(top[1] / g.size, 4) if g.size else None,
            }
        )
    return out


def recommendation_summary(rows: Sequence[RecommendationRow]) -> list[dict[str, Any]]:
    """Return recommendations in their documented order, evidence status intact."""
    return [
        {
            "rule": r.rule,
            "action": r.action,
            "status": r.status,
            "actionable": bool(r.actionable),
            "affected": r.affected,
            "priority": r.priority,
            "group": r.cluster_label,
            "rationale": r.rationale,
        }
        for r in rows
    ]


def benchmark_summary(rows: Sequence[BenchmarkRow]) -> list[dict[str, Any]]:
    """Return each benchmark with the device it belongs to."""
    return [
        {
            "device": r.device,
            "image_size": r.image_size,
            "measured_images": r.measured_images,
            **{k: v for k, v in r.measurements.items()},
        }
        for r in rows
    ]


def run_identity(run: RunRecord, *, include_paths: bool = False) -> dict[str, Any]:
    """Return who a run is: model, dataset and configuration.

    Paths are omitted by default. They are local filesystem locations, which
    a hosted consumer has no use for and which say more about the operator's
    machine than about the run.
    """
    model: dict[str, Any] = {
        "name": model_name(run),
        "sha256_short": run.model_sha256[:8],
        "sha256": run.model_sha256,
    }
    dataset: dict[str, Any] = {"name": dataset_name(run), "split": run.split}
    if include_paths:
        model["path"] = run.model_path
        dataset["yaml_path"] = run.dataset_yaml
    return {
        "run_id": run.id,
        "created_at": run.created_at,
        "model": model,
        "dataset": dataset,
        "config": {
            "image_size": run.image_size,
            "confidence_threshold": run.confidence_threshold,
            "match_iou_threshold": run.match_iou_threshold,
            "localization_iou_floor": run.localization_iou_floor,
        },
    }


# ---------------------------------------------------------------------------
# Cross-run
# ---------------------------------------------------------------------------


def _config_value(run: RunRecord, field_name: str) -> Any:
    if field_name == "dataset":
        return dataset_name(run)
    return getattr(run, field_name)


def config_differences(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Return every setting on which the runs disagree, with each run's value.

    Reported rather than judged: a different image size is a legitimate
    experiment, and a different checkpoint is a different model. What the
    difference *means* is the reader's call; that it exists is not.
    """
    out = []
    for name in CONFIG_FIELDS:
        values = {run.id: _config_value(run, name) for run in runs}
        if len({repr(v) for v in values.values()}) > 1:
            out.append({"field": name, "values": values})
    return out


def comparability(
    runs: Sequence[RunRecord],
    evaluations: Mapping[int, Sequence[EvaluationRow]],
    benchmarks: Mapping[int, Sequence[BenchmarkRow]],
) -> dict[str, Any]:
    """Say what the runs share, and what makes their numbers comparable.

    ``same_dataset.by_hash`` is ``None`` because no dataset hash is stored:
    identity is by directory name only, and that is stated rather than hidden.
    """
    shas = {r.model_sha256 for r in runs}
    datasets = {dataset_name(r) for r in runs}
    splits = {r.split for r in runs}

    warnings: list[str] = []
    eval_rows = [e for rid in evaluations for e in evaluations[rid]]
    by_task: dict[str, list[EvaluationRow]] = {}
    for e in eval_rows:
        by_task.setdefault(e.task, []).append(e)
    eval_match: bool | None = None
    if by_task:
        eval_match = all(
            comparable_evaluations(rows[0], other)
            for rows in by_task.values()
            for other in rows[1:]
        )
        if not eval_match:
            warnings.append(
                "Evaluations were produced under different settings; their "
                "mAP figures are not comparable."
            )
    bench_rows = [b for rid in benchmarks for b in benchmarks[rid]]
    if bench_rows:
        devices = {b.device for b in bench_rows}
        if len(devices) > 1:
            warnings.append(
                "Benchmarks span devices "
                f"({', '.join(sorted(devices))}); latency and memory are "
                "only comparable within one device."
            )
    if len(shas) > 1:
        warnings.append(
            "Runs use different checkpoints; disagreements between them are "
            "different observations, not failed replications."
        )
    if len(datasets) > 1 or len(splits) > 1:
        warnings.append(
            "Runs cover different datasets or splits; outcome counts are not "
            "over the same objects."
        )

    # Which of the runs being compared are re-executions of each other. Stated
    # here because a reader looking at a table of deltas that are all zero
    # should be told why they are zero, rather than concluding the pattern is
    # unusually stable.
    groups_by_print: dict[str, list[int]] = {}
    for r in runs:
        groups_by_print.setdefault(run_fingerprint(r), []).append(r.id)
    repeated = {
        fp: sorted(ids) for fp, ids in groups_by_print.items() if len(ids) > 1
    }
    if repeated:
        described = "; ".join(
            ", ".join(f"#{i}" for i in ids) for ids in repeated.values()
        )
        warnings.append(
            f"These runs re-execute one configuration ({described}); their "
            "deltas are structurally zero and carry no replication weight."
        )

    return {
        "independent_configurations": len(groups_by_print),
        "reproduction_groups": [
            {"fingerprint_of": ids[0], "runs": ids} for ids in repeated.values()
        ],
        "same_model": len(shas) == 1,
        "same_dataset": {
            "by_name": len(datasets) == 1,
            "by_hash": None,
            "note": "dataset identity is a path; no content hash is stored",
        },
        "same_split": len(splits) == 1,
        "evaluation_settings_match": eval_match,
        "config_differences": config_differences(runs),
        "warnings": warnings,
    }


def _delta(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    d = b - a
    return round(d, 4) if isinstance(d, float) else d


def cross_run(
    baseline_id: int,
    outcomes: Mapping[int, Mapping[str, int]],
    evaluations: Mapping[int, Sequence[EvaluationRow]],
    factors: Mapping[int, Sequence[FactorRateRow]],
    groups: Mapping[int, Sequence[GroupRow]],
    recommendations: Mapping[int, Sequence[RecommendationRow]],
    fingerprints: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Return every run measured against the baseline, and what replicates.

    Deltas are ``other - baseline``, so a negative false-positive delta is an
    improvement and a positive mAP delta is one too. Group replication uses the
    same test the recommendations engine uses, on the same counts, so this
    module and that one cannot disagree about whether a pattern held.

    **Replication is judged only between runs of different configurations.**
    When ``fingerprints`` is given, each label's runs are split into those that
    could confirm the baseline's group and those that merely re-execute its
    configuration, and ``outcome_agrees`` is decided on the first set alone.
    Without it, every run is treated as independent — which is what this
    function did before the distinction existed, and is why a group present in
    runs 5, 7 and 10 read as replicated three times over.
    """
    base_counts = normalise_outcomes(outcomes.get(baseline_id, {}))
    base_eval = {e.task: e for e in evaluations.get(baseline_id, ())}

    outcome_deltas: dict[int, dict[str, int]] = {}
    metric_deltas: dict[int, dict[str, float | None]] = {}
    for rid, counts in outcomes.items():
        if rid == baseline_id:
            continue
        here = normalise_outcomes(counts)
        outcome_deltas[rid] = {k: here[k] - base_counts[k] for k in OUTCOMES}
        md: dict[str, float | None] = {}
        for e in evaluations.get(rid, ()):
            base = base_eval.get(e.task)
            for key in ("map50", "map50_95", "map75"):
                md[f"{key}_{e.task}"] = _delta(
                    measured(base.metrics.get(key)) if base else None,
                    measured(e.metrics.get(key)),
                )
        metric_deltas[rid] = md

    factor_lift: dict[str, dict[int, float | None]] = {}
    for rid, rows in factors.items():
        for r in rows:
            factor_lift.setdefault(r.factor, {})[rid] = (
                None if r.lift is None else round(r.lift, 3)
            )

    # Replication: for each label, which runs carry it and whether every pair
    # agrees on what the group does — the same test app.recommendations runs.
    by_label: dict[str, list[tuple[int, GroupRow]]] = {}
    for rid, rows in groups.items():
        for g in rows:
            by_label.setdefault(g.label, []).append((rid, g))
    replication: dict[str, Any] = {}
    for label, present in by_label.items():
        present.sort(key=lambda p: p[0])
        first_rid, first = present[0]
        # Anything sharing the anchor's fingerprint repeats it rather than
        # testing it. With no fingerprints supplied nothing is a reproduction,
        # which preserves the previous behaviour exactly.
        anchor = fingerprints.get(first_rid) if fingerprints else None
        independent: list[int] = []
        reproductions: list[int] = []
        for rid, _ in present[1:]:
            same = anchor is not None and fingerprints.get(rid) == anchor
            (reproductions if same else independent).append(rid)

        verdict: bool | None = None
        # Judged on independent runs alone: a re-execution cannot disagree, so
        # counting its agreement would turn a repeated command into evidence.
        verdicts = [
            outcomes_agree(
                normalise_outcomes(first.outcomes)["false_negative"],
                first.size,
                normalise_outcomes(g.outcomes)["false_negative"],
                g.size,
            )
            for rid, g in present[1:]
            if rid in independent and g.size and first.size
        ]
        if verdicts:
            verdict = all(verdicts)

        replication[label] = {
            "present_in": [rid for rid, _ in present],
            "independent_runs": independent,
            "reproductions": reproductions,
            "sizes": {rid: g.size for rid, g in present},
            "outcome_agrees": verdict,
        }

    shared = None
    per_run_actionable = [
        {f"{r.rule}:{r.cluster_label}" for r in rows if r.actionable}
        for rows in recommendations.values()
    ]
    if per_run_actionable:
        shared = sorted(set.intersection(*per_run_actionable))

    return {
        "baseline": baseline_id,
        "outcome_deltas": outcome_deltas,
        "metric_deltas": metric_deltas,
        "factor_lift": factor_lift,
        "group_replication": replication,
        "shared_actionable_recommendations": shared or [],
    }


def evidence_gaps(
    run_ids: Iterable[int], present: Mapping[int, Mapping[str, bool]]
) -> list[dict[str, str | int]]:
    """Return what each run lacks, with the command that produces it."""
    gaps: list[dict[str, str | int]] = []
    for rid in run_ids:
        for kind, remedy in REMEDIES.items():
            if not present.get(rid, {}).get(kind, False):
                gaps.append(
                    {"run_id": rid, "missing": kind, "how": remedy.format(run_id=rid)}
                )
    return gaps


# ---------------------------------------------------------------------------
# Image-level summary (schema version 11)
# ---------------------------------------------------------------------------

#: Verdicts in the order they are reported: the good state first, then the
#: shapes a mistake can take, roughly by how much they say about the model.
IMAGE_VERDICT_ORDER: tuple[str, ...] = (
    "clean",
    "empty",
    "zero_prediction",
    "merged",
    "split",
    "merged_and_split",
    "partly_missed",
    "spurious",
    "partly_missed_and_spurious",
    "partial_coverage",
)


def image_summary(rows: Sequence[Any]) -> dict[str, Any] | None:
    """Aggregate one run's image diagnoses, or ``None`` when not measured.

    **``empty`` is reported separately and never counted as clean.** An image
    with nothing to find and nothing found is correct behaviour, but folding it
    into the clean rate would let a test set of blank photographs score
    perfectly. The clean rate is therefore over images that had something to
    find or something predicted.

    The single/multi split is reported because an object's chances depend on
    what else is in frame: on this project's data an object in a multi-object
    image failed 2.1x to 2.8x more often, significantly on three runs of five.
    It is offered as a measurement, not as a conclusion.

    Thresholds travel with the numbers, since merge and split counts are
    sensitive to ``cover_hit``.
    """
    if not rows:
        return None

    verdicts: dict[str, int] = {}
    for row in rows:
        verdicts[row.verdict] = verdicts.get(row.verdict, 0) + 1

    empty = verdicts.get("empty", 0)
    scored = [r for r in rows if r.verdict != "empty"]
    clean = verdicts.get("clean", 0)

    def failures(group: Sequence[Any]) -> int:
        return sum(
            r.outcomes["false_negative"]
            + r.outcomes["false_positive"]
            + r.outcomes["poor_localization"]
            + r.outcomes["wrong_class"]
            for r in group
        )

    def block(group: Sequence[Any]) -> dict[str, Any]:
        objects = sum(r.gt_count for r in group)
        failed = failures(group)
        return {
            "images": len(group),
            "objects": objects,
            "failures": failed,
            "failures_per_object": round(failed / objects, 4) if objects else None,
            "clean_images": sum(1 for r in group if r.verdict == "clean"),
        }

    single = [r for r in scored if r.gt_count == 1]
    multi = [r for r in scored if r.gt_count > 1]
    first = rows[0]
    return {
        "total": len(rows),
        "scored": len(scored),
        "empty": empty,
        "clean": clean,
        "clean_rate": round(clean / len(scored), 4) if scored else None,
        "affected": len(scored) - clean,
        "verdicts": {v: verdicts[v] for v in IMAGE_VERDICT_ORDER if v in verdicts},
        "merged_images": sum(1 for r in rows if r.merged),
        "split_images": sum(1 for r in rows if r.split),
        "by_object_count": {"single": block(single), "multi": block(multi)},
        "thresholds": {
            "cover_hit": first.cover_hit,
            "cover_miss": first.cover_miss,
            "method": first.method,
        },
    }
