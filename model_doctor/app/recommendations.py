"""Turn failure groups into suggested actions, or say why there are none.

**Every recommendation is traceable to the evidence that caused it.** The chain
is navigable in SQL — a recommendation names a `cluster_id`; the cluster names
its factors in its label and its members through `cluster_members`; each factor
carries its rate among failures *and* among correct findings in `factor_rates`.
No rule may fire on a factor's raw frequency: qualification is re-checked
against measured lift and significance before any action is proposed, even
though the discriminating grouping has already filtered for it.

That double check is deliberate. A factor describing 71% of failures reads as a
cause until the control group shows it describes 76% of correct detections
(D-031), and the whole point of this module is not to make that mistake in
public.

**Evidence status is part of the output, not a footnote.** A pattern seen once
is reported as provisional. A pattern that two runs of the same model disagree
about is reported as conflicting and is *not* actionable — `thin_structure` is
85% false positive on one split and evenly divided on another, differing at
p = 0.011, and a tool without that status would emit a confident fix for it.

**Absence of evidence is stored, not omitted.** A group with nothing to say
gets a row saying so. Silence is indistinguishable from "the pass never ran",
which is the same reason `images.error` exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.clustering import DISCRIMINATING_METHOD, UNEXPLAINED_LABEL
from model_doctor.app.comparison import (
    factor_qualifies,
    outcomes_agree,
    partition_by_fingerprint,
)
from model_doctor.utils.exceptions import ModelDoctorError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RecommendationError(ModelDoctorError):
    """Recommendations could not be produced, with a reason the caller can act on."""


# Evidence status. Ordered from strongest to weakest; the first two are
# actionable and the last two are deliberately not.
REPLICATED: str = "replicated"
#: Agreed with by re-executions of this run's own configuration, and nothing
#: else. Evidentially a single observation: inference is deterministic, so a
#: run with the same fingerprint cannot disagree, and its agreement confirms
#: only that the command was run twice. Named rather than folded into
#: `provisional` so a reader can see that the agreement exists and why it does
#: not count.
REPRODUCED: str = "reproduced"
PROVISIONAL: str = "provisional"
CONFLICTING: str = "conflicting"
INSUFFICIENT: str = "insufficient_evidence"

ACTIONABLE_STATUSES: frozenset[str] = frozenset(
    {REPLICATED, REPRODUCED, PROVISIONAL}
)

# Rule identifiers, stored verbatim in `recommendations.rule`. Part of what a
# consumer reads, so changing one is a contract change.
RECALL_RULE: str = "recall_on_factor"
PRECISION_RULE: str = "precision_on_factor"
BACKLOG_RULE: str = "unexplained_backlog"
UNSTABLE_RULE: str = "unstable_pattern"
TOO_SMALL_RULE: str = "group_too_small"


@dataclass(frozen=True)
class GroupEvidence:
    """Everything known about one failure group, gathered before any rule runs.

    Assembled once so each rule reads the same facts. A rule that gathered its
    own evidence could disagree with its neighbour about the same group.
    """

    cluster_id: int
    label: str
    size: int
    factors: tuple[str, ...]
    false_negatives: int
    false_positives: int
    dominant_class: str
    dominant_class_share: float
    qualified_factors: tuple[str, ...]
    excluded_factors: tuple[tuple[str, float | None, float], ...]
    comparison_runs: tuple[int, ...]
    outcome_agrees: bool | None
    #: Runs that re-execute this run's own configuration. They are carried so
    #: the rationale can say they exist without counting them as agreement.
    reproduction_runs: tuple[int, ...] = ()
    #: Whether those re-executions match. Expected to be True; False means two
    #: identical configurations produced different groups, which is a
    #: data-integrity problem rather than a disagreement about the pattern.
    reproductions_agree: bool | None = None

    @property
    def false_negative_share(self) -> float:
        """Share of this group's failures that are misses."""
        return self.false_negatives / self.size if self.size else 0.0

    @property
    def false_positive_share(self) -> float:
        """Share of this group's failures that are inventions."""
        return self.false_positives / self.size if self.size else 0.0


def _qualifies(lift: float | None, p_value: float) -> bool:
    """Return whether a factor is more common in failures than in successes.

    The rule itself lives in :func:`model_doctor.app.comparison.factor_qualifies`,
    shared
    with every other consumer that asks the question, so this module and the
    MCP server cannot disagree about which factors are worth citing.
    """
    return factor_qualifies(lift, p_value)


def comparable_runs(
    connection: Any, run_id: int, candidate_run_ids: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Split other runs into those that can test a pattern and those that repeat it.

    Two filters, for two different ways of manufacturing confidence.

    A run only counts if it has measured factor rates: a run analysed before
    those existed cannot agree or disagree, and treating its silence as
    agreement would invent replication out of missing data.

    A run only counts as *independent* if its fingerprint differs. Inference is
    deterministic, so a re-execution of the same checkpoint, dataset, split and
    thresholds returns the same numbers by construction — runs 5, 6, 7 and 10
    of the reference database differ by exactly zero on every measured
    quantity. Counting those as confirmations turned one observation into four.

    Returns:
        ``(independent, reproductions)``, both sorted. Only the first may
        decide whether a pattern replicated.
    """
    subject = storage.load_run(connection, run_id)
    usable = [
        other
        for other in candidate_run_ids
        if other != run_id and storage.load_factor_rates(connection, other)
    ]
    if subject is None:
        return sorted(usable), []
    records = [
        run for run in (storage.load_run(connection, o) for o in usable) if run
    ]
    return partition_by_fingerprint(subject, records)


def gather_evidence(
    connection: Any,
    run_id: int,
    comparison_run_ids: Sequence[int],
    reproduction_run_ids: Sequence[int] = (),
) -> list[GroupEvidence]:
    """Collect the facts each rule needs, for every discriminating group.

    Args:
        connection: An open database connection.
        run_id: Run whose groups are being examined.
        comparison_run_ids: Independently configured runs of the same model,
            used to decide whether a pattern replicates. Runs without measured
            factor rates contribute nothing — missing data is not evidence of
            qualification.
        reproduction_run_ids: Runs re-executing this run's own configuration.
            Reported so a reader knows they exist, and never allowed to decide
            replication: a deterministic re-run cannot disagree.

    Returns:
        One record per group, ordered largest first.
    """
    groups = storage.load_clusters(connection, run_id, DISCRIMINATING_METHOD)
    rates = {row.factor: row for row in storage.load_factor_rates(connection, run_id)}

    # Only runs that actually measured rates can confirm or contradict.
    comparisons: dict[int, dict[str, Any]] = {}
    for other in comparison_run_ids:
        if other == run_id:
            continue
        other_rates = {
            row.factor: row for row in storage.load_factor_rates(connection, other)
        }
        if other_rates:
            comparisons[other] = other_rates

    # Kept in a separate list, never merged into `comparisons`: the two are
    # asked the same question but only one of them may answer it.
    reproductions = [
        other
        for other in reproduction_run_ids
        if other != run_id and storage.load_factor_rates(connection, other)
    ]

    evidence: list[GroupEvidence] = []
    for group in groups:
        members = storage.load_cluster_members(connection, group.id)
        outcomes: dict[str, int] = {}
        classes: dict[str, int] = {}
        for member in members:
            outcomes[member["outcome"]] = outcomes.get(member["outcome"], 0) + 1
            classes[member["class_name"]] = classes.get(member["class_name"], 0) + 1

        factors = (
            ()
            if group.label == UNEXPLAINED_LABEL
            else tuple(part.strip() for part in group.label.split("+"))
        )

        qualified: list[str] = []
        excluded: list[tuple[str, float | None, float]] = []
        for factor in factors:
            rate = rates.get(factor)
            if rate is not None and _qualifies(rate.lift, rate.p_value):
                qualified.append(factor)
            elif rate is not None:
                excluded.append((factor, rate.lift, rate.p_value))

        top_class = max(classes.items(), key=lambda item: item[1], default=("", 0))
        evidence.append(
            GroupEvidence(
                cluster_id=group.id,
                label=group.label,
                size=group.size,
                factors=factors,
                false_negatives=outcomes.get("false_negative", 0),
                false_positives=outcomes.get("false_positive", 0),
                dominant_class=top_class[0],
                dominant_class_share=top_class[1] / group.size if group.size else 0.0,
                qualified_factors=tuple(qualified),
                excluded_factors=tuple(excluded),
                comparison_runs=tuple(sorted(comparisons)),
                outcome_agrees=_outcome_agrees(
                    connection, group.label, outcomes, group.size, comparisons
                ),
                reproduction_runs=tuple(sorted(reproductions)),
                # The same test, run separately, so a re-execution's agreement
                # is visible without ever being mixed into the verdict above.
                reproductions_agree=_outcome_agrees(
                    connection, group.label, outcomes, group.size, reproductions
                ),
            )
        )
    return evidence


def _outcome_agrees(
    connection: Any,
    label: str,
    outcomes: dict[str, int],
    size: int,
    comparisons: dict[int, dict[str, Any]],
) -> bool | None:
    """Return whether other runs agree about what this group *does*.

    A group can be stable while its meaning is not. ``thin_structure`` is the
    same share of failures on two splits, yet 85% false positive on one and
    evenly divided on the other. Membership replicating is not the same as the
    pattern replicating, and only the second justifies an action.

    Returns:
        ``True`` when every comparable run agrees, ``False`` when one differs
        significantly, and ``None`` when no run has the same group to compare.
    """
    here = outcomes.get("false_negative", 0)
    verdicts: list[bool] = []
    for other_run in comparisons:
        other = connection.execute(
            """
            SELECT c.id, c.size FROM clusters c
            WHERE c.run_id = ? AND c.method = ? AND c.label = ?
            """,
            (other_run, DISCRIMINATING_METHOD, label),
        ).fetchone()
        if other is None or not other["size"]:
            continue
        there = connection.execute(
            """
            SELECT COUNT(*) AS n FROM cluster_members cm
            JOIN findings f ON f.id = cm.finding_id
            WHERE cm.cluster_id = ? AND f.outcome = 'false_negative'
            """,
            (other["id"],),
        ).fetchone()["n"]
        # The statistical decision is model_doctor.app.comparison's, so a cross-run
        # report
        # and this pass reach the same verdict from the same counts.
        verdicts.append(outcomes_agree(here, size, there, other["size"]))

    if not verdicts:
        return None
    return all(verdicts)


def _rationale(evidence: GroupEvidence, headline: str) -> str:
    """Compose the evidence trail behind one recommendation.

    States what was found, what was ruled out, and which runs were compared.
    Naming the excluded factors matters as much as naming the cause: it is the
    difference between explaining a failure and pattern-matching on frequency.
    """
    parts = [headline]
    if evidence.dominant_class:
        parts.append(
            f"{evidence.dominant_class} accounts for "
            f"{100 * evidence.dominant_class_share:.0f}% of the group."
        )
    if evidence.excluded_factors:
        ruled_out = "; ".join(
            f"{factor} (lift "
            + ("n/a" if lift is None else f"{lift:.2f}x")
            + f", p={p:.2f})"
            for factor, lift, p in evidence.excluded_factors
        )
        parts.append(f"Ruled out as not distinguishing failures: {ruled_out}.")
    if evidence.comparison_runs:
        runs = ", ".join(str(r) for r in evidence.comparison_runs)
        agreement = {
            True: "which agree on the outcome mix",
            False: "which disagree on the outcome mix",
            None: "which have no comparable group",
        }[evidence.outcome_agrees]
        parts.append(
            f"Compared against independently configured run(s) {runs}, "
            f"{agreement}."
        )
    else:
        parts.append(
            "No independently configured run of this model has measured "
            "factor rates, so this rests on a single observation."
        )
    # Named but never counted: a reader who knows four runs exist should be
    # told what happened to the other three, not left to assume they were
    # overlooked.
    if evidence.reproduction_runs:
        repeats = ", ".join(str(r) for r in evidence.reproduction_runs)
        outcome = {
            True: "reproduce it exactly, which repeats the observation rather "
            "than confirming it",
            False: "re-execute the same configuration yet differ, which is a "
            "data-integrity problem rather than a failed replication",
            None: "re-execute the same configuration but have no comparable "
            "group",
        }[evidence.reproductions_agree]
        parts.append(f"Run(s) {repeats} {outcome}.")
    return " ".join(parts)


def _status_for(evidence: GroupEvidence) -> str:
    """Decide how much weight a group's pattern can bear.

    **Only a run with a different fingerprint can replicate.** A re-execution
    of the same configuration returns the same numbers by construction, so its
    agreement is reproduction and is reported as such. Four cases:

    * an independent run disagrees        -> conflicting
    * every independent run agrees        -> replicated
    * no independent run, but a
      re-execution agrees                 -> reproduced
    * nothing to compare against          -> provisional

    A re-execution that *disagrees* is not a failed replication — it means two
    identical configurations produced different groups, which is a problem with
    the data rather than with the pattern. It is reported as conflicting so it
    cannot be acted on until someone has looked.
    """
    if evidence.outcome_agrees is False:
        return CONFLICTING
    if evidence.outcome_agrees is True:
        return REPLICATED
    if evidence.reproductions_agree is False:
        return CONFLICTING
    if evidence.reproductions_agree is True:
        return REPRODUCED
    return PROVISIONAL


def evaluate(evidence: GroupEvidence) -> list[tuple[str, str, str, str, bool]]:
    """Apply every rule to one group.

    Pure: it reads the gathered facts and returns proposals. No database, so
    the decision each rule makes is testable directly on constructed input.

    Returns:
        Tuples of ``(rule, action, rationale, status, actionable)``. A group
        always yields at least one, because "nothing to recommend" is itself a
        result worth recording.
    """
    if evidence.size < config.MIN_RECOMMENDATION_GROUP_SIZE:
        return [
            (
                TOO_SMALL_RULE,
                "No action — too few failures to draw a conclusion.",
                _rationale(
                    evidence,
                    f"{evidence.size} failure(s), below the minimum of "
                    f"{config.MIN_RECOMMENDATION_GROUP_SIZE} needed for a pattern.",
                ),
                INSUFFICIENT,
                False,
            )
        ]

    if evidence.label == UNEXPLAINED_LABEL:
        return [
            (
                BACKLOG_RULE,
                "Investigate — no measured condition accounts for these failures.",
                _rationale(
                    evidence,
                    f"{evidence.size} failures carry no factor that distinguishes "
                    "them from correct detections. The current detectors cannot "
                    "explain this group; a new factor is needed.",
                ),
                INSUFFICIENT,
                False,
            )
        ]

    if not evidence.qualified_factors:
        return [
            (
                TOO_SMALL_RULE,
                "No action — this group's conditions do not distinguish failures.",
                _rationale(
                    evidence,
                    f"{evidence.size} failures, but no factor in "
                    f"'{evidence.label}' is measurably more common among "
                    "failures than among correct detections.",
                ),
                INSUFFICIENT,
                False,
            )
        ]

    status = _status_for(evidence)
    named = " and ".join(evidence.qualified_factors)

    if status == CONFLICTING:
        return [
            (
                UNSTABLE_RULE,
                "Do not act yet — runs disagree about this pattern.",
                _rationale(
                    evidence,
                    f"{evidence.size} failures share {named}, but the balance of "
                    "misses to false positives differs significantly between "
                    "runs. Collect another run before acting.",
                ),
                CONFLICTING,
                False,
            )
        ]

    proposals: list[tuple[str, str, str, str, bool]] = []
    share = config.RECOMMENDATION_OUTCOME_SHARE

    if evidence.false_negative_share >= share:
        proposals.append(
            (
                RECALL_RULE,
                f"Improve recall for {named}.",
                _rationale(
                    evidence,
                    f"{evidence.false_negatives} of {evidence.size} failures "
                    f"({100 * evidence.false_negative_share:.0f}%) are missed "
                    f"objects sharing {named}. The model is not detecting them "
                    "at all, so this is a coverage problem — more examples of "
                    "this condition, augmentation that produces it, or a larger "
                    "inference size.",
                ),
                status,
                True,
            )
        )

    if evidence.false_positive_share >= share:
        proposals.append(
            (
                PRECISION_RULE,
                f"Improve precision for {named}.",
                _rationale(
                    evidence,
                    f"{evidence.false_positives} of {evidence.size} failures "
                    f"({100 * evidence.false_positive_share:.0f}%) are objects "
                    f"the model invented, sharing {named}. It is over-predicting "
                    "under this condition — raise the confidence threshold or "
                    "add hard negatives that show it.",
                ),
                status,
                True,
            )
        )

    if not proposals:
        proposals.append(
            (
                TOO_SMALL_RULE,
                "No single action — this group mixes misses and false positives.",
                _rationale(
                    evidence,
                    f"{evidence.size} failures share {named}, but neither misses "
                    f"nor false positives reach {100 * share:.0f}% of the group. "
                    "That is two problems, not one.",
                ),
                INSUFFICIENT,
                False,
            )
        )
    return proposals


@dataclass(frozen=True)
class RecommendationReport:
    """What one recommendation pass produced."""

    run_id: int
    group_count: int
    actionable_count: int
    written: int
    comparison_runs: tuple[int, ...]

    def describe(self) -> str:
        """Render the report for console output."""
        width = 70
        compared = (
            ", ".join(str(r) for r in self.comparison_runs)
            if self.comparison_runs
            else "none — single observation"
        )
        return "\n".join(
            [
                "",
                "Recommendations",
                "=" * width,
                f"  Run                : {self.run_id}",
                f"  Failure groups     : {self.group_count}",
                f"  Actionable         : {self.actionable_count}",
                f"  Recorded           : {self.written}",
                f"  Compared with runs : {compared}",
                "=" * width,
                "",
            ]
        )


def recommend_run(connection: Any, run_id: int) -> RecommendationReport:
    """Produce and store recommendations for a saved run's failure groups.

    Args:
        connection: An open database connection.
        run_id: Run to advise on. Its failures must already be grouped under
            the discriminating method.

    Returns:
        A description of what was produced.

    Raises:
        RecommendationError: If the run has no discriminating failure groups.
    """
    run = storage.load_run(connection, run_id)
    if run is None:
        raise RecommendationError(f"Run {run_id} does not exist.")

    groups = storage.load_clusters(connection, run_id, DISCRIMINATING_METHOD)
    if not groups:
        raise RecommendationError(
            f"Run {run_id} has no failure groups. "
            f"Run: python -m model_doctor.app.clustering --run {run_id}"
        )

    candidates = storage.load_runs_for_model(connection, run.model_sha256)
    comparison_runs, reproduction_runs = comparable_runs(
        connection, run_id, candidates
    )
    counted = len(comparison_runs) + len(reproduction_runs)
    if len(candidates) - 1 > counted:
        logger.info(
            "Ignoring %d run(s) of this model with no measured factor rates; "
            "missing data cannot confirm a pattern",
            len(candidates) - 1 - counted,
        )
    if reproduction_runs:
        logger.info(
            "Run(s) %s re-execute run %d's configuration; they are reported "
            "but cannot replicate it",
            ", ".join(str(r) for r in reproduction_runs),
            run_id,
        )
    evidence = gather_evidence(
        connection, run_id, comparison_runs, reproduction_runs
    )

    entries = []
    actionable = 0
    for group in evidence:
        for rule, action, rationale, status, is_actionable in evaluate(group):
            # Ordering is documented and deterministic: actionable first, then
            # by failures addressed. `priority` carries the second term only.
            entries.append(
                (
                    group.cluster_id,
                    rule,
                    action,
                    rationale,
                    status,
                    is_actionable,
                    group.size,
                    float(group.size),
                )
            )
            actionable += 1 if is_actionable else 0

    written = storage.save_recommendations(connection, run_id, entries)
    logger.info("Recorded %d recommendation(s) for run %d", written, run_id)

    return RecommendationReport(
        run_id=run_id,
        group_count=len(groups),
        actionable_count=actionable,
        written=written,
        comparison_runs=tuple(comparison_runs),
    )


def format_recommendations(rows: Sequence[Any]) -> str:
    """Render stored recommendations in the documented display order."""
    if not rows:
        return "\n  Nothing recorded.\n"
    lines = ["", "Suggested actions", "-" * 70]
    for row in rows:
        marker = " " if row.actionable else "·"
        lines.append(f"{marker} [{row.status}] {row.action}")
        lines.append(f"    group   : {row.cluster_label} ({row.affected} failures)")
        lines.append(f"    evidence: {row.rationale}")
        lines.append("")
    lines.append("-" * 70)
    lines.append(
        "  Rows marked · are not actionable: the evidence does not support an\n"
        "  action yet. They are shown rather than hidden."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the recommendations command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-recommend",
        description="Turn failure groups into suggested actions, with evidence.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=str, help="Database path override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to advise on.
    """
    args = build_parser().parse_args(argv)

    with storage.connect(Path(args.db) if args.db else None) as connection:
        run_id = args.run
        if run_id is None:
            runs = storage.list_runs(connection)
            if not runs:
                logger.error(
                    "No runs found. Run: python -m model_doctor.app.diagnosis "
                    "--split test --save"
                )
                return 1
            run_id = runs[0].id
            logger.info("Using newest run %d", run_id)

        try:
            report = recommend_run(connection, run_id)
        except RecommendationError as exc:
            logger.error("%s", exc)
            return 1

        stored = storage.load_recommendations(connection, run_id)

    print(report.describe())
    print(format_recommendations(stored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
