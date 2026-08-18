"""Group a run's failures so an engineer fixes patterns, not individual images.

**This is not unsupervised clustering, and the distinction matters.** The
default method assigns each failure to a group by its attributed root-cause
factors, deterministically. Two failures share a group when they share a
factor set — nothing is learned, nothing is fitted, and running it twice on the
same data gives the same answer.

That choice was made against measurement, not taste. K-means over the CLIP
embeddings of failure crops was tried first and scored a silhouette of
0.13–0.23 across every *k* from 2 to 8 and both PCA-10 and PCA-20 projections.
Below 0.25 there is no substantial structure to find. Worse, the clusters it
did produce largely re-encoded class and outcome — facts already in the
``findings`` table — and arrived without labels, so every one of them would
have needed a human to work out what it meant. Factor signatures are
interpretable by construction: a group called ``blur + edge_truncation`` needs
no interpretation step. See ``docs/DECISIONS.md`` D-030 for the full numbers.

The embeddings are untouched and still earn their place — as nearest-neighbour
retrieval in :mod:`app.similarity`, which is statistically sound at this sample
size in a way that clustering is not.

Grouping runs as a pass over an already-saved run, like feature extraction,
explanation, and root-cause analysis before it. The write path is untouched and
historical runs can be grouped without re-running inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from app import storage
from utils.exceptions import ModelDoctorError
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Identifies the grouping method in `clusters.method`. Stored rather than
# implied so a second method can be added later and the two sets of groups stay
# distinguishable without a schema change.
FACTOR_SIGNATURE_METHOD: str = "factor-signature"

# The same rule applied to a restricted factor set: only those shown to occur
# more often in failures than in correct findings, on more than one split
# (`config.DISCRIMINATING_FACTORS`). Stored alongside the full signature rather
# than replacing it — `clusters.method` exists so two partitions of the same
# findings can coexist, and each answers a different question. The full
# signature is the complete descriptive record; this one is what an engineer
# should act on (D-033).
DISCRIMINATING_METHOD: str = "discriminating-signature"

# The group for failures no factor explained. Named rather than omitted: these
# are the failures the current detectors cannot account for, which makes them
# the most interesting ones to look at, not the ones to hide.
UNEXPLAINED_LABEL: str = "unexplained"

# Joins factor names into a group label. Chosen to read as prose in a UI
# ("blur + edge_truncation") rather than as a machine key.
SIGNATURE_SEPARATOR: str = " + "


class FailureGroupingError(ModelDoctorError):
    """A run cannot be grouped, with a reason the caller can act on."""


def signature_for(factors: Iterable[str]) -> str:
    """Build the group label for one failure's set of factors.

    Factors are sorted, so a failure attributed ``blur`` then ``crowding``
    lands in the same group as one attributed ``crowding`` then ``blur``.
    Without that, group membership would depend on row order and the same run
    could group differently on re-analysis.

    Duplicates are collapsed. The database's ``UNIQUE (finding_id, factor)``
    constraint should already prevent them, but a caller passing a plain list
    should not be able to produce ``blur + blur``.

    Args:
        factors: Factor names attributed to one finding.

    Returns:
        The group label, or :data:`UNEXPLAINED_LABEL` when there are none.
    """
    unique = sorted(set(factors))
    return SIGNATURE_SEPARATOR.join(unique) if unique else UNEXPLAINED_LABEL


def group_by_factor_signature(
    failure_ids: Sequence[int],
    factors_by_finding: Mapping[int, Sequence[str]],
) -> dict[str, list[int]]:
    """Partition failures into groups by their factor signature.

    Pure: it touches no database and reads no images, so the grouping rule can
    be tested directly on constructed input rather than through a saved run.

    Every failure lands in exactly one group. A failure absent from
    ``factors_by_finding`` is unexplained rather than skipped — dropping it
    would make the group sizes silently fail to sum to the failure count, which
    is the kind of arithmetic a reader is entitled to trust.

    Args:
        failure_ids: Every failure in the run. This is the authority on what
            must be grouped; the factor mapping only says how.
        factors_by_finding: Attributed factors, keyed by finding id.

    Returns:
        Mapping of group label to member finding ids, each list in ascending
        id order. Groups are ordered largest first, then by label, so a
        consumer that iterates without sorting still sees the biggest problem
        first.
    """
    groups: dict[str, list[int]] = {}
    for finding_id in failure_ids:
        label = signature_for(factors_by_finding.get(finding_id, ()))
        groups.setdefault(label, []).append(finding_id)

    for members in groups.values():
        members.sort()

    return dict(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    )


@dataclass(frozen=True)
class GroupingReport:
    """What grouping a run produced, in terms a reader can check.

    Deliberately reports the arithmetic that should hold — grouped equals the
    failure count — rather than leaving it implied.
    """

    run_id: int
    method: str
    failure_count: int
    group_count: int
    grouped_count: int
    unexplained_count: int
    largest_label: str
    largest_size: int

    def describe(self) -> str:
        """Render the report as a fixed-width block for the CLI."""
        share = (
            100 * self.largest_size / self.failure_count
            if self.failure_count
            else 0.0
        )
        unexplained_share = (
            100 * self.unexplained_count / self.failure_count
            if self.failure_count
            else 0.0
        )
        width = 66
        lines = [
            "",
            "Failure grouping",
            "-" * width,
            f"  Run                : {self.run_id}",
            f"  Method             : {self.method}",
            f"  Failures           : {self.failure_count}",
            f"  Grouped            : {self.grouped_count}",
            f"  Groups             : {self.group_count}",
            f"  Largest group      : {self.largest_label} "
            f"({self.largest_size}, {share:.1f}%)",
            f"  Unexplained        : {self.unexplained_count} "
            f"({unexplained_share:.1f}%)",
            "-" * width,
            "",
        ]
        return "\n".join(lines)


def group_run(
    connection: Any,
    run_id: int,
    method: str = FACTOR_SIGNATURE_METHOD,
    allowed_factors: Iterable[str] | None = None,
) -> GroupingReport:
    """Group a saved run's failures and store the result.

    Args:
        connection: An open database connection.
        run_id: Run to group. Its findings must already be saved, and its
            root causes analysed — grouping reads attribution, it does not
            compute it.
        method: Value recorded in ``clusters.method``. Groups stored under one
            method never disturb those stored under another, so two partitions
            of the same findings coexist.
        allowed_factors: Restrict grouping to these factors. Others are still
            attributed and stored in ``root_causes``; they simply do not split
            failures. ``None`` uses every factor.

    Returns:
        A description of the grouping produced.

    Raises:
        FailureGroupingError: If the run has no failures to group.
    """
    failure_ids = storage.load_failure_ids(connection, run_id)
    if not failure_ids:
        raise FailureGroupingError(
            f"Run {run_id} has no failures to group. Diagnose and save it first."
        )

    factors_by_finding = storage.load_factors_by_finding(connection, run_id)
    if not factors_by_finding:
        logger.warning(
            "Run %d has no attributed root causes, so every failure will be "
            "'%s'. Run: python -m app.root_cause --run %d",
            run_id,
            UNEXPLAINED_LABEL,
            run_id,
        )

    if allowed_factors is not None:
        permitted = frozenset(allowed_factors)
        factors_by_finding = {
            finding_id: [f for f in factors if f in permitted]
            for finding_id, factors in factors_by_finding.items()
        }

    groups = group_by_factor_signature(failure_ids, factors_by_finding)
    storage.save_clusters(connection, run_id, method, groups.items())

    largest_label, largest_members = next(iter(groups.items()))
    return GroupingReport(
        run_id=run_id,
        method=method,
        failure_count=len(failure_ids),
        group_count=len(groups),
        grouped_count=sum(len(members) for members in groups.values()),
        unexplained_count=len(groups.get(UNEXPLAINED_LABEL, ())),
        largest_label=largest_label,
        largest_size=len(largest_members),
    )


def format_group_detail(
    connection: Any, run_id: int, method: str, limit: int = 10
) -> str:
    """Render the largest groups with their outcome mix, for the CLI.

    Args:
        connection: An open database connection.
        run_id: Run to describe.
        method: Grouping method to read.
        limit: How many groups to show, largest first.

    Returns:
        A printable block, or a note when nothing has been grouped.
    """
    groups = storage.load_clusters(connection, run_id, method)
    if not groups:
        return "\n  Nothing grouped yet.\n"

    lines = ["", f"Largest groups (top {min(limit, len(groups))})", "-" * 66]
    for group in groups[:limit]:
        members = storage.load_cluster_members(connection, group.id)
        outcomes: dict[str, int] = {}
        classes: dict[str, int] = {}
        for member in members:
            outcomes[member["outcome"]] = outcomes.get(member["outcome"], 0) + 1
            classes[member["class_name"]] = classes.get(member["class_name"], 0) + 1
        outcome_text = ", ".join(
            f"{count} {name.replace('_', ' ')}"
            for name, count in sorted(outcomes.items(), key=lambda i: -i[1])
        )
        class_text = ", ".join(
            f"{name} {count}"
            for name, count in sorted(classes.items(), key=lambda i: -i[1])
        )
        lines.append(f"  {group.size:>4}  {group.label}")
        lines.append(f"        {outcome_text}")
        lines.append(f"        {class_text}")
    lines.append("-" * 66)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the failure-grouping command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-grouping",
        description="Group a run's failures by their attributed root causes.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=str, help="Database path override.")
    parser.add_argument(
        "--detail", type=int, default=10, help="Groups to list. 0 to skip."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to group.
    """
    args = build_parser().parse_args(argv)

    with storage.connect(Path(args.db) if args.db else None) as connection:
        run_id = args.run
        if run_id is None:
            runs = storage.list_runs(connection)
            if not runs:
                logger.error(
                    "No runs found. Run: python -m app.diagnosis --split test --save"
                )
                return 1
            run_id = runs[0].id
            logger.info("Using newest run %d", run_id)

        try:
            # Both partitions are stored. The full signature is the complete
            # descriptive record; the discriminating one is what to act on.
            full_report = group_run(connection, run_id)
            report = group_run(
                connection,
                run_id,
                method=DISCRIMINATING_METHOD,
                allowed_factors=config.DISCRIMINATING_FACTORS,
            )
        except FailureGroupingError as exc:
            logger.error("%s", exc)
            return 1

        detail = (
            format_group_detail(
                connection, run_id, DISCRIMINATING_METHOD, args.detail
            )
            if args.detail
            else ""
        )

    print(report.describe())
    print(
        f"  Also stored under '{FACTOR_SIGNATURE_METHOD}': "
        f"{full_report.group_count} group(s) from every attributed factor, "
        f"{full_report.unexplained_count} unexplained.\n"
        "  That partition describes; this one discriminates. See D-033.\n"
    )
    if detail:
        print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
