"""A read-only MCP projection of the diagnosis schema, for a reasoning model.

**This adds no analysis.** Every number a tool returns is a stored measurement
or an arithmetic combination of stored counts whose rule is stated beside it,
assembled by :mod:`model_doctor.app.comparison` from rows
:mod:`model_doctor.app.storage` already knows
how to read. Model Doctor is the evidence layer; whatever reads these tools is
the reasoning layer. Keeping that boundary sharp is the point: a tool that
offered its own conclusions would compete with the model it serves.

**Read-only, structurally.** Connections come from
:func:`model_doctor.app.storage.connect_read_only`, the same ``mode=ro`` opener the HTTP
API uses, so "read-only" has one implementation (D-037). There is no tool that
deletes, mutates, trains, or triggers inference, and the transport is stdio —
no port, no network surface. Runs are addressed by id, never by path, and the
database location comes from configuration rather than from any argument, for
the reason :func:`model_doctor.app.api.database_path` already states: letting a caller
name
the file would let it name any file.

**Compact by design.** Findings, root-cause rows, images and heatmaps are never
returned. They are per-object — thousands of rows in a modest database — and
the aggregates already carry the evidence: outcome counts, factor lift against
a control rate, failure groups with their outcome mix, and recommendations with
their evidential status. A comparison of several runs fits in a few kilobytes.

Run it::

    MD_DB_PATH=/path/to/model_doctor.db python -m model_doctor.app.mcp_server
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from model_doctor import config
from model_doctor.app import capabilities, comparison, storage
from model_doctor.app.clustering import DISCRIMINATING_METHOD, FACTOR_SIGNATURE_METHOD
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Optional tables, and the evidence kind each one carries.
_EVIDENCE_TABLES: dict[str, str] = {
    "run_evaluations": "evaluation",
    "factor_rates": "factor_rates",
    "clusters": "groups",
    "recommendations": "recommendations",
    "mask_findings": "mask_findings",
    "run_benchmarks": "benchmarks",
    "heatmaps": "heatmaps",
    "image_diagnoses": "image_diagnoses",
    "finding_relations": "relations",
}

#: Every tool here is safe to call repeatedly and changes nothing.
_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

INSTRUCTIONS = """Model Doctor diagnoses why an object-detection model fails.
These tools expose its stored analysis, read-only. `list_runs` says which runs
exist and what evidence each carries; `get_analysis` returns the evidence for a
set of runs in one comparison-friendly structure.

Treat every figure as evidence and keep your own reasoning separate from it.
Outcome counts are not COCO mAP and will not agree with it. Factors are
correlations against a control rate, not proven causes. A recommendation's
`status` says how much weight it can bear: only `replicated`, `reproduced` and
`provisional` are actionable, and `reproduced` means the only runs that agreed
were re-executions of the same configuration. `null` means not measured, never
zero.

The `relations` block describes how two findings relate — which prediction
covers which object — rather than which condition co-occurred with a failure.
It is measured geometry, not a cause. `duplicate_prediction` inside it is a
provisional reading of the continuous `prediction_on_matched_object`
measurement; it does not establish suppression settings, decoding, assignment
order, architecture or anything else as the mechanism, and none of those is
measured."""


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    """Open the configured database read-only, or fail in tool vocabulary."""
    try:
        with storage.connect_read_only() as connection:
            yield connection
    except FileNotFoundError as error:
        raise ToolError(str(error)) from error
    except sqlite3.Error as error:
        raise ToolError(f"Cannot open the database at {config.DB_PATH}.") from error


def _resolve_run_ids(run_ids: Sequence[int]) -> list[int]:
    """Validate a caller's run ids before anything is read.

    Duplicates are dropped rather than rejected, because ``[4, 4, 5]`` is a
    harmless slip; an empty list, a non-integer, or more than the configured
    maximum is refused with the reason, because each would produce a response
    that is either empty or unreadable.
    """
    if not run_ids:
        raise ToolError("run_ids must name at least one run.")
    cleaned: list[int] = []
    for value in run_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"run_ids must be integers; got {value!r}.")
        if value not in cleaned:
            cleaned.append(value)
    if len(cleaned) > config.MCP_MAX_RUNS:
        raise ToolError(
            f"get_analysis compares at most {config.MCP_MAX_RUNS} runs at once; "
            f"{len(cleaned)} were given. Split them across calls."
        )
    return cleaned


def _evidence_present(connection: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Say which optional evidence a run carries, checking tables first."""
    present: dict[str, Any] = {}
    for table, kind in _EVIDENCE_TABLES.items():
        if not storage.has_table(connection, table):
            present[kind] = False
            continue
        if kind == "evaluation":
            rows = storage.load_evaluations(connection, run_id)
            present[kind] = [r.task for r in rows] or False
        elif kind == "factor_rates":
            present[kind] = bool(storage.load_factor_rates(connection, run_id))
        elif kind == "groups":
            present[kind] = bool(
                storage.load_clusters(connection, run_id, DISCRIMINATING_METHOD)
            )
        elif kind == "recommendations":
            present[kind] = len(storage.load_recommendations(connection, run_id))
        elif kind == "mask_findings":
            present[kind] = bool(storage.load_mask_findings(connection, run_id))
        elif kind == "benchmarks":
            present[kind] = sorted(
                {b.device for b in storage.load_benchmarks(connection, run_id)}
            )
        elif kind == "heatmaps":
            present[kind] = len(storage.load_heatmaps(connection, run_id))
        elif kind == "image_diagnoses":
            present[kind] = bool(storage.load_image_diagnoses(connection, run_id))
        elif kind == "relations":
            present[kind] = bool(storage.load_finding_relations(connection, run_id))
    return present


def _run_row(
    connection: sqlite3.Connection, run: storage.RunRecord, *, include_paths: bool
) -> dict[str, Any]:
    """One run as ``list_runs`` reports it."""
    counts = storage.outcome_counts(connection, run.id)
    findings = storage.load_findings(connection, run.id)
    classes = sorted({f.class_name for f in findings if f.class_name})
    job = storage.load_job_for_run(connection, run.id)
    evaluations = (
        storage.load_evaluations(connection, run.id)
        if storage.has_table(connection, "run_evaluations")
        else []
    )
    bbox = next((e for e in evaluations if e.task == "bbox"), None)

    row = comparison.run_identity(run, include_paths=include_paths)
    # Family comes from the job that produced the run, and from nowhere else:
    # a CLI run has no job, and detecting the family means loading the
    # checkpoint, which a read-only surface must never do.
    row["model"]["family"] = job.detector if job else None
    row["dataset"]["classes"] = classes
    row["dataset"]["classes_source"] = "findings"
    row["counts"] = {
        "images": len(storage.load_images(connection, run.id)),
        **{
            k: v
            for k, v in comparison.outcome_summary(counts).items()
            if k != "derived"
        },
    }
    def headline(key: str) -> float | None:
        value = comparison.measured(bbox.metrics.get(key)) if bbox else None
        return None if value is None else round(value, 4)

    row["headline"] = {
        "map50_box": headline("map50"),
        "map50_95_box": headline("map50_95"),
    }
    row["evidence"] = _evidence_present(connection, run.id)
    row["same_model_runs"] = storage.load_runs_for_model(connection, run.model_sha256)
    return row


def _groups_for(
    connection: sqlite3.Connection, run_id: int, method: str
) -> list[comparison.GroupRow]:
    """Assemble each group with its members' outcome and class mix."""
    if not storage.has_table(connection, "clusters"):
        return []
    rows: list[comparison.GroupRow] = []
    for cluster in storage.load_clusters(connection, run_id, method):
        members = storage.load_cluster_members(connection, cluster.id)
        outcomes = Counter(m["outcome"] for m in members)
        classes = Counter(m["class_name"] for m in members if m["class_name"])
        rows.append(
            comparison.GroupRow(
                cluster_id=cluster.id,
                label=cluster.label,
                size=cluster.size,
                outcomes=dict(outcomes),
                classes=dict(classes),
            )
        )
    return rows


def _mask_summary(connection: sqlite3.Connection, run_id: int) -> dict[str, int] | None:
    """Outline-level outcomes, with unmeasured pairs named as such."""
    if not storage.has_table(connection, "mask_findings"):
        return None
    rows = storage.load_mask_findings(connection, run_id)
    if not rows:
        return None
    tally: Counter[str] = Counter()
    for row in rows:
        tally[row.mask_outcome or "not_measured"] += 1
    return dict(tally)


def create_server() -> MCPServer:
    """Build the server.

    A factory rather than a module-level instance so tests can construct one
    against a different database without reimporting the module — the same
    reason :func:`model_doctor.app.api.create_app` is a factory.
    """
    server = MCPServer(
        name="model-doctor",
        title="Model Doctor",
        instructions=INSTRUCTIONS,
        version=str(storage.SCHEMA_VERSION),
    )

    @server.tool(
        name="list_runs",
        title="List analysed runs",
        description=(
            "Every run in the Model Doctor database, newest first, with what "
            "each one is (model, dataset, split, configuration), its outcome "
            "counts, its headline mAP if evaluated, which optional evidence it "
            "carries, and which other runs share its checkpoint. Use this to "
            "choose runs for get_analysis; runs of the same model are the ones "
            "that can replicate or contradict each other."
        ),
        annotations=_READ_ONLY,
    )
    def list_runs(
        include_paths: Annotated[
            bool,
            Field(
                description=(
                    "Include local filesystem paths for the checkpoint and "
                    "dataset. Off by default: they describe the operator's "
                    "machine, not the run."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        with _database() as connection:
            runs = storage.list_runs(connection)
            return {
                "schema_version": storage.SCHEMA_VERSION,
                "runs": [
                    _run_row(connection, run, include_paths=include_paths)
                    for run in runs
                ],
            }

    @server.tool(
        name="get_analysis",
        title="Get comparable analysis for runs",
        description=(
            "The stored evidence for one or more runs, in one structure built "
            "for comparison: identity and configuration, outcome counts with "
            "derived precision/recall and the rule behind them, COCO mAP/AR "
            "with the protocol that produced it, each factor's rate on "
            "failures against its rate on correct detections with lift and "
            "significance, failure groups with their outcome mix, and "
            "recommendations with their evidential status. A cross-run section "
            "gives deltas against a baseline, which factors and groups "
            "replicate, and what every run lacks. Never returns per-object "
            "rows or images."
        ),
        annotations=_READ_ONLY,
    )
    def get_analysis(
        run_ids: Annotated[
            list[int],
            Field(
                description=(
                    "Runs to analyse, by id from list_runs. One id gives a "
                    "single-run report; several give a comparison."
                ),
                min_length=1,
            ),
        ],
        baseline: Annotated[
            int | None,
            Field(
                description=(
                    "Run every delta is measured against. Defaults to the "
                    "first id given."
                )
            ),
        ] = None,
        include_paths: Annotated[
            bool, Field(description="Include local filesystem paths. Off by default.")
        ] = False,
        include_descriptive_groups: Annotated[
            bool,
            Field(
                description=(
                    "Also return the full factor-signature grouping, which "
                    "describes every failure. Off by default: the "
                    "discriminating grouping is the one to act on."
                )
            ),
        ] = False,
    ) -> dict[str, Any]:
        ids = _resolve_run_ids(run_ids)
        base = ids[0] if baseline is None else baseline
        if base not in ids:
            raise ToolError(f"baseline {base} is not among run_ids {ids}.")

        with _database() as connection:
            runs: dict[int, storage.RunRecord] = {}
            missing: list[int] = []
            for rid in ids:
                run = storage.load_run(connection, rid)
                if run is None:
                    missing.append(rid)
                else:
                    runs[rid] = run
            if missing:
                raise ToolError(
                    f"No such run: {', '.join(str(m) for m in missing)}. "
                    "Call list_runs to see what exists."
                )

            outcomes = {rid: storage.outcome_counts(connection, rid) for rid in ids}
            evaluations = {
                rid: (
                    storage.load_evaluations(connection, rid)
                    if storage.has_table(connection, "run_evaluations")
                    else []
                )
                for rid in ids
            }
            benchmarks = {
                rid: (
                    storage.load_benchmarks(connection, rid)
                    if storage.has_table(connection, "run_benchmarks")
                    else []
                )
                for rid in ids
            }
            factors = {
                rid: (
                    storage.load_factor_rates(connection, rid)
                    if storage.has_table(connection, "factor_rates")
                    else []
                )
                for rid in ids
            }
            groups = {
                rid: _groups_for(connection, rid, DISCRIMINATING_METHOD) for rid in ids
            }
            recommendations = {
                rid: (
                    storage.load_recommendations(connection, rid)
                    if storage.has_table(connection, "recommendations")
                    else []
                )
                for rid in ids
            }

            per_run: dict[int, dict[str, Any]] = {}
            present: dict[int, dict[str, Any]] = {}
            for rid in ids:
                evidence = _evidence_present(connection, rid)
                present[rid] = evidence
                block: dict[str, Any] = {
                    "outcomes": comparison.outcome_summary(outcomes[rid]),
                    "evaluation": comparison.evaluation_summary(evaluations[rid]),
                    "mask_summary": _mask_summary(connection, rid),
                    "factors": comparison.factor_summary(factors[rid]),
                    "groups": comparison.group_summary(groups[rid]),
                    "recommendations": comparison.recommendation_summary(
                        recommendations[rid]
                    ),
                    "benchmarks": comparison.benchmark_summary(benchmarks[rid]),
                    # A second lens over the same findings: how many whole
                    # photographs were handled correctly, and what shape the
                    # mistakes took. None when the pass has not been run.
                    "images": comparison.image_summary(
                        storage.load_image_diagnoses(connection, rid)
                        if storage.has_table(connection, "image_diagnoses")
                        else []
                    ),
                    # A third lens: not which object failed, nor what shape the
                    # image's mistake took, but how two findings relate — which
                    # prediction covers which object. None when unmeasured.
                    "relations": comparison.relation_summary(
                        storage.load_finding_relations(connection, rid)
                        if storage.has_table(connection, "finding_relations")
                        else []
                    ),
                }
                if include_descriptive_groups:
                    block["descriptive_groups"] = comparison.group_summary(
                        _groups_for(connection, rid, FACTOR_SIGNATURE_METHOD)
                    )
                per_run[rid] = block

            return {
                "schema_version": storage.SCHEMA_VERSION,
                "runs": [
                    _run_row(connection, runs[rid], include_paths=include_paths)
                    for rid in ids
                ],
                "comparability": comparison.comparability(
                    [runs[rid] for rid in ids], evaluations, benchmarks
                ),
                "per_run": per_run,
                # Fingerprints are passed so replication is judged only between
                # differently configured runs. Without them a group present in
                # three re-executions of one command reads as replicated three
                # times over.
                "cross_run": comparison.cross_run(
                    base,
                    outcomes,
                    evaluations,
                    factors,
                    groups,
                    recommendations,
                    fingerprints={
                        rid: comparison.run_fingerprint(runs[rid]) for rid in ids
                    },
                ),
                "evidence_gaps": comparison.evidence_gaps(ids, present),
                "caveats": list(comparison.CAVEATS),
            }

    @server.tool(
        name="experiment_feasibility",
        title="Can varying this knob change anything?",
        description=(
            "Whether an inference parameter can move these runs' output at "
            "all, asked before an experiment is designed rather than "
            "discovered after it is run. Reports three separate facts per run: "
            "whether the detector accepts the parameter, whether Model "
            "Doctor's own path forwards it to the model, and whether changing "
            "it changes the output — which are routinely different answers. "
            "`actuation` is 'actuates', 'inert' or 'unknown'; 'unknown' is "
            "never a synonym for 'inert', and an observed absence of effect "
            "without an architectural reason stays 'unknown'. Use it to avoid "
            "spending runs on a parameter nothing reads. It proposes no "
            "experiment and draws no conclusion about the model."
        ),
        annotations=_READ_ONLY,
    )
    def experiment_feasibility(
        knob: Annotated[
            str,
            Field(
                description=(
                    "Inference knob to interrogate: "
                    f"{' or '.join(repr(k) for k in capabilities.KNOBS)}."
                )
            ),
        ],
        run_ids: Annotated[
            list[int],
            Field(
                description="Runs the experiment would span, by id.",
                min_length=1,
            ),
        ],
        probe: Annotated[
            bool,
            Field(
                description=(
                    "Run the bounded empirical probe as well as reading the "
                    "architecture. On by default. Turning it off is faster and "
                    "loads no images, at the cost of an 'unknown' wherever the "
                    "architecture alone does not settle the question."
                )
            ),
        ] = True,
    ) -> dict[str, Any]:
        ids = _resolve_run_ids(run_ids)
        with _database() as connection:
            try:
                return capabilities.experiment_feasibility(
                    connection, knob, ids, probe=probe
                )
            except capabilities.CapabilityError as error:
                raise ToolError(str(error)) from error

    return server


def main() -> None:
    """Serve over stdio. The transport is fixed: no port is ever opened."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
