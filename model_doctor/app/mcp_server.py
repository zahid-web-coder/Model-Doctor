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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from model_doctor import config
from model_doctor.app import capabilities, mcp_payloads, storage
from model_doctor.app.mcp_payloads import PayloadError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

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
            try:
                return mcp_payloads.list_runs_payload(
                    connection, include_paths=include_paths
                )
            except PayloadError as error:
                raise _as_tool_error(error) from error

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
        with _database() as connection:
            try:
                return mcp_payloads.get_analysis_payload(
                    connection,
                    run_ids,
                    baseline=baseline,
                    include_paths=include_paths,
                    include_descriptive_groups=include_descriptive_groups,
                )
            except PayloadError as error:
                raise _as_tool_error(error) from error

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
        try:
            ids = mcp_payloads.resolve_run_ids(run_ids)
        except PayloadError as error:
            raise _as_tool_error(error) from error
        with _database() as connection:
            try:
                return capabilities.experiment_feasibility(
                    connection, knob, ids, probe=probe
                )
            except capabilities.CapabilityError as error:
                raise ToolError(str(error)) from error

    return server


def _as_tool_error(error: PayloadError) -> ToolError:
    """Map a payload refusal into this transport's vocabulary.

    The payload layer cannot raise `ToolError` — it must not import `mcp` at
    all — so the translation happens here, once, and the message it wrote for
    the reader passes through unaltered.
    """
    return ToolError(str(error))


def main() -> None:
    """Serve over stdio. The transport is fixed: no port is ever opened."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
