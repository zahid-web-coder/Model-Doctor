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

import base64
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, ToolAnnotations
from pydantic import Field

from model_doctor import config
from model_doctor.app import capabilities, mcp_guidance, mcp_payloads, storage
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

#: The shared text, so this server and Ramanujan's cannot tell a reader
#: two different things about the same numbers. Lives in
#: :mod:`model_doctor.app.mcp_guidance`, which both vendor.
INSTRUCTIONS = mcp_guidance.INSTRUCTIONS


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
        name="list_findings",
        title="List a run's findings",
        description=(
            "A page of one run's findings: every prediction and every "
            "annotation the matcher accounted for, with its outcome, class, "
            "confidence, IoU and boxes, named by image filename. Filter by "
            "outcome or by image. The run's full outcome counts come back with "
            "every page, so a page of 50 false positives is never mistaken for "
            "the whole run. Use get_analysis first for rates and factors; this "
            "is for looking at the individual cases behind them. Returns no "
            "filesystem paths and starts no work."
        ),
        annotations=_READ_ONLY,
    )
    def list_findings(
        run_id: Annotated[int, Field(description="Which run's findings to list.")],
        outcome: Annotated[
            str | None,
            Field(
                description=(
                    "Only findings with this outcome: correct, wrong_class, "
                    "poor_localization, false_positive or false_negative."
                )
            ),
        ] = None,
        image_id: Annotated[
            int | None,
            Field(
                description=(
                    "Only findings on this image — every object and every "
                    "prediction on one photograph."
                )
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(description="How many findings to return."),
        ] = mcp_payloads.DEFAULT_FINDING_LIMIT,
        offset: Annotated[
            int, Field(description="How many to skip, for paging.")
        ] = 0,
    ) -> dict[str, Any]:
        try:
            page, skip, kind = mcp_payloads.resolve_finding_page(
                limit, offset, outcome
            )
        except PayloadError as error:
            raise _as_tool_error(error) from error
        with _database() as connection:
            try:
                return mcp_payloads.list_findings_payload(
                    connection,
                    run_id,
                    outcome=kind,
                    image_id=image_id,
                    limit=page,
                    offset=skip,
                )
            except PayloadError as error:
                raise _as_tool_error(error) from error

    @server.tool(
        name="get_finding",
        title="Get one finding with its evidence",
        description=(
            "One finding and everything the stored passes attributed to it: "
            "the factors it carries, its outline result if masks were "
            "measured, its measured relationships to other findings, the "
            "failure groups it was placed in, and whether a heatmap exists. "
            "Both ids are required — finding ids are unique across the whole "
            "database, so a finding is always asked for within its run. Read "
            "the caveats: one finding is an anecdote, and a factor is an "
            "attributed condition rather than a cause."
        ),
        annotations=_READ_ONLY,
    )
    def get_finding(
        run_id: Annotated[int, Field(description="The run the finding belongs to.")],
        finding_id: Annotated[
            int, Field(description="Which finding, from list_findings.")
        ],
    ) -> dict[str, Any]:
        with _database() as connection:
            try:
                return mcp_payloads.get_finding_payload(
                    connection, run_id, finding_id
                )
            except PayloadError as error:
                raise _as_tool_error(error) from error

    @server.tool(
        name="get_image",
        title="Get the photograph a finding was measured on",
        description=(
            "The source photograph for one image of a run, as image content "
            "you can look at. Ask by run_id and image_id, both from "
            "list_findings; there is no way to name a file. Use this to see "
            "what the model actually saw — whether an object called a false "
            "negative is visible at all, whether a scene is dark or cluttered. "
            "Returns the picture and nothing about where it is stored."
        ),
        annotations=_READ_ONLY,
    )
    def get_image(
        run_id: Annotated[int, Field(description="The run the image belongs to.")],
        image_id: Annotated[
            int, Field(description="Which image, from a finding in list_findings.")
        ],
    ) -> ImageContent:
        with _database() as connection:
            try:
                found = mcp_payloads.image_for_run(connection, run_id, image_id)
            except PayloadError as error:
                raise _as_tool_error(error) from error
        return ImageContent(
            type="image",
            data=base64.b64encode(found["data"]).decode("ascii"),
            mimeType=found["media_type"],
        )

    @server.tool(
        name="get_heatmap_image",
        title="Get the overlay explaining one finding",
        description=(
            "The Grad-CAM overlay for one finding, as image content you can "
            "look at: where the model was attending when it produced this "
            "outcome. Ask by run_id and finding_id. Not every run has one — "
            "explanation is implemented for some detector families and not "
            "others — and a run without them says so rather than failing. "
            "Read it as attention, not as a cause."
        ),
        annotations=_READ_ONLY,
    )
    def get_heatmap_image(
        run_id: Annotated[int, Field(description="The run the finding belongs to.")],
        finding_id: Annotated[
            int, Field(description="Which finding, from list_findings.")
        ],
        full_resolution: Annotated[
            bool,
            Field(
                description=(
                    "Return the full-resolution overlay instead of the "
                    "downscaled companion. Off by default: the original can be "
                    "several megabytes."
                )
            ),
        ] = False,
    ) -> ImageContent:
        with _database() as connection:
            try:
                found = mcp_payloads.heatmap_for_finding(
                    connection,
                    run_id,
                    finding_id,
                    prefer_preview=not full_resolution,
                )
            except PayloadError as error:
                raise _as_tool_error(error) from error
        return ImageContent(
            type="image",
            data=base64.b64encode(found["data"]).decode("ascii"),
            mimeType=found["media_type"],
        )

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

    # The same three workflows Ramanujan's surface offers, from the same
    # definitions. Registered with a loop rather than three decorated
    # functions: the bodies would be identical but for which entry they read,
    # and a copy per prompt is how the two surfaces would drift.
    for _name, _entry in mcp_guidance.PROMPTS.items():
        _register_prompt(server, _name, _entry)

    return server


def _register_prompt(server: MCPServer, name: str, entry: dict[str, Any]) -> None:
    """Expose one shared prompt through this server's decorator."""
    definition = entry["definition"]

    @server.prompt(
        name=name, title=definition["title"], description=definition["description"]
    )
    def _prompt(**arguments: str) -> str:
        try:
            return mcp_guidance.build_prompt(name, arguments)["text"]
        except mcp_guidance.PromptError as error:
            raise ToolError(str(error)) from error


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
