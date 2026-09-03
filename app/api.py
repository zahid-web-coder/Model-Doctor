"""A read-only HTTP projection of the published schema.

**This adds no analysis.** Every endpoint is a documented query from
``docs/SCHEMA.md``, served through the readers already in
:mod:`app.storage`. No SQL is duplicated here, and no business logic lives
here. When the two consumers of this backend — the Streamlit dashboard and a
browser front end — disagree about a number, it will not be because one of them
reimplemented a query.

**Why a service at all.** The dashboard opens the SQLite file directly; a
browser cannot. It also cannot reach :mod:`app.similarity`, whose
nearest-neighbour search is Python rather than SQL and has been unreachable
since it was built. That endpoint is the first thing a new front end can do
that the old one could not.

**Read-only, structurally.** Connections are opened with SQLite's ``mode=ro``
URI rather than through :func:`app.storage.connect`, which creates tables and
can upgrade ``schema_info`` — behaviour that is right for a CLI pass and wrong
for a request handler, where two workers could race on it. A reader that ever
began writing would fail loudly at this boundary instead of quietly mutating
analysis history (D-037).

**Files are served by id, never by path.** A path parameter would be a
traversal hole. Even the *stored* path is verified against the configured roots
before it is opened, because it came from a database file the operator chose
rather than from this process.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import config
from app import storage
from app.clustering import DISCRIMINATING_METHOD
from app.explainability import preview_path
from app.similarity import SimilarityError, nearest_neighbours
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Findings run to hundreds per run and would run to millions on a large
# dataset. A default cap keeps an unparameterised request from materialising
# the whole table; the ceiling keeps a hostile one from doing it deliberately.
DEFAULT_PAGE_SIZE: int = 200
MAX_PAGE_SIZE: int = 1000

# Media types for the two kinds of file this serves. Anything else is a bug in
# the pipeline rather than something to guess at.
_IMAGE_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def database_path() -> Path:
    """Return the database this service reads.

    A single configured location rather than a per-request parameter: letting a
    caller name the file would let it name any file.
    """
    return config.DB_PATH


@contextmanager
def read_only(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database read-only, or explain why it cannot be opened.

    Deliberately not :func:`app.storage.connect`, which applies the schema and
    can bump the recorded version. A request handler must not do either. The
    ``mode=ro`` open itself is :func:`app.storage.connect_read_only`, shared
    with the MCP server so that "read-only" is defined once.

    Yields:
        A connection that cannot create or modify anything.

    Raises:
        HTTPException: 503 when the database is absent or unreadable, carrying
            the command that produces one.
    """
    target = path or database_path()
    # The opener lives in storage and is shared with the MCP server, so the
    # read-only guarantee has one implementation. This wrapper only translates
    # its failures into HTTP.
    try:
        with storage.connect_read_only(target) as connection:
            yield connection
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503, detail=f"Cannot open the database at {target}."
        ) from error


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    """Report whether an optional table exists in this database.

    Only ``runs``, ``images`` and ``findings`` are guaranteed. A database saved
    before a later schema version simply lacks the rest, and the surface that
    needs one should return empty rather than fail (SCHEMA.md §6).
    """
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def serialise(value: Any) -> Any:
    """Convert storage's return types into JSON-ready structures.

    Rows are returned as the schema defines them. Nothing here reshapes data
    for a particular screen: a view-model would encode assumptions about a
    front end that does not exist yet, and both consumers would then have to
    agree with those assumptions.
    """
    if isinstance(value, sqlite3.Row):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list | tuple):
        return [serialise(item) for item in value]
    return value


def resolve_run(connection: sqlite3.Connection, run_id: int) -> Any:
    """Return a run, or 404 rather than an empty body for a nonexistent id."""
    run = storage.load_run(connection, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} does not exist.")
    return run


def verified_file(stored_path: str) -> Path:
    """Return a stored path only if it is safe and present.

    Three failure modes, deliberately distinguished:

    - **403** — the path resolves outside every configured root. The stored
      path came from a database the operator selected, not from this process,
      so it is input and is treated as such. Symlinks are followed before the
      check, so a link out of an allowed root does not escape it.
    - **404** — the path is allowed but the file is gone. Images move; a run
      diagnosed from a temporary directory outlives it. That is an ordinary
      state, not a server fault.
    - **200** — allowed and present.

    Args:
        stored_path: A path from ``images.path`` or ``heatmaps.path``.

    Returns:
        The resolved path, safe to open.

    Raises:
        HTTPException: 403 or 404 as described.
    """
    # Relocation happens before anything else, because every later step —
    # the root check, the existence check, the error message — should speak
    # about the file this machine will actually open, not the one the
    # diagnosing machine recorded.
    candidate = Path(config.remap_path(stored_path)).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as error:
        raise HTTPException(status_code=404, detail="File is not readable.") from error

    permitted = any(
        resolved == root or root in resolved.parents
        for root in config.API_FILE_ROOTS
    )
    if not permitted:
        logger.warning("Refused a file outside the configured roots: %s", resolved)
        raise HTTPException(
            status_code=403,
            detail=(
                "That file is outside the directories this service may read. "
                "Start the API with MD_DATASETS_DIR set to the location the "
                "run was diagnosed from, or add it to MD_API_FILE_ROOTS. If "
                "this database was copied from another machine, its stored "
                "paths belong to that machine — set MD_PATH_REMAP to point "
                "them at this one."
            ),
        )
    if not resolved.is_file():
        raise HTTPException(
            status_code=404, detail=f"File is no longer present at {resolved}."
        )
    return resolved


def create_app() -> FastAPI:
    """Build the application.

    A factory rather than a module-level instance so tests can construct one
    with different configuration without reimporting the module.
    """
    application = FastAPI(
        title="Model Doctor API",
        description=(
            "Read-only projection of the diagnosis schema documented in "
            "docs/SCHEMA.md. Adds no analysis."
        ),
        version=str(storage.SCHEMA_VERSION),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Runs
    # -----------------------------------------------------------------------
    @application.get("/health")
    def health() -> dict[str, Any]:
        """Report whether the database is present and which schema it holds."""
        target = database_path()
        if not target.is_file():
            return {"status": "no-database", "database": str(target)}
        with read_only() as connection:
            version = connection.execute(
                "SELECT version FROM schema_info"
            ).fetchone()
        return {
            "status": "ok",
            "database": str(target),
            "schema_version": version["version"] if version else None,
            "expected_schema_version": storage.SCHEMA_VERSION,
        }

    @application.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        """Every saved run, newest first."""
        with read_only() as connection:
            return serialise(storage.list_runs(connection))

    @application.get("/runs/{run_id}")
    def get_run(run_id: int) -> dict[str, Any]:
        """One run's provenance — model hash, split, thresholds, image size."""
        with read_only() as connection:
            return serialise(resolve_run(connection, run_id))

    @application.get("/runs/{run_id}/outcomes")
    def get_outcomes(run_id: int) -> dict[str, int]:
        """Finding counts by outcome. Not COCO mAP — see SCHEMA.md §4."""
        with read_only() as connection:
            resolve_run(connection, run_id)
            return storage.outcome_counts(connection, run_id)

    @application.get("/runs/{run_id}/images")
    def get_images(run_id: int) -> list[dict[str, Any]]:
        """Every image the run attempted, including errored and empty ones.

        The only honest source for image totals. A count derived from findings
        misses images that were processed and genuinely contained nothing, and
        cannot report images that failed to process at all — ``error`` exists
        to keep those apart.

        ``path`` is the location as diagnosed, for reference. Fetch the bytes
        from ``/images/{id}``; the path is not addressable by this service.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            return serialise(storage.load_images(connection, run_id))

    # -----------------------------------------------------------------------
    # Findings
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/findings")
    def get_findings(
        run_id: int,
        limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        """A page of findings, with the total so a caller can page properly."""
        with read_only() as connection:
            resolve_run(connection, run_id)
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM findings WHERE run_id = ?", (run_id,)
            ).fetchone()["n"]
            rows = storage.load_findings_page(connection, run_id, limit, offset)
        return {
            "items": serialise(rows),
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # -----------------------------------------------------------------------
    # Failure groups
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/groups")
    def get_groups(
        run_id: int, method: str = DISCRIMINATING_METHOD
    ) -> list[dict[str, Any]]:
        """Failure groups, largest first.

        Defaults to the discriminating partition, which SCHEMA.md instructs
        consumers to show: the full signature produces many small groups built
        partly from factors that do not distinguish failures (D-033).
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "clusters"):
                return []
            return serialise(storage.load_clusters(connection, run_id, method))

    @application.get("/groups/{cluster_id}/members")
    def get_group_members(cluster_id: int) -> list[dict[str, Any]]:
        """The findings in one group, with their image filename."""
        with read_only() as connection:
            if not has_table(connection, "cluster_members"):
                return []
            return serialise(storage.load_cluster_members(connection, cluster_id))

    # -----------------------------------------------------------------------
    # Root causes and their base rates
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/root-causes")
    def get_root_causes(
        run_id: int, factor: str | None = None
    ) -> list[dict[str, Any]]:
        """Attributed conditions, strongest evidence first."""
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "root_causes"):
                return []
            return serialise(storage.load_root_causes(connection, run_id, factor))

    @application.get("/runs/{run_id}/factor-rates")
    def get_factor_rates(run_id: int) -> list[dict[str, Any]]:
        """Each factor's rate among failures against its rate among successes.

        A count without this is not evidence: ``edge_truncation`` describes 71%
        of failures and 76% of correct detections (D-031). ``lift`` is null when
        undefined and must not be rendered as a number.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "factor_rates"):
                return []
            return serialise(storage.load_factor_rates(connection, run_id))

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/recommendations")
    def get_recommendations(run_id: int) -> list[dict[str, Any]]:
        """Suggested actions in the documented order.

        Actionable first, then by failures addressed. Rows with status
        ``conflicting`` or ``insufficient_evidence`` are refusals to advise and
        are returned deliberately — a consumer must not filter them (D-035).
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "recommendations"):
                return []
            return serialise(storage.load_recommendations(connection, run_id))

    # -----------------------------------------------------------------------
    # Mask-level diagnosis
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/mask-findings")
    def get_mask_findings(
        run_id: int, disagreements: bool = False
    ) -> list[dict[str, Any]]:
        """Outline-level results beside their box-level verdict.

        ``disagreements=true`` returns only findings the box called correct and
        the outline did not — the set mask diagnosis exists to surface.
        ``mask_iou`` is null when never measured, which is not zero overlap.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "mask_findings"):
                return []
            return serialise(
                storage.load_mask_findings(connection, run_id, disagreements)
            )

    # -----------------------------------------------------------------------
    # Measured performance
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/evaluation")
    def get_evaluation(run_id: int) -> list[dict[str, Any]]:
        """A run's mAP and AR, one row per task, from the shared evaluator.

        Empty when the run has never been evaluated — which is a different
        statement from a score of zero, and the caller must render it as one.
        Each row carries the confidence sweep, IoU range, detection cap and
        ground truth that produced it, because two mAPs are comparable only if
        those agree.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "run_evaluations"):
                return []
            return serialise(storage.load_evaluations(connection, run_id))

    @application.get("/runs/{run_id}/benchmarks")
    def get_benchmarks(run_id: int) -> list[dict[str, Any]]:
        """A run's measured latency, memory and checkpoint size, per device.

        One row per device the run was benchmarked on, because latency and
        memory are properties of a model *on a device*: a figure from one is
        not evidence about another. Empty when never benchmarked.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "run_benchmarks"):
                return []
            return serialise(storage.load_benchmarks(connection, run_id))

    # -----------------------------------------------------------------------
    # Similar failures — unreachable from a SQL-only consumer
    # -----------------------------------------------------------------------
    @application.get("/runs/{run_id}/findings/{finding_id}/neighbours")
    def get_neighbours(
        run_id: int, finding_id: int, limit: int = Query(5, ge=1, le=50)
    ) -> list[dict[str, Any]]:
        """Failures that look like this one, by CLIP embedding similarity.

        Computed rather than stored: the answer depends on which finding is
        asked about, so a table of them would be answers to questions nobody
        has asked.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "embeddings"):
                return []
            try:
                neighbours = nearest_neighbours(
                    connection, run_id, finding_id, limit=limit
                )
            except SimilarityError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
        return serialise(neighbours)

    # -----------------------------------------------------------------------
    # Files — by id, never by path
    # -----------------------------------------------------------------------
    @application.get("/images/{image_id}")
    def get_image(image_id: int) -> FileResponse:
        """The source image for one row of ``images``."""
        with read_only() as connection:
            row = connection.execute(
                "SELECT path, filename FROM images WHERE id = ?", (image_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404, detail=f"Image {image_id} does not exist."
            )

        resolved = verified_file(str(row["path"]))
        return FileResponse(
            resolved,
            media_type=_IMAGE_TYPES.get(resolved.suffix.lower(), "image/jpeg"),
            filename=str(row["filename"]),
        )

    @application.get("/runs/{run_id}/image-diagnoses")
    def get_image_diagnoses(
        run_id: int, verdict: str | None = None, coverage: bool = False
    ) -> list[dict[str, Any]]:
        """What shape the model's mistake took on each image (schema v11).

        A second lens over the same findings, never a replacement: the
        outcome counts on each row are the finding-level ones, unchanged.
        `coverage` attaches the object/prediction pairs behind each verdict,
        which is how a merge is distinguished from an independent miss.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "image_diagnoses"):
                return []
            return serialise(
                storage.load_image_diagnoses(
                    connection, run_id, verdict, with_coverage=coverage
                )
            )

    @application.get("/runs/{run_id}/images/{image_id}/coverage")
    def get_image_coverage(run_id: int, image_id: int) -> list[dict[str, Any]]:
        """Every measured object/prediction overlap on one image, strongest first."""
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "image_coverage"):
                return []
            return serialise(storage.load_image_coverage(connection, run_id, image_id))

    @application.get("/runs/{run_id}/heatmaps")
    def get_heatmaps(run_id: int, method: str = "grad-cam") -> list[dict[str, Any]]:
        """Every heatmap recorded for a run, joined to the finding it explains.

        Empty is a meaningful answer, not a fault: explanation is implemented
        for some detector families and not others, so a run can be complete and
        still have none. A caller showing heatmaps needs to tell that apart
        from "this run was never explained", and both from a broken image.
        """
        with read_only() as connection:
            resolve_run(connection, run_id)
            if not has_table(connection, "heatmaps"):
                return []
            return serialise(storage.load_heatmaps(connection, run_id, method))

    @application.get("/findings/{finding_id}/heatmap")
    def get_heatmap(
        finding_id: int, method: str = "grad-cam", preview: bool = False
    ) -> FileResponse:
        """The Grad-CAM overlay explaining one finding, if one was generated.

        ``preview=true`` serves a downscaled companion image instead, for
        callers rendering many at once — a grid of tiles a few hundred pixels
        wide otherwise downloads full-resolution overlays it cannot display.
        The full-resolution image remains the default and is what any close
        reading gets. A run explained before previews existed has none on disk,
        so the request falls back to the original rather than failing.
        """
        with read_only() as connection:
            if not has_table(connection, "heatmaps"):
                raise HTTPException(
                    status_code=404, detail="This database has no heatmaps."
                )
            row = connection.execute(
                "SELECT path FROM heatmaps WHERE finding_id = ? AND method = ?",
                (finding_id, method),
            ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No {method} heatmap for finding {finding_id}.",
            )

        stored = Path(config.remap_path(str(row["path"])))
        if preview:
            companion = preview_path(stored)
            if companion.is_file():
                return FileResponse(
                    verified_file(str(companion)), media_type="image/jpeg"
                )
        return FileResponse(verified_file(str(stored)), media_type="image/png")

    return application


app = create_app()
