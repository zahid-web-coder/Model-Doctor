"""What the two HTTP applications do, minus the HTTP.

:mod:`model_doctor.app.api` and :mod:`model_doctor.app.control` each pair a
transport with a body of work: open the database read-only, check a stored path
against the configured roots, render a job, delete a run and the files it owns.
That work is not web-specific, and a host that embeds this engine rather than
calling it over a socket needs exactly the same answers.

So it lives here, framework-free, and the two applications import it. There is
no second implementation to keep in step — which matters most for
:func:`verified_file`, the check that decides whether a stored path may be
opened at all. A duplicated security check is one that eventually drifts.

Failures are raised as :class:`ServiceError`, carrying the status and wording
the HTTP applications already produced; each translates it into its own
exception type at its boundary. Nothing here imports FastAPI, so this module is
usable from a Flask process, a worker thread, or a script.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from model_doctor import config
from model_doctor.app import comparison, jobs, storage, workspace
from model_doctor.app.clustering import DISCRIMINATING_METHOD
from model_doctor.app.detectors import SUPPORTED_FAMILIES
from model_doctor.app.explainability import preview_path
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Paging bounds for a page of findings, as the read API has always applied them.
DEFAULT_PAGE_SIZE: int = 200
MAX_PAGE_SIZE: int = 1000

#: Media types for the image extensions this project reads.
IMAGE_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


class ServiceError(Exception):
    """A refusal with the status and wording the HTTP surface would have used."""

    def __init__(self, status: int, detail: Any) -> None:
        """Record the status and the wording a caller should be shown."""
        super().__init__(detail if isinstance(detail, str) else str(detail))
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def database_path() -> Path:
    """The database this installation reads.

    A single configured location rather than a per-request parameter: letting a
    caller name the file would let it name any file.
    """
    return config.DB_PATH


@contextmanager
def read_only(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database read-only, or explain why it cannot be opened.

    Deliberately not :func:`model_doctor.app.storage.connect`, which applies the
    schema and can bump the recorded version. A read path must do neither. The
    ``mode=ro`` open itself is
    :func:`model_doctor.app.storage.connect_read_only`, shared with the MCP
    server so that "read-only" is defined once.

    Raises:
        ServiceError: 503 when the database is absent or unreadable.
    """
    target = path or database_path()
    try:
        with storage.connect_read_only(target) as connection:
            yield connection
    except FileNotFoundError as error:
        raise ServiceError(503, str(error)) from error
    except sqlite3.Error as error:
        raise ServiceError(503, f"Cannot open the database at {target}.") from error


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
    front end that does not exist yet, and every consumer would then have to
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
    """Return a run, or refuse rather than answer emptily for a missing id."""
    run = storage.load_run(connection, run_id)
    if run is None:
        raise ServiceError(404, f"Run {run_id} does not exist.")
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
    - **404** — the path cannot be resolved at all.
    """
    # Relocation happens before anything else, because every later step — the
    # root check, the existence check, the error message — should speak about
    # the file this machine will actually open, not the one the diagnosing
    # machine recorded.
    candidate = Path(config.remap_path(stored_path)).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as error:
        raise ServiceError(404, "File is not readable.") from error

    permitted = any(
        resolved == root or root in resolved.parents
        for root in config.API_FILE_ROOTS
    )
    if not permitted:
        logger.warning("Refused a file outside the configured roots: %s", resolved)
        raise ServiceError(
            403,
            "That file is outside the directories this service may read. "
            "Start the API with MD_DATASETS_DIR set to the location the run "
            "was diagnosed from, or add it to MD_API_FILE_ROOTS. If this "
            "database was copied from another machine, its stored paths "
            "belong to that machine — set MD_PATH_REMAP to point them at "
            "this one.",
        )
    if not resolved.is_file():
        raise ServiceError(404, f"File is no longer present at {resolved}.")
    return resolved


# ---------------------------------------------------------------------------
# Run resources
# ---------------------------------------------------------------------------
#
# One function per resource, each taking the connection and the query
# parameters the read API declares for it. The dispatch below is what an
# embedding host calls instead of issuing a GET; the API's own handlers keep
# their signatures and their documentation.

def _as_int(params: Mapping[str, Any], key: str, default: int | None) -> int | None:
    raw = params.get(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise ServiceError(422, f"{key} must be an integer.") from error


def _as_bool(params: Mapping[str, Any], key: str) -> bool:
    """Query strings carry booleans as text; match the HTTP surface's reading."""
    raw = params.get(key)
    if raw in (None, ""):
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def run_outcomes(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """Finding counts by outcome. Not COCO mAP — see SCHEMA.md §4."""
    resolve_run(connection, run_id)
    return storage.outcome_counts(connection, run_id)


def run_images(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """Every image the run attempted, including errored and empty ones."""
    resolve_run(connection, run_id)
    return serialise(storage.load_images(connection, run_id))


def run_findings(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """A page of findings, with the total so a caller can page properly."""
    resolve_run(connection, run_id)
    limit = _as_int(params, "limit", DEFAULT_PAGE_SIZE)
    offset = _as_int(params, "offset", 0)
    image_id = _as_int(params, "image_id", None)
    if not (1 <= limit <= MAX_PAGE_SIZE):
        raise ServiceError(422, f"limit must be between 1 and {MAX_PAGE_SIZE}.")
    if offset < 0:
        raise ServiceError(422, "offset must not be negative.")

    if image_id is None:
        total = connection.execute(
            "SELECT COUNT(*) AS n FROM findings WHERE run_id = ?", (run_id,)
        ).fetchone()["n"]
    else:
        total = connection.execute(
            "SELECT COUNT(*) AS n FROM findings WHERE run_id = ? AND image_id = ?",
            (run_id, image_id),
        ).fetchone()["n"]
    rows = storage.load_findings_page(
        connection, run_id, limit, offset, image_id=image_id
    )
    return {"items": serialise(rows), "total": total, "limit": limit, "offset": offset}


def run_groups(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """Failure groups, largest first, from the discriminating partition."""
    resolve_run(connection, run_id)
    if not has_table(connection, "clusters"):
        return []
    method = str(params.get("method") or DISCRIMINATING_METHOD)
    return serialise(storage.load_clusters(connection, run_id, method))


def run_root_causes(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """Attributed conditions, strongest evidence first."""
    resolve_run(connection, run_id)
    if not has_table(connection, "root_causes"):
        return []
    factor = params.get("factor") or None
    return serialise(storage.load_root_causes(connection, run_id, factor))


def run_factor_rates(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """Each factor's rate among failures against its rate among successes."""
    resolve_run(connection, run_id)
    if not has_table(connection, "factor_rates"):
        return []
    return serialise(storage.load_factor_rates(connection, run_id))


def run_recommendations(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """Suggested actions in the documented order."""
    resolve_run(connection, run_id)
    if not has_table(connection, "recommendations"):
        return []
    return serialise(storage.load_recommendations(connection, run_id))


def run_mask_findings(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """Outline-level results beside their box-level verdict."""
    resolve_run(connection, run_id)
    if not has_table(connection, "mask_findings"):
        return []
    return serialise(
        storage.load_mask_findings(
            connection, run_id, _as_bool(params, "disagreements")
        )
    )


def run_evaluation(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """A run's mAP and AR, one row per task, from the shared evaluator."""
    resolve_run(connection, run_id)
    if not has_table(connection, "run_evaluations"):
        return []
    return serialise(storage.load_evaluations(connection, run_id))


def run_image_diagnoses(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """What shape the model's mistake took on each image."""
    resolve_run(connection, run_id)
    if not has_table(connection, "image_diagnoses"):
        return []
    return serialise(
        storage.load_image_diagnoses(
            connection, run_id, params.get("verdict") or None,
            with_coverage=_as_bool(params, "coverage"),
        )
    )


def run_relation_summary(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """Aggregate counts per relation; None when the pass never ran."""
    resolve_run(connection, run_id)
    if not has_table(connection, "finding_relations"):
        return None
    return comparison.relation_summary(
        storage.load_finding_relations(connection, run_id)
    )


def run_heatmaps(connection, run_id: int, params: Mapping[str, Any]) -> Any:
    """Every heatmap recorded for a run, joined to the finding it explains."""
    resolve_run(connection, run_id)
    if not has_table(connection, "heatmaps"):
        return []
    method = str(params.get("method") or "grad-cam")
    return serialise(storage.load_heatmaps(connection, run_id, method))


def run_summary(connection, run_id: int, _params: Mapping[str, Any]) -> Any:
    """One run's provenance — model hash, split, thresholds, image size."""
    return serialise(resolve_run(connection, run_id))


#: Resource name to reader, matching the read API's own paths under a run.
RUN_RESOURCES: Mapping[str, Any] = {
    "": run_summary,
    "outcomes": run_outcomes,
    "images": run_images,
    "findings": run_findings,
    "groups": run_groups,
    "root-causes": run_root_causes,
    "factor-rates": run_factor_rates,
    "recommendations": run_recommendations,
    "mask-findings": run_mask_findings,
    "evaluation": run_evaluation,
    "image-diagnoses": run_image_diagnoses,
    "relations/summary": run_relation_summary,
    "heatmaps": run_heatmaps,
}


def read_run_resource(
    run_id: int, resource: str, params: Mapping[str, Any] | None = None
):
    """Read one resource of one run, as the read API would have returned it."""
    reader = RUN_RESOURCES.get(resource)
    if reader is None:
        raise ServiceError(404, f"No such resource: {resource}.")
    with read_only() as connection:
        return reader(connection, run_id, params or {})


def read_group_members(cluster_id: int) -> Any:
    """The findings in one group, with their image filename."""
    with read_only() as connection:
        if not has_table(connection, "cluster_members"):
            return []
        return serialise(storage.load_cluster_members(connection, cluster_id))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def image_file(image_id: int) -> tuple[Path, str, str]:
    """The source image for one row of ``images``: path, media type, filename."""
    with read_only() as connection:
        row = connection.execute(
            "SELECT path, filename FROM images WHERE id = ?", (image_id,)
        ).fetchone()
    if row is None:
        raise ServiceError(404, f"Image {image_id} does not exist.")
    resolved = verified_file(str(row["path"]))
    media = IMAGE_TYPES.get(resolved.suffix.lower(), "image/jpeg")
    return resolved, media, str(row["filename"])


def heatmap_file(
    finding_id: int, method: str = "grad-cam", preview: bool = False
) -> tuple[Path, str]:
    """The overlay explaining one finding: path and media type.

    ``preview`` serves the downscaled companion when one exists, and otherwise
    falls back to the original — a run explained before previews existed has
    none on disk, and that is not a failure.
    """
    with read_only() as connection:
        if not has_table(connection, "heatmaps"):
            raise ServiceError(404, "This database has no heatmaps.")
        row = connection.execute(
            "SELECT path FROM heatmaps WHERE finding_id = ? AND method = ?",
            (finding_id, method),
        ).fetchone()
    if row is None:
        raise ServiceError(404, f"No {method} heatmap for finding {finding_id}.")

    stored = Path(config.remap_path(str(row["path"])))
    if preview:
        companion = preview_path(stored)
        if companion.is_file():
            return verified_file(str(companion)), "image/jpeg"
    return verified_file(str(stored)), "image/png"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def job_payload(record: storage.JobRecord) -> dict[str, Any]:
    """Render a job for the browser, including where it is in the pipeline."""
    stages = list(storage.JOB_STAGES)
    if record.detector not in jobs.EXPLAINABLE_FAMILIES:
        # Never advertise a stage this run will not perform. RF-DETR has no
        # validated attribution here, and showing a pending "explainability"
        # step that silently never happens would be a lie told by a progress
        # bar.
        stages = [s for s in stages if s != "explainability"]

    return {
        "token": record.token,
        "status": record.status,
        "stage": record.stage,
        "stages": stages,
        "stage_index": stages.index(record.stage) if record.stage in stages else None,
        "run_id": record.run_id,
        "detector": record.detector,
        "split": record.split,
        "model_name": record.model_name,
        "dataset_name": record.dataset_name,
        "image_size": record.image_size,
        "confidence": record.confidence,
        "error": record.error,
        "log_tail": record.log_tail,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "explainability_supported": record.detector in jobs.EXPLAINABLE_FAMILIES,
    }


def read_job(token: str) -> dict[str, Any] | None:
    """One job's progress, or None when this installation has no such token."""
    with read_only() as connection:
        record = storage.load_job(connection, token)
    return job_payload(record) if record is not None else None


def capabilities_payload() -> dict[str, Any]:
    """What this installation can actually accept and do.

    A caller reads this rather than hardcoding a detector list, so adding an
    adapter makes it selectable without a client change — and, more
    importantly, so a client can never offer a family this build does not
    implement.
    """
    return {
        "detectors": [
            {
                "family": family,
                "explainability": family in jobs.EXPLAINABLE_FAMILIES,
                "explainability_note": (
                    "Grad-CAM is validated for this family."
                    if family in jobs.EXPLAINABLE_FAMILIES
                    else (
                        "No validated attribution method for this family, so "
                        "heatmaps are not generated. An attribution method was "
                        "tested and did not meet the faithfulness criteria set "
                        "for it."
                    )
                ),
            }
            for family in SUPPORTED_FAMILIES
        ],
        "dataset_format": {
            "name": "YOLO",
            "detail": (
                "A data.yaml naming the classes, beside split directories each "
                "holding images/ and labels/. Annotations are normalised YOLO "
                "text, box or polygon."
            ),
            "archives": sorted(workspace.ARCHIVE_SUFFIXES),
        },
        "limits": {
            "model_mb": config.MAX_MODEL_BYTES // 1_000_000,
            "dataset_mb": config.MAX_DATASET_BYTES // 1_000_000,
        },
        "queue_depth": jobs.RUNNER.pending(),
    }


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def remove_within_results(paths: Sequence[str]) -> tuple[int, int]:
    """Delete each path that lies inside ``config.RESULTS_DIR``.

    Returns the number removed and the number refused for sitting outside it.
    Containment is checked after resolving symlinks, so a link pointing out of
    the results directory is refused rather than followed.

    A file already gone counts as removed: the goal is its absence, and a run
    whose images were cleaned up by hand should not fail to delete.

    **Every directory emptied is pruned, not just the first one.** Artifacts for
    a run are conventionally written under a single ``run_<id>`` directory, but
    the ``heatmaps`` table records absolute paths and nothing constrains them to
    agree — a database written by an older layout, or one relocated through
    ``MD_PATH_REMAP``, can spread a run's files across several. Pruning only the
    first path's parent left the rest as empty directories that accumulate
    silently.

    Only directories that actually held something removed are considered, so a
    directory emptied by somebody else is not swept up as a side effect.
    Deepest first, so a nested directory and the parent it empties both go in
    one pass. ``RESULTS_DIR`` itself is never removed, however empty it gets.
    """
    root = config.RESULTS_DIR.resolve()
    deleted = 0
    refused = 0
    emptied: set[Path] = set()

    for raw in paths:
        candidate = Path(config.remap_path(raw))
        try:
            resolved = candidate.resolve()
        except OSError:
            refused += 1
            continue
        if not resolved.is_relative_to(root):
            refused += 1
            continue
        try:
            resolved.unlink(missing_ok=True)
            deleted += 1
            emptied.add(resolved.parent)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", resolved, exc)
            refused += 1

    prune_empty_directories(emptied, root)
    return deleted, refused


def prune_empty_directories(directories: Iterable[Path], root: Path) -> None:
    """Remove each directory that is now empty, deepest first.

    Walks upward from each one so a directory emptied only by the removal of
    its last subdirectory is collected too. Stops at ``root``, which is the
    project's own results directory and belongs to the installation rather than
    to any run.

    A directory that is not empty, not inside ``root``, or cannot be removed is
    left alone. Failing to prune is untidy; removing the wrong directory is not
    recoverable, so every check here refuses rather than assumes.
    """
    candidates: set[Path] = set()
    for directory in directories:
        current = directory
        while current != root and root in current.parents:
            candidates.add(current)
            current = current.parent

    for directory in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError as exc:  # noqa: PERF203 - each directory decided on its own
            logger.warning("Could not prune %s: %s", directory, exc)


def remove_run(run_id: int) -> dict[str, Any]:
    """Delete one run, everything it owns, and the images it wrote.

    **Irreversible, and the only destructive operation in the project.** Files
    are removed only after the database delete succeeds, and only when they sit
    inside this project's own results directory. A stored path is a string from
    a table, not a warrant to unlink anything on the machine — a database
    copied from elsewhere can name paths this process should never touch.

    Raises:
        ServiceError: 404 when there is no such run.
    """
    from model_doctor.utils.exceptions import ResourceNotFoundError

    with storage.connect() as connection:
        try:
            paths = storage.heatmap_paths(connection, run_id)
            removed = storage.delete_run(connection, run_id)
        except ResourceNotFoundError as exc:
            raise ServiceError(404, str(exc)) from exc

    deleted_files, refused = remove_within_results(paths)
    if refused:
        logger.warning(
            "Left %d heatmap file(s) in place: outside %s", refused, config.RESULTS_DIR
        )
    return {
        "run_id": run_id,
        "rows": {table: n for table, n in removed.items() if n},
        "total_rows": sum(removed.values()),
        "files_deleted": deleted_files,
        "files_refused": refused,
    }
