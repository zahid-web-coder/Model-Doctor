"""The write half of Model Doctor: uploads, validation, and starting analyses.

**A separate application from :mod:`model_doctor.app.api`, deliberately.** That
module is
read-only in a way that is structural rather than promised: it opens SQLite
through the ``mode=ro`` URI, so a write fails at the driver, and it declares no
non-GET route. D-037 gives the reason — a reader that ever began writing would
silently mutate analysis history instead of failing loudly.

Adding upload endpoints there would have dissolved that guarantee, turning a
property of the code into a convention about which handlers are careful. So
writes live here, in their own app with its own connections, and the reader
keeps its guarantee unchanged. The two can be mounted on one process or run
separately; neither knows about the other.

**Nothing here runs a model.** Requests validate, store, and enqueue. The work
happens in :mod:`model_doctor.app.jobs`, on a worker thread, in subprocesses. A request
that
started inference inline would hold a connection open for the length of an
inference run and time out in every proxy between the browser and the server.

.. warning::

   **This service has no authentication and must not be exposed to a network.**

   Anyone who can reach it can upload a checkpoint and cause this process to
   execute it. That is not a flaw in the validation — identifying a file can be
   done safely, but *running* a model means constructing it, and for
   Ultralytics that is a full unpickle of attacker-supplied data. No amount of
   checking upstream changes that.

   Bind it to a loopback address. It is a local tool, and adding a token here
   would create the appearance of a security boundary without the substance of
   one: a bearer token in front of arbitrary code execution buys very little,
   and it would invite exactly the network exposure this warning exists to
   prevent. If it ever needs to serve more than one machine, the answer is a
   sandboxed execution environment and a real identity system — not a header
   check bolted onto this module.

.. warning::

   **Run exactly one worker process.** The job queue lives in this process, and
   start-up closes out any job still marked running as belonging to a dead
   process. A second worker would therefore mark the first worker's live job as
   failed the moment it booted, while that job carried on writing results. Use
   ``uvicorn model_doctor.app.control:app`` without ``--workers``; the queue is
   serial by
   design and a second worker would not make anything finish sooner.

**Benchmarking is deliberately not one of the stages.** Latency, throughput and
memory are properties of a model *on a device under controlled conditions*, and
a figure measured on a shared machine while other work is running is not a
measurement — it is a number that looks like one. The worker cannot promise
those conditions: it runs wherever the service runs, possibly beside a browser
and a dev server. So benchmarking stays in ``scripts/benchmark_run.py``, run
deliberately on a quiet machine, and a browser-started run has no
``run_benchmarks`` row rather than an unreliable one. The Compare screen
already renders that absence as "not measured".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_doctor import config
from model_doctor.app import jobs, service, storage, validation, workspace
from model_doctor.app.detectors import SUPPORTED_FAMILIES
from model_doctor.utils.exceptions import ResourceNotFoundError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# FastAPI declares uploads and form fields through call expressions in the
# signature. Hoisting them to module level is what satisfies B008 without
# departing from the framework's documented idiom.
_UPLOADED_FILE = File(...)
_OPTIONAL_FAMILY = Form(default=None)
_SPLIT_FIELD = Form(default="test")


class RunNameBody(BaseModel):
    """The body of a rename.

    ``None`` and a blank string both clear the name — the field is optional so
    that clearing is expressible, rather than requiring a separate route to
    undo what this one did. Storage does the trimming and the length check, so
    the rule lives with the column it protects; the bound here only stops an
    unreasonable payload before it reaches the database.
    """

    name: str | None = Field(default=None, max_length=1000)


def _checks(result: validation.ValidationResult) -> dict[str, Any]:
    """Render a validation result as the shape the UI consumes."""
    return {
        "ok": result.ok,
        "checks": [
            {"name": c.name, "ok": c.ok, "detail": c.detail} for c in result.checks
        ],
        "facts": result.facts,
    }


#: Rendering a job and the results-directory fence are shared with every
#: non-HTTP caller of this engine, so both live in `service`.
_job_payload = service.job_payload


def create_app() -> FastAPI:
    """Build the control application.

    Reconciles jobs left behind by a previous process before serving anything.
    Doing it here rather than on first request means a restarted service tells
    the truth immediately, instead of only once somebody happens to ask.
    """
    try:
        with storage.connect() as connection:
            storage.reconcile_stale_jobs(connection)
    except Exception:  # noqa: BLE001 - never block start-up on bookkeeping
        logger.exception("Could not reconcile jobs from a previous process")

    application = FastAPI(
        title="Model Doctor control API",
        description=(
            "Uploads, validation and analysis execution. The read-only "
            "projection of results lives in model_doctor.app.api and is "
            "unaffected by this."
        ),
        version=str(storage.SCHEMA_VERSION),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_credentials=False,
        # Enumerated rather than "*", so the browser is told exactly what this
        # service accepts and nothing is enabled by accident. DELETE and PATCH
        # are here because deleting and renaming a run live on this API; the
        # reader still allows GET alone, which is what makes its read-only
        # guarantee visible from outside the process.
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    @application.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        """What this installation can actually accept and do.

        The UI reads this rather than hardcoding a detector list, so adding an
        adapter makes it selectable without a frontend change — and, more
        importantly, so the UI can never offer a family this build does not
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
                            "No validated attribution method for this family, "
                            "so heatmaps are not generated. An attribution "
                            "method was tested and did not meet the "
                            "faithfulness criteria set for it."
                        )
                    ),
                }
                for family in SUPPORTED_FAMILIES
            ],
            "dataset_format": {
                "name": "YOLO",
                "detail": (
                    "A data.yaml naming the classes, beside split directories "
                    "each holding images/ and labels/. Annotations are "
                    "normalised YOLO text, box or polygon."
                ),
                "archives": sorted(workspace.ARCHIVE_SUFFIXES),
            },
            "limits": {
                "model_mb": config.MAX_MODEL_BYTES // 1_000_000,
                "dataset_mb": config.MAX_DATASET_BYTES // 1_000_000,
            },
            "queue_depth": jobs.RUNNER.pending(),
        }

    # -----------------------------------------------------------------------
    @application.post("/uploads")
    async def create_upload() -> dict[str, str]:
        """Open a workspace for one analysis and return its token."""
        space = workspace.create()
        return {"token": space.token, "workspace": str(space.root)}

    def _space(token: str) -> workspace.Workspace:
        """Resolve a token to a workspace, refusing anything invented."""
        safe = workspace.safe_name(token, fallback="")
        candidate = (config.WORKSPACE_DIR / safe).resolve()
        if safe != token or not candidate.is_dir():
            raise HTTPException(status_code=404, detail="No such upload session.")
        return workspace.Workspace(token=token, root=candidate)

    @application.post("/uploads/{token}/model")
    async def upload_model(
        token: str,
        file: UploadFile = _UPLOADED_FILE,
        family: str | None = _OPTIONAL_FAMILY,
    ) -> dict[str, Any]:
        """Store a checkpoint and report whether it can actually be run."""
        space = _space(token)
        try:
            stored = workspace.store_upload(
                space.models,
                file.filename or "model.pt",
                await file.read(),
                allowed_suffixes=workspace.MODEL_SUFFIXES,
                max_bytes=config.MAX_MODEL_BYTES,
            )
        except workspace.WorkspaceError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        result = validation.validate_model(stored, family)
        return {"path": str(stored), "name": stored.name, **_checks(result)}

    @application.post("/uploads/{token}/dataset")
    async def upload_dataset(
        token: str,
        file: UploadFile = _UPLOADED_FILE,
        split: str = _SPLIT_FIELD,
    ) -> dict[str, Any]:
        """Unpack a dataset archive and report whether it can be diagnosed."""
        space = _space(token)
        try:
            archive = workspace.store_upload(
                space.dataset,
                file.filename or "dataset.zip",
                await file.read(),
                allowed_suffixes=workspace.ARCHIVE_SUFFIXES,
                max_bytes=config.MAX_DATASET_BYTES,
            )
            root = workspace.extract_archive(
                archive, space.dataset / "unpacked", max_bytes=config.MAX_DATASET_BYTES
            )
            archive.unlink(missing_ok=True)  # the copy on disk is what matters now
        except workspace.WorkspaceError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        result = validation.validate_dataset(root, require_split=split)
        return {"root": str(root), "name": root.name, **_checks(result)}

    # -----------------------------------------------------------------------
    @application.post("/analyses")
    def start_analysis(payload: dict[str, Any]) -> dict[str, Any]:
        """Validate one more time, record a job, and queue it.

        Revalidating here is not redundant. The upload endpoints validated what
        was uploaded; this validates what is about to run, and between the two
        a caller could have changed the split, the family, or which files it
        points at. The check that matters is the one immediately before the
        work.
        """
        token = str(payload.get("token", ""))
        space = _space(token)

        detector = str(payload.get("detector", "")).strip().lower()
        if detector not in SUPPORTED_FAMILIES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{detector!r} is not a detector family this build "
                    f"implements. Supported: {', '.join(SUPPORTED_FAMILIES)}."
                ),
            )

        model_path = Path(str(payload.get("model_path", "")))
        data_yaml = Path(str(payload.get("data_yaml", "")))
        for label, path in (("model", model_path), ("dataset", data_yaml)):
            resolved = path.resolve()
            if space.root not in resolved.parents or not resolved.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"The {label} path is not part of this upload session.",
                )

        split = str(payload.get("split", "test"))
        model_check = validation.validate_model(model_path, detector)
        dataset_check = validation.validate_dataset(
            data_yaml.parent, require_split=split
        )
        if not (model_check.ok and dataset_check.ok):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "The inputs did not pass validation.",
                    "model": _checks(model_check),
                    "dataset": _checks(dataset_check),
                },
            )

        image_size = int(payload.get("image_size") or config.IMAGE_SIZE)
        confidence = float(payload.get("confidence") or config.CONFIDENCE_THRESHOLD)

        with storage.connect() as connection:
            storage.create_job(
                connection,
                token=token,
                detector=detector,
                split=split,
                model_name=model_path.name,
                dataset_name=data_yaml.parent.name,
                workspace=str(space.root),
                image_size=image_size,
                confidence=confidence,
            )

        # Bound the disk before adding to it. Done here rather than after a job
        # finishes so that a crashed or abandoned session is still eventually
        # collected, and never while an upload into an older workspace is in
        # flight — by this point every upload for this analysis is complete.
        try:
            workspace.prune(config.WORKSPACE_KEEP)
        except OSError:
            logger.warning("Could not prune old workspaces", exc_info=True)

        jobs.RUNNER.submit(
            jobs.JobRequest(
                token=token,
                detector=detector,
                model_path=model_path.resolve(),
                data_yaml=data_yaml.resolve(),
                split=split,
                image_size=image_size,
                confidence=confidence,
                database=config.DB_PATH,
                workspace=space.root,
            )
        )
        logger.info("Queued analysis %s (%s, %s)", token, detector, split)

        with storage.connect() as connection:
            record = storage.load_job(connection, token)
        assert record is not None  # noqa: S101 - just written in this request
        return _job_payload(record)

    @application.get("/analyses/{token}")
    def analysis_status(token: str) -> dict[str, Any]:
        """Where one analysis has got to, and why it stopped if it did."""
        with storage.connect() as connection:
            record = storage.load_job(connection, token)
        if record is None:
            raise HTTPException(status_code=404, detail="No such analysis.")
        return _job_payload(record)

    @application.get("/analyses")
    def list_analyses(limit: int = 25) -> list[dict[str, Any]]:
        """Recent analyses, newest first."""
        with storage.connect() as connection:
            return [_job_payload(r) for r in storage.load_jobs(connection, limit)]

    @application.get("/runs/{run_id}/footprint")
    def run_footprint(run_id: int) -> dict[str, Any]:
        """What deleting this run would destroy, counted per table.

        Served so the confirmation a user sees carries quantities rather than a
        bare question. It lives here rather than on the read API because it
        exists only to support a write, and the reader has no business
        describing destruction it cannot perform.
        """
        with storage.connect() as connection:
            row = connection.execute(
                "SELECT id FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"No run with id {run_id}.")
            counts = storage.run_footprint(connection, run_id)
            files = len(storage.heatmap_paths(connection, run_id))
        return {
            "run_id": run_id,
            "rows": {table: n for table, n in counts.items() if n},
            "total_rows": sum(counts.values()),
            "heatmap_files": files,
        }

    @application.patch("/runs/{run_id}")
    def rename_run(run_id: int, body: RunNameBody) -> dict[str, Any]:
        """Set or clear a run's name.

        **The only field on a run that can be written.** Everything else
        records what an analysis pass did, and rewriting any of it would make
        the stored parameters disagree with the findings derived from them.
        What a run is *called* belongs to the reader instead: `best.pt` is
        Ultralytics' default filename, so several runs are otherwise
        indistinguishable without reading their ids.

        On the control API for the same reason delete is: the read API opens
        SQLite read-only and declares no non-GET route (D-037).
        """
        with storage.connect() as connection:
            try:
                run = storage.rename_run(connection, run_id, body.name)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if run is None:
                raise HTTPException(status_code=404, detail=f"No run with id {run_id}.")
        return {"run_id": run.id, "name": run.name}

    @application.delete("/runs/{run_id}")
    def delete_run(run_id: int) -> dict[str, Any]:
        """Delete one run, everything it owns, and the images it wrote.

        **Irreversible, and the only destructive route in the project.** It is
        on the control API because the reader opens SQLite read-only and
        declares no non-GET route (D-037); adding a delete there would dissolve
        a guarantee that is structural rather than conventional.

        Files are removed only after the database delete succeeds, and only
        when they sit inside this project's own results directory. A stored
        path is a string from a table, not a warrant to unlink anything on the
        machine — a database copied from elsewhere can name paths this process
        should never touch.
        """
        with storage.connect() as connection:
            try:
                paths = storage.heatmap_paths(connection, run_id)
                removed = storage.delete_run(connection, run_id)
            except ResourceNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        deleted_files, refused = _remove_within_results(paths)
        if refused:
            logger.warning(
                "Left %d heatmap file(s) in place: outside %s",
                refused,
                config.RESULTS_DIR,
            )
        return {
            "run_id": run_id,
            "rows": {table: n for table, n in removed.items() if n},
            "total_rows": sum(removed.values()),
            "files_deleted": deleted_files,
            "files_refused": refused,
        }

    return application


_remove_within_results = service.remove_within_results
