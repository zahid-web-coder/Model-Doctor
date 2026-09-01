"""Runs the analysis pipeline for a self-service request, off the request path.

**Stages are the existing command-line entry points, run as subprocesses.**
Not a reimplementation of the pipeline, and not an import of it either. Three
reasons, in order of weight:

* The CLI is already the tested interface to each stage. Calling it means the
  browser workflow and a terminal produce identical runs, and a fix to either
  reaches both.
* Detector libraries hold global state — CUDA/MPS contexts, module-level locks,
  Ultralytics' own configuration. A subprocess ends when its stage ends, so
  none of it accumulates across jobs in the API process.
* A stage that segfaults inside a native library takes down a subprocess and
  is reported, rather than taking down the service.

The dataset and database reach each stage through **environment variables**
rather than flags, because `config` is already built that way: every path is
overridable so that "the same code runs on a laptop, a CI runner, or a
container without edits". The worker sets that environment per job.

**One job at a time.** The queue is serial, and that is a decision rather than
a simplification. Three constraints make concurrency actively wrong here:
SQLite is a single-writer store, RF-DETR is unstable under concurrent
Metal use, and a second job would contend for the same GPU memory rather than
finish sooner. A second request queues, and the UI says so.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import config
from app import storage
from utils.logging_utils import get_logger

logger = get_logger(__name__)

#: Explainability is Grad-CAM, which targets a convolutional detection head.
#: RF-DETR predicts through decoder queries and has no validated attribution
#: in this project, so the stage is skipped rather than run to produce a map
#: that would not correspond to the prediction. Skipping is reported.
EXPLAINABLE_FAMILIES: frozenset[str] = frozenset({"yolo"})


@dataclass(frozen=True)
class JobRequest:
    """Everything the worker needs to run one analysis."""

    token: str
    detector: str
    model_path: Path
    data_yaml: Path
    split: str
    image_size: int
    confidence: float
    database: Path
    workspace: Path


def _timestamp() -> str:
    """Current UTC time, matching how storage records its own timestamps."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def stage_commands(
    request: JobRequest, run_id: int | None
) -> list[tuple[str, list[str]]]:
    """Return the command for each stage, in execution order.

    ``run_id`` is None before the first stage has produced one, which is why
    the later stages are only assembled once it exists.
    """
    python = sys.executable
    database = str(request.database)

    if run_id is None:
        return [(
            "inference+diagnosis",
            [
                python, "-m", "app.diagnosis",
                "--split", request.split,
                "--detector", request.detector,
                "--model", str(request.model_path),
                "--conf", str(request.confidence),
                "--imgsz", str(request.image_size),
                "--save", "--db", database,
            ],
        )]

    run = str(run_id)
    stages: list[tuple[str, list[str]]] = [
        ("evaluation", [
            python, "scripts/evaluate_run.py", "--run-id", run, "--db", database,
        ]),
        ("mask-diagnosis", [
            python, "-m", "app.mask_diagnosis", "--run", run, "--db", database,
            "--detector", request.detector, "--model", str(request.model_path),
        ]),
        ("root-cause", [
            python, "-m", "app.root_cause", "--run", run, "--db", database,
        ]),
        ("clustering", [
            python, "-m", "app.clustering", "--run", run, "--db", database,
        ]),
        ("recommendations", [
            python, "-m", "app.recommendations", "--run", run, "--db", database,
        ]),
    ]
    if request.detector in EXPLAINABLE_FAMILIES:
        stages.append((
            "explainability",
            [
                python, "-m", "app.explainability", "--run", run, "--db", database,
                "--model", str(request.model_path),
            ],
        ))
    return stages


def stage_environment(request: JobRequest) -> dict[str, str]:
    """The environment one stage runs under.

    Inherited from the parent and then overridden, so a deployment's own
    settings survive while the job's dataset and database take precedence.
    """
    environment = dict(os.environ)
    environment.update({
        "MD_DATA_YAML": str(request.data_yaml),
        "MD_DB_PATH": str(request.database),
        "MD_DATASETS_DIR": str(request.data_yaml.parent),
        # Uploaded runs must not silently inherit a developer's detector choice.
        "MD_DETECTOR": request.detector,
    })
    return environment


class JobRunner:
    """A single background worker draining a queue of analyses.

    Started lazily on the first submission so that importing this module —
    which the read-only API does not do, but tests might — never spawns a
    thread as a side effect.
    """

    def __init__(self) -> None:
        """Start empty, with no thread until the first submission."""
        self._queue: queue.Queue[JobRequest] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, request: JobRequest) -> None:
        """Queue one analysis and make sure the worker is running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._drain, name="model-doctor-jobs", daemon=True
                )
                self._thread.start()
        self._queue.put(request)

    def pending(self) -> int:
        """How many analyses are waiting behind the one running."""
        return self._queue.qsize()

    def _drain(self) -> None:
        """Run queued analyses one at a time, forever."""
        while True:
            request = self._queue.get()
            try:
                self.run(request)
            except Exception:  # noqa: BLE001 - a worker must never die
                logger.exception("Job %s crashed the worker loop", request.token)
            finally:
                self._queue.task_done()

    # -----------------------------------------------------------------------
    def run(self, request: JobRequest) -> None:
        """Execute every stage for one job, recording progress as it goes."""
        self._record(request.token, status="running", started_at=_timestamp())

        run_id: int | None = None
        try:
            for name, command in stage_commands(request, None):
                before = self._max_run_id(request.database)
                self._run_stage(request, name, command)
                run_id = self._max_run_id(request.database)
                if run_id is None or run_id == before:
                    raise RuntimeError(
                        "The diagnosis stage finished without saving a run. "
                        "Its output is in the log below."
                    )
                self._verify_usable(request.database, run_id)
                self._record(request.token, run_id=run_id, stage=name)

            for name, command in stage_commands(request, run_id):
                self._record(request.token, stage=name)
                self._run_stage(request, name, command)

        except Exception as error:  # noqa: BLE001 - reported, not raised
            logger.warning("Job %s failed: %s", request.token, error)
            self._record(
                request.token,
                status="failed",
                error=str(error)[: config.LOG_TAIL_CHARS],
                finished_at=_timestamp(),
            )
            return

        self._record(
            request.token, status="succeeded", stage=None, finished_at=_timestamp()
        )

    def _run_stage(self, request: JobRequest, name: str, command: list[str]) -> None:
        """Run one stage, capturing its output and raising on failure."""
        logger.info("Job %s: %s", request.token, name)
        log_file = request.workspace / "logs" / f"{name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command,
                cwd=str(config.PROJECT_ROOT),
                env=stage_environment(request),
                capture_output=True,
                text=True,
                timeout=config.STAGE_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"{name} exceeded the {config.STAGE_TIMEOUT_S}s limit and was "
                "stopped. A larger dataset may need MD_STAGE_TIMEOUT raised."
            ) from error

        output = (completed.stdout or "") + (completed.stderr or "")
        log_file.write_text(output)
        tail = output[-config.LOG_TAIL_CHARS :]
        self._record(request.token, log_tail=tail)

        if completed.returncode != 0:
            raise RuntimeError(
                f"{name} exited with status {completed.returncode}. "
                "The end of its output is shown below."
            )

    # -----------------------------------------------------------------------
    @staticmethod
    def _record(token: str, **fields: object) -> None:
        """Persist a job update, never letting bookkeeping fail the job."""
        try:
            with storage.connect() as connection:
                storage.update_job(connection, token, **fields)
        except Exception:  # noqa: BLE001 - progress reporting is not the work
            logger.exception("Could not update job %s", token)

    @staticmethod
    def _verify_usable(database: Path, run_id: int) -> None:
        """Fail the job when the run that was saved is not worth analysing.

        The diagnosis pass records a per-image failure and keeps going, which
        is right when one image is unreadable and wrong when *none* of them
        could be processed. A checkpoint that will not load produces exactly
        that: every image errors identically, no findings are written, and the
        stage still exits zero. Without this check the job runs on, and the
        first thing to actually fail is a later stage — reporting the wrong
        culprit three steps from the real one — while the dashboard gains an
        empty run that looks like a real result.

        The underlying error is lifted out of the images table so the operator
        is told what went wrong ("could not load the checkpoint") rather than
        what went wrong downstream of it.

        Raises:
            RuntimeError: Every image failed, carrying the reason they gave.
        """
        with storage.connect(database) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS failed,
                       MAX(error) AS reason
                FROM images WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        total = int(row["total"] or 0)
        failed = int(row["failed"] or 0)
        if total == 0:
            raise RuntimeError(
                "The run recorded no images at all. Check that the split you "
                "chose contains readable image files."
            )
        if failed == total:
            reason = (row["reason"] or "no reason was recorded").strip()
            raise RuntimeError(
                f"Every one of the {total} image(s) failed to process, so the "
                f"run holds no usable results. The first reason given was: "
                f"{reason}"
            )

    @staticmethod
    def _max_run_id(database: Path) -> int | None:
        """Highest run id currently in the database, or None if there are none.

        Used to identify the run the diagnosis stage just created. Safe only
        because jobs are serialised — with a concurrent worker this would be a
        race, which is one of the reasons the queue is serial.
        """
        with storage.connect(database) as connection:
            row = connection.execute("SELECT MAX(id) AS newest FROM runs").fetchone()
        return int(row["newest"]) if row and row["newest"] is not None else None


#: The process-wide worker. One queue, one thread, by design.
RUNNER = JobRunner()
