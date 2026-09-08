"""SQLite persistence for diagnosis results.

A diagnosis that exists only in memory dies with the process. This module makes
it durable, queryable, and — most importantly — **available to code that is not
this codebase**. The dashboard is being built by a second developer against the
database, not against these Python objects, so the schema is a published
interface. See ``docs/SCHEMA.md``, which is the contract; this module is one
implementation of it.

**Why this lives in ``app/`` and not ``utils/``.** Persistence needs the
diagnosis domain types — ``Finding``, ``Outcome``, ``ImageDiagnosis`` — which
live in ``model_doctor.app.diagnosis``. A ``utils`` module importing from ``app`` would
be
the first upward dependency in the project and would break the layering every
other module has held to. The direction is one-way and must stay that way::

    app/storage.py → app/diagnosis.py → app/inference.py → utils/* → config.py

``model_doctor.app.diagnosis`` must never import this module. Callers orchestrate:
diagnose
first, then persist.

**Why the schema is normalised.** Later milestones add clustering, root-cause
analysis, and recommendations. Each will attach its own data to a finding. If
those were nullable columns on ``findings``, every one of them would alter a
table another developer is already querying. Instead they arrive as new tables
keyed on ``finding_id``, and ``findings`` never changes shape.

Nothing speculative is stored. Prediction outlines, cluster ids, root causes,
and recommendations have no columns here because nothing produces them yet.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_doctor import config
from model_doctor.app.diagnosis import (
    DatasetDiagnosis,
    Finding,
    ImageDiagnosis,
    Outcome,
)
from model_doctor.utils.exceptions import ModelDoctorError, ResourceNotFoundError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Bumped when the schema changes in a way that existing readers must know
# about. Recorded in the database so a consumer can detect a mismatch instead
# of failing on a missing column.
SCHEMA_VERSION: int = 15

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_info (
        version     INTEGER NOT NULL,
        applied_at  TEXT    NOT NULL
    )
    """,
    # One row per diagnosis execution. Everything needed to reproduce the run
    # lives here: without it a finding is an assertion with no provenance, and
    # comparing two runs is impossible.
    """
    CREATE TABLE IF NOT EXISTS runs (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at              TEXT    NOT NULL,
        model_path              TEXT    NOT NULL,
        model_sha256            TEXT    NOT NULL,
        dataset_yaml            TEXT    NOT NULL,
        split                   TEXT    NOT NULL,
        confidence_threshold    REAL    NOT NULL,
        match_iou_threshold     REAL    NOT NULL,
        localization_iou_floor  REAL    NOT NULL,
        image_size              INTEGER NOT NULL,
        name                    TEXT
    )
    """,
    # One row per image attempted. `error` is populated when an image could not
    # be processed, so a failed image is recorded rather than silently absent —
    # "no findings" and "never ran" must be distinguishable.
    """
    CREATE TABLE IF NOT EXISTS images (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        path              TEXT    NOT NULL,
        filename          TEXT    NOT NULL,
        width             INTEGER,
        height            INTEGER,
        prediction_count  INTEGER NOT NULL,
        truth_count       INTEGER NOT NULL,
        error             TEXT
    )
    """,
    # One row per prediction *or* per ground-truth annotation. Both boxes are
    # nullable because a false positive has no ground truth and a false
    # negative has no prediction; which columns are populated is determined by
    # `outcome`, and that rule is documented in SCHEMA.md.
    """
    CREATE TABLE IF NOT EXISTS findings (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        image_id       INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        outcome        TEXT    NOT NULL,
        class_id       INTEGER,
        class_name     TEXT    NOT NULL,
        confidence     REAL,
        iou            REAL,
        pred_x1        REAL,
        pred_y1        REAL,
        pred_x2        REAL,
        pred_y2        REAL,
        truth_x1       REAL,
        truth_y1       REAL,
        truth_x2       REAL,
        truth_y2       REAL,
        truth_polygon  TEXT
    )
    """,
    # Added at schema version 2. A separate table rather than columns on
    # `findings`, so the published contract other developers query stays exactly
    # as it was (D-020). A finding has at most one embedding per model.
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        model_name  TEXT    NOT NULL,
        dimensions  INTEGER NOT NULL,
        vector      BLOB    NOT NULL,
        UNIQUE (finding_id, model_name)
    )
    """,
    # Added at schema version 3. Heatmaps are image files; this records which
    # finding each explains and how it was produced, so a dashboard finds them
    # with one left join instead of probing the filesystem per row. A separate
    # table for the same reason embeddings is one (D-020).
    """
    CREATE TABLE IF NOT EXISTS heatmaps (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id    INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        path          TEXT    NOT NULL,
        method        TEXT    NOT NULL,
        target_layers TEXT    NOT NULL,
        UNIQUE (finding_id, method)
    )
    """,
    # Added at schema version 4. Attached to findings rather than to clusters,
    # so that once failures are grouped a per-cluster summary is a GROUP BY
    # over these rows and needs no schema change (D-026).
    """
    CREATE TABLE IF NOT EXISTS root_causes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        factor      TEXT    NOT NULL,
        score       REAL    NOT NULL,
        evidence    TEXT    NOT NULL,
        UNIQUE (finding_id, factor)
    )
    """,
    # Added at schema version 5. Failure groups: the label *is* the
    # explanation, so a consumer needs no second lookup to name a group.
    # `method` records how the grouping was produced and is what allows a
    # second method to be added later without a schema change — the same seam
    # `heatmaps.method` provides (D-030).
    """
    CREATE TABLE IF NOT EXISTS clusters (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        method  TEXT    NOT NULL,
        label   TEXT    NOT NULL,
        size    INTEGER NOT NULL,
        UNIQUE (run_id, method, label)
    )
    """,
    # Membership is its own table because a group has many findings. The
    # composite primary key enforces the rule that a finding belongs to at most
    # one group *per method*: a second method may group the same finding
    # differently, which is the point of keeping `method` on `clusters`.
    """
    CREATE TABLE IF NOT EXISTS cluster_members (
        cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
        finding_id INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        PRIMARY KEY (cluster_id, finding_id)
    )
    """,
    # Added at schema version 6. How often each factor appears among failures
    # *and* among correct findings, so a count can be read against its base
    # rate. `root_causes` alone cannot answer this: it is only ever populated
    # for failures, so a factor describing 71% of them looks damning until the
    # control group shows it describes 76% of successes (D-031).
    #
    # One row per run and factor — a different grain from `root_causes`, which
    # is per finding. Correct findings deliberately get no `root_causes` rows;
    # only these aggregates, so every existing query returns what it always did.
    """
    CREATE TABLE IF NOT EXISTS factor_rates (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        factor         TEXT    NOT NULL,
        failure_count  INTEGER NOT NULL,
        failure_total  INTEGER NOT NULL,
        correct_count  INTEGER NOT NULL,
        correct_total  INTEGER NOT NULL,
        lift           REAL,
        p_value        REAL    NOT NULL,
        UNIQUE (run_id, factor)
    )
    """,
    # Added at schema version 7. One row per failure group per rule that fired.
    # Attached to `clusters`, not to findings or factors: a recommendation is
    # about a pattern, and the group is the unit that already means "failures
    # sharing a cause" (D-034). Cluster ids are not stable across regrouping,
    # so the cascade is deliberate — regrouping invalidates advice derived from
    # the old partition, and stale advice is worse than none.
    #
    # `actionable` is derived from `status` and stored anyway, so ordering is a
    # plain column sort rather than a CASE every consumer must get right. The
    # same denormalisation `clusters.size` already uses.
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        cluster_id  INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
        rule        TEXT    NOT NULL,
        action      TEXT    NOT NULL,
        rationale   TEXT    NOT NULL,
        status      TEXT    NOT NULL,
        actionable  INTEGER NOT NULL,
        affected    INTEGER NOT NULL,
        priority    REAL    NOT NULL,
        UNIQUE (cluster_id, rule)
    )
    """,
    # Added at schema version 8. Mask-level diagnosis, keyed on the finding it
    # re-examines, exactly as D-022 planned when this was deferred. `findings`
    # keeps its shape, so every box-level query returns what it always did.
    #
    # `mask_iou` and `mask_outcome` are nullable on purpose: an outline that
    # was never produced is not an outline that failed. A detection model has
    # no masks at all, and a false positive has no ground-truth outline to
    # compare against — both are "not measured", not "measured as zero".
    """
    CREATE TABLE IF NOT EXISTS mask_findings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        finding_id    INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
        run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        mask_iou      REAL,
        mask_outcome  TEXT,
        pred_polygon  TEXT,
        UNIQUE (finding_id)
    )
    """,
    # Added at schema version 9. Evaluation and benchmarking are separate
    # tables because they are separate measurements with separate validity:
    # mAP is a property of a model on a dataset and travels between machines,
    # while latency and memory are properties of a model on *one* device and do
    # not. Folding them together would invite a row that is half portable.
    #
    # Neither becomes a column on `runs`. The migration path here is additive —
    # `initialise_database` creates missing tables and bumps the version, and
    # cannot add a column to a table that already exists — so a new table is
    # both the safer and the only supported shape (D-020).
    #
    # The settings that produced the numbers are stored beside them. An mAP
    # without its confidence sweep, IoU range and detection cap is not a
    # measurement, it is a rumour.
    """
    CREATE TABLE IF NOT EXISTS run_evaluations (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        task              TEXT    NOT NULL,
        evaluator         TEXT    NOT NULL,
        evaluator_version TEXT,
        sweep_confidence  REAL    NOT NULL,
        iou_thresholds    TEXT    NOT NULL,
        max_detections    INTEGER NOT NULL,
        ground_truth      TEXT    NOT NULL,
        gt_images         INTEGER NOT NULL,
        gt_annotations    INTEGER NOT NULL,
        prediction_count  INTEGER NOT NULL,
        mask_source       TEXT,
        map50_95          REAL,
        map50             REAL,
        map75             REAL,
        map_small         REAL,
        map_medium        REAL,
        map_large         REAL,
        ar1               REAL,
        ar10              REAL,
        ar100             REAL,
        ar_small          REAL,
        ar_medium         REAL,
        ar_large          REAL,
        created_at        TEXT    NOT NULL,
        UNIQUE (run_id, task)
    )
    """,
    # `device` is part of the key because `runs` does not record one: the
    # device is resolved at run time from `config.DEVICE`. A latency without
    # its device is meaningless, and silently comparing an MPS number against a
    # CPU number is the worst thing this table could enable — so the device is
    # mandatory, and the same run may hold one row per device it was measured
    # on. `image_size` joins the key because resolution changes both.
    """
    CREATE TABLE IF NOT EXISTS run_benchmarks (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id             INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        device             TEXT    NOT NULL,
        image_size         INTEGER NOT NULL,
        torch_threads      INTEGER,
        warmup_images      INTEGER NOT NULL,
        measured_images    INTEGER NOT NULL,
        cold_start_s       REAL,
        weights_load_s     REAL,
        latency_mean_ms    REAL,
        latency_median_ms  REAL,
        latency_p90_ms     REAL,
        latency_min_ms     REAL,
        fps                REAL,
        rss_baseline_mb    REAL,
        rss_after_load_mb  REAL,
        rss_peak_mb        REAL,
        gpu_allocated_mb   REAL,
        gpu_driver_mb      REAL,
        checkpoint_bytes   INTEGER,
        host_platform      TEXT,
        created_at         TEXT    NOT NULL,
        UNIQUE (run_id, device, image_size)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        -- Opaque to callers and unguessable, because a job id is handed to a
        -- browser and used to address an upload workspace.
        token          TEXT    NOT NULL UNIQUE,
        -- queued | running | succeeded | failed | cancelled
        status         TEXT    NOT NULL,
        -- Which pipeline stage is executing, or the one that failed.
        stage          TEXT,
        -- The run this job produced. Null until diagnosis has saved one, which
        -- is what makes a job observable before it has any results to show.
        run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
        detector       TEXT    NOT NULL,
        split          TEXT    NOT NULL,
        model_name     TEXT    NOT NULL,
        dataset_name   TEXT    NOT NULL,
        workspace      TEXT    NOT NULL,
        image_size     INTEGER NOT NULL,
        confidence     REAL    NOT NULL,
        -- Operator-facing failure text. Kept separate from the log tail so a
        -- reader gets the reason without reading output meant for a developer.
        error          TEXT,
        log_tail       TEXT,
        created_at     TEXT    NOT NULL,
        started_at     TEXT,
        finished_at    TEXT
    )
    """,
    # Indexes chosen for the queries a dashboard actually issues: filter by
    # run, then by outcome or class. Without them every filter is a full scan.
    "CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_findings_run ON findings(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_findings_image ON findings(image_id)",
    "CREATE INDEX IF NOT EXISTS idx_findings_outcome ON findings(run_id, outcome)",
    "CREATE INDEX IF NOT EXISTS idx_findings_class ON findings(run_id, class_name)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_run ON embeddings(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_heatmaps_run ON heatmaps(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_root_causes_run ON root_causes(run_id, factor)",
    "CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id, method)",
    "CREATE INDEX IF NOT EXISTS idx_factor_rates_run ON factor_rates(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_run "
    "ON recommendations(run_id, actionable, priority)",
    "CREATE INDEX IF NOT EXISTS idx_mask_findings_run ON mask_findings(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_cluster_members_finding "
    "ON cluster_members(finding_id)",
    "CREATE INDEX IF NOT EXISTS idx_run_evaluations_run ON run_evaluations(run_id)",
    # Added at schema version 11. A second lens over the *same* findings: one
    # row per image saying what shape the model's mistake took, where the
    # finding-level tables say which object failed. Nothing here changes a
    # finding, an outcome or a count — D-017 and every existing query are
    # untouched (D-040).
    #
    # The thresholds that produced a verdict are stored beside it, for the
    # reason `run_evaluations` stores its confidence sweep: a figure without
    # its protocol is not a measurement. Merge and split counts are genuinely
    # sensitive to `cover_hit`, so a consumer must be able to see what it was.
    """
    CREATE TABLE IF NOT EXISTS image_diagnoses (
        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id                 INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        image_id               INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        gt_count               INTEGER NOT NULL,
        pred_count             INTEGER NOT NULL,
        correct                INTEGER NOT NULL,
        false_negative         INTEGER NOT NULL,
        false_positive         INTEGER NOT NULL,
        poor_localization      INTEGER NOT NULL,
        wrong_class            INTEGER NOT NULL,
        verdict                TEXT    NOT NULL,
        merged                 INTEGER NOT NULL,
        split                  INTEGER NOT NULL,
        objects_untouched      INTEGER NOT NULL,
        objects_partial        INTEGER NOT NULL DEFAULT 0,
        predictions_partial    INTEGER NOT NULL DEFAULT 0,
        predictions_on_nothing INTEGER NOT NULL,
        cover_hit              REAL    NOT NULL,
        cover_miss             REAL    NOT NULL,
        method                 TEXT    NOT NULL,
        created_at             TEXT    NOT NULL,
        UNIQUE (run_id, image_id)
    )
    """,
    # The relationships behind a verdict, so no aggregate is reported without
    # its constituents. One prediction covering two annotated objects is a
    # *merge*, and this table is what lets a reader see that rather than infer
    # an independent miss for the object the matcher could not pair.
    #
    # Only pairs above `IMAGE_COVERAGE_FLOOR` are stored: a row saying two
    # shapes do not overlap carries no more than its own absence, and keeping
    # every combination would be quadratic in a crowded image.
    """
    CREATE TABLE IF NOT EXISTS image_coverage (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        image_diagnosis_id  INTEGER NOT NULL
                            REFERENCES image_diagnoses(id) ON DELETE CASCADE,
        truth_finding_id    INTEGER NOT NULL
                            REFERENCES findings(id) ON DELETE CASCADE,
        pred_finding_id     INTEGER NOT NULL
                            REFERENCES findings(id) ON DELETE CASCADE,
        coverage            REAL    NOT NULL,
        UNIQUE (image_diagnosis_id, truth_finding_id, pred_finding_id)
    )
    """,
    # One relationship between findings, measured on masks (schema version 13).
    #
    # **A relation is not a factor.** A factor is a property of one object,
    # measured on failures *and* on correct detections, and admitted only when
    # its lift over that control rate is significant. A relation describes what
    # a failure *is* — a missed object that another prediction already covers,
    # a box outcome its own mask contradicts — and by construction it cannot
    # attach to a correct detection, so it has no control rate and no lift.
    # Routing one through `factor_rates` would therefore silently discard it.
    # They are stored apart for that reason, and never enter that pipeline.
    #
    # `UNIQUE (run_id, finding_id, relation)` is the anti-double-counting rule
    # as a constraint rather than a convention: one relationship produces one
    # row, attributed to one side of it, with `span` recording multiplicity
    # instead of extra rows.
    #
    # The thresholds that produced the row travel with it, as they do on
    # `image_diagnoses`: a run analysed at one threshold must never be read as
    # though it had been analysed at another.
    """
    CREATE TABLE IF NOT EXISTS finding_relations (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        finding_id          INTEGER NOT NULL
                            REFERENCES findings(id) ON DELETE CASCADE,
        relation            TEXT    NOT NULL,
        direction           TEXT,
        partner_finding_id  INTEGER REFERENCES findings(id) ON DELETE CASCADE,
        value               REAL    NOT NULL,
        best_coverage       REAL,
        span                INTEGER,
        threshold           REAL,
        coverage_floor      REAL,
        qualifies           INTEGER NOT NULL,
        cover_hit           REAL    NOT NULL,
        method              TEXT    NOT NULL DEFAULT 'mask',
        created_at          TEXT    NOT NULL,
        UNIQUE (run_id, finding_id, relation)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_run_benchmarks_run ON run_benchmarks(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_image_diagnoses_run "
    "ON image_diagnoses(run_id, verdict)",
    "CREATE INDEX IF NOT EXISTS idx_image_coverage_diag "
    "ON image_coverage(image_diagnosis_id)",
    "CREATE INDEX IF NOT EXISTS idx_finding_relations_run "
    "ON finding_relations(run_id, relation)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_token ON jobs(token)",
)


# ---------------------------------------------------------------------------
# Records returned on read
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunContext:
    """Everything that identifies and reproduces one diagnosis run.

    Assembled by the caller because only the caller knows which model and
    thresholds were actually used. Storing it is what makes a finding
    interpretable months later, and what makes two runs comparable.
    """

    model_path: str
    model_sha256: str
    dataset_yaml: str
    split: str
    confidence_threshold: float
    match_iou_threshold: float
    localization_iou_floor: float
    image_size: int
    #: Schema 15. What was passed to the detector, not what it did with it —
    #: a value can be passed and ignored, which is exactly what
    #: :mod:`model_doctor.app.capabilities` exists to detect. ``None`` when
    #: the caller
    # did
    #: not record one.
    nms_iou_threshold: float | None = None
    inference_max_detections: int | None = None
    inference_library_version: str | None = None


@dataclass(frozen=True)
class RunRecord:
    """A run as stored, including its assigned id and timestamp."""

    id: int
    created_at: str
    model_path: str
    model_sha256: str
    dataset_yaml: str
    split: str
    confidence_threshold: float
    match_iou_threshold: float
    localization_iou_floor: float
    image_size: int
    #: What a reader called this run, or None if never named. Never a blank
    #: string: clearing a name restores the id as the run's only identity,
    #: and "" would render as a run with an invisible name.
    name: str | None = None
    #: Schema 15 provenance. ``None`` means *not recorded*, and is never
    #: replaced by a configured default when read: what an environment held
    #: when a months-old run executed is not recoverable from what it holds
    #: now, and a plausible substitute would be indistinguishable from a fact.
    nms_iou_threshold: float | None = None
    inference_max_detections: int | None = None
    inference_library_version: str | None = None


@dataclass(frozen=True)
class FindingRecord:
    """A finding as stored, row-shaped.

    Deliberately mirrors the table rather than reconstructing
    :class:`~model_doctor.app.diagnosis.Finding`. Tests assert that every column
    survives a
    round trip, and a row-shaped type makes a missing column an obvious test
    failure instead of a quietly defaulted object attribute.
    """

    id: int
    run_id: int
    image_id: int
    outcome: str
    class_id: int | None
    class_name: str
    confidence: float | None
    iou: float | None
    pred_box: tuple[float, float, float, float] | None
    truth_box: tuple[float, float, float, float] | None
    truth_polygon: list[list[float]] | None


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------
@contextmanager
def connect(path: Path | None = None):
    """Open a connection with the schema applied and foreign keys enforced.

    SQLite ignores foreign-key constraints unless they are switched on *per
    connection*. Without the pragma the constraints in the schema are
    decorative and orphan rows accumulate silently, so it is set here rather
    than left to callers to remember.

    Args:
        path: Database file. Defaults to :data:`config.DB_PATH`. The parent
            directory is created if needed.

    Yields:
        An open :class:`sqlite3.Connection`, committed on clean exit and rolled
        back if the block raises.
    """
    db_path = Path(path) if path is not None else config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # Write-ahead logging so a reader is not blocked by a writer. Without it
    # the read-only API returns "database is locked" for the duration of an
    # analysis — precisely while a user is watching the job they just started
    # and clicking into the dashboard. The pragma is persistent, so setting it
    # here applies to every later connection, including read-only ones.
    connection.execute("PRAGMA journal_mode = WAL")
    # Wait rather than fail the moment two writers overlap. Jobs are
    # serialised, but a CLI run and a job can still coincide.
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        initialise_database(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def connect_read_only(path: Path | None = None):
    """Open the database so that nothing can be created or modified.

    **The one implementation of "read-only" in the project.** The HTTP API and
    the MCP server are both projections of the schema that must never write,
    and each used to carry its own opener. Two copies of a guarantee drift —
    one grows a pragma the other lacks — and the guarantee is worth exactly as
    much as its weakest copy.

    Deliberately not :func:`connect`, which applies the schema and can bump the
    recorded version. That is right for a CLI pass and wrong for a request
    handler, where two workers could race on it (D-037). SQLite's ``mode=ro``
    URI refuses writes at the engine, so a reader that ever began writing
    fails loudly at this boundary rather than quietly mutating history.

    Args:
        path: Database file. Defaults to :data:`config.DB_PATH`. Unlike
            :func:`connect`, nothing is created: an absent file is an error.

    Yields:
        A connection with :class:`sqlite3.Row` rows that cannot write.

    Raises:
        FileNotFoundError: If the file does not exist. Callers translate this
            into their own vocabulary — a 503 for HTTP, a tool error for MCP —
            because the remedy is the same in both and belongs with them.
        sqlite3.Error: If the file exists but cannot be opened.
    """
    target = Path(path) if path is not None else config.DB_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"No diagnosis database at {target}. Produce one with: "
            "python -m model_doctor.app.diagnosis --split test --save"
        )
    connection = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


#: Columns added to tables that already existed, as ``(table, column, type)``.
#:
#: Every schema change before version 12 added a whole table, which
#: ``CREATE TABLE IF NOT EXISTS`` handles on its own. A column added to an
#: existing table does not work that way: the create statement is skipped for a
#: database that already has the table, so the column would appear only on
#: freshly created files and every existing database would keep reading and
#: writing without it. Each entry here is applied with ``ALTER TABLE`` when
#: absent.
#:
#: Nullable and without a default, so the migration cannot alter a stored value
#: — an unnamed run reads as ``NULL``, which is what it was before the column
#: existed.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("runs", "name", "TEXT"),
    ("finding_relations", "coverage_floor", "REAL"),
    # Schema 15. What Model Doctor *passed* at inference, which nothing else
    # records. Everything a checkpoint can answer about itself — its family,
    # its head, whether that head is end-to-end, the size it was trained at —
    # is deliberately absent: `model_doctor.app.detectors.detect_family` established
    # that
    # duplicating a fact the weights already hold only creates two versions of
    # it that can disagree. These three cannot be recovered from the weights.
    # `inference_max_detections` is named apart from
    # `run_evaluations.max_detections`, which is COCOeval's `maxDets` and a
    # different number entirely.
    ("runs", "nms_iou_threshold", "REAL"),
    ("runs", "inference_max_detections", "INTEGER"),
    ("runs", "inference_library_version", "TEXT"),
)


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    """Add any column in :data:`_ADDED_COLUMNS` the database does not have.

    Idempotent, and safe on a database created before the column existed: the
    table is inspected first, so this neither fails on a second call nor
    depends on the recorded schema version to decide what to do.
    """
    for table, column, declaration in _ADDED_COLUMNS:
        present = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not present:
            continue  # The table itself is absent; nothing to widen.
        if column in present:
            continue
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        logger.info("Added column %s.%s", table, column)


def initialise_database(connection: sqlite3.Connection) -> None:
    """Create tables and indexes if they are not already present.

    Idempotent: every statement is ``IF NOT EXISTS``, so calling this on an
    existing database is a no-op and opening a database is always safe.
    """
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    _add_missing_columns(connection)

    existing = connection.execute("SELECT version FROM schema_info").fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO schema_info (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _timestamp()),
        )
    elif existing["version"] < SCHEMA_VERSION:
        # Every change so far has been a new table, which the statements above
        # have already created. Recording the new version is all that remains.
        # A future change that is not purely additive must migrate here rather
        # than simply bumping the number.
        connection.execute(
            "UPDATE schema_info SET version = ?, applied_at = ?",
            (SCHEMA_VERSION, _timestamp()),
        )
        logger.info(
            "Upgraded database schema to version %d (additive)", SCHEMA_VERSION
        )


def _timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string.

    UTC rather than local time so runs recorded on different machines sort
    correctly against each other.
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Used to fingerprint the model. A path alone is not enough to identify which
    weights produced a run — files get overwritten in place, and then two runs
    that disagree look like a regression rather than a different model.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def save_run(connection: sqlite3.Connection, context: RunContext) -> int:
    """Insert a run and return its id.

    Each call creates a **new** run. This is deliberately not idempotent: runs
    are events, and keeping every one is what allows a model to be compared
    against its predecessor.

    Args:
        connection: An open connection.
        context: The parameters that identify this run.

    Returns:
        The assigned run id.
    """
    cursor = connection.execute(
        """
        INSERT INTO runs (
            created_at, model_path, model_sha256, dataset_yaml, split,
            confidence_threshold, match_iou_threshold, localization_iou_floor,
            image_size, nms_iou_threshold, inference_max_detections,
            inference_library_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _timestamp(),
            context.model_path,
            context.model_sha256,
            context.dataset_yaml,
            context.split,
            context.confidence_threshold,
            context.match_iou_threshold,
            context.localization_iou_floor,
            context.image_size,
            context.nms_iou_threshold,
            context.inference_max_detections,
            context.inference_library_version,
        ),
    )
    run_id = int(cursor.lastrowid)
    logger.debug("Recorded run %d", run_id)
    return run_id


def save_image(
    connection: sqlite3.Connection, run_id: int, diagnosis: ImageDiagnosis
) -> int:
    """Insert one image's row and return its id.

    An errored image is stored with its error text and zero counts, so that
    "processed and found nothing" stays distinguishable from "never processed".
    """
    cursor = connection.execute(
        """
        INSERT INTO images (
            run_id, path, filename, width, height,
            prediction_count, truth_count, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            str(diagnosis.image_path),
            diagnosis.image_path.name,
            diagnosis.image_width or None,
            diagnosis.image_height or None,
            len(diagnosis.predictions),
            len(diagnosis.truths),
            diagnosis.error,
        ),
    )
    return int(cursor.lastrowid)


def save_findings(
    connection: sqlite3.Connection,
    run_id: int,
    image_id: int,
    findings: Iterable[Finding],
) -> int:
    """Insert an image's findings and return how many were written.

    Args:
        connection: An open connection.
        run_id: Owning run.
        image_id: Owning image.
        findings: The findings to store.

    Returns:
        Number of rows inserted.
    """
    rows = []
    for finding in findings:
        prediction, truth = finding.prediction, finding.truth
        # class_id is taken from ground truth when present, matching how
        # Finding.class_name attributes a failure to the class that should have
        # been found rather than the one wrongly predicted.
        source = truth if truth is not None else prediction
        rows.append(
            (
                run_id,
                image_id,
                finding.outcome.value,
                source.class_id if source is not None else None,
                finding.class_name,
                prediction.confidence if prediction is not None else None,
                finding.iou,
                *(prediction.xyxy if prediction is not None else (None,) * 4),
                *(truth.xyxy if truth is not None else (None,) * 4),
                json.dumps([list(p) for p in truth.polygon])
                if truth is not None and truth.polygon is not None
                else None,
            )
        )

    if rows:
        connection.executemany(
            """
            INSERT INTO findings (
                run_id, image_id, outcome, class_id, class_name,
                confidence, iou,
                pred_x1, pred_y1, pred_x2, pred_y2,
                truth_x1, truth_y1, truth_x2, truth_y2,
                truth_polygon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def save_dataset_diagnosis(
    connection: sqlite3.Connection,
    context: RunContext,
    summary: DatasetDiagnosis,
) -> int:
    """Persist a whole dataset diagnosis as one run, and return its run id.

    The convenience entry point the CLI uses. Everything is written inside the
    caller's transaction, so a failure part-way leaves no partial run behind.
    """
    run_id = save_run(connection, context)
    total = 0
    for diagnosis in summary.diagnoses:
        image_id = save_image(connection, run_id, diagnosis)
        total += save_findings(connection, run_id, image_id, diagnosis.findings)

    logger.info(
        "Saved run %d: %d image(s), %d finding(s)",
        run_id,
        len(summary.diagnoses),
        total,
    )
    return run_id


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _row_to_run(row: sqlite3.Row) -> RunRecord:
    """Convert a ``runs`` row into a :class:`RunRecord`."""
    columns = row.keys()
    return RunRecord(
        id=row["id"],
        created_at=row["created_at"],
        model_path=row["model_path"],
        model_sha256=row["model_sha256"],
        dataset_yaml=row["dataset_yaml"],
        split=row["split"],
        confidence_threshold=row["confidence_threshold"],
        match_iou_threshold=row["match_iou_threshold"],
        localization_iou_floor=row["localization_iou_floor"],
        image_size=row["image_size"],
        # Absent from databases written before schema 12, which is why this is
        # read defensively rather than by subscript. `in` on the row itself
        # would search its *values*, so the column list is asked explicitly.
        name=row["name"] if "name" in columns else None,
        # Absent before schema 15, and NULL on every run written before it.
        # Read defensively for the same reason `name` is, and left as None
        # rather than defaulted: see `RunRecord`.
        nms_iou_threshold=(
            row["nms_iou_threshold"] if "nms_iou_threshold" in columns else None
        ),
        inference_max_detections=(
            row["inference_max_detections"]
            if "inference_max_detections" in columns
            else None
        ),
        inference_library_version=(
            row["inference_library_version"]
            if "inference_library_version" in columns
            else None
        ),
    )


def load_run(connection: sqlite3.Connection, run_id: int) -> RunRecord | None:
    """Return one run by id, or ``None`` if it does not exist."""
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row is not None else None


def list_runs(connection: sqlite3.Connection) -> list[RunRecord]:
    """Return every run, newest first."""
    rows = connection.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [_row_to_run(row) for row in rows]


#: Longest name a run may carry. Generous for a label, short enough that a
#: pasted paragraph cannot make the header unreadable.
MAX_RUN_NAME = 80


def rename_run(
    connection: sqlite3.Connection, run_id: int, name: str | None
) -> RunRecord | None:
    """Set or clear a run's name, returning the run as it now stands.

    A name is the only writable field on a run. Everything else records what an
    analysis pass did, and rewriting that would make the stored evidence
    disagree with the findings derived from it — but what a run is *called* is
    the reader's, not the pass's.

    Blank input clears the name rather than storing ``""``. An empty string is
    a name that renders as nothing, which is indistinguishable on screen from
    an unnamed run while behaving differently everywhere else.

    Args:
        connection: An open connection.
        run_id: The run to rename.
        name: The new name, or None/blank to clear it.

    Returns:
        The updated run, or None if no run has that id.

    Raises:
        ValueError: If the name exceeds :data:`MAX_RUN_NAME` characters.
    """
    cleaned = (name or "").strip()
    if len(cleaned) > MAX_RUN_NAME:
        raise ValueError(
            f"A run name may be at most {MAX_RUN_NAME} characters, got {len(cleaned)}."
        )
    cursor = connection.execute(
        "UPDATE runs SET name = ? WHERE id = ?", (cleaned or None, run_id)
    )
    if cursor.rowcount == 0:
        return None
    return load_run(connection, run_id)


@dataclass(frozen=True)
class ImageRow:
    """One image as attempted, including images that produced nothing.

    Every image the run touched appears here, which is what makes the counts
    trustworthy: an image processed cleanly with nothing in it has no finding,
    and an image that never ran has ``error`` set. Deriving image totals from
    ``findings`` silently loses both cases.
    """

    id: int
    run_id: int
    path: str
    filename: str
    width: int | None
    height: int | None
    prediction_count: int
    truth_count: int
    error: str | None


def load_images(connection: sqlite3.Connection, run_id: int) -> list[ImageRow]:
    """Return every image attempted in a run, in insertion order.

    Includes images that errored and images that produced no findings. Both are
    dropped by any count derived from ``findings``, and ``error`` exists
    precisely so "processed and found nothing" stays distinguishable from
    "never ran".

    Args:
        connection: An open connection.
        run_id: Which run to read.

    Returns:
        Image rows exactly as the schema defines them.
    """
    rows = connection.execute(
        """
        SELECT id, run_id, path, filename, width, height,
               prediction_count, truth_count, error
        FROM images
        WHERE run_id = ?
        ORDER BY id
        """,
        (run_id,),
    ).fetchall()
    return [
        ImageRow(
            id=row["id"],
            run_id=row["run_id"],
            path=row["path"],
            filename=row["filename"],
            width=row["width"],
            height=row["height"],
            prediction_count=row["prediction_count"],
            truth_count=row["truth_count"],
            error=row["error"],
        )
        for row in rows
    ]


def load_findings(
    connection: sqlite3.Connection,
    run_id: int,
    outcome: Outcome | None = None,
) -> list[FindingRecord]:
    """Return a run's findings, optionally filtered to one outcome.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        outcome: When given, only findings of this outcome are returned.

    Returns:
        Findings in insertion order.
    """
    if outcome is None:
        rows = connection.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM findings WHERE run_id = ? AND outcome = ? ORDER BY id",
            (run_id, outcome.value),
        ).fetchall()

    def box(prefix: str) -> tuple[float, float, float, float] | None:
        values = [row[f"{prefix}_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
        return tuple(values) if values[0] is not None else None  # type: ignore[return-value]

    records = []
    for row in rows:
        polygon = json.loads(row["truth_polygon"]) if row["truth_polygon"] else None
        records.append(
            FindingRecord(
                id=row["id"],
                run_id=row["run_id"],
                image_id=row["image_id"],
                outcome=row["outcome"],
                class_id=row["class_id"],
                class_name=row["class_name"],
                confidence=row["confidence"],
                iou=row["iou"],
                pred_box=box("pred"),
                truth_box=box("truth"),
                truth_polygon=polygon,
            )
        )
    return records


def load_findings_page(
    connection: sqlite3.Connection,
    run_id: int,
    limit: int,
    offset: int,
    outcome: Outcome | None = None,
    image_id: int | None = None,
) -> list[FindingRecord]:
    """Return a page of findings using SQL-level pagination.

    Unlike :func:`load_findings`, only the requested slice leaves SQLite,
    which is critical for the API endpoint where loading tens of thousands of
    rows per page request would blow the memory budget.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        limit: Maximum number of rows to return.
        offset: Number of rows to skip.
        outcome: When given, only findings of this outcome are returned.
        image_id: When given, only findings on this image are returned. Asked
            for by a viewer showing one photograph in full: every annotated
            object and every prediction on it, including those the matcher
            paired with a different finding. Without it the caller would have
            to hold a whole run's findings to draw one image.

    Returns:
        At most *limit* findings, starting from *offset*, in insertion order.
    """
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if outcome is not None:
        clauses.append("outcome = ?")
        params.append(outcome.value if hasattr(outcome, "value") else str(outcome))
    if image_id is not None:
        clauses.append("image_id = ?")
        params.append(image_id)
    sql = (
        "SELECT * FROM findings WHERE "
        + " AND ".join(clauses)
        + " ORDER BY id LIMIT ? OFFSET ?"
    )
    rows = connection.execute(sql, (*params, limit, offset)).fetchall()
    return [_finding_record(row) for row in rows]


def _finding_record(row: sqlite3.Row) -> FindingRecord:
    """Build one finding from a stored row.

    Extracted so every reader of `findings` produces the same shape; the paged
    query previously inlined it and would have needed a second copy once a
    filter was added.
    """

    def box(prefix: str) -> tuple[float, float, float, float] | None:
        values = [row[f"{prefix}_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
        return tuple(values) if values[0] is not None else None  # type: ignore[return-value]

    return FindingRecord(
        id=row["id"],
        run_id=row["run_id"],
        image_id=row["image_id"],
        outcome=row["outcome"],
        class_id=row["class_id"],
        class_name=row["class_name"],
        confidence=row["confidence"],
        iou=row["iou"],
        pred_box=box("pred"),
        truth_box=box("truth"),
        truth_polygon=(
            json.loads(row["truth_polygon"]) if row["truth_polygon"] else None
        ),
    )


def outcome_counts(connection: sqlite3.Connection, run_id: int) -> dict[str, int]:
    """Return finding counts by outcome for one run.

    The aggregate a dashboard opens with. Provided here so the same numbers
    come from one query rather than being re-derived, and slightly differently,
    by each consumer.
    """
    rows = connection.execute(
        "SELECT outcome, COUNT(*) AS n FROM findings WHERE run_id = ? GROUP BY outcome",
        (run_id,),
    ).fetchall()
    return {row["outcome"]: row["n"] for row in rows}


def build_run_context(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    confidence: float,
    match_iou: float,
    localization_floor: float,
    image_size: int,
    nms_iou: float | None = None,
    max_detections: int | None = None,
    library_version: str | None = None,
) -> RunContext:
    """Assemble a :class:`RunContext`, hashing the model file.

    Args:
        model_path: Weights used for the run.
        dataset_yaml: Dataset descriptor used.
        split: Split diagnosed.
        confidence: Confidence threshold applied at inference.
        match_iou: IoU at which a pair counted as correctly localised.
        localization_floor: Floor for the poor-localisation pass.
        image_size: Inference image size.
        nms_iou: NMS IoU threshold *passed* to the detector. Recording it says
            nothing about whether the model acted on it — a value can be
            forwarded and ignored, which is what :mod:`model_doctor.app.capabilities`
            determines. Omitted rather than defaulted when the caller does not
            know it.
        max_detections: Detection cap passed at inference. Distinct from
            ``run_evaluations.max_detections``, which is COCOeval's ``maxDets``.
        library_version: Version of the inference library that ran. Not the
            version recorded inside the checkpoint, which is the one that
            *trained* it and is routinely older.

    Returns:
        A populated run context.
    """
    return RunContext(
        model_path=str(model_path),
        model_sha256=file_sha256(model_path),
        dataset_yaml=str(dataset_yaml),
        split=split,
        confidence_threshold=confidence,
        match_iou_threshold=match_iou,
        localization_iou_floor=localization_floor,
        image_size=image_size,
        nms_iou_threshold=nms_iou,
        inference_max_detections=max_detections,
        inference_library_version=library_version,
    )


def format_run_summary(runs: Sequence[RunRecord]) -> str:
    """Render stored runs as a table."""
    if not runs:
        return "\n  No runs recorded yet.\n"
    lines = ["", "Stored runs", "-" * 78]
    lines.append(f"  {'ID':>4}  {'WHEN':<20} {'SPLIT':<8} {'MODEL':<24} {'SHA':<10}")
    for run in runs:
        lines.append(
            f"  {run.id:>4}  {run.created_at:<20} {run.split:<8} "
            f"{Path(run.model_path).name[:24]:<24} {run.model_sha256[:8]:<10}"
        )
    lines.append("-" * 78)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embeddings (schema version 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingRow:
    """One stored embedding, joined to the finding it describes.

    Carries the finding's outcome and class so a consumer clustering these
    vectors does not need a second query to label them.
    """

    finding_id: int
    run_id: int
    outcome: str
    class_name: str
    model_name: str
    dimensions: int
    vector: tuple[float, ...]


def save_embeddings(
    connection: sqlite3.Connection,
    run_id: int,
    model_name: str,
    vectors: Iterable[tuple[int, Sequence[float]]],
) -> int:
    """Store embeddings for a run's findings.

    Vectors are written as raw float32 bytes rather than JSON: 512 floats cost
    2 KB exactly, with no precision loss and no parsing on read.

    Re-running for the same finding and model replaces the previous vector.
    Embeddings are derived data — unlike a run, recomputing one is a correction,
    not a second observation, so replacing is right here where it would be wrong
    for runs (D-021).

    Args:
        connection: An open connection.
        run_id: Owning run.
        model_name: Identifier of the model that produced these vectors, stored
            so vectors from different encoders are never mixed in one cluster.
        vectors: Pairs of ``(finding_id, vector)``.

    Returns:
        Number of embeddings written.
    """
    import array

    rows = []
    for finding_id, vector in vectors:
        payload = array.array("f", vector).tobytes()
        rows.append((finding_id, run_id, model_name, len(vector), payload))

    if rows:
        connection.executemany(
            """
            INSERT INTO embeddings (finding_id, run_id, model_name, dimensions, vector)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (finding_id, model_name) DO UPDATE SET
                vector = excluded.vector,
                dimensions = excluded.dimensions,
                run_id = excluded.run_id
            """,
            rows,
        )
    return len(rows)


def load_embeddings(
    connection: sqlite3.Connection, run_id: int, model_name: str | None = None
) -> list[EmbeddingRow]:
    """Return a run's embeddings with their findings' outcome and class.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        model_name: Restrict to one encoder. Recommended whenever more than one
            has been used, since vectors from different encoders are not
            comparable.

    Returns:
        Embeddings in finding order.
    """
    import array

    sql = """
        SELECT e.finding_id, e.run_id, e.model_name, e.dimensions, e.vector,
               f.outcome, f.class_name
        FROM embeddings e JOIN findings f ON f.id = e.finding_id
        WHERE e.run_id = ?
    """
    params: list[object] = [run_id]
    if model_name is not None:
        sql += " AND e.model_name = ?"
        params.append(model_name)
    sql += " ORDER BY e.finding_id"

    records = []
    for row in connection.execute(sql, params).fetchall():
        values = array.array("f")
        values.frombytes(row["vector"])
        records.append(
            EmbeddingRow(
                finding_id=row["finding_id"],
                run_id=row["run_id"],
                outcome=row["outcome"],
                class_name=row["class_name"],
                model_name=row["model_name"],
                dimensions=row["dimensions"],
                vector=tuple(values),
            )
        )
    return records


def load_findings_for_embedding(
    connection: sqlite3.Connection, run_id: int, failures_only: bool = True
) -> list[sqlite3.Row]:
    """Return findings joined to their image, ready for region extraction.

    Reads from the database rather than from in-memory diagnosis objects, so
    feature extraction is a pass over an already-saved run. That means it needs
    no changes to the write path, and can be re-run over historical runs.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        failures_only: Restrict to findings that represent mistakes, which is
            what the roadmap asks embeddings to cover.

    Returns:
        Rows carrying the finding's geometry plus the image path and size.
    """
    sql = """
        SELECT f.id AS finding_id, f.outcome, f.class_id, f.class_name,
               f.pred_x1, f.pred_y1, f.pred_x2, f.pred_y2,
               f.truth_x1, f.truth_y1, f.truth_x2, f.truth_y2,
               f.truth_polygon,
               i.path, i.width, i.height
        FROM findings f JOIN images i ON i.id = f.image_id
        WHERE f.run_id = ? AND i.error IS NULL
    """
    if failures_only:
        sql += " AND f.outcome != 'correct'"
    sql += " ORDER BY f.id"
    return connection.execute(sql, (run_id,)).fetchall()


# ---------------------------------------------------------------------------
# Heatmaps (schema version 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatmapRow:
    """One stored heatmap, joined to the finding it explains."""

    finding_id: int
    run_id: int
    outcome: str
    class_name: str
    path: str
    method: str
    target_layers: str


def save_heatmaps(
    connection: sqlite3.Connection,
    run_id: int,
    method: str,
    target_layers: str,
    entries: Iterable[tuple[int, str]],
) -> int:
    """Record generated heatmaps for a run's findings.

    Re-running replaces a finding's heatmap for the same method, matching how
    embeddings behave: a regenerated explanation is a correction, not a second
    observation.

    Args:
        connection: An open connection.
        run_id: Owning run.
        method: How the map was produced, e.g. ``"grad-cam"``.
        target_layers: Adapter identifier describing which layers were used.
        entries: Pairs of ``(finding_id, path)``.

    Returns:
        Number of rows written.
    """
    rows = [
        (finding_id, run_id, str(path), method, target_layers)
        for finding_id, path in entries
    ]
    if rows:
        connection.executemany(
            """
            INSERT INTO heatmaps (finding_id, run_id, path, method, target_layers)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (finding_id, method) DO UPDATE SET
                path = excluded.path,
                target_layers = excluded.target_layers,
                run_id = excluded.run_id
            """,
            rows,
        )
    return len(rows)


def load_heatmaps(
    connection: sqlite3.Connection, run_id: int, method: str | None = None
) -> list[HeatmapRow]:
    """Return a run's heatmaps with their findings' outcome and class."""
    sql = """
        SELECT h.finding_id, h.run_id, h.path, h.method, h.target_layers,
               f.outcome, f.class_name
        FROM heatmaps h JOIN findings f ON f.id = h.finding_id
        WHERE h.run_id = ?
    """
    params: list[object] = [run_id]
    if method is not None:
        sql += " AND h.method = ?"
        params.append(method)
    sql += " ORDER BY h.finding_id"

    return [
        HeatmapRow(
            finding_id=row["finding_id"],
            run_id=row["run_id"],
            outcome=row["outcome"],
            class_name=row["class_name"],
            path=row["path"],
            method=row["method"],
            target_layers=row["target_layers"],
        )
        for row in connection.execute(sql, params).fetchall()
    ]


# ---------------------------------------------------------------------------
# Root causes (schema version 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RootCauseRow:
    """One attributed factor, joined to the finding it concerns."""

    finding_id: int
    run_id: int
    outcome: str
    class_name: str
    factor: str
    score: float
    evidence: str


def save_root_causes(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[tuple[int, str, float, str]],
) -> int:
    """Record attributed factors for a run's findings.

    Re-running replaces a finding's evidence for the same factor. Attribution
    is derived data: recomputing it with different thresholds is a correction,
    not a second observation.

    Args:
        connection: An open connection.
        run_id: Owning run.
        entries: Tuples of ``(finding_id, factor, score, evidence)``.

    Returns:
        Number of rows written.
    """
    rows = [
        (finding_id, run_id, factor, float(score), evidence)
        for finding_id, factor, score, evidence in entries
    ]
    if rows:
        connection.executemany(
            """
            INSERT INTO root_causes (finding_id, run_id, factor, score, evidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (finding_id, factor) DO UPDATE SET
                score = excluded.score,
                evidence = excluded.evidence,
                run_id = excluded.run_id
            """,
            rows,
        )
    return len(rows)


def load_root_causes(
    connection: sqlite3.Connection, run_id: int, factor: str | None = None
) -> list[RootCauseRow]:
    """Return a run's attributed factors, strongest first."""
    sql = """
        SELECT rc.finding_id, rc.run_id, rc.factor, rc.score, rc.evidence,
               f.outcome, f.class_name
        FROM root_causes rc JOIN findings f ON f.id = rc.finding_id
        WHERE rc.run_id = ?
    """
    params: list[object] = [run_id]
    if factor is not None:
        sql += " AND rc.factor = ?"
        params.append(factor)
    sql += " ORDER BY rc.score DESC, rc.finding_id"

    return [
        RootCauseRow(
            finding_id=row["finding_id"],
            run_id=row["run_id"],
            outcome=row["outcome"],
            class_name=row["class_name"],
            factor=row["factor"],
            score=row["score"],
            evidence=row["evidence"],
        )
        for row in connection.execute(sql, params).fetchall()
    ]


def factor_counts(connection: sqlite3.Connection, run_id: int) -> dict[str, int]:
    """Return how many findings each factor was attributed to."""
    rows = connection.execute(
        "SELECT factor, COUNT(*) AS n FROM root_causes "
        "WHERE run_id = ? GROUP BY factor",
        (run_id,),
    ).fetchall()
    return {row["factor"]: row["n"] for row in rows}


# ---------------------------------------------------------------------------
# Failure groups (schema version 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterRow:
    """One stored failure group.

    Named for the table it comes from. The concept is a *failure group*: the
    default method assigns membership deterministically rather than by
    unsupervised clustering, and the prose in ``docs/SCHEMA.md`` says so
    (D-030).
    """

    id: int
    run_id: int
    method: str
    label: str
    size: int


def save_clusters(
    connection: sqlite3.Connection,
    run_id: int,
    method: str,
    groups: Iterable[tuple[str, Sequence[int]]],
) -> int:
    """Replace a run's failure groups for one method.

    Existing groups for this ``(run_id, method)`` are deleted first, and their
    memberships fall with them by cascade. Replacement rather than upsert is
    correct here because regrouping changes which findings belong together —
    an upsert would leave members of a group that no longer exists. Grouping is
    derived data, so recomputing it is a correction, not a second observation
    (D-021). Groups produced by a *different* method are untouched.

    Args:
        connection: An open connection.
        run_id: Owning run.
        method: How the grouping was produced, e.g. ``"factor-signature"``.
        groups: Pairs of ``(label, finding_ids)``. Empty groups are skipped —
            a group nothing belongs to would be a row that explains nothing.

    Returns:
        Number of groups written.
    """
    connection.execute(
        "DELETE FROM clusters WHERE run_id = ? AND method = ?", (run_id, method)
    )

    written = 0
    for label, finding_ids in groups:
        members = list(finding_ids)
        if not members:
            continue
        cursor = connection.execute(
            "INSERT INTO clusters (run_id, method, label, size) VALUES (?, ?, ?, ?)",
            (run_id, method, label, len(members)),
        )
        cluster_id = int(cursor.lastrowid)
        connection.executemany(
            "INSERT INTO cluster_members (cluster_id, finding_id) VALUES (?, ?)",
            [(cluster_id, finding_id) for finding_id in members],
        )
        written += 1
    return written


def load_clusters(
    connection: sqlite3.Connection, run_id: int, method: str | None = None
) -> list[ClusterRow]:
    """Return a run's failure groups, largest first.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        method: Restrict to one grouping method. Recommended once more than one
            has been run, since two methods partition the same findings
            differently and their groups are not comparable.

    Returns:
        Groups ordered by descending size, then label for a stable tie-break.
    """
    sql = "SELECT id, run_id, method, label, size FROM clusters WHERE run_id = ?"
    params: list[object] = [run_id]
    if method is not None:
        sql += " AND method = ?"
        params.append(method)
    sql += " ORDER BY size DESC, label"

    return [
        ClusterRow(
            id=row["id"],
            run_id=row["run_id"],
            method=row["method"],
            label=row["label"],
            size=row["size"],
        )
        for row in connection.execute(sql, params).fetchall()
    ]


def load_cluster_members(
    connection: sqlite3.Connection, cluster_id: int
) -> list[sqlite3.Row]:
    """Return the findings in one group, with the columns needed to show them.

    Args:
        connection: An open connection.
        cluster_id: Group to read.

    Returns:
        Rows carrying each finding's outcome, class, and owning image.
    """
    return connection.execute(
        """
        SELECT f.id AS finding_id, f.outcome, f.class_name, f.confidence, f.iou,
               i.id AS image_id, i.filename, i.path
        FROM cluster_members cm
        JOIN findings f ON f.id = cm.finding_id
        JOIN images i ON i.id = f.image_id
        WHERE cm.cluster_id = ?
        ORDER BY f.id
        """,
        (cluster_id,),
    ).fetchall()


def load_factors_by_finding(
    connection: sqlite3.Connection, run_id: int
) -> dict[int, list[str]]:
    """Return each failure's attributed factors, keyed by finding id.

    The input to factor-signature grouping. Findings with no attributed factor
    are absent from the mapping rather than present with an empty list — the
    caller supplies the full set of failures, so absence is what marks a
    finding unexplained.

    Args:
        connection: An open connection.
        run_id: Which run to read.

    Returns:
        Mapping of finding id to its factors, each list sorted so that the
        signature built from it does not depend on insertion order.
    """
    rows = connection.execute(
        "SELECT finding_id, factor FROM root_causes WHERE run_id = ? "
        "ORDER BY finding_id, factor",
        (run_id,),
    ).fetchall()

    factors: dict[int, list[str]] = {}
    for row in rows:
        factors.setdefault(row["finding_id"], []).append(row["factor"])
    return factors


@dataclass(frozen=True)
class FactorRateRow:
    """How often one factor appears among failures, against its base rate.

    ``lift`` is ``None`` when it cannot be computed — no correct finding
    carried the factor, so the ratio is undefined rather than very large.
    Presenting that as a number would invite it being read as a measurement.
    """

    run_id: int
    factor: str
    failure_count: int
    failure_total: int
    correct_count: int
    correct_total: int
    lift: float | None
    p_value: float

    @property
    def failure_rate(self) -> float:
        """Share of failures carrying this factor."""
        return self.failure_count / self.failure_total if self.failure_total else 0.0

    @property
    def correct_rate(self) -> float:
        """Share of correct findings carrying this factor."""
        return self.correct_count / self.correct_total if self.correct_total else 0.0


def save_factor_rates(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[tuple[str, int, int, int, int, float | None, float]],
) -> int:
    """Record how often each factor appears in failures and in correct findings.

    Re-running replaces a run's rates for the same factor, matching how the
    other derived tables behave: recomputing is a correction, not a second
    observation (D-021).

    Args:
        connection: An open connection.
        run_id: Owning run.
        entries: Tuples of ``(factor, failure_count, failure_total,
            correct_count, correct_total, lift, p_value)``.

    Returns:
        Number of rows written.
    """
    rows = [
        (
            run_id,
            factor,
            int(failure_count),
            int(failure_total),
            int(correct_count),
            int(correct_total),
            None if lift_value is None else float(lift_value),
            float(p_value),
        )
        for factor, failure_count, failure_total, correct_count, correct_total, (
            lift_value
        ), p_value in entries
    ]
    if rows:
        connection.executemany(
            """
            INSERT INTO factor_rates (
                run_id, factor, failure_count, failure_total,
                correct_count, correct_total, lift, p_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, factor) DO UPDATE SET
                failure_count = excluded.failure_count,
                failure_total = excluded.failure_total,
                correct_count = excluded.correct_count,
                correct_total = excluded.correct_total,
                lift = excluded.lift,
                p_value = excluded.p_value
            """,
            rows,
        )
    return len(rows)


def load_factor_rates(
    connection: sqlite3.Connection, run_id: int
) -> list[FactorRateRow]:
    """Return a run's factor rates, most over-represented in failures first.

    Ordered by lift descending so the factors that actually distinguish
    failures from successes appear before the ones that merely describe the
    dataset. Rows with an undefined lift sort last, since an unknown ratio is
    not evidence of a large one.
    """
    rows = connection.execute(
        """
        SELECT run_id, factor, failure_count, failure_total,
               correct_count, correct_total, lift, p_value
        FROM factor_rates
        WHERE run_id = ?
        ORDER BY lift IS NULL, lift DESC, factor
        """,
        (run_id,),
    ).fetchall()
    return [
        FactorRateRow(
            run_id=row["run_id"],
            factor=row["factor"],
            failure_count=row["failure_count"],
            failure_total=row["failure_total"],
            correct_count=row["correct_count"],
            correct_total=row["correct_total"],
            lift=row["lift"],
            p_value=row["p_value"],
        )
        for row in rows
    ]


def load_failure_ids(connection: sqlite3.Connection, run_id: int) -> list[int]:
    """Return the ids of every finding in a run that represents a mistake.

    Grouping covers failures, not correct findings: a group of things the model
    got right names no problem to act on.
    """
    rows = connection.execute(
        "SELECT id FROM findings WHERE run_id = ? AND outcome != 'correct' "
        "ORDER BY id",
        (run_id,),
    ).fetchall()
    return [int(row["id"]) for row in rows]


# ---------------------------------------------------------------------------
# Recommendations (schema version 7)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecommendationRow:
    """One suggested action, with the evidence that produced it.

    ``actionable`` is derived from ``status`` and stored so ordering is a plain
    column sort. The documented order is
    ``actionable DESC, priority DESC, id`` — actionable advice first, then by
    how many failures it addresses, then by id so repeated reads agree.
    """

    id: int
    run_id: int
    cluster_id: int
    cluster_label: str
    rule: str
    action: str
    rationale: str
    status: str
    actionable: bool
    affected: int
    priority: float


def save_recommendations(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[tuple[int, str, str, str, str, bool, int, float]],
) -> int:
    """Store suggested actions for a run's failure groups.

    Re-running replaces a group's recommendation for the same rule. Advice is
    derived data: recomputing it against fresh evidence is a correction, not a
    second opinion (D-021).

    Args:
        connection: An open connection.
        run_id: Owning run.
        entries: Tuples of ``(cluster_id, rule, action, rationale, status,
            actionable, affected, priority)``.

    Returns:
        Number of rows written.
    """
    rows = [
        (
            run_id,
            int(cluster_id),
            rule,
            action,
            rationale,
            status,
            1 if actionable else 0,
            int(affected),
            float(priority),
        )
        for cluster_id, rule, action, rationale, status, actionable, (
            affected
        ), priority in entries
    ]
    if rows:
        connection.executemany(
            """
            INSERT INTO recommendations (
                run_id, cluster_id, rule, action, rationale,
                status, actionable, affected, priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (cluster_id, rule) DO UPDATE SET
                action = excluded.action,
                rationale = excluded.rationale,
                status = excluded.status,
                actionable = excluded.actionable,
                affected = excluded.affected,
                priority = excluded.priority,
                run_id = excluded.run_id
            """,
            rows,
        )
    return len(rows)


def load_recommendations(
    connection: sqlite3.Connection, run_id: int
) -> list[RecommendationRow]:
    """Return a run's recommendations in the documented display order.

    Actionable advice first, then by how many failures it addresses, then by id.
    Non-actionable rows — ``conflicting`` and ``insufficient_evidence`` — are
    returned too, below the rest. They are the honest part of the output and a
    consumer must not filter them away.
    """
    rows = connection.execute(
        """
        SELECT r.id, r.run_id, r.cluster_id, c.label AS cluster_label,
               r.rule, r.action, r.rationale, r.status,
               r.actionable, r.affected, r.priority
        FROM recommendations r
        JOIN clusters c ON c.id = r.cluster_id
        WHERE r.run_id = ?
        ORDER BY r.actionable DESC, r.priority DESC, r.id
        """,
        (run_id,),
    ).fetchall()
    return [
        RecommendationRow(
            id=row["id"],
            run_id=row["run_id"],
            cluster_id=row["cluster_id"],
            cluster_label=row["cluster_label"],
            rule=row["rule"],
            action=row["action"],
            rationale=row["rationale"],
            status=row["status"],
            actionable=bool(row["actionable"]),
            affected=row["affected"],
            priority=row["priority"],
        )
        for row in rows
    ]


def load_runs_for_model(
    connection: sqlite3.Connection, model_sha256: str
) -> list[int]:
    """Return ids of every run produced by one set of weights, oldest first.

    Replication is only meaningful between runs of the *same* model. Two runs
    with different weights that disagree are not a failed replication, they are
    two different observations.
    """
    rows = connection.execute(
        "SELECT id FROM runs WHERE model_sha256 = ? ORDER BY id", (model_sha256,)
    ).fetchall()
    return [int(row["id"]) for row in rows]


# ---------------------------------------------------------------------------
# Mask findings (schema version 8)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskFindingRow:
    """One finding re-examined at outline level, joined to its box result.

    ``mask_iou`` is ``None`` when the comparison could not be made — no
    predicted outline, or no ground-truth outline to compare against. That is
    distinct from an outline that overlapped nothing, which is ``0.0``.
    """

    finding_id: int
    run_id: int
    outcome: str
    class_name: str
    box_iou: float | None
    mask_iou: float | None
    mask_outcome: str | None
    pred_polygon: list[list[float]] | None


def save_mask_findings(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[tuple[int, float | None, str | None, object]],
) -> int:
    """Store outline-level results for a run's findings.

    Re-running replaces a finding's mask result. Like every derived table here,
    recomputing is a correction rather than a second observation (D-021).

    Args:
        connection: An open connection.
        run_id: Owning run.
        entries: Tuples of ``(finding_id, mask_iou, mask_outcome, polygon)``
            where ``polygon`` is a sequence of points or ``None``.

    Returns:
        Number of rows written.
    """
    rows = [
        (
            finding_id,
            run_id,
            None if mask_iou is None else float(mask_iou),
            mask_outcome,
            None if polygon is None else json.dumps([list(p) for p in polygon]),
        )
        for finding_id, mask_iou, mask_outcome, polygon in entries
    ]
    if rows:
        connection.executemany(
            """
            INSERT INTO mask_findings (
                finding_id, run_id, mask_iou, mask_outcome, pred_polygon
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (finding_id) DO UPDATE SET
                mask_iou = excluded.mask_iou,
                mask_outcome = excluded.mask_outcome,
                pred_polygon = excluded.pred_polygon,
                run_id = excluded.run_id
            """,
            rows,
        )
    return len(rows)


def load_mask_findings(
    connection: sqlite3.Connection, run_id: int, disagreements_only: bool = False
) -> list[MaskFindingRow]:
    """Return a run's outline-level results beside their box-level verdict.

    Args:
        connection: An open connection.
        run_id: Which run to read.
        disagreements_only: Return only findings the box called correct and the
            outline did not. That set is the reason this milestone exists.

    Returns:
        Rows ordered by ascending mask IoU, worst outlines first.
    """
    sql = """
        SELECT m.finding_id, m.run_id, m.mask_iou, m.mask_outcome, m.pred_polygon,
               f.outcome, f.class_name, f.iou AS box_iou
        FROM mask_findings m JOIN findings f ON f.id = m.finding_id
        WHERE m.run_id = ?
    """
    if disagreements_only:
        sql += (
            " AND f.outcome = 'correct' AND m.mask_outcome IS NOT NULL"
            " AND m.mask_outcome != 'correct'"
        )
    sql += " ORDER BY m.mask_iou IS NULL, m.mask_iou, m.finding_id"

    return [
        MaskFindingRow(
            finding_id=row["finding_id"],
            run_id=row["run_id"],
            outcome=row["outcome"],
            class_name=row["class_name"],
            box_iou=row["box_iou"],
            mask_iou=row["mask_iou"],
            mask_outcome=row["mask_outcome"],
            pred_polygon=(
                json.loads(row["pred_polygon"]) if row["pred_polygon"] else None
            ),
        )
        for row in connection.execute(sql, (run_id,)).fetchall()
    ]


# ---------------------------------------------------------------------------
# Evaluation — mAP and AR, from one shared evaluator
# ---------------------------------------------------------------------------
_EVALUATION_METRICS: tuple[str, ...] = (
    "map50_95", "map50", "map75", "map_small", "map_medium", "map_large",
    "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large",
)


@dataclass(frozen=True)
class EvaluationRow:
    """One task's evaluation for one run, with the settings behind it.

    The settings are not decoration. Two mAP figures are comparable only if
    they share a confidence sweep, an IoU range, a detection cap and a ground
    truth, so each is stored beside the number it produced rather than assumed
    from a constant that may since have changed.
    """

    id: int
    run_id: int
    task: str
    evaluator: str
    evaluator_version: str | None
    sweep_confidence: float
    iou_thresholds: str
    max_detections: int
    ground_truth: str
    gt_images: int
    gt_annotations: int
    prediction_count: int
    mask_source: str | None
    metrics: dict[str, float | None]
    created_at: str


def save_evaluation(
    connection: sqlite3.Connection,
    run_id: int,
    task: str,
    evaluator: str,
    evaluator_version: str | None,
    sweep_confidence: float,
    iou_thresholds: str,
    max_detections: int,
    ground_truth: str,
    gt_images: int,
    gt_annotations: int,
    prediction_count: int,
    metrics: Mapping[str, float],
    mask_source: str | None = None,
) -> int:
    """Persist one task's evaluation for a run, replacing any earlier one.

    Replacing rather than accumulating: an evaluation is a statement about a
    run as it stands, and two contradictory statements about the same run and
    task would leave a reader with no way to choose. Re-running deliberately
    supersedes.

    Returns:
        The number of rows written.
    """
    columns = ", ".join(_EVALUATION_METRICS)
    placeholders = ", ".join("?" for _ in _EVALUATION_METRICS)
    connection.execute(
        f"""
        INSERT INTO run_evaluations (
            run_id, task, evaluator, evaluator_version, sweep_confidence,
            iou_thresholds, max_detections, ground_truth, gt_images,
            gt_annotations, prediction_count, mask_source, {columns}, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {placeholders}, ?)
        ON CONFLICT (run_id, task) DO UPDATE SET
            evaluator = excluded.evaluator,
            evaluator_version = excluded.evaluator_version,
            sweep_confidence = excluded.sweep_confidence,
            iou_thresholds = excluded.iou_thresholds,
            max_detections = excluded.max_detections,
            ground_truth = excluded.ground_truth,
            gt_images = excluded.gt_images,
            gt_annotations = excluded.gt_annotations,
            prediction_count = excluded.prediction_count,
            mask_source = excluded.mask_source,
            {", ".join(f"{name} = excluded.{name}" for name in _EVALUATION_METRICS)},
            created_at = excluded.created_at
        """,
        (
            run_id, task, evaluator, evaluator_version, sweep_confidence,
            iou_thresholds, max_detections, ground_truth, gt_images,
            gt_annotations, prediction_count, mask_source,
            *(metrics.get(name) for name in _EVALUATION_METRICS),
            _timestamp(),
        ),
    )
    return 1


def load_evaluations(
    connection: sqlite3.Connection, run_id: int
) -> list[EvaluationRow]:
    """Return every task evaluated for a run, or an empty list."""
    rows = connection.execute(
        "SELECT * FROM run_evaluations WHERE run_id = ? ORDER BY task", (run_id,)
    ).fetchall()
    return [
        EvaluationRow(
            id=row["id"],
            run_id=row["run_id"],
            task=row["task"],
            evaluator=row["evaluator"],
            evaluator_version=row["evaluator_version"],
            sweep_confidence=row["sweep_confidence"],
            iou_thresholds=row["iou_thresholds"],
            max_detections=row["max_detections"],
            ground_truth=row["ground_truth"],
            gt_images=row["gt_images"],
            gt_annotations=row["gt_annotations"],
            prediction_count=row["prediction_count"],
            mask_source=row["mask_source"],
            metrics={name: row[name] for name in _EVALUATION_METRICS},
            created_at=row["created_at"],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Benchmark — latency, memory and checkpoint size, per device
# ---------------------------------------------------------------------------
_BENCHMARK_FIELDS: tuple[str, ...] = (
    "torch_threads", "cold_start_s",
    "weights_load_s", "latency_mean_ms", "latency_median_ms", "latency_p90_ms",
    "latency_min_ms", "fps", "rss_baseline_mb", "rss_after_load_mb",
    "rss_peak_mb", "gpu_allocated_mb", "gpu_driver_mb", "checkpoint_bytes",
    "host_platform",
)


@dataclass(frozen=True)
class BenchmarkRow:
    """One controlled compute measurement of a run, on one device.

    ``device`` is never optional. A latency without it cannot be compared with
    anything, and comparing across devices is the mistake this field exists to
    make impossible to commit by accident.
    """

    id: int
    run_id: int
    device: str
    image_size: int
    warmup_images: int
    measured_images: int
    created_at: str
    measurements: dict[str, float | int | str | None]


def save_benchmark(
    connection: sqlite3.Connection,
    run_id: int,
    device: str,
    image_size: int,
    warmup_images: int,
    measured_images: int,
    measurements: Mapping[str, float | int | str | None],
) -> int:
    """Persist one device's benchmark of a run, replacing any earlier one.

    ``device``, ``warmup_images`` and ``measured_images`` are named parameters
    rather than entries in ``measurements`` because a latency is not a claim
    without them. A mean over three images and a mean over thirty are different
    statements, and one measured without a warm-up is mostly reporting kernel
    compilation. Requiring them here means a caller cannot omit them by
    forgetting a dictionary key.

    Raises:
        ValueError: If ``device`` is empty, or the image counts are negative or
            leave nothing measured.
    """
    if not device or not device.strip():
        raise ValueError(
            "A benchmark must record the device it ran on; latency and memory "
            "mean nothing without it."
        )
    if warmup_images < 0 or measured_images < 1:
        raise ValueError(
            "A benchmark needs a non-negative warm-up and at least one "
            f"measured image; got warmup={warmup_images}, "
            f"measured={measured_images}."
        )

    assignments = ", ".join(f"{name} = excluded.{name}" for name in _BENCHMARK_FIELDS)
    connection.execute(
        f"""
        INSERT INTO run_benchmarks (
            run_id, device, image_size, warmup_images, measured_images,
            {", ".join(_BENCHMARK_FIELDS)}, created_at
        ) VALUES (?, ?, ?, ?, ?, {", ".join("?" for _ in _BENCHMARK_FIELDS)}, ?)
        ON CONFLICT (run_id, device, image_size) DO UPDATE SET
            warmup_images = excluded.warmup_images,
            measured_images = excluded.measured_images,
            {assignments},
            created_at = excluded.created_at
        """,
        (
            run_id, device.strip(), image_size, warmup_images, measured_images,
            *(measurements.get(name) for name in _BENCHMARK_FIELDS),
            _timestamp(),
        ),
    )
    return 1


def load_benchmarks(
    connection: sqlite3.Connection, run_id: int
) -> list[BenchmarkRow]:
    """Return every benchmark recorded for a run, or an empty list."""
    rows = connection.execute(
        "SELECT * FROM run_benchmarks WHERE run_id = ? ORDER BY device, image_size",
        (run_id,),
    ).fetchall()
    return [
        BenchmarkRow(
            id=row["id"],
            run_id=row["run_id"],
            device=row["device"],
            image_size=row["image_size"],
            warmup_images=row["warmup_images"],
            measured_images=row["measured_images"],
            created_at=row["created_at"],
            measurements={name: row[name] for name in _BENCHMARK_FIELDS},
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
#: Pipeline stages, in the order the worker runs them. The first is the only
#: one that creates a run; every later stage enriches it, which is why a job
#: becomes viewable in the dashboard before it has finished.
JOB_STAGES: tuple[str, ...] = (
    "inference+diagnosis",
    "evaluation",
    "mask-diagnosis",
    "root-cause",
    "clustering",
    "recommendations",
    "image-diagnosis",
    "relations",
    "explainability",
)

#: Terminal states. A job in any of these will not change again.
JOB_TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class JobRecord:
    """One self-service analysis, as stored."""

    id: int
    token: str
    status: str
    stage: str | None
    run_id: int | None
    detector: str
    split: str
    model_name: str
    dataset_name: str
    workspace: str
    image_size: int
    confidence: float
    error: str | None
    log_tail: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


def create_job(
    connection: sqlite3.Connection,
    *,
    token: str,
    detector: str,
    split: str,
    model_name: str,
    dataset_name: str,
    workspace: str,
    image_size: int,
    confidence: float,
) -> int:
    """Record a queued job and return its row id.

    The job exists before any work starts so that a caller always has
    something to poll. A job that fails in its first second is still a job with
    a status and a reason, not a request that vanished.
    """
    cursor = connection.execute(
        """
        INSERT INTO jobs (
            token, status, stage, run_id, detector, split, model_name,
            dataset_name, workspace, image_size, confidence, created_at
        ) VALUES (?, 'queued', NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            token, detector, split, model_name, dataset_name, workspace,
            image_size, confidence, _timestamp(),
        ),
    )
    return int(cursor.lastrowid or 0)


def update_job(
    connection: sqlite3.Connection,
    token: str,
    **fields: Any,
) -> None:
    """Patch one job's mutable columns.

    Only the columns a running job legitimately changes are writable here. A
    caller cannot rewrite the workspace or the detector through this, because
    those describe what was *asked for* and rewriting them would make the
    record disagree with what actually ran.
    """
    writable = {
        "status", "stage", "run_id", "error", "log_tail",
        "started_at", "finished_at",
    }
    unknown = set(fields) - writable
    if unknown:
        raise ValueError(f"Not writable on a job: {', '.join(sorted(unknown))}")
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE jobs SET {assignments} WHERE token = ?",
        (*fields.values(), token),
    )


def load_job(connection: sqlite3.Connection, token: str) -> JobRecord | None:
    """Return one job by token, or None if there is no such job."""
    row = connection.execute("SELECT * FROM jobs WHERE token = ?", (token,)).fetchone()
    return _job_record(row) if row is not None else None


def load_jobs(connection: sqlite3.Connection, limit: int = 50) -> list[JobRecord]:
    """Return recent jobs, newest first."""
    rows = connection.execute(
        "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_job_record(row) for row in rows]


def _job_record(row: sqlite3.Row) -> JobRecord:
    """Build a :class:`JobRecord` from a row."""
    return JobRecord(
        id=row["id"],
        token=row["token"],
        status=row["status"],
        stage=row["stage"],
        run_id=row["run_id"],
        detector=row["detector"],
        split=row["split"],
        model_name=row["model_name"],
        dataset_name=row["dataset_name"],
        workspace=row["workspace"],
        image_size=row["image_size"],
        confidence=row["confidence"],
        error=row["error"],
        log_tail=row["log_tail"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def reconcile_stale_jobs(connection: sqlite3.Connection) -> int:
    """Fail jobs left mid-flight by a service that stopped, and say how many.

    A job's status lives in the database while the work runs in this process.
    If the process ends — a restart, a crash, a laptop closing — the row stays
    ``running`` and nothing will ever move it, so the UI polls a job that no
    longer exists for as long as anyone leaves the tab open.

    Called at start-up, when no job can legitimately be in flight yet: this
    process has just begun and the queue is empty, so anything already marked
    running belongs to a previous life and is known to be dead.

    Returns:
        How many jobs were closed out.
    """
    stale = connection.execute(
        "SELECT token FROM jobs WHERE status IN ('queued', 'running')"
    ).fetchall()
    for row in stale:
        update_job(
            connection,
            row["token"],
            status="failed",
            error=(
                "The analysis service stopped while this job was running, so "
                "it did not finish. Any results it had already written are "
                "kept. Start the analysis again to complete it."
            ),
            finished_at=_timestamp(),
        )
    if stale:
        logger.warning(
            "Closed %d job(s) left running by a previous process", len(stale)
        )
    return len(stale)


#: Tables a run owns, in the order a reader would want them reported.
#:
#: Used only to *describe* what a delete will remove. The delete itself does not
#: walk this list — see :func:`delete_run`.
#:
#: ``cluster_members`` and ``image_coverage`` reach their run indirectly, through
#: a cluster and a diagnosis: both record a relationship and carry no ``run_id``
#: of their own. :func:`run_footprint` counts them through that join, and a
#: table added here without one would raise on its first count.
RUN_OWNED_TABLES: tuple[str, ...] = (
    "images",
    "image_diagnoses",
    "image_coverage",
    "findings",
    "mask_findings",
    "root_causes",
    "clusters",
    "cluster_members",
    "recommendations",
    "finding_relations",
    "heatmaps",
    "embeddings",
    "factor_rates",
    "run_evaluations",
    "run_benchmarks",
)


def has_table(connection: sqlite3.Connection, name: str) -> bool:
    """Report whether an optional table exists in this database.

    Only ``runs``, ``images`` and ``findings`` are guaranteed. A database saved
    before a later schema version simply lacks the rest, and the surface that
    needs one should return empty rather than fail (SCHEMA.md §6). Public so
    that every read-only consumer asks the same question the same way.
    """
    return _table_exists(connection, name)


def load_job_for_run(connection: sqlite3.Connection, run_id: int) -> JobRecord | None:
    """Return the job that produced ``run_id``, or ``None`` for a CLI run.

    A run started from the browser is linked to its job by ``jobs.run_id``,
    and that row is the only place the detector family and the human-readable
    model and dataset names are recorded. A run produced by the CLI has no
    job, and a consumer must report the family as unknown rather than guess —
    detecting it from the checkpoint means loading it, which a read-only
    surface must never do.
    """
    if not _table_exists(connection, "jobs"):
        return None
    row = connection.execute(
        "SELECT * FROM jobs WHERE run_id = ? ORDER BY id DESC LIMIT 1", (run_id,)
    ).fetchone()
    return _job_record(row) if row is not None else None


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    """Return whether a table is present.

    Local rather than imported from ``model_doctor.app.api``: storage sits below the API
    in
    the layering documented at the top of this module, and reaching upward for
    a three-line helper would invert that for no gain.
    """
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def run_footprint(connection: sqlite3.Connection, run_id: int) -> dict[str, int]:
    """Return how many rows in each table belong to ``run_id``.

    Exists so a caller can say what a delete will destroy *before* destroying
    it. A confirmation that names no quantity is not informed consent — "delete
    this run?" and "delete this run, its 231 findings and 73 heatmaps?" ask
    different questions.

    ``cluster_members`` is counted through its cluster, because it is the one
    owned table with no ``run_id`` of its own.

    Missing tables are reported as ``0`` rather than raising: the optional
    passes create their tables on first use, so a run analysed before one
    existed legitimately has none.
    """
    counts: dict[str, int] = {}
    for table in RUN_OWNED_TABLES:
        if not _table_exists(connection, table):
            counts[table] = 0
            continue
        if table == "cluster_members":
            row = connection.execute(
                "SELECT COUNT(*) FROM cluster_members WHERE cluster_id IN "
                "(SELECT id FROM clusters WHERE run_id = ?)",
                (run_id,),
            ).fetchone()
        elif table == "image_coverage":
            # Reached through its diagnosis, like cluster_members through its
            # cluster: both record a relationship and carry no run of their own.
            row = connection.execute(
                "SELECT COUNT(*) FROM image_coverage WHERE image_diagnosis_id IN "
                "(SELECT id FROM image_diagnoses WHERE run_id = ?)",
                (run_id,),
            ).fetchone()
        else:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
            ).fetchone()
        counts[table] = int(row[0])
    return counts


def delete_run(connection: sqlite3.Connection, run_id: int) -> dict[str, int]:
    """Delete one run and everything it owns, returning what was removed.

    **One statement, not twelve.** Every owned table declares
    ``ON DELETE CASCADE`` against ``runs`` or against ``findings``/``clusters``,
    and :func:`connect` enables ``PRAGMA foreign_keys``, so deleting the run
    row removes the rest. Hand-writing a delete per table would duplicate a
    rule the schema already states, and the copy would drift the first time a
    milestone adds a table — silently leaving orphans that only surface as
    wrong counts much later.

    ``jobs`` is deliberately not cascaded. Its foreign key is ``SET NULL``, so
    the record of an analysis having been requested and run survives the
    deletion of its output. The job is history; the run is data.

    Raises:
        ResourceNotFoundError: If no such run exists, so a caller cannot
            report a successful delete of something that was never there.
    """
    row = connection.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ResourceNotFoundError(f"No run with id {run_id}.")

    removed = run_footprint(connection, run_id)
    with connection:
        # Asserted rather than assumed: this connection is the one doing the
        # delete, and without the pragma the cascade silently does nothing,
        # leaving every child row orphaned behind a deleted parent.
        enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        if not enabled:
            raise ModelDoctorError(
                "Refusing to delete a run without foreign keys enabled: the "
                "cascade would not fire and every owned row would be orphaned."
            )
        connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    removed["runs"] = 1
    logger.info("Deleted run %d and %d owned row(s)", run_id, sum(removed.values()) - 1)
    return removed


def heatmap_paths(connection: sqlite3.Connection, run_id: int) -> list[str]:
    """Return the heatmap files recorded for ``run_id``.

    Read before a delete, so a caller can remove the images the run wrote.
    Returns what the database *records* rather than what a path convention
    implies: the rows are the only authority on which files this run owns, and
    a directory named after a run could hold anything.

    The caller is responsible for refusing paths outside its own output
    directory — this returns stored strings and makes no claim about them.
    """
    if not _table_exists(connection, "heatmaps"):
        return []
    rows = connection.execute(
        "SELECT path FROM heatmaps WHERE run_id = ? AND path IS NOT NULL", (run_id,)
    ).fetchall()
    return [str(row[0]) for row in rows]


# ---------------------------------------------------------------------------
# Image-level diagnosis (schema version 11)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageCoverageRow:
    """How much of one annotated object a single prediction covers.

    Attributes:
        truth_finding_id: The finding carrying the ground-truth object.
        pred_finding_id: The finding carrying the prediction. This is often a
            *different* finding — that is the whole point: it is how one
            prediction spanning two objects becomes visible.
        coverage: Share of the ground-truth mask intersected, in ``[0, 1]``.
            Measured on masks, never boxes: an axis-aligned box around a
            diagonal object sweeps across its neighbours and manufactures
            overlap that the objects do not have.
    """

    truth_finding_id: int
    pred_finding_id: int
    coverage: float


@dataclass(frozen=True)
class ImageDiagnosisRow:
    """What shape the model's mistake took on one image.

    A second lens over the same findings. The finding-level tables say which
    object failed; this says whether the image was handled cleanly, whether one
    prediction was stretched across several objects, or several predictions
    landed on one.

    Attributes:
        verdict: One of :data:`IMAGE_VERDICTS`.
        cover_hit: Coverage at which a prediction counted as finding an object,
            stored because merge and split counts are sensitive to it.
        cover_miss: Coverage below which an object counted as untouched.
        method: How geometry was compared. ``"mask"`` is the only value; the
            column exists so a later variant cannot be confused with this one.
        coverage: The pairs behind the verdict, strongest first.
    """

    id: int
    run_id: int
    image_id: int
    gt_count: int
    pred_count: int
    outcomes: dict[str, int]
    verdict: str
    merged: bool
    split: bool
    objects_untouched: int
    objects_partial: int
    predictions_partial: int
    predictions_on_nothing: int
    cover_hit: float
    cover_miss: float
    method: str
    created_at: str
    coverage: tuple[ImageCoverageRow, ...] = ()


def save_image_diagnoses(
    connection: sqlite3.Connection,
    run_id: int,
    entries: Iterable[tuple[int, Mapping[str, Any], Sequence[ImageCoverageRow]]],
    *,
    cover_hit: float,
    cover_miss: float,
    method: str = "mask",
) -> int:
    """Persist one verdict per image, with the pairs that produced it.

    Replaces any previous diagnosis for the same run, because this is a pass
    over an already-saved run: re-running it with a different threshold must
    not leave two contradictory verdicts for one image.

    Args:
        connection: An open connection.
        run_id: Run being diagnosed.
        entries: ``(image_id, fields, coverage)`` where ``fields`` carries the
            counts and verdict, and ``coverage`` the pairwise measurements.
        cover_hit: Threshold recorded with every row.
        cover_miss: Threshold recorded with every row.
        method: Geometry used. Only ``"mask"`` is produced today.

    Returns:
        How many image diagnoses were written.
    """
    connection.execute(
        "DELETE FROM image_diagnoses WHERE run_id = ?", (run_id,)
    )  # image_coverage cascades
    written = 0
    now = _timestamp()
    for image_id, fields, coverage in entries:
        cursor = connection.execute(
            """
            INSERT INTO image_diagnoses (
                run_id, image_id, gt_count, pred_count, correct, false_negative,
                false_positive, poor_localization, wrong_class, verdict, merged,
                split, objects_untouched, objects_partial, predictions_partial,
                predictions_on_nothing, cover_hit, cover_miss, method, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, image_id, fields["gt_count"], fields["pred_count"],
                fields["correct"], fields["false_negative"],
                fields["false_positive"], fields["poor_localization"],
                fields["wrong_class"], fields["verdict"],
                int(fields["merged"]), int(fields["split"]),
                fields["objects_untouched"], fields.get("objects_partial", 0),
                fields.get("predictions_partial", 0),
                fields["predictions_on_nothing"],
                cover_hit, cover_miss, method, now,
            ),
        )
        diagnosis_id = cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO image_coverage (
                image_diagnosis_id, truth_finding_id, pred_finding_id, coverage
            ) VALUES (?,?,?,?)
            """,
            [(diagnosis_id, c.truth_finding_id, c.pred_finding_id, c.coverage)
             for c in coverage],
        )
        written += 1
    return written


def load_image_diagnoses(
    connection: sqlite3.Connection,
    run_id: int,
    verdict: str | None = None,
    with_coverage: bool = False,
) -> list[ImageDiagnosisRow]:
    """Return one row per image, optionally filtered to a single verdict.

    ``with_coverage`` is off by default: the pairs are the evidence behind a
    verdict and are wanted when inspecting one image, not when listing a run.
    """
    if not _table_exists(connection, "image_diagnoses"):
        return []
    sql = "SELECT * FROM image_diagnoses WHERE run_id = ?"
    params: list[Any] = [run_id]
    if verdict is not None:
        sql += " AND verdict = ?"
        params.append(verdict)
    rows = connection.execute(sql + " ORDER BY image_id", params).fetchall()

    pairs: dict[int, list[ImageCoverageRow]] = {}
    if with_coverage and rows and _table_exists(connection, "image_coverage"):
        ids = tuple(int(r["id"]) for r in rows)
        placeholders = ",".join("?" * len(ids))
        for c in connection.execute(
            "SELECT * FROM image_coverage WHERE image_diagnosis_id IN "
            f"({placeholders}) ORDER BY coverage DESC",
            ids,
        ):
            pairs.setdefault(int(c["image_diagnosis_id"]), []).append(
                ImageCoverageRow(
                    truth_finding_id=int(c["truth_finding_id"]),
                    pred_finding_id=int(c["pred_finding_id"]),
                    coverage=float(c["coverage"]),
                )
            )
    return [_image_diagnosis_row(r, tuple(pairs.get(int(r["id"]), ()))) for r in rows]


def load_image_coverage(
    connection: sqlite3.Connection, run_id: int, image_id: int
) -> list[ImageCoverageRow]:
    """Return the object/prediction pairs measured on one image, strongest first.

    Both tables are checked: the query joins them, so the presence of one
    without the other — reachable on a partially migrated database — would
    otherwise raise instead of degrading to "not measured" (SCHEMA.md §6).
    """
    if not _table_exists(connection, "image_coverage") or not _table_exists(
        connection, "image_diagnoses"
    ):
        return []
    rows = connection.execute(
        """
        SELECT c.* FROM image_coverage c
        JOIN image_diagnoses d ON d.id = c.image_diagnosis_id
        WHERE d.run_id = ? AND d.image_id = ?
        ORDER BY c.coverage DESC
        """,
        (run_id, image_id),
    ).fetchall()
    return [
        ImageCoverageRow(
            truth_finding_id=int(r["truth_finding_id"]),
            pred_finding_id=int(r["pred_finding_id"]),
            coverage=float(r["coverage"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Finding relations (schema version 13)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FindingRelationRow:
    """One measured relationship between findings.

    **Additive evidence, never a reclassification.** The finding keeps the
    outcome the matcher gave it; this says something further about *why* that
    outcome looks the way it does — that the missed object is covered by a
    prediction the matcher assigned elsewhere, or that a box-based verdict
    disagrees with the mask underneath it.

    Attributes:
        finding_id: The finding this is attributed to. Which side of a
            relationship gets the row is fixed per relation, so one
            relationship can never be counted twice.
        relation: One of :data:`RELATIONS`.
        direction: Which way a two-sided relation went, or ``None``.
        partner_finding_id: The other finding involved, or ``None`` for a
            relation a finding has with itself.
        value: The continuous measurement, in ``[0, 1]``.
        best_coverage: Strongest coverage behind the relation, where that is a
            different quantity from ``value``.
        span: How many objects are involved, for a relation that can span more
            than one. Multiplicity is recorded here rather than as extra rows.
        threshold: The value ``qualifies`` was decided at, or ``None`` when the
            relation has no threshold of its own.
        coverage_floor: The second bound a two-part rule was decided at, or
            ``None``. Stored beside ``threshold`` for the same reason
            ``image_diagnoses`` stores both of its own: a row must carry every
            number that produced it, or a later reader cannot tell why a
            measurement did or did not qualify.
        qualifies: Whether the measurement met the rule. What that rule *means*
            is the caller's to name; this only records that it was met.
        cover_hit: The run's stored coverage threshold, carried so a row is
            never read as though it had been measured at another.
        method: Geometry used. Only ``"mask"`` is produced today.
    """

    finding_id: int
    relation: str
    value: float
    qualifies: bool
    cover_hit: float
    direction: str | None = None
    partner_finding_id: int | None = None
    best_coverage: float | None = None
    span: int | None = None
    threshold: float | None = None
    coverage_floor: float | None = None
    method: str = "mask"


def save_finding_relations(
    connection: sqlite3.Connection,
    run_id: int,
    rows: Iterable[FindingRelationRow],
) -> int:
    """Replace this run's relations with the ones given.

    Replaces rather than appends, for the same reason
    :func:`save_image_diagnoses` does: this is a pass over an already-saved
    run, and re-running it at a different threshold must not leave two
    contradictory rows for one finding.

    Writes to ``finding_relations`` and nothing else. No finding, outcome,
    metric, factor rate, cluster or image verdict is touched by this call.

    Args:
        connection: An open connection.
        run_id: Run whose relations these are.
        rows: The relations to store.

    Returns:
        How many rows were written.
    """
    connection.execute("DELETE FROM finding_relations WHERE run_id = ?", (run_id,))
    stamp = _timestamp()
    written = 0
    for row in rows:
        connection.execute(
            """
            INSERT INTO finding_relations (
                run_id, finding_id, relation, direction, partner_finding_id,
                value, best_coverage, span, threshold, coverage_floor,
                qualifies, cover_hit, method, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                row.finding_id,
                row.relation,
                row.direction,
                row.partner_finding_id,
                row.value,
                row.best_coverage,
                row.span,
                row.threshold,
                row.coverage_floor,
                1 if row.qualifies else 0,
                row.cover_hit,
                row.method,
                stamp,
            ),
        )
        written += 1
    return written


def load_finding_relations(
    connection: sqlite3.Connection,
    run_id: int,
    relation: str | None = None,
) -> list[FindingRelationRow]:
    """Return a run's relations, ordered by finding then relation.

    Empty for a run analysed before the pass existed, which a caller must
    report as not measured rather than as "no relationships found" — the same
    distinction ``images.error`` exists to preserve (SCHEMA.md §6).
    """
    if not _table_exists(connection, "finding_relations"):
        return []
    clauses = ["run_id = ?"]
    params: list[Any] = [run_id]
    if relation is not None:
        clauses.append("relation = ?")
        params.append(relation)
    sql = (
        "SELECT * FROM finding_relations WHERE "
        + " AND ".join(clauses)
        + " ORDER BY finding_id, relation"
    )
    rows = connection.execute(sql, params).fetchall()
    columns = set(rows[0].keys()) if rows else set()
    return [
        FindingRelationRow(
            finding_id=int(r["finding_id"]),
            relation=str(r["relation"]),
            value=float(r["value"]),
            qualifies=bool(r["qualifies"]),
            cover_hit=float(r["cover_hit"]),
            direction=r["direction"],
            partner_finding_id=(
                None
                if r["partner_finding_id"] is None
                else int(r["partner_finding_id"])
            ),
            best_coverage=(
                None if r["best_coverage"] is None else float(r["best_coverage"])
            ),
            span=None if r["span"] is None else int(r["span"]),
            threshold=None if r["threshold"] is None else float(r["threshold"]),
            # Absent from databases written before schema 14, so read
            # defensively rather than by subscript.
            coverage_floor=(
                float(r["coverage_floor"])
                if "coverage_floor" in columns and r["coverage_floor"] is not None
                else None
            ),
            method=str(r["method"]),
        )
        for r in rows
    ]


def _image_diagnosis_row(
    row: sqlite3.Row, coverage: tuple[ImageCoverageRow, ...]
) -> ImageDiagnosisRow:
    """Build a record from a stored row."""
    return ImageDiagnosisRow(
        id=int(row["id"]),
        run_id=int(row["run_id"]),
        image_id=int(row["image_id"]),
        gt_count=int(row["gt_count"]),
        pred_count=int(row["pred_count"]),
        outcomes={
            "correct": int(row["correct"]),
            "false_negative": int(row["false_negative"]),
            "false_positive": int(row["false_positive"]),
            "poor_localization": int(row["poor_localization"]),
            "wrong_class": int(row["wrong_class"]),
        },
        verdict=str(row["verdict"]),
        merged=bool(row["merged"]),
        split=bool(row["split"]),
        objects_untouched=int(row["objects_untouched"]),
        objects_partial=int(row["objects_partial"]),
        predictions_partial=int(row["predictions_partial"]),
        predictions_on_nothing=int(row["predictions_on_nothing"]),
        cover_hit=float(row["cover_hit"]),
        cover_miss=float(row["cover_miss"]),
        method=str(row["method"]),
        created_at=str(row["created_at"]),
        coverage=coverage,
    )
