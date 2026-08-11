"""SQLite persistence for diagnosis results.

A diagnosis that exists only in memory dies with the process. This module makes
it durable, queryable, and — most importantly — **available to code that is not
this codebase**. The dashboard is being built by a second developer against the
database, not against these Python objects, so the schema is a published
interface. See ``docs/SCHEMA.md``, which is the contract; this module is one
implementation of it.

**Why this lives in ``app/`` and not ``utils/``.** Persistence needs the
diagnosis domain types — ``Finding``, ``Outcome``, ``ImageDiagnosis`` — which
live in ``app.diagnosis``. A ``utils`` module importing from ``app`` would be
the first upward dependency in the project and would break the layering every
other module has held to. The direction is one-way and must stay that way::

    app/storage.py → app/diagnosis.py → app/inference.py → utils/* → config.py

``app.diagnosis`` must never import this module. Callers orchestrate: diagnose
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
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import config
from app.diagnosis import DatasetDiagnosis, Finding, ImageDiagnosis, Outcome
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Bumped when the schema changes in a way that existing readers must know
# about. Recorded in the database so a consumer can detect a mismatch instead
# of failing on a missing column.
SCHEMA_VERSION: int = 4

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
        image_size              INTEGER NOT NULL
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


@dataclass(frozen=True)
class FindingRecord:
    """A finding as stored, row-shaped.

    Deliberately mirrors the table rather than reconstructing
    :class:`~app.diagnosis.Finding`. Tests assert that every column survives a
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
    try:
        initialise_database(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialise_database(connection: sqlite3.Connection) -> None:
    """Create tables and indexes if they are not already present.

    Idempotent: every statement is ``IF NOT EXISTS``, so calling this on an
    existing database is a no-op and opening a database is always safe.
    """
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)

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
            image_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    )


def load_run(connection: sqlite3.Connection, run_id: int) -> RunRecord | None:
    """Return one run by id, or ``None`` if it does not exist."""
    row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row is not None else None


def list_runs(connection: sqlite3.Connection) -> list[RunRecord]:
    """Return every run, newest first."""
    rows = connection.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [_row_to_run(row) for row in rows]


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
