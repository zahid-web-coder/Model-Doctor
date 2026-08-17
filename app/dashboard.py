"""Provide an interactive, read-only explorer for saved diagnosis runs.

The dashboard queries the published SQLite schema instead of the diagnosis
implementation, which keeps its interface stable as the backend evolves.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

OUTCOME_LABELS: Final = {
    "correct": "Correct",
    "wrong_class": "Wrong class",
    "poor_localization": "Poor localization",
    "false_positive": "False positive",
    "false_negative": "False negative",
}
FAILURE_OUTCOMES: Final = tuple(
    outcome for outcome in OUTCOME_LABELS if outcome != "correct"
)
OUTCOME_COLORS: Final = {
    "correct": "#2dd4bf",
    "wrong_class": "#f59e0b",
    "poor_localization": "#a78bfa",
    "false_positive": "#fb7185",
    "false_negative": "#f97316",
}
PREDICTION_COLOR: Final = "#fb7185"
GROUND_TRUTH_COLOR: Final = "#2dd4bf"
REQUIRED_TABLES: Final = frozenset({"runs", "images", "findings"})
OPTIONAL_TABLES: Final = frozenset(
    {"heatmaps", "root_causes", "clusters", "cluster_members"}
)


class DashboardDataError(RuntimeError):
    """Explain why a database cannot safely be rendered by the dashboard."""


@dataclass(frozen=True)
class RunSummary:
    """Hold run-level values that are independent of the active UI filter."""

    image_count: int
    processed_image_count: int
    errored_image_count: int
    finding_count: int
    failure_count: int


def default_database_path() -> Path:
    """Return the schema's standard database location within this checkout."""
    return Path(__file__).resolve().parents[1] / "db" / "model_doctor.db"


@contextmanager
def read_connection(database: Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database in read-only mode to protect diagnosis history.

    Args:
        database: Existing database file to inspect.

    Yields:
        A connection that cannot create or modify the database.

    Raises:
        DashboardDataError: If SQLite cannot open the supplied database.
    """
    try:
        connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro",
            uri=True,
        )
    except sqlite3.Error as error:
        message = f"Could not open the diagnosis database at {database}."
        raise DashboardDataError(message) from error

    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def validate_database(database: Path) -> None:
    """Confirm that a saved diagnosis database satisfies the published contract.

    Args:
        database: File selected in the dashboard sidebar.

    Raises:
        DashboardDataError: If the path is absent, unreadable, or lacks the
            ``runs``, ``images``, and ``findings`` tables.
    """
    if not database.is_file():
        message = (
            f"No diagnosis database exists at {database}. Run diagnosis with "
            "`--save`, or select the database created by an earlier run."
        )
        raise DashboardDataError(message)

    try:
        with read_connection(database) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
    except sqlite3.Error as error:
        message = f"The file at {database} is not a readable SQLite database."
        raise DashboardDataError(message) from error

    table_names = {str(row["name"]) for row in rows}
    missing_tables = REQUIRED_TABLES - table_names
    if missing_tables:
        missing_text = ", ".join(sorted(missing_tables))
        message = (
            "The selected database does not match Model Doctor's schema. "
            f"Missing table(s): {missing_text}."
        )
        raise DashboardDataError(message)


def query_rows(
    database: Path,
    query: str,
    parameters: Sequence[object] = (),
) -> list[dict[str, object]]:
    """Run one read-only query after checking the dashboard data contract.

    Args:
        database: Saved diagnosis database to query.
        query: Parameterised SQL statement.
        parameters: Values bound to the statement placeholders.

    Returns:
        Rows represented as ordinary dictionaries for chart and UI rendering.

    Raises:
        DashboardDataError: If the query cannot run against the selected file.
    """
    validate_database(database)
    try:
        with read_connection(database) as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
    except sqlite3.Error as error:
        message = "The selected database could not answer a dashboard query."
        raise DashboardDataError(message) from error
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Cached data-loading functions
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def available_tables(database: Path) -> frozenset[str]:
    """Report which optional schema tables the selected database provides.

    Heatmaps and root causes arrived in later schema versions. A database saved
    before those milestones still satisfies the required contract enforced by
    ``validate_database``, so the surfaces that consume them degrade to empty
    instead of failing the whole page.

    Args:
        database: Saved diagnosis database to inspect.

    Returns:
        The subset of ``OPTIONAL_TABLES`` this database actually contains.
    """
    rows = query_rows(
        database,
        "SELECT name FROM sqlite_master WHERE type = 'table'",
    )
    return OPTIONAL_TABLES & {str(row["name"]) for row in rows}


@st.cache_data(show_spinner=False)
def load_runs(database: Path) -> list[dict[str, object]]:
    """List runs newest first with enough context to distinguish them.

    Args:
        database: Saved diagnosis database to query.

    Returns:
        Run rows ordered from most to least recent identifier.
    """
    return query_rows(
        database,
        """
        SELECT id, created_at, model_path, model_sha256, dataset_yaml, split,
               confidence_threshold, match_iou_threshold,
               localization_iou_floor, image_size
        FROM runs
        ORDER BY id DESC
        """,
    )


@st.cache_data(show_spinner=False)
def load_run_summary(
    database: Path, run_id: int, min_confidence: float = 0.0
) -> RunSummary:
    """Compute transparent, non-mAP totals for a single diagnosis run.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        min_confidence: Minimum confidence threshold.

    Returns:
        Totals for images and findings in the run.
    """
    image_row = query_rows(
        database,
        """
        SELECT COUNT(*) AS image_count,
               SUM(CASE WHEN error IS NULL THEN 1 ELSE 0 END)
                   AS processed_image_count,
               SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)
                   AS errored_image_count
        FROM images
        WHERE run_id = ?
        """,
        (run_id,),
    )[0]
    finding_row = query_rows(
        database,
        """
        SELECT COUNT(*) AS finding_count,
               SUM(CASE WHEN outcome != 'correct' THEN 1 ELSE 0 END)
                   AS failure_count
        FROM findings
        WHERE run_id = ? AND (confidence IS NULL OR confidence >= ?)
        """,
        (run_id, min_confidence),
    )[0]
    return RunSummary(
        image_count=int(image_row["image_count"] or 0),
        processed_image_count=int(image_row["processed_image_count"] or 0),
        errored_image_count=int(image_row["errored_image_count"] or 0),
        finding_count=int(finding_row["finding_count"] or 0),
        failure_count=int(finding_row["failure_count"] or 0),
    )


@st.cache_data(show_spinner=False)
def load_outcome_counts(
    database: Path,
    run_id: int,
    outcomes: Sequence[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Return diagnosis outcome counts, optionally narrowed by the UI filter.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Outcomes to include. ``None`` retains every schema outcome.
        min_confidence: Minimum confidence threshold.

    Returns:
        Outcome and count rows sorted by the schema's display order.
    """
    if outcomes is not None and not outcomes:
        return []

    filter_sql = ""
    parameters: list[object] = [run_id]
    if outcomes is not None:
        placeholders = ", ".join("?" for _ in outcomes)
        filter_sql = f" AND outcome IN ({placeholders})"
        parameters.extend(outcomes)

    confidence_sql = " AND (confidence IS NULL OR confidence >= ?)"
    parameters.append(min_confidence)

    rows = query_rows(
        database,
        f"""
        SELECT outcome, COUNT(*) AS count
        FROM findings
        WHERE run_id = ?{filter_sql}{confidence_sql}
        GROUP BY outcome
        """,
        parameters,
    )
    rank = {outcome: index for index, outcome in enumerate(OUTCOME_LABELS)}
    return sorted(rows, key=lambda row: rank.get(str(row["outcome"]), 99))


@st.cache_data(show_spinner=False)
def load_class_statistics(
    database: Path,
    run_id: int,
    outcomes: Sequence[str],
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Aggregate filtered findings by their ground-truth-attributed class.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Diagnosis outcomes currently enabled in the sidebar filter.
        min_confidence: Minimum confidence threshold.

    Returns:
        Per-class outcome counts and mean IoU, alphabetically ordered.
    """
    if not outcomes:
        return []

    placeholders = ", ".join("?" for _ in outcomes)
    return query_rows(
        database,
        f"""
        SELECT class_name,
               SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN outcome = 'wrong_class' THEN 1 ELSE 0 END)
                   AS wrong_class,
               SUM(CASE WHEN outcome = 'poor_localization' THEN 1 ELSE 0 END)
                   AS poor_localization,
               SUM(CASE WHEN outcome = 'false_positive' THEN 1 ELSE 0 END)
                   AS false_positive,
               SUM(CASE WHEN outcome = 'false_negative' THEN 1 ELSE 0 END)
                   AS false_negative,
               ROUND(AVG(iou), 4) AS mean_iou
        FROM findings
        WHERE run_id = ? AND outcome IN ({placeholders})
          AND (confidence IS NULL OR confidence >= ?)
        GROUP BY class_name
        ORDER BY class_name
        """,
        [run_id, *outcomes, min_confidence],
    )


@st.cache_data(show_spinner=False)
def load_worst_images(
    database: Path,
    run_id: int,
    outcomes: Sequence[str],
    limit: int = 50,
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Rank images by selected failure count for rapid visual investigation.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Failure outcomes currently enabled in the sidebar filter.
        limit: Maximum number of rows to return for the explorer.
        min_confidence: Minimum confidence threshold.

    Returns:
        Images with at least one selected failure, ordered worst first.
    """
    selected_failures = [outcome for outcome in outcomes if outcome in FAILURE_OUTCOMES]
    if not selected_failures:
        return []

    placeholders = ", ".join("?" for _ in selected_failures)
    return query_rows(
        database,
        f"""
        SELECT i.id, i.filename, i.path, i.width, i.height, i.error,
               COUNT(*) AS failure_count
        FROM findings AS f
        JOIN images AS i ON i.id = f.image_id
        WHERE f.run_id = ? AND f.outcome IN ({placeholders})
          AND (f.confidence IS NULL OR f.confidence >= ?)
        GROUP BY i.id
        ORDER BY failure_count DESC, i.filename ASC
        LIMIT ?
        """,
        [run_id, *selected_failures, min_confidence, limit],
    )


@st.cache_data(show_spinner=False)
def load_image_findings(
    database: Path,
    image_id: int,
    outcomes: Sequence[str],
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Load all selected annotations needed to render one image's overlays.

    Args:
        database: Saved diagnosis database to query.
        image_id: Identifier of the image being inspected.
        outcomes: Outcomes currently enabled in the sidebar filter.
        min_confidence: Minimum confidence threshold.

    Returns:
        Finding rows with prediction, truth, and optional polygon geometry.
    """
    if not outcomes:
        return []

    placeholders = ", ".join("?" for _ in outcomes)
    optional = available_tables(database)

    # These fragments are literals selected by table presence, never user input,
    # so they carry no injection risk. Values stay parameterised.
    if "heatmaps" in optional:
        heatmap_select = "h.path AS heatmap_path"
        heatmap_join = (
            "LEFT JOIN heatmaps h ON h.finding_id = f.id AND h.method = 'grad-cam'"
        )
    else:
        heatmap_select = "NULL AS heatmap_path"
        heatmap_join = ""

    if "root_causes" in optional:
        root_cause_select = """(
                   SELECT json_group_array(json_object(
                       'factor', factor, 'score', score, 'evidence', evidence
                   ))
                   FROM (
                       SELECT factor, score, evidence
                       FROM root_causes
                       WHERE finding_id = f.id
                       ORDER BY score DESC
                   )
               ) AS root_causes_json"""
    else:
        root_cause_select = "NULL AS root_causes_json"

    return query_rows(
        database,
        f"""
        SELECT f.id AS finding_id, f.outcome, f.class_name, f.confidence, f.iou,
               f.pred_x1, f.pred_y1, f.pred_x2, f.pred_y2,
               f.truth_x1, f.truth_y1, f.truth_x2, f.truth_y2, f.truth_polygon,
               {heatmap_select},
               {root_cause_select}
        FROM findings f
        {heatmap_join}
        WHERE f.image_id = ? AND f.outcome IN ({placeholders})
          AND (f.confidence IS NULL OR f.confidence >= ?)
        ORDER BY f.outcome, f.class_name
        """,
        [image_id, *outcomes, min_confidence],
    )


@st.cache_data(show_spinner=False)
def load_errored_images(database: Path, run_id: int) -> list[dict[str, object]]:
    """List images that never completed so they are not counted as clean images.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        Failed image rows ordered by filename.
    """
    return query_rows(
        database,
        """
        SELECT filename, path, error
        FROM images
        WHERE run_id = ? AND error IS NOT NULL
        ORDER BY filename
        """,
        (run_id,),
    )


@st.cache_data(show_spinner=False)
def load_class_names(database: Path, run_id: int) -> list[str]:
    """Return distinct class names in alphabetical order for the class filter.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        Sorted list of class name strings.
    """
    rows = query_rows(
        database,
        """
        SELECT DISTINCT class_name
        FROM findings
        WHERE run_id = ?
        ORDER BY class_name
        """,
        (run_id,),
    )
    return [str(row["class_name"]) for row in rows]


@st.cache_data(show_spinner=False)
def load_class_statistics_filtered(
    database: Path,
    run_id: int,
    outcomes: Sequence[str],
    class_names: Sequence[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Per-class statistics with optional class name filter.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Outcomes currently enabled.
        class_names: Class names to include. ``None`` retains all classes.
        min_confidence: Minimum confidence threshold.

    Returns:
        Per-class outcome counts and mean IoU.
    """
    if not outcomes:
        return []

    outcome_placeholders = ", ".join("?" for _ in outcomes)
    parameters: list[object] = [run_id, *outcomes]

    class_sql = ""
    if class_names is not None and class_names:
        class_placeholders = ", ".join("?" for _ in class_names)
        class_sql = f" AND class_name IN ({class_placeholders})"
        parameters.extend(class_names)
    elif class_names is not None:
        return []

    return query_rows(
        database,
        f"""
        SELECT class_name,
               SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN outcome = 'wrong_class' THEN 1 ELSE 0 END)
                   AS wrong_class,
               SUM(CASE WHEN outcome = 'poor_localization' THEN 1 ELSE 0 END)
                   AS poor_localization,
               SUM(CASE WHEN outcome = 'false_positive' THEN 1 ELSE 0 END)
                   AS false_positive,
               SUM(CASE WHEN outcome = 'false_negative' THEN 1 ELSE 0 END)
                   AS false_negative,
               ROUND(AVG(iou), 4) AS mean_iou
        FROM findings
        WHERE run_id = ? AND outcome IN ({outcome_placeholders}){class_sql}
          AND (confidence IS NULL OR confidence >= ?)
        GROUP BY class_name
        ORDER BY class_name
        """,
        parameters + [min_confidence],
    )


@st.cache_data(show_spinner=False)
def load_root_causes_summary(database: Path, run_id: int) -> list[dict[str, object]]:
    """Count how many findings each root-cause factor explains in the run.

    A finding may carry several factors, so findings are counted distinctly.
    Summing the counts across factors would exceed the number of findings.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        Factor and finding-count rows, most common first. Empty when the
        database predates the root-cause table.
    """
    if "root_causes" not in available_tables(database):
        return []
    return query_rows(
        database,
        """
        SELECT factor, COUNT(DISTINCT finding_id) AS count
        FROM root_causes
        WHERE run_id = ?
        GROUP BY factor
        ORDER BY count DESC
        """,
        [run_id],
    )


@st.cache_data(show_spinner=False)
def load_root_causes_by_class(database: Path, run_id: int) -> list[dict[str, object]]:
    """Count findings per factor broken down by class, with percentage.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        List of factor, class_name, count, and percentage.
    """
    if "root_causes" not in available_tables(database):
        return []

    class_totals_rows = query_rows(
        database,
        """
        SELECT class_name, COUNT(*) AS total_failures
        FROM findings
        WHERE run_id = ? AND outcome != 'correct'
        GROUP BY class_name
        """,
        [run_id],
    )
    class_totals = {
        str(row["class_name"]): int(row["total_failures"])
        for row in class_totals_rows
    }

    breakdown_rows = query_rows(
        database,
        """
        SELECT rc.factor, f.class_name, COUNT(DISTINCT rc.finding_id) AS count
        FROM root_causes rc
        JOIN findings f ON f.id = rc.finding_id
        WHERE rc.run_id = ?
        GROUP BY rc.factor, f.class_name
        ORDER BY rc.factor, count DESC
        """,
        [run_id],
    )

    result = []
    for row in breakdown_rows:
        factor = str(row["factor"])
        class_name = str(row["class_name"])
        count = int(row["count"])
        total = class_totals.get(class_name, 1)
        percentage = round((count / total) * 100, 1)
        result.append({
            "factor": factor,
            "class_name": class_name,
            "count": count,
            "percentage": percentage,
        })
    return result


@st.cache_data(show_spinner=False)
def load_root_causes_by_outcome(database: Path, run_id: int) -> list[dict[str, object]]:
    """Count findings per factor broken down by outcome, with percentage.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        List of factor, outcome, count, and percentage.
    """
    if "root_causes" not in available_tables(database):
        return []

    outcome_totals_rows = query_rows(
        database,
        """
        SELECT outcome, COUNT(*) AS total_failures
        FROM findings
        WHERE run_id = ? AND outcome != 'correct'
        GROUP BY outcome
        """,
        [run_id],
    )
    outcome_totals = {
        str(row["outcome"]): int(row["total_failures"])
        for row in outcome_totals_rows
    }

    breakdown_rows = query_rows(
        database,
        """
        SELECT rc.factor, f.outcome, COUNT(DISTINCT rc.finding_id) AS count
        FROM root_causes rc
        JOIN findings f ON f.id = rc.finding_id
        WHERE rc.run_id = ?
        GROUP BY rc.factor, f.outcome
        ORDER BY rc.factor, count DESC
        """,
        [run_id],
    )

    result = []
    for row in breakdown_rows:
        factor = str(row["factor"])
        outcome = str(row["outcome"])
        count = int(row["count"])
        total = outcome_totals.get(outcome, 1)
        percentage = round((count / total) * 100, 1)
        result.append({
            "factor": factor,
            "outcome": outcome,
            "count": count,
            "percentage": percentage,
        })
    return result


@st.cache_data(show_spinner=False)
def load_failure_groups(database: Path, run_id: int) -> list[dict[str, object]]:
    """Load the list of failure groups for the run.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        List of cluster id, label, and size.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return []

    rows = query_rows(
        database,
        """
        SELECT id, label, size
        FROM clusters
        WHERE run_id = ? AND method = 'factor-signature'
        ORDER BY size DESC, label;
        """,
        [run_id],
    )
    return [
        {
            "id": int(row["id"]),
            "label": str(row["label"]),
            "size": int(row["size"]),
        }
        for row in rows
    ]


@st.cache_data(show_spinner=False)
def load_failure_group_members(
    database: Path, cluster_id: int
) -> list[dict[str, object]]:
    """Load the finding members of a failure group.

    Args:
        database: Saved diagnosis database to query.
        cluster_id: Identifier of the failure group (cluster).

    Returns:
        List of findings in the group.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return []

    rows = query_rows(
        database,
        """
        SELECT f.id, f.outcome, f.class_name, f.confidence, f.iou, i.filename, i.path
        FROM cluster_members cm
        JOIN findings f ON f.id = cm.finding_id
        JOIN images i ON i.id = f.image_id
        WHERE cm.cluster_id = ?
        ORDER BY f.id;
        """,
        [cluster_id],
    )
    return [
        {
            "id": int(row["id"]),
            "outcome": str(row["outcome"]),
            "class_name": str(row["class_name"]),
            "confidence": (
                float(row["confidence"]) if row["confidence"] is not None else None
            ),
            "iou": float(row["iou"]) if row["iou"] is not None else None,
            "filename": str(row["filename"]),
            "path": str(row["path"]),
        }
        for row in rows
    ]


@st.cache_data(show_spinner=False)
def load_failure_group_outcomes(database: Path, run_id: int) -> list[dict[str, object]]:
    """Count distinct findings per failure group, broken down by outcome.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        List of label, outcome, and count.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return []

    rows = query_rows(
        database,
        """
        SELECT c.label, f.outcome, COUNT(DISTINCT f.id) AS n
        FROM clusters c
        JOIN cluster_members cm ON cm.cluster_id = c.id
        JOIN findings f ON f.id = cm.finding_id
        WHERE c.run_id = ? AND c.method = 'factor-signature'
        GROUP BY c.label, f.outcome
        ORDER BY c.label, n DESC;
        """,
        [run_id],
    )
    return [
        {
            "label": str(row["label"]),
            "outcome": str(row["outcome"]),
            "n": int(row["n"]),
        }
        for row in rows
    ]


@st.cache_data(show_spinner=False)
def load_failure_group_classes(database: Path, run_id: int) -> list[dict[str, object]]:
    """Count distinct findings per failure group, broken down by class.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        List of label, class_name, and count.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return []

    rows = query_rows(
        database,
        """
        SELECT c.label, f.class_name, COUNT(DISTINCT f.id) AS n
        FROM clusters c
        JOIN cluster_members cm ON cm.cluster_id = c.id
        JOIN findings f ON f.id = cm.finding_id
        WHERE c.run_id = ? AND c.method = 'factor-signature'
        GROUP BY c.label, f.class_name
        ORDER BY n DESC;
        """,
        [run_id],
    )
    return [
        {
            "label": str(row["label"]),
            "class_name": str(row["class_name"]),
            "n": int(row["n"]),
        }
        for row in rows
    ]


@st.cache_data(show_spinner=False)
def load_finding_failure_group(database: Path, finding_id: int) -> str | None:
    """Find which failure group a given finding belongs to.

    Args:
        database: Saved diagnosis database to query.
        finding_id: Identifier of the finding.

    Returns:
        The label of the failure group, or None if not found/grouped.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return None

    rows = query_rows(
        database,
        """
        SELECT c.label
        FROM cluster_members cm
        JOIN clusters c ON c.id = cm.cluster_id
        WHERE cm.finding_id = ? AND c.method = 'factor-signature';
        """,
        [finding_id],
    )
    if not rows:
        return None
    return str(rows[0]["label"])


@st.cache_data(show_spinner=False)
def check_failure_groups_integrity(database: Path, run_id: int) -> str | None:
    """Check if the grouped failures sum exactly to the total run failures.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

    Returns:
        An error message if the counts differ, otherwise None.
    """
    if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
        return None

    rows = query_rows(
        database,
        """
        SELECT
          (SELECT COALESCE(SUM(size), 0) FROM clusters
            WHERE run_id = ? AND method = 'factor-signature')          AS grouped,
          (SELECT COUNT(*) FROM findings
            WHERE run_id = ? AND outcome != 'correct')                 AS failures;
        """,
        [run_id, run_id],
    )

    if not rows:
        return "Failed to evaluate arithmetic integrity check."

    grouped = int(rows[0]["grouped"] or 0)
    failures = int(rows[0]["failures"] or 0)

    if grouped != failures:
        return (
            f"Integrity check failed: grouped failures ({grouped}) do not "
            f"equal total failures ({failures})."
        )

    return None


@st.cache_data(show_spinner=False)
def load_run_comparison(
    database: Path,
    run_id_a: int,
    run_id_b: int,
    min_confidence: float = 0.0,
) -> list[dict[str, object]]:
    """Load outcome counts for two runs side by side (SCHEMA.md §5).

    Args:
        database: Saved diagnosis database to query.
        run_id_a: Identifier of the baseline run.
        run_id_b: Identifier of the candidate run.
        min_confidence: Minimum confidence threshold.

    Returns:
        Rows with run_id, outcome, and count.
    """
    return query_rows(
        database,
        """
        SELECT run_id, outcome, COUNT(*) AS n
        FROM findings WHERE run_id IN (?, ?) AND (confidence IS NULL OR confidence >= ?)
        GROUP BY run_id, outcome
        """,
        (run_id_a, run_id_b, min_confidence),
    )


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------


def image_with_overlays(
    image_path: Path,
    findings: Sequence[dict[str, object]],
) -> Image.Image:
    """Draw prediction, ground-truth, and segmentation overlays on an image.

    Geometry in the schema is in source-image pixels. Images are downscaled for
    dashboard responsiveness and every coordinate is scaled by the same factor.

    Args:
        image_path: Image file stored in the ``images.path`` column.
        findings: Selected finding rows returned by ``load_image_findings``.

    Returns:
        An RGBA image ready for Streamlit display.
    """
    source = Image.open(image_path).convert("RGBA")
    source_width, source_height = source.size
    largest_side = max(source_width, source_height)
    scale = min(1.0, 1800 / largest_side)
    if scale < 1.0:
        target_size = (round(source_width * scale), round(source_height * scale))
        source = source.resize(target_size)

    canvas = ImageDraw.Draw(source)
    line_width = max(2, round(3 * scale))
    for finding in findings:
        outcome = str(finding.get("outcome", ""))
        outcome_color = OUTCOME_COLORS.get(outcome, PREDICTION_COLOR)

        _draw_box(
            canvas,
            finding,
            prefix="pred",
            color=outcome_color,
            label="Pred",
            scale=scale,
            line_width=line_width,
        )
        _draw_box(
            canvas,
            finding,
            prefix="truth",
            color=GROUND_TRUTH_COLOR,
            label="Truth",
            scale=scale,
            line_width=line_width,
        )
        _draw_polygon(canvas, finding, scale, line_width)
    return source


def _draw_box(
    canvas: ImageDraw.ImageDraw,
    finding: dict[str, object],
    prefix: str,
    color: str,
    label: str,
    scale: float,
    line_width: int,
) -> None:
    """Render one optional schema box and a compact, class-aware label."""
    coordinate_keys = tuple(f"{prefix}_{axis}" for axis in ("x1", "y1", "x2", "y2"))
    coordinates = [finding[key] for key in coordinate_keys]
    if any(value is None for value in coordinates):
        return

    scaled = tuple(float(value) * scale for value in coordinates)
    canvas.rectangle(scaled, outline=color, width=line_width)
    class_name = str(finding["class_name"])

    outcome = str(finding.get("outcome", ""))
    confidence = finding["confidence"]
    iou = finding["iou"]
    confidence_text = f" {float(confidence):.2f}" if confidence is not None else ""
    iou_text = f" IoU:{float(iou):.2f}" if iou is not None else ""

    if outcome == "wrong_class" and label == "Pred":
        text = f"Pred: NOT '{class_name}' (predicted class not stored)"
    else:
        text = f"{label}: {class_name}{confidence_text}{iou_text}"

    text_origin = (scaled[0] + 3, max(0, scaled[1] - 14))
    canvas.text(text_origin, text, fill=color, stroke_width=1, stroke_fill="#0b1020")


def _draw_polygon(
    canvas: ImageDraw.ImageDraw,
    finding: dict[str, object],
    scale: float,
    line_width: int,
) -> None:
    """Render an optional ground-truth polygon without trusting malformed JSON."""
    polygon_json = finding["truth_polygon"]
    if polygon_json is None:
        return
    try:
        points = json.loads(str(polygon_json))
        scaled_points = [
            (float(point[0]) * scale, float(point[1]) * scale) for point in points
        ]
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return
    if len(scaled_points) >= 3:
        canvas.line(
            [*scaled_points, scaled_points[0]],
            fill=GROUND_TRUTH_COLOR,
            width=line_width,
        )


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def run_label(run: dict[str, object]) -> str:
    """Format a concise selector label while retaining reproducibility context."""
    model_hash = str(run["model_sha256"])[:8]
    return f"Run {run['id']} · {run['split']} · {run['created_at']} · {model_hash}"


def _outcome_badge_html(outcome: str) -> str:
    """Return an inline HTML pill badge coloured by outcome type."""
    color = OUTCOME_COLORS.get(outcome, "#94a3b8")
    label = OUTCOME_LABELS.get(outcome, outcome)
    return (
        f'<span style="background:{color}22; color:{color}; '
        f"padding:2px 10px; border-radius:20px; font-size:.82rem; "
        f'font-weight:600; border:1px solid {color}44;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def inject_theme() -> None:
    """Apply a dark-mode glassmorphism design system with Inter font.

    This transforms the default Streamlit appearance into a premium
    portfolio-quality interface with consistent spacing, subtle animations,
    and a refined colour palette.
    """
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

          /* ── Global ─────────────────────────────────────────── */
          .stApp {
            background: linear-gradient(135deg, #0b1120 0%, #101d35 50%, #0f172a 100%);
            color: #e2e8f0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          }
          /* Streamlit text elements */
          .stApp p, .stApp span, .stApp label, .stApp div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          }

          /* ── Sidebar ────────────────────────────────────────── */
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.08);
          }
          [data-testid="stSidebar"] * { color: #cbd5e1; }
          [data-testid="stSidebar"] .stSelectbox label,
          [data-testid="stSidebar"] .stMultiSelect label {
            color: #94a3b8; font-weight: 600; font-size: .82rem;
            text-transform: uppercase; letter-spacing: .08em;
          }

          /* ── Metric cards (glassmorphism) ────────────────── */
          [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            padding: 20px 18px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
          }
          [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
          }
          [data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-weight: 600; font-size: .78rem;
            letter-spacing: .06em; text-transform: uppercase;
          }
          [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
            font-weight: 800; font-size: 1.8rem;
          }
          [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-weight: 600;
          }

          /* ── Glass panels ──────────────────────────────────── */
          .glass-panel {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 1rem;
          }

          /* ── Typography ────────────────────────────────────── */
          .eyebrow {
            color: #2dd4bf; font-size: 0.78rem; font-weight: 700;
            letter-spacing: .14em; text-transform: uppercase;
            margin-bottom: .35rem;
          }
          .hero-title {
            color: #f1f5f9; font-size: 2.6rem; font-weight: 800;
            letter-spacing: -.04em; margin: 0 0 .2rem;
            background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
          }
          .hero-subtitle {
            color: #94a3b8; font-size: 1.02rem; margin-bottom: 1.5rem;
            line-height: 1.6;
          }
          .section-title {
            color: #e2e8f0; font-size: 1.25rem; font-weight: 700;
            margin: 1.8rem 0 .6rem;
            padding-bottom: .4rem;
            border-bottom: 2px solid rgba(45, 212, 191, 0.2);
          }

          /* ── Legend ─────────────────────────────────────────── */
          .legend { color: #94a3b8; font-size: .9rem; }
          .legend-dot {
            display: inline-block; width: 10px; height: 10px;
            border-radius: 50%; margin: 0 6px 0 14px;
          }

          /* ── Dataframes ────────────────────────────────────── */
          [data-testid="stDataFrame"] {
            border-radius: 12px; overflow: hidden;
          }

          /* ── Expanders ─────────────────────────────────────── */
          .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 12px;
            color: #94a3b8 !important;
          }

          /* ── Divider ───────────────────────────────────────── */
          hr { border-color: rgba(148, 163, 184, 0.1); }

          /* ── Tabs ──────────────────────────────────────────── */
          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 41, 59, 0.3);
            border-radius: 12px;
            padding: 4px;
          }
          .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
          }
          .stTabs [aria-selected="true"] {
            background: rgba(45, 212, 191, 0.15);
            color: #2dd4bf !important;
          }

          /* ── Onboarding card ───────────────────────────────── */
          .onboard-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(45, 212, 191, 0.15);
            border-radius: 20px;
            padding: 48px 40px;
            text-align: center;
            max-width: 600px;
            margin: 80px auto;
          }
          .onboard-card h2 {
            color: #f1f5f9; font-size: 1.8rem; margin-bottom: .8rem;
          }
          .onboard-card p {
            color: #94a3b8; font-size: 1rem; line-height: 1.7;
          }
          .onboard-card code {
            background: rgba(45, 212, 191, 0.1);
            color: #2dd4bf;
            padding: 2px 8px; border-radius: 6px;
          }

          /* ── Comparison delta badges ───────────────────────── */
          .delta-up { color: #2dd4bf; font-weight: 700; }
          .delta-down { color: #fb7185; font-weight: 700; }
          .delta-neutral { color: #94a3b8; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------


def render_outcome_chart(counts: Sequence[dict[str, object]]) -> None:
    """Render a readable outcome mix without ever calling it mAP."""
    if not counts:
        st.info("Choose at least one outcome to populate this chart.")
        return
    chart_rows = [
        {
            "Outcome": OUTCOME_LABELS.get(str(row["outcome"]), str(row["outcome"])),
            "Findings": int(row["count"]),
        }
        for row in counts
    ]
    colors = [OUTCOME_COLORS[str(row["outcome"])] for row in counts]
    figure = px.pie(
        chart_rows,
        names="Outcome",
        values="Findings",
        hole=0.68,
        color="Outcome",
        color_discrete_sequence=colors,
    )
    figure.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            font=dict(color="#94a3b8"),
        ),
    )
    figure.update_traces(
        textinfo="percent+label",
        textfont=dict(color="#e2e8f0"),
        hovertemplate="%{label}: %{value}<extra></extra>",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_class_chart(statistics: Sequence[dict[str, object]]) -> None:
    """Render per-class outcome composition for quick failure pattern scanning."""
    if not statistics:
        st.info("No per-class findings match the active outcome filter.")
        return
    chart_rows: list[dict[str, object]] = []
    for statistic in statistics:
        for outcome in OUTCOME_LABELS:
            value = int(statistic[outcome] or 0)
            if value:
                chart_rows.append(
                    {
                        "Class": statistic["class_name"],
                        "Outcome": OUTCOME_LABELS[outcome],
                        "Findings": value,
                    }
                )
    if not chart_rows:
        st.info("No per-class findings match the active outcome filter.")
        return
    figure = px.bar(
        chart_rows,
        x="Class",
        y="Findings",
        color="Outcome",
        barmode="stack",
        color_discrete_map={
            label: OUTCOME_COLORS[key] for key, label in OUTCOME_LABELS.items()
        },
    )
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(
            title=None,
            color="#94a3b8",
            gridcolor="rgba(148,163,184,0.08)",
        ),
        yaxis=dict(
            title="Findings",
            color="#94a3b8",
            gridcolor="rgba(148,163,184,0.08)",
        ),
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def render_comparison_chart(
    run_a: dict[str, object],
    run_b: dict[str, object],
    comparison_rows: Sequence[dict[str, object]],
) -> None:
    """Render side-by-side outcome comparison of two runs with delta indicators."""
    run_a_id = int(run_a["id"])
    run_b_id = int(run_b["id"])
    run_a_hash = str(run_a["model_sha256"])[:8]
    run_b_hash = str(run_b["model_sha256"])[:8]

    if run_a["model_sha256"] != run_b["model_sha256"]:
        st.warning(
            f"⚠️ These runs use different models "
            f"(`{run_a_hash}…` vs `{run_b_hash}…`). "
            "Outcome deltas may reflect model changes, not data changes."
        )

    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}
    for row in comparison_rows:
        rid = int(row["run_id"])
        outcome = str(row["outcome"])
        count = int(row["n"])
        if rid == run_a_id:
            counts_a[outcome] = count
        elif rid == run_b_id:
            counts_b[outcome] = count

    outcomes_in_order = list(OUTCOME_LABELS.keys())
    labels = [OUTCOME_LABELS[o] for o in outcomes_in_order]
    vals_a = [counts_a.get(o, 0) for o in outcomes_in_order]
    vals_b = [counts_b.get(o, 0) for o in outcomes_in_order]
    colors = [OUTCOME_COLORS[o] for o in outcomes_in_order]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name=f"Run {run_a_id}",
            x=labels,
            y=vals_a,
            marker_color=colors,
            marker_line=dict(width=0),
            opacity=0.75,
        )
    )
    figure.add_trace(
        go.Bar(
            name=f"Run {run_b_id}",
            x=labels,
            y=vals_b,
            marker_color=colors,
            marker_line=dict(width=2, color="#f1f5f9"),
            marker_pattern_shape="/",
            opacity=0.5,
        )
    )
    figure.update_layout(
        barmode="group",
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            font=dict(color="#94a3b8"),
        ),
        xaxis=dict(
            color="#94a3b8",
            gridcolor="rgba(148,163,184,0.08)",
        ),
        yaxis=dict(
            title="Findings",
            color="#94a3b8",
            gridcolor="rgba(148,163,184,0.08)",
        ),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    # Delta table
    delta_rows = []
    for outcome in outcomes_in_order:
        val_a = counts_a.get(outcome, 0)
        val_b = counts_b.get(outcome, 0)
        diff = val_b - val_a
        if outcome == "correct":
            if diff > 0:
                css = "delta-up"
            elif diff < 0:
                css = "delta-down"
            else:
                css = "delta-neutral"
        else:
            if diff > 0:
                css = "delta-down"
            elif diff < 0:
                css = "delta-up"
            else:
                css = "delta-neutral"
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "—")
        delta_rows.append(
            f"<tr><td>{_outcome_badge_html(outcome)}</td>"
            f'<td style="text-align:right">{val_a}</td>'
            f'<td style="text-align:right">{val_b}</td>'
            f'<td style="text-align:right" class="{css}">'
            f"{arrow} {abs(diff)}</td></tr>"
        )
    table_html = (
        '<div class="glass-panel">'
        '<table style="width:100%; border-collapse:collapse; color:#e2e8f0;">'
        '<thead><tr style="border-bottom:1px solid rgba(148,163,184,0.15);">'
        '<th style="text-align:left;padding:8px;color:#94a3b8;">Outcome</th>'
        f'<th style="text-align:right;padding:8px;color:#94a3b8;">Run {run_a_id}</th>'
        f'<th style="text-align:right;padding:8px;color:#94a3b8;">Run {run_b_id}</th>'
        '<th style="text-align:right;padding:8px;color:#94a3b8;">Delta</th>'
        "</tr></thead><tbody>" + "".join(delta_rows) + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------


def render_onboarding() -> None:
    """Show a premium empty state when no database or no runs exist."""
    st.markdown(
        """
        <div class="onboard-card">
            <h2>🔬 Welcome to Model Doctor</h2>
            <p>
                No diagnosis runs found yet.<br>
                Generate your first run with:
            </p>
            <p><code>python -m app.diagnosis --split test --save</code></p>
            <p style="margin-top:1.5rem; font-size:.88rem;">
                The dashboard will populate automatically once a run is saved
                to the configured SQLite database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(database: Path) -> None:
    """Render all dashboard surfaces after a usable database is selected."""
    runs = load_runs(database)
    if not runs:
        render_onboarding()
        return

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔬 Explore diagnosis runs")
        selected_run = st.selectbox("Run", runs, format_func=run_label)
        selected_outcomes = st.multiselect(
            "Outcome filter",
            options=list(OUTCOME_LABELS),
            default=list(OUTCOME_LABELS),
            format_func=lambda outcome: OUTCOME_LABELS[outcome],
        )

        # Class filter (new)
        run_id = int(selected_run["id"])
        available_classes = load_class_names(database, run_id)
        selected_classes = st.multiselect(
            "Class filter",
            options=available_classes,
            default=[],
            help="Leave empty to include all classes.",
        )

        min_confidence = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(selected_run["confidence_threshold"]),
            step=0.05,
            help=(
                "Filter out predictions below this threshold. "
                "Does not affect false negatives."
            ),
        )

        st.divider()
        st.caption(
            "Outcome and class filters refine the charts and the image "
            "explorer. The confidence threshold also narrows the finding and "
            "failure counts on the summary cards."
        )

    # ── Data queries ─────────────────────────────────────────────────────
    summary = load_run_summary(database, run_id, min_confidence=min_confidence)
    filtered_outcomes = load_outcome_counts(
        database,
        run_id,
        tuple(selected_outcomes) if selected_outcomes else None,
        min_confidence=min_confidence,
    )

    effective_classes = tuple(selected_classes) if selected_classes else None
    class_statistics = load_class_statistics_filtered(
        database,
        run_id,
        tuple(selected_outcomes),
        effective_classes,
        min_confidence=min_confidence,
    )

    # ── Hero header ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="eyebrow">Diagnosis intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">Model Doctor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Inspect saved model runs, isolate failure '
        "modes, and open the images that need attention.</div>",
        unsafe_allow_html=True,
    )

    # ── Metric cards ─────────────────────────────────────────────────────
    metrics = st.columns(5)
    metrics[0].metric("Images", summary.image_count)
    metrics[1].metric("Processed", summary.processed_image_count)
    metrics[2].metric("Image errors", summary.errored_image_count)
    metrics[3].metric("Findings", summary.finding_count)
    metrics[4].metric("Failures", summary.failure_count)
    st.caption(
        "Diagnosis findings are not COCO mAP. They classify outcomes at this "
        "run's fixed thresholds."
    )

    # ── Tabs ─────────────────────────────────────────────────────────────
    (
        tab_overview,
        tab_classes,
        tab_images,
        tab_compare,
        tab_root_causes,
        tab_failure_groups,
    ) = st.tabs(
        [
            "📊 Overview",
            "📋 Per-class",
            "🔍 Image explorer",
            "⚖️ Run comparison",
            "🌱 Root causes",
            "🧩 Failure Groups",
        ]
    )

    # ── Tab 1: Overview ──────────────────────────────────────────────────
    with tab_overview:
        chart_col, provenance_col = st.columns((1.15, 0.85), gap="large")
        with chart_col:
            st.markdown(
                '<div class="section-title">Outcome mix</div>',
                unsafe_allow_html=True,
            )
            render_outcome_chart(filtered_outcomes)
        with provenance_col:
            st.markdown(
                '<div class="section-title">Run provenance</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                {
                    "Setting": [
                        "Split",
                        "Model hash",
                        "Confidence threshold",
                        "Match IoU",
                        "Localization floor",
                        "Image size",
                    ],
                    "Value": [
                        selected_run["split"],
                        str(selected_run["model_sha256"])[:12],
                        selected_run["confidence_threshold"],
                        selected_run["match_iou_threshold"],
                        selected_run["localization_iou_floor"],
                        selected_run["image_size"],
                    ],
                },
                hide_index=True,
                use_container_width=True,
            )

        errored_images = load_errored_images(database, run_id)
        if errored_images:
            with st.expander(f"⚠️ {len(errored_images)} image processing error(s)"):
                st.dataframe(errored_images, hide_index=True, use_container_width=True)

    # ── Tab 2: Per-class ─────────────────────────────────────────────────
    with tab_classes:
        st.markdown(
            '<div class="section-title">Per-class statistics</div>',
            unsafe_allow_html=True,
        )
        if selected_classes:
            st.info(
                f"Filtered to {len(selected_classes)} class(es): "
                + ", ".join(selected_classes)
            )
        render_class_chart(class_statistics)
        if class_statistics:
            display_rows = []
            for row in class_statistics:
                total = sum(int(row[o] or 0) for o in OUTCOME_LABELS)
                correct_count = int(row["correct"] or 0)
                success_rate = (
                    round(100 * correct_count / total, 1) if total > 0 else 0.0
                )
                display_rows.append(
                    {
                        "Class": row["class_name"],
                        "Correct": row["correct"],
                        "Wrong class": row["wrong_class"],
                        "Poor localization": row["poor_localization"],
                        "False positive": row["false_positive"],
                        "False negative": row["false_negative"],
                        "Mean IoU": row["mean_iou"],
                        "Success %": success_rate,
                    }
                )
            st.dataframe(display_rows, hide_index=True, use_container_width=True)

    # ── Tab 3: Image explorer ────────────────────────────────────────────
    with tab_images:
        st.markdown(
            '<div class="section-title">Worst-image explorer</div>',
            unsafe_allow_html=True,
        )
        worst_images = load_worst_images(
            database, run_id, tuple(selected_outcomes), min_confidence=min_confidence
        )
        if not worst_images:
            st.info(
                "No failures match the active filter. Select a failure outcome to "
                "explore images."
            )
        else:
            selected_image = st.selectbox(
                "Image ranked by failure count",
                worst_images,
                format_func=lambda row: (
                    f"{row['filename']} · {row['failure_count']} failures"
                ),
            )
            findings = load_image_findings(
                database,
                int(selected_image["id"]),
                tuple(selected_outcomes),
                min_confidence=min_confidence,
            )
            image_column, detail_column = st.columns((1.55, 0.85), gap="large")
            with image_column:
                # Colour legend by outcome
                legend_items = []
                for outcome_key, outcome_label in OUTCOME_LABELS.items():
                    colour = OUTCOME_COLORS[outcome_key]
                    legend_items.append(
                        f'<span class="legend-dot" '
                        f'style="background:{colour}"></span>{outcome_label}'
                    )
                legend_html = (
                    '<div class="legend">Boxes coloured by outcome: '
                    + " ".join(legend_items)
                    + "</div>"
                )
                st.markdown(legend_html, unsafe_allow_html=True)

                image_path = Path(str(selected_image["path"]))
                if image_path.is_file():
                    show_heatmaps = st.toggle("Show heatmaps", value=False)
                    if show_heatmaps:
                        img_col, heat_col = st.columns(2)
                        with img_col:
                            st.image(
                                image_with_overlays(image_path, findings),
                                use_container_width=True,
                            )
                        with heat_col:
                            heatmaps = [f for f in findings if f.get("heatmap_path")]
                            if not heatmaps:
                                st.info("No heatmaps available for these findings.")
                            else:
                                for h in heatmaps:
                                    h_path = Path(str(h["heatmap_path"]))
                                    if h_path.is_file():
                                        st.image(
                                            str(h_path),
                                            caption=(
                                                f"Heatmap: {h['class_name']} "
                                                f"({h.get('outcome', '')})"
                                            ),
                                            use_container_width=True,
                                        )
                                    else:
                                        st.warning(
                                            f"Heatmap file unavailable at `{h_path}`"
                                        )
                    else:
                        st.image(
                            image_with_overlays(image_path, findings),
                            use_container_width=True,
                        )
                else:
                    st.warning(
                        f"Image file is unavailable at `{image_path}`. "
                        "The stored findings remain visible."
                    )
            with detail_column:
                st.markdown("#### Selected image")
                st.write(selected_image["filename"])
                width = selected_image.get("width")
                height = selected_image.get("height")
                if width and height:
                    st.caption(f"{width} × {height} px")
                st.metric("Selected failures", selected_image["failure_count"])

                # Outcome breakdown mini-badges
                outcome_counts: dict[str, int] = {}
                for finding in findings:
                    outcome_val = str(finding["outcome"])
                    outcome_counts[outcome_val] = outcome_counts.get(outcome_val, 0) + 1
                if outcome_counts:
                    badges_html = " ".join(
                        f"{_outcome_badge_html(o)} <span "
                        f'style="color:#94a3b8;font-size:.85rem;margin-right:8px;">'
                        f"×{c}</span>"
                        for o, c in outcome_counts.items()
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)

                st.markdown("---")
                finding_rows = []
                for row in findings:
                    rc_text = ""
                    if row.get("root_causes_json"):
                        try:
                            rc_list = json.loads(str(row["root_causes_json"]))
                            if isinstance(rc_list, list) and rc_list:
                                factors = []
                                for rc in rc_list:
                                    if isinstance(rc, dict) and "factor" in rc:
                                        factors.append(
                                            f"{rc['factor']}\n"
                                            f"Score: {rc['score']:.2f}\n"
                                            f"Evidence: {rc['evidence']}"
                                        )
                                if factors:
                                    rc_text = "\n\n".join(factors)
                        except json.JSONDecodeError:
                            pass

                    finding_rows.append(
                        {
                            "Outcome": OUTCOME_LABELS.get(
                                str(row["outcome"]), row["outcome"]
                            ),
                            "Class": row["class_name"],
                            "Confidence": row["confidence"],
                            "IoU": row["iou"],
                            "Root Causes": rc_text,
                        }
                    )

                st.dataframe(finding_rows, hide_index=True, use_container_width=True)

    # ── Tab 4: Run comparison ────────────────────────────────────────────
    with tab_compare:
        st.markdown(
            '<div class="section-title">Run comparison</div>',
            unsafe_allow_html=True,
        )
        if len(runs) < 2:
            st.info(
                "Run comparison requires at least two saved runs. "
                "Execute diagnosis again with different settings to compare."
            )
        else:
            compare_col_a, compare_col_b = st.columns(2)
            with compare_col_a:
                run_a = st.selectbox(
                    "Baseline run (A)",
                    runs,
                    index=min(1, len(runs) - 1),
                    format_func=run_label,
                    key="compare_run_a",
                )
            with compare_col_b:
                run_b = st.selectbox(
                    "Candidate run (B)",
                    runs,
                    index=0,
                    format_func=run_label,
                    key="compare_run_b",
                )
            if int(run_a["id"]) == int(run_b["id"]):
                st.info("Select two different runs to compare.")
            else:
                comparison_data = load_run_comparison(
                    database,
                    int(run_a["id"]),
                    int(run_b["id"]),
                    min_confidence=min_confidence,
                )
                render_comparison_chart(run_a, run_b, comparison_data)

    # ── Tab 5: Root causes ───────────────────────────────────────────────
    with tab_root_causes:
        st.markdown(
            '<div class="section-title">Root causes</div>',
            unsafe_allow_html=True,
        )
        rc_summary = load_root_causes_summary(database, run_id)
        if not rc_summary:
            st.info(
                "No root causes recorded for this run. Run "
                "`python -m app.root_cause --run <id>` to generate them."
            )
        else:
            rc_rows = [
                {"Factor": row["factor"], "Findings": row["count"]}
                for row in rc_summary
            ]
            fig = px.bar(
                rc_rows,
                x="Factor",
                y="Findings",
                color_discrete_sequence=["#a78bfa"],
            )
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e2e8f0"),
                xaxis=dict(
                    title=None,
                    color="#94a3b8",
                    gridcolor="rgba(148,163,184,0.08)",
                ),
                yaxis=dict(
                    title="Findings",
                    color="#94a3b8",
                    gridcolor="rgba(148,163,184,0.08)",
                ),
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )
            st.dataframe(rc_rows, hide_index=True, use_container_width=True)

            st.markdown(
                '<div class="section-title" style="font-size: 1.1rem;">'
                "Breakdowns</div>",
                unsafe_allow_html=True,
            )

            class_breakdown = load_root_causes_by_class(database, run_id)
            outcome_breakdown = load_root_causes_by_outcome(database, run_id)

            rc_col_a, rc_col_b = st.columns(2)
            with rc_col_a:
                st.markdown("#### By Class")
                if class_breakdown:
                    st.dataframe(
                        [
                            {
                                "Factor": r["factor"],
                                "Class": r["class_name"],
                                "Findings": r["count"],
                                "Percentage": f"{r['percentage']}%",
                            }
                            for r in class_breakdown
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No class breakdowns available.")

            with rc_col_b:
                st.markdown("#### By Outcome")
                if outcome_breakdown:
                    st.dataframe(
                        [
                            {
                                "Factor": r["factor"],
                                "Outcome": OUTCOME_LABELS.get(
                                    str(r["outcome"]), str(r["outcome"])
                                ),
                                "Findings": r["count"],
                                "Percentage": f"{r['percentage']}%",
                            }
                            for r in outcome_breakdown
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No outcome breakdowns available.")

    # ── Tab 6: Failure Groups ────────────────────────────────────────────
    with tab_failure_groups:
        st.markdown(
            '<div class="section-title">Failure Groups</div>',
            unsafe_allow_html=True,
        )

        if not {"clusters", "cluster_members"}.issubset(available_tables(database)):
            st.info("Failure groups have not been generated for this database.")
        else:
            integrity_error = check_failure_groups_integrity(database, run_id)
            if integrity_error:
                st.error(integrity_error)
            else:
                failure_groups = load_failure_groups(database, run_id)
                if not failure_groups:
                    st.info(
                        "No failure groups recorded for this run. Run "
                        "`python -m app.clustering --run <id>` to generate them."
                    )
                else:
                    fg_rows = [
                        {"Group": fg["label"], "Failures": fg["size"]}
                        for fg in failure_groups
                    ]

                    fig = px.bar(
                        fg_rows,
                        x="Group",
                        y="Failures",
                        color_discrete_sequence=["#f43f5e"],
                    )
                    fig.update_layout(
                        xaxis_title=None,
                        yaxis_title="Failures",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=300,
                        dragmode=False,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, config={"displayModeBar": False}
                    )
                    st.dataframe(fg_rows, hide_index=True, use_container_width=True)

                    st.markdown(
                        '<div class="section-title" style="font-size: 1.1rem;">'
                        "Group Breakdowns</div>",
                        unsafe_allow_html=True,
                    )

                    fg_class_bd = load_failure_group_classes(database, run_id)
                    fg_outcome_bd = load_failure_group_outcomes(database, run_id)

                    fg_col_a, fg_col_b = st.columns(2)
                    with fg_col_a:
                        st.markdown("#### By Class")
                        if fg_class_bd:
                            st.dataframe(
                                [
                                    {
                                        "Group": r["label"],
                                        "Class": r["class_name"],
                                        "Failures": r["n"],
                                    }
                                    for r in fg_class_bd
                                ],
                                hide_index=True,
                                use_container_width=True,
                            )
                        else:
                            st.info("No class breakdowns available.")

                    with fg_col_b:
                        st.markdown("#### By Outcome")
                        if fg_outcome_bd:
                            st.dataframe(
                                [
                                    {
                                        "Group": r["label"],
                                        "Outcome": OUTCOME_LABELS.get(
                                    str(r["outcome"]), str(r["outcome"])
                                ),
                                        "Failures": r["n"],
                                    }
                                    for r in fg_outcome_bd
                                ],
                                hide_index=True,
                                use_container_width=True,
                            )
                        else:
                            st.info("No outcome breakdowns available.")


def main() -> None:
    """Start the Streamlit dashboard with a database picker and safe error state."""
    st.set_page_config(
        page_title="Model Doctor",
        page_icon="🔬",
        layout="wide",
    )
    inject_theme()
    with st.sidebar:
        st.markdown("## 🔬 Model Doctor")
        database_text = st.text_input(
            "SQLite database",
            value=str(default_database_path()),
        )
        st.caption("Read-only access — no run data is changed here.")

    try:
        render_dashboard(Path(database_text).expanduser())
    except DashboardDataError as error:
        st.markdown(
            '<div class="hero-title">Model Doctor</div>',
            unsafe_allow_html=True,
        )
        st.error(str(error))


if __name__ == "__main__":
    main()
