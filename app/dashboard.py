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

from PIL import Image, ImageDraw
import plotly.express as px
import streamlit as st


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


def load_run_summary(database: Path, run_id: int) -> RunSummary:
    """Compute transparent, non-mAP totals for a single diagnosis run.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.

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
        WHERE run_id = ?
        """,
        (run_id,),
    )[0]
    return RunSummary(
        image_count=int(image_row["image_count"] or 0),
        processed_image_count=int(image_row["processed_image_count"] or 0),
        errored_image_count=int(image_row["errored_image_count"] or 0),
        finding_count=int(finding_row["finding_count"] or 0),
        failure_count=int(finding_row["failure_count"] or 0),
    )


def load_outcome_counts(
    database: Path,
    run_id: int,
    outcomes: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Return diagnosis outcome counts, optionally narrowed by the UI filter.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Outcomes to include. ``None`` retains every schema outcome.

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

    rows = query_rows(
        database,
        f"""
        SELECT outcome, COUNT(*) AS count
        FROM findings
        WHERE run_id = ?{filter_sql}
        GROUP BY outcome
        """,
        parameters,
    )
    rank = {outcome: index for index, outcome in enumerate(OUTCOME_LABELS)}
    return sorted(rows, key=lambda row: rank.get(str(row["outcome"]), 99))


def load_class_statistics(
    database: Path,
    run_id: int,
    outcomes: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate filtered findings by their ground-truth-attributed class.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Diagnosis outcomes currently enabled in the sidebar filter.

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
        GROUP BY class_name
        ORDER BY class_name
        """,
        [run_id, *outcomes],
    )


def load_worst_images(
    database: Path,
    run_id: int,
    outcomes: Sequence[str],
    limit: int = 50,
) -> list[dict[str, object]]:
    """Rank images by selected failure count for rapid visual investigation.

    Args:
        database: Saved diagnosis database to query.
        run_id: Identifier of the selected run.
        outcomes: Failure outcomes currently enabled in the sidebar filter.
        limit: Maximum number of rows to return for the explorer.

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
        GROUP BY i.id
        ORDER BY failure_count DESC, i.filename ASC
        LIMIT ?
        """,
        [run_id, *selected_failures, limit],
    )


def load_image_findings(
    database: Path,
    image_id: int,
    outcomes: Sequence[str],
) -> list[dict[str, object]]:
    """Load all selected annotations needed to render one image's overlays.

    Args:
        database: Saved diagnosis database to query.
        image_id: Identifier of the image being inspected.
        outcomes: Outcomes currently enabled in the sidebar filter.

    Returns:
        Finding rows with prediction, truth, and optional polygon geometry.
    """
    if not outcomes:
        return []

    placeholders = ", ".join("?" for _ in outcomes)
    return query_rows(
        database,
        f"""
        SELECT outcome, class_name, confidence, iou,
               pred_x1, pred_y1, pred_x2, pred_y2,
               truth_x1, truth_y1, truth_x2, truth_y2, truth_polygon
        FROM findings
        WHERE image_id = ? AND outcome IN ({placeholders})
        ORDER BY outcome, class_name
        """,
        [image_id, *outcomes],
    )


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
        _draw_box(
            canvas,
            finding,
            prefix="pred",
            color=PREDICTION_COLOR,
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
    confidence = finding["confidence"]
    confidence_text = f" {float(confidence):.2f}" if confidence is not None else ""
    text = f"{label}: {class_name}{confidence_text}"
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


def run_label(run: dict[str, object]) -> str:
    """Format a concise selector label while retaining reproducibility context."""
    model_hash = str(run["model_sha256"])[:8]
    return (
        f"Run {run['id']} · {run['split']} · {run['created_at']} · {model_hash}"
    )


def inject_theme() -> None:
    """Apply a restrained visual system.

    This keeps the dashboard from looking default.
    """
    st.markdown(
        """
        <style>
          .stApp { background: #f5f7fb; color: #172033; }
          [data-testid="stSidebar"] { background: #101828; }
          [data-testid="stSidebar"] * { color: #e5e7eb; }
          [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e6eaf0; border-radius: 14px;
            padding: 16px; box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
          }
          .eyebrow { color: #64748b; font-size: 0.78rem; font-weight: 700;
            letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }
          .hero-title { color: #101828; font-size: 2.4rem; font-weight: 750;
            letter-spacing: -.05em; margin: 0 0 .2rem; }
          .hero-subtitle { color: #667085; font-size: 1.02rem; margin-bottom: 1.5rem; }
          .section-title { color: #172033; font-size: 1.2rem; font-weight: 700;
            margin: 1.4rem 0 .4rem; }
          .legend { color: #475467; font-size: .9rem; }
          .legend-dot { display: inline-block; width: 10px; height: 10px;
            border-radius: 50%; margin: 0 6px 0 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    figure.update_traces(
        textinfo="percent+label",
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        xaxis_title=None,
        yaxis_title="Findings",
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def render_dashboard(database: Path) -> None:
    """Render all dashboard surfaces after a usable database is selected."""
    runs = load_runs(database)
    if not runs:
        st.info(
            "No saved runs yet. Run `python -m app.diagnosis --split test "
            "--save` first."
        )
        return

    with st.sidebar:
        st.markdown("### Explore diagnosis runs")
        selected_run = st.selectbox("Run", runs, format_func=run_label)
        selected_outcomes = st.multiselect(
            "Outcome filter",
            options=list(OUTCOME_LABELS),
            default=list(OUTCOME_LABELS),
            format_func=lambda outcome: OUTCOME_LABELS[outcome],
        )
        st.caption(
            "Filters refine charts and the image explorer. Summary cards show "
            "the complete run."
        )

    run_id = int(selected_run["id"])
    summary = load_run_summary(database, run_id)
    all_outcomes = load_outcome_counts(database, run_id)
    filtered_outcomes = load_outcome_counts(database, run_id, selected_outcomes)
    class_statistics = load_class_statistics(database, run_id, selected_outcomes)

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
        'modes, and open the images that need attention.</div>',
        unsafe_allow_html=True,
    )

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

    chart_column, run_column = st.columns((1.15, 0.85), gap="large")
    with chart_column:
        st.markdown(
            '<div class="section-title">Outcome mix</div>',
            unsafe_allow_html=True,
        )
        render_outcome_chart(filtered_outcomes)
    with run_column:
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
        st.caption(f"{len(all_outcomes)} outcome types recorded in this run.")

    st.markdown(
        '<div class="section-title">Per-class statistics</div>',
        unsafe_allow_html=True,
    )
    render_class_chart(class_statistics)
    if class_statistics:
        display_rows = [
            {
                "Class": row["class_name"],
                "Correct": row["correct"],
                "Wrong class": row["wrong_class"],
                "Poor localization": row["poor_localization"],
                "False positive": row["false_positive"],
                "False negative": row["false_negative"],
                "Mean IoU": row["mean_iou"],
            }
            for row in class_statistics
        ]
        st.dataframe(display_rows, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="section-title">Worst-image explorer</div>',
        unsafe_allow_html=True,
    )
    worst_images = load_worst_images(database, run_id, selected_outcomes)
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
                f"{row['filename']} · {row['failure_count']} selected failures"
            ),
        )
        findings = load_image_findings(
            database,
            int(selected_image["id"]),
            selected_outcomes,
        )
        image_column, detail_column = st.columns((1.55, 0.85), gap="large")
        with image_column:
            st.markdown(
                '<div class="legend"><span class="legend-dot" '
                'style="background:#fb7185"></span>Prediction '
                '<span class="legend-dot" style="background:#2dd4bf"></span>'
                'Ground truth / polygon</div>',
                unsafe_allow_html=True,
            )
            image_path = Path(str(selected_image["path"]))
            if image_path.is_file():
                st.image(
                    image_with_overlays(image_path, findings),
                    use_column_width=True,
                )
            else:
                st.warning(
                    f"Image file is unavailable at {image_path}. The stored "
                    "findings remain visible."
                )
        with detail_column:
            st.markdown("#### Selected image")
            st.write(selected_image["filename"])
            st.caption(f"{selected_image['width']} × {selected_image['height']} px")
            st.metric("Selected failures", selected_image["failure_count"])
            finding_rows = [
                {
                    "Outcome": OUTCOME_LABELS.get(str(row["outcome"]), row["outcome"]),
                    "Class": row["class_name"],
                    "Confidence": row["confidence"],
                    "IoU": row["iou"],
                }
                for row in findings
            ]
            st.dataframe(finding_rows, hide_index=True, use_container_width=True)

    errored_images = load_errored_images(database, run_id)
    if errored_images:
        with st.expander(f"{len(errored_images)} image processing error(s)"):
            st.dataframe(errored_images, hide_index=True, use_container_width=True)


def main() -> None:
    """Start the Streamlit dashboard with a database picker and safe error state."""
    st.set_page_config(
        page_title="Model Doctor",
        page_icon="◈",
        layout="wide",
    )
    inject_theme()
    with st.sidebar:
        st.markdown("## Model Doctor")
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
