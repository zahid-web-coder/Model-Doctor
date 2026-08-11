"""Behavioural tests for the dashboard's read-only SQLite data layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.dashboard import (
    DashboardDataError,
    load_class_statistics,
    load_outcome_counts,
    load_run_summary,
    load_runs,
    load_worst_images,
    validate_database,
)


def create_dashboard_database(database: Path) -> None:
    """Create the smallest schema-valid database needed to pin UI query contracts."""
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_sha256 TEXT NOT NULL,
            dataset_yaml TEXT NOT NULL,
            split TEXT NOT NULL,
            confidence_threshold REAL NOT NULL,
            match_iou_threshold REAL NOT NULL,
            localization_iou_floor REAL NOT NULL,
            image_size INTEGER NOT NULL
        );
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            filename TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            prediction_count INTEGER NOT NULL,
            truth_count INTEGER NOT NULL,
            error TEXT
        );
        CREATE TABLE findings (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            class_id INTEGER,
            class_name TEXT NOT NULL,
            confidence REAL,
            iou REAL,
            pred_x1 REAL,
            pred_y1 REAL,
            pred_x2 REAL,
            pred_y2 REAL,
            truth_x1 REAL,
            truth_y1 REAL,
            truth_x2 REAL,
            truth_y2 REAL,
            truth_polygon TEXT
        );
        """
    )
    connection.execute(
        """
        INSERT INTO runs VALUES (1, '2026-08-11T10:00:00+00:00', 'model.pt',
        'abcdef123456', 'data.yaml', 'test', 0.25, 0.50, 0.10, 640)
        """
    )
    connection.executemany(
        """
        INSERT INTO images VALUES (?, 1, ?, ?, 640, 480, 0, 0, ?)
        """,
        [
            (1, '/data/one.jpg', 'one.jpg', None),
            (2, '/data/two.jpg', 'two.jpg', None),
            (3, '/data/broken.jpg', 'broken.jpg', 'Unreadable image'),
        ],
    )
    connection.executemany(
        """
        INSERT INTO findings (
            id, run_id, image_id, outcome, class_id, class_name, confidence, iou
        ) VALUES (?, 1, ?, ?, 0, ?, ?, ?)
        """,
        [
            (1, 1, 'correct', 'crack', 0.90, 0.80),
            (2, 1, 'false_positive', 'crack', 0.70, None),
            (3, 1, 'false_positive', 'crack', 0.60, None),
            (4, 2, 'false_negative', 'spall', None, None),
        ],
    )
    connection.commit()
    connection.close()


@pytest.fixture
def dashboard_database(tmp_path: Path) -> Path:
    """Provide a schema-valid saved run with varied outcomes and an image error."""
    database = tmp_path / "model_doctor.db"
    create_dashboard_database(database)
    return database


def test_validate_database_explains_missing_database(tmp_path: Path) -> None:
    """A missing database is an actionable dashboard state, not a SQLite crash."""
    with pytest.raises(DashboardDataError, match="Run diagnosis with `--save`"):
        validate_database(tmp_path / "missing.db")


def test_dashboard_queries_show_run_statistics_and_worst_images(
    dashboard_database: Path,
) -> None:
    """The explorer derives run totals and ranking from the published schema."""
    runs = load_runs(dashboard_database)
    summary = load_run_summary(dashboard_database, 1)
    outcomes = load_outcome_counts(dashboard_database, 1)
    statistics = load_class_statistics(
        dashboard_database,
        1,
        ["correct", "false_positive", "false_negative"],
    )
    worst_images = load_worst_images(
        dashboard_database,
        1,
        ["false_positive", "false_negative"],
    )

    assert runs[0]["split"] == "test"
    assert summary.image_count == 3
    assert summary.processed_image_count == 2
    assert summary.errored_image_count == 1
    assert summary.finding_count == 4
    assert summary.failure_count == 3
    assert outcomes == [
        {"outcome": "correct", "count": 1},
        {"outcome": "false_positive", "count": 2},
        {"outcome": "false_negative", "count": 1},
    ]
    assert statistics[0]["class_name"] == "crack"
    assert statistics[0]["false_positive"] == 2
    assert worst_images[0]["filename"] == "one.jpg"
    assert worst_images[0]["failure_count"] == 2
