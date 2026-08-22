"""Behavioural tests for the dashboard's read-only SQLite data layer."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dashboard import (
    DashboardDataError,
    available_tables,
    check_failure_groups_integrity,
    load_class_names,
    load_class_statistics,
    load_class_statistics_filtered,
    load_factor_rates,
    load_failure_group_classes,
    load_failure_group_members,
    load_failure_group_outcomes,
    load_failure_groups,
    load_finding_failure_group,
    load_image_findings,
    load_mask_class_summary,
    load_mask_findings,
    load_outcome_counts,
    load_recommendations,
    load_root_causes_by_class,
    load_root_causes_by_outcome,
    load_root_causes_summary,
    load_run_comparison,
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


def create_two_run_database(database: Path) -> None:
    """Create a database with two runs for comparison testing."""
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
    connection.execute(
        """
        INSERT INTO runs VALUES (2, '2026-08-12T10:00:00+00:00', 'model.pt',
        'abcdef123456', 'data.yaml', 'test', 0.25, 0.50, 0.10, 640)
        """
    )
    connection.executemany(
        "INSERT INTO images VALUES (?, ?, ?, ?, 640, 480, 0, 0, NULL)",
        [
            (1, 1, '/data/one.jpg', 'one.jpg'),
            (2, 2, '/data/one.jpg', 'one.jpg'),
        ],
    )
    connection.executemany(
        """
        INSERT INTO findings (
            id, run_id, image_id, outcome, class_id, class_name, confidence, iou
        ) VALUES (?, ?, ?, ?, 0, ?, ?, ?)
        """,
        [
            (1, 1, 1, 'correct', 'crack', 0.90, 0.80),
            (2, 1, 1, 'false_positive', 'crack', 0.70, None),
            (3, 2, 2, 'correct', 'crack', 0.92, 0.85),
            (4, 2, 2, 'correct', 'crack', 0.88, 0.78),
        ],
    )
    connection.commit()
    connection.close()


def create_optional_dashboard_database(database: Path) -> None:
    """Create a database that includes the optional tables."""
    create_dashboard_database(database)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE heatmaps (
            id INTEGER PRIMARY KEY,
            finding_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            method TEXT NOT NULL,
            target_layers TEXT NOT NULL
        );
        CREATE TABLE root_causes (
            id INTEGER PRIMARY KEY,
            finding_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            factor TEXT NOT NULL,
            score REAL NOT NULL,
            evidence TEXT NOT NULL
        );
        CREATE TABLE clusters (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            method TEXT NOT NULL,
            label TEXT NOT NULL,
            size INTEGER NOT NULL
        );
        CREATE TABLE cluster_members (
            cluster_id INTEGER NOT NULL,
            finding_id INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, finding_id)
        );
        CREATE TABLE factor_rates (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            factor TEXT NOT NULL,
            failure_count INTEGER NOT NULL,
            failure_total INTEGER NOT NULL,
            correct_count INTEGER NOT NULL,
            correct_total INTEGER NOT NULL,
            lift REAL,
            p_value REAL NOT NULL
        );
        CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            rule TEXT NOT NULL,
            action TEXT NOT NULL,
            rationale TEXT NOT NULL,
            status TEXT NOT NULL,
            actionable INTEGER NOT NULL,
            affected INTEGER NOT NULL,
            priority REAL NOT NULL
        );
        CREATE TABLE mask_findings (
            id INTEGER PRIMARY KEY,
            finding_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            mask_iou REAL,
            mask_outcome TEXT,
            pred_polygon TEXT
        );
        """
    )

    # A qualifying factor, and one whose lift is undefined because no correct
    # finding carried it — the "n/a" path the UI must never render as a number.
    connection.executemany(
        "INSERT INTO factor_rates VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 'small_object', 8, 10, 2, 40, 2.5, 0.001),
            (2, 'edge_truncation', 7, 10, 30, 40, 0.93, 0.34),
            (3, 'sparse_factor', 2, 10, 0, 40, None, 0.21),
        ],
    )
    # One action and one refusal. The refusal must survive to the UI.
    connection.executemany(
        "INSERT INTO recommendations VALUES (?, 1, 1, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 'recall_on_factor', 'Improve recall for small_object.',
             '8 of 10 are misses.', 'replicated', 1, 10, 10.0),
            (2, 'unexplained_backlog', 'Investigate.',
             'Nothing accounts for these.', 'insufficient_evidence', 0, 40, 40.0),
        ],
    )
    # A measured outline, and one never measured.
    connection.executemany(
        "INSERT INTO mask_findings VALUES (?, ?, 1, ?, ?, ?)",
        [
            (1, 1, 0.12, 'poor_localization', '[[0,0],[5,0],[5,5]]'),
            (2, 4, None, None, None),
        ],
    )

    # Insert heatmap for Finding 2
    connection.execute(
        "INSERT INTO heatmaps VALUES "
        "(1, 2, 1, '/data/heatmap_2.jpg', 'grad-cam', 'layer4')"
    )

    # Insert multiple root causes for Finding 2
    connection.executemany(
        "INSERT INTO root_causes VALUES (?, ?, 1, ?, ?, ?)",
        [
            (1, 2, 'blur', 0.9, 'Laplacian variance 5.1 < 100'),
            (2, 2, 'edge_truncation', 0.85, 'touches left, top, bottom'),
        ]
    )

    # Insert clusters and members
    connection.executemany(
        "INSERT INTO clusters VALUES (?, 1, 'discriminating-signature', ?, ?)",
        [
            (1, 'blur + edge_truncation', 1),
            (2, 'unexplained', 2),
        ]
    )
    connection.executemany(
        "INSERT INTO cluster_members VALUES (?, ?)",
        [
            (1, 2),
            (2, 3),
            (2, 4),
        ]
    )

    connection.commit()
    connection.close()


@pytest.fixture
def dashboard_database(tmp_path: Path) -> Path:
    """Provide a schema-valid saved run with varied outcomes and an image error."""
    database = tmp_path / "model_doctor.db"
    create_dashboard_database(database)
    return database


@pytest.fixture
def optional_dashboard_database(tmp_path: Path) -> Path:
    """Provide a database with all optional schema tables populated."""
    database = tmp_path / "optional_model_doctor.db"
    create_optional_dashboard_database(database)
    return database


@pytest.fixture
def two_run_database(tmp_path: Path) -> Path:
    """Provide a database with two runs for comparison testing."""
    database = tmp_path / "two_run.db"
    create_two_run_database(database)
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


def test_class_names_returns_distinct_sorted_classes(
    dashboard_database: Path,
) -> None:
    """The class filter needs a sorted list of unique class names."""
    names = load_class_names(dashboard_database, 1)
    assert names == ["crack", "spall"]


def test_class_statistics_filtered_narrows_to_selected_classes(
    dashboard_database: Path,
) -> None:
    """Class filter must restrict statistics to selected class names only."""
    stats = load_class_statistics_filtered(
        dashboard_database,
        1,
        ("correct", "false_positive", "false_negative"),
        ("crack",),
    )
    assert len(stats) == 1
    assert stats[0]["class_name"] == "crack"
    assert stats[0]["false_positive"] == 2


def test_class_statistics_filtered_returns_all_when_no_class_filter(
    dashboard_database: Path,
) -> None:
    """Passing ``None`` for class_names disables class filtering."""
    stats = load_class_statistics_filtered(
        dashboard_database,
        1,
        ("correct", "false_positive", "false_negative"),
        None,
    )
    assert len(stats) == 2
    class_names = [row["class_name"] for row in stats]
    assert "crack" in class_names
    assert "spall" in class_names


def test_class_statistics_filtered_empty_class_list_returns_empty(
    dashboard_database: Path,
) -> None:
    """An explicit empty class list is a no-results filter, not 'all classes'."""
    stats = load_class_statistics_filtered(
        dashboard_database,
        1,
        ("correct",),
        (),
    )
    assert stats == []


def test_run_comparison_returns_grouped_outcome_counts(
    two_run_database: Path,
) -> None:
    """Comparing two runs groups outcomes by run_id per SCHEMA.md §5."""
    data = load_run_comparison(two_run_database, 1, 2)
    run_1_rows = [row for row in data if int(row["run_id"]) == 1]
    run_2_rows = [row for row in data if int(row["run_id"]) == 2]

    run_1_outcomes = {str(r["outcome"]): int(r["n"]) for r in run_1_rows}
    run_2_outcomes = {str(r["outcome"]): int(r["n"]) for r in run_2_rows}

    assert run_1_outcomes.get("correct") == 1
    assert run_1_outcomes.get("false_positive") == 1
    assert run_2_outcomes.get("correct") == 2


def test_outcome_counts_empty_filter_returns_empty(
    dashboard_database: Path,
) -> None:
    """An empty outcome list is an explicit 'show nothing' filter."""
    counts = load_outcome_counts(dashboard_database, 1, ())
    assert counts == []


def test_validate_database_rejects_missing_tables(tmp_path: Path) -> None:
    """A database without the required tables is clearly reported."""
    database = tmp_path / "bad.db"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    with pytest.raises(DashboardDataError, match="Missing table"):
        validate_database(database)


def test_load_runs_returns_newest_first(two_run_database: Path) -> None:
    """Runs are ordered newest first by descending id."""
    runs = load_runs(two_run_database)
    assert len(runs) == 2
    assert int(runs[0]["id"]) == 2
    assert int(runs[1]["id"]) == 1


def test_image_findings_degrade_when_optional_tables_are_absent(
    dashboard_database: Path,
) -> None:
    """A database without heatmaps or root causes still renders its findings.

    ``validate_database`` requires only runs, images, and findings, so a run
    saved before those tables existed must not take the whole page down.
    """
    findings = load_image_findings(
        dashboard_database, image_id=1, outcomes=("correct", "false_positive")
    )

    assert len(findings) == 3
    assert all(row["heatmap_path"] is None for row in findings)
    assert all(row["root_causes_json"] is None for row in findings)


def test_root_causes_summary_is_empty_when_the_table_is_absent(
    dashboard_database: Path,
) -> None:
    """The root-cause surface reports nothing rather than raising."""
    assert load_root_causes_summary(dashboard_database, run_id=1) == []


def test_load_image_findings_with_optional_data(
    optional_dashboard_database: Path,
) -> None:
    """Findings optionally carry heatmaps and root causes."""
    findings = load_image_findings(
        optional_dashboard_database, image_id=1, outcomes=("correct", "false_positive")
    )
    assert len(findings) == 3

    # Finding 1: Case A (correct, no optional data)
    finding_1 = next(f for f in findings if f["finding_id"] == 1)
    assert finding_1["heatmap_path"] is None
    assert finding_1["root_causes_json"] == "[]"

    # Finding 2: Case B (heatmap and multiple root causes)
    finding_2 = next(f for f in findings if f["finding_id"] == 2)
    assert finding_2["heatmap_path"] == "/data/heatmap_2.jpg"
    import json
    rc_list = json.loads(str(finding_2["root_causes_json"]))
    assert len(rc_list) == 2
    factors = [rc["factor"] for rc in rc_list]
    assert "blur" in factors
    assert "edge_truncation" in factors

    # Finding 3: Case C (no heatmap, no root causes)
    finding_3 = next(f for f in findings if f["finding_id"] == 3)
    assert finding_3["heatmap_path"] is None
    assert finding_3["root_causes_json"] == "[]"


def test_root_causes_breakdowns_with_optional_data(
    optional_dashboard_database: Path,
) -> None:
    """Breakdowns calculate percentages correctly and use COUNT(DISTINCT)."""
    # Total failures for crack: findings #2 and #3, both false_positive -> 2
    # spall (1 false_negative finding #4) -> total 1
    # Note: Finding 1 is correct (not a failure). Finding 2 has blur & edge_truncation.

    class_breakdown = load_root_causes_by_class(optional_dashboard_database, run_id=1)

    blur_class = next(
        r
        for r in class_breakdown
        if r["factor"] == "blur" and r["class_name"] == "crack"
    )
    assert blur_class["count"] == 1
    # total crack failures = 2 (finding 2 & 3). So 1/2 = 50.0%
    assert blur_class["percentage"] == 50.0

    outcome_breakdown = load_root_causes_by_outcome(
        optional_dashboard_database, run_id=1
    )
    blur_outcome = next(
        r
        for r in outcome_breakdown
        if r["factor"] == "blur" and r["outcome"] == "false_positive"
    )
    assert blur_outcome["count"] == 1
    # total false_positive = 2. So 1/2 = 50.0%
    assert blur_outcome["percentage"] == 50.0


def test_available_tables_reports_every_optional_table_it_guards(
    optional_dashboard_database: Path,
) -> None:
    """Every table guarded by ``available_tables`` must be in ``OPTIONAL_TABLES``.

    The loaders guard themselves with
    ``{"clusters", "cluster_members"}.issubset(available_tables(db))``, but
    ``available_tables`` returns ``OPTIONAL_TABLES & <tables present>``. A table
    missing from that constant can therefore never appear in the result, so the
    guard is permanently false and the feature silently returns nothing on a
    database that does have the table. This asserts the two stay in step.
    """
    reported = available_tables(optional_dashboard_database)

    assert reported == {
        "heatmaps",
        "root_causes",
        "clusters",
        "cluster_members",
        "factor_rates",
        "recommendations",
        "mask_findings",
    }


def test_failure_groups_degrade_when_optional_tables_are_absent(
    dashboard_database: Path,
) -> None:
    """Missing clusters/cluster_members tables result in graceful empty returns."""
    assert load_failure_groups(dashboard_database, 1) == []
    assert load_failure_group_members(dashboard_database, 1) == []
    assert load_failure_group_outcomes(dashboard_database, 1) == []
    assert load_failure_group_classes(dashboard_database, 1) == []
    assert load_finding_failure_group(dashboard_database, 1) is None
    assert check_failure_groups_integrity(dashboard_database, 1) is None


def test_failure_groups_queries(
    optional_dashboard_database: Path,
) -> None:
    """Failure group endpoints execute correctly over populated schema."""
    # Group list
    groups = load_failure_groups(optional_dashboard_database, 1)
    assert len(groups) == 2
    # unexplained group check
    unexplained = next(g for g in groups if g["label"] == "unexplained")
    assert unexplained["size"] == 2
    blur_edge = next(g for g in groups if g["label"] == "blur + edge_truncation")
    assert blur_edge["size"] == 1

    # Group members
    members = load_failure_group_members(optional_dashboard_database, unexplained["id"])
    assert len(members) == 2
    # Finding 3 (false_positive) and Finding 4 (false_negative) are in unexplained
    member_ids = [m["id"] for m in members]
    assert 3 in member_ids
    assert 4 in member_ids

    # Outcome breakdown
    outcomes = load_failure_group_outcomes(optional_dashboard_database, 1)
    # unexplained has 1 false_positive (finding 3) and 1 false_negative (finding 4)
    unexp_fp = next(
        o
        for o in outcomes
        if o["label"] == "unexplained" and o["outcome"] == "false_positive"
    )
    assert unexp_fp["n"] == 1

    # Class breakdown
    classes = load_failure_group_classes(optional_dashboard_database, 1)
    unexp_crack = next(
        c
        for c in classes
        if c["label"] == "unexplained" and c["class_name"] == "crack"
    )
    assert unexp_crack["n"] == 1

    # Finding lookup
    label_3 = load_finding_failure_group(optional_dashboard_database, 3)
    assert label_3 == "unexplained"
    label_2 = load_finding_failure_group(optional_dashboard_database, 2)
    assert label_2 == "blur + edge_truncation"

    # Integrity check
    assert check_failure_groups_integrity(optional_dashboard_database, 1) is None





# ---------------------------------------------------------------------------
# Factor rates, recommendations, outlines
# ---------------------------------------------------------------------------
def test_factor_rates_are_ranked_by_lift_not_count(
    optional_dashboard_database: Path,
) -> None:
    """edge_truncation has the larger count and the smaller lift.

    Ranking by count puts the misleading factor first, which is what the tab
    used to do (D-031).
    """
    rates = load_factor_rates(optional_dashboard_database, run_id=1)

    assert [row["factor"] for row in rates][:2] == ["small_object", "edge_truncation"]


def test_an_undefined_lift_sorts_last_and_stays_null(
    optional_dashboard_database: Path,
) -> None:
    """NULL means the ratio is undefined, never that it is large."""
    rates = load_factor_rates(optional_dashboard_database, run_id=1)

    assert rates[-1]["factor"] == "sparse_factor"
    assert rates[-1]["lift"] is None


def test_recommendations_keep_the_documented_order(
    optional_dashboard_database: Path,
) -> None:
    """Actionable first, then by failures addressed."""
    rows = load_recommendations(optional_dashboard_database, run_id=1)

    assert [bool(row["actionable"]) for row in rows] == [True, False]


def test_the_refusal_is_returned_not_filtered(
    optional_dashboard_database: Path,
) -> None:
    """Two of four statuses decline to advise, and are the honest half (D-035)."""
    rows = load_recommendations(optional_dashboard_database, run_id=1)

    assert any(row["status"] == "insufficient_evidence" for row in rows)


def test_recommendations_carry_their_group_label(
    optional_dashboard_database: Path,
) -> None:
    """A consumer should not need a second query to name the group."""
    rows = load_recommendations(optional_dashboard_database, run_id=1)

    assert all(row["group_label"] for row in rows)


def test_mask_findings_preserve_the_unmeasured_null(
    optional_dashboard_database: Path,
) -> None:
    """A false negative has no predicted outline. That is not zero overlap."""
    rows = load_mask_findings(optional_dashboard_database, run_id=1)

    scores = {row["finding_id"]: row["mask_iou"] for row in rows}
    assert scores[1] == pytest.approx(0.12)
    assert scores[4] is None


def test_the_mask_class_summary_excludes_unmeasured_findings(
    optional_dashboard_database: Path,
) -> None:
    """Averaging a null as zero would drag the mean down with absent data."""
    summary = load_mask_class_summary(optional_dashboard_database, run_id=1)

    assert summary
    assert all(row["pairs"] >= 1 for row in summary)
    assert all(row["mean_mask_iou"] is not None for row in summary)


def test_the_new_surfaces_degrade_when_their_tables_are_absent(
    dashboard_database: Path,
) -> None:
    """A pre-v6 database still renders; these surfaces are simply empty."""
    assert load_factor_rates(dashboard_database, run_id=1) == []
    assert load_recommendations(dashboard_database, run_id=1) == []
    assert load_mask_findings(dashboard_database, run_id=1) == []
    assert load_mask_class_summary(dashboard_database, run_id=1) == []
