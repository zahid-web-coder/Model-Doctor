"""Tests for the SQLite persistence layer.

The database is a *published interface* — a second developer builds a dashboard
against it without reading this code. So these tests check the contract, not
just that writes succeed: every field survives a round trip, foreign keys are
actually enforced, the schema is safe to re-apply, and multiple runs coexist
without interfering.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage
from app.diagnosis import DatasetDiagnosis, Outcome, diagnose_image
from utils.annotations import ObjectAnnotation


def _truth(x1, y1, x2, y2, cls: int = 0) -> ObjectAnnotation:
    """Build a ground-truth annotation."""
    names = {0: "door", 1: "door_frame"}
    return ObjectAnnotation.from_box(cls, names[cls], x1, y1, x2, y2)


def _pred(x1, y1, x2, y2, cls: int = 0, conf: float = 0.9) -> ObjectAnnotation:
    """Build a prediction."""
    return ObjectAnnotation.from_box(
        cls, {0: "door", 1: "door_frame"}[cls], x1, y1, x2, y2, confidence=conf
    )


def _context(split: str = "test") -> storage.RunContext:
    """Build a run context without touching the filesystem."""
    return storage.RunContext(
        model_path="models/best.pt",
        model_sha256="a" * 64,
        dataset_yaml="datasets/data.yaml",
        split=split,
        confidence_threshold=0.25,
        match_iou_threshold=0.5,
        localization_iou_floor=0.1,
        image_size=672,
    )


def _sample(width: int = 640, height: int = 480) -> DatasetDiagnosis:
    """A diagnosis containing every outcome at least once."""
    return DatasetDiagnosis(
        diagnoses=[
            diagnose_image(
                Path("a.jpg"),
                [_pred(0, 0, 10, 10), _pred(500, 500, 510, 510)],
                [_truth(0, 0, 10, 10)],
                image_width=width,
                image_height=height,
            ),
            diagnose_image(
                Path("b.jpg"),
                [_pred(6, 0, 16, 10, cls=1)],
                [_truth(0, 0, 10, 10, cls=1), _truth(900, 900, 910, 910, cls=0)],
                image_width=width,
                image_height=height,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_creates_expected_tables(tmp_path: Path) -> None:
    """The three contract tables exist after opening a fresh database."""
    with storage.connect(tmp_path / "d.db") as conn:
        names = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"runs", "images", "findings", "schema_info"} <= names


def test_schema_creation_is_idempotent(tmp_path: Path) -> None:
    """Re-opening an existing database is safe and does not duplicate rows."""
    path = tmp_path / "d.db"
    for _ in range(3):
        with storage.connect(path) as conn:
            pass
    with storage.connect(path) as conn:
        rows = conn.execute("SELECT COUNT(*) AS n FROM schema_info").fetchone()
    assert rows["n"] == 1


def test_schema_version_is_recorded(tmp_path: Path) -> None:
    """Consumers can detect a schema they were not written against."""
    with storage.connect(tmp_path / "d.db") as conn:
        row = conn.execute("SELECT version FROM schema_info").fetchone()
    assert row["version"] == storage.SCHEMA_VERSION


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    """A finding cannot reference a run that does not exist.

    SQLite ignores foreign keys unless the pragma is set per connection, so
    this asserts the pragma is actually applied rather than assumed.
    """
    with storage.connect(tmp_path / "d.db") as conn, pytest.raises(
        sqlite3.IntegrityError
    ):
        conn.execute(
            "INSERT INTO findings (run_id, image_id, outcome, class_name) "
            "VALUES (999, 999, 'correct', 'x')"
        )


def test_deleting_a_run_removes_its_rows(tmp_path: Path) -> None:
    """Cascade delete keeps the database free of orphans."""
    path = tmp_path / "d.db"
    with storage.connect(path) as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
    with storage.connect(path) as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    with storage.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM findings").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 0


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_run_round_trips_every_field(tmp_path: Path) -> None:
    """Every reproducibility field survives the write/read cycle."""
    context = _context()
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_run(conn, context)
        loaded = storage.load_run(conn, run_id)

    assert loaded is not None
    assert loaded.model_path == context.model_path
    assert loaded.model_sha256 == context.model_sha256
    assert loaded.dataset_yaml == context.dataset_yaml
    assert loaded.split == context.split
    assert loaded.confidence_threshold == context.confidence_threshold
    assert loaded.match_iou_threshold == context.match_iou_threshold
    assert loaded.localization_iou_floor == context.localization_iou_floor
    assert loaded.image_size == context.image_size
    assert loaded.created_at


def test_findings_round_trip_with_boxes_and_iou(tmp_path: Path) -> None:
    """Geometry and overlap survive persistence exactly."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        records = storage.load_findings(conn, run_id)

    correct = [r for r in records if r.outcome == Outcome.CORRECT.value]
    assert len(correct) == 1
    assert correct[0].pred_box == (0.0, 0.0, 10.0, 10.0)
    assert correct[0].truth_box == (0.0, 0.0, 10.0, 10.0)
    assert correct[0].iou == pytest.approx(1.0)
    assert correct[0].confidence == pytest.approx(0.9)


def test_false_positive_has_no_truth_box(tmp_path: Path) -> None:
    """The null pattern documented in SCHEMA.md holds in practice."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        records = storage.load_findings(conn, run_id, Outcome.FALSE_POSITIVE)

    assert records
    for record in records:
        assert record.pred_box is not None
        assert record.truth_box is None
        assert record.iou is None


def test_false_negative_has_no_prediction(tmp_path: Path) -> None:
    """A miss stores ground truth only, with no confidence."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        records = storage.load_findings(conn, run_id, Outcome.FALSE_NEGATIVE)

    assert records
    for record in records:
        assert record.truth_box is not None
        assert record.pred_box is None
        assert record.confidence is None


def test_polygon_round_trips(tmp_path: Path) -> None:
    """Ground-truth outlines survive as JSON without losing precision."""
    polygon = ((1.5, 2.5), (30.25, 4.0), (20.0, 40.75))
    truth = ObjectAnnotation.from_polygon(0, "door", polygon)
    diagnosis = DatasetDiagnosis(
        diagnoses=[
            diagnose_image(
                Path("p.jpg"), [], [truth], image_width=100, image_height=100
            )
        ]
    )
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), diagnosis)
        records = storage.load_findings(conn, run_id)

    assert records[0].truth_polygon == [[1.5, 2.5], [30.25, 4.0], [20.0, 40.75]]


def test_box_only_annotation_stores_no_polygon(tmp_path: Path) -> None:
    """A detection-only dataset leaves the polygon column null."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        records = storage.load_findings(conn, run_id)
    assert all(r.truth_polygon is None for r in records)


def test_image_rows_record_dimensions_and_counts(tmp_path: Path) -> None:
    """Images carry the frame their pixel boxes are relative to."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample(800, 600))
        rows = conn.execute(
            "SELECT * FROM images WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()

    assert len(rows) == 2
    assert rows[0]["width"] == 800 and rows[0]["height"] == 600
    assert rows[0]["filename"] == "a.jpg"
    assert rows[0]["prediction_count"] == 2
    assert rows[0]["truth_count"] == 1
    assert rows[0]["error"] is None


def test_errored_image_is_recorded_not_dropped(tmp_path: Path) -> None:
    """"Processed and found nothing" must differ from "never processed"."""
    from app.diagnosis import ImageDiagnosis

    failed = DatasetDiagnosis(
        diagnoses=[ImageDiagnosis(image_path=Path("bad.jpg"), error="unreadable")]
    )
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), failed)
        row = conn.execute(
            "SELECT * FROM images WHERE run_id = ?", (run_id,)
        ).fetchone()

    assert row["error"] == "unreadable"
    assert row["prediction_count"] == 0


# ---------------------------------------------------------------------------
# Multiple runs
# ---------------------------------------------------------------------------
def test_runs_accumulate_and_stay_isolated(tmp_path: Path) -> None:
    """Saving twice creates two runs whose findings do not mix.

    Runs are events, not idempotent upserts — keeping every one is what makes
    model-versus-model comparison possible later.
    """
    path = tmp_path / "d.db"
    with storage.connect(path) as conn:
        first = storage.save_dataset_diagnosis(conn, _context("val"), _sample())
        second = storage.save_dataset_diagnosis(conn, _context("test"), _sample())

    assert first != second
    with storage.connect(path) as conn:
        assert len(storage.load_findings(conn, first)) == len(
            storage.load_findings(conn, second)
        )
        assert {r.split for r in storage.list_runs(conn)} == {"val", "test"}


def test_list_runs_is_newest_first(tmp_path: Path) -> None:
    """The dashboard's run picker gets the most recent run at the top."""
    path = tmp_path / "d.db"
    with storage.connect(path) as conn:
        for split in ("train", "val", "test"):
            storage.save_run(conn, _context(split))
    with storage.connect(path) as conn:
        runs = storage.list_runs(conn)
    assert [r.split for r in runs] == ["test", "val", "train"]


def test_outcome_counts_matches_the_engine(tmp_path: Path) -> None:
    """Stored aggregates equal what the diagnosis engine reported.

    Guards the seam between analysis and persistence: if a finding were dropped
    on write, the totals would silently disagree.
    """
    summary = _sample()
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), summary)
        stored = storage.outcome_counts(conn, run_id)

    expected = {o.value: n for o, n in summary.counts.items() if n}
    assert stored == expected


def test_load_run_returns_none_for_unknown_id(tmp_path: Path) -> None:
    """Asking for a run that does not exist is not an error."""
    with storage.connect(tmp_path / "d.db") as conn:
        assert storage.load_run(conn, 4242) is None


def test_failed_write_leaves_no_partial_run(tmp_path: Path) -> None:
    """A crash mid-write rolls back rather than leaving half a run behind."""
    path = tmp_path / "d.db"
    with pytest.raises(RuntimeError), storage.connect(path) as conn:
        storage.save_run(conn, _context())
        raise RuntimeError("interrupted")
    with storage.connect(path) as conn:
        assert storage.list_runs(conn) == []


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    """Model fingerprinting is deterministic and content-sensitive."""
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"weights")
    b.write_bytes(b"weights")
    assert storage.file_sha256(a) == storage.file_sha256(b)
    b.write_bytes(b"different")
    assert storage.file_sha256(a) != storage.file_sha256(b)


def test_format_run_summary_handles_empty() -> None:
    """The run listing renders when nothing has been saved."""
    assert "No runs recorded" in storage.format_run_summary([])


def test_findings_page_matches_slicing_the_whole_run(tmp_path: Path) -> None:
    """Paging in SQL must return exactly what paging in Python returned.

    The endpoint used to load every finding and slice the list. That is correct
    and unaffordable: the cost of one page grew with the size of the run rather
    than the size of the page. Moving the window into SQL is only safe if the
    two agree on order as well as content, so this compares them directly
    rather than asserting a hand-written expectation.
    """
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        every = storage.load_findings(conn, run_id)

        for size in (1, 2, len(every), len(every) + 5):
            paged = []
            for offset in range(0, len(every), max(1, size)):
                paged.extend(storage.load_findings_page(conn, run_id, size, offset))
            assert [r.id for r in paged] == [r.id for r in every], (
                f"page size {size} did not reproduce the full ordering"
            )

        # Past the end is empty, not an error and not a wrapped-around page.
        assert storage.load_findings_page(conn, run_id, 10, len(every) + 1) == []


def test_findings_page_respects_the_outcome_filter(tmp_path: Path) -> None:
    """Filtering and paging together must not drop or duplicate rows."""
    with storage.connect(tmp_path / "d.db") as conn:
        run_id = storage.save_dataset_diagnosis(conn, _context(), _sample())
        expected = storage.load_findings(conn, run_id, Outcome.CORRECT)
        paged = storage.load_findings_page(
            conn, run_id, 100, 0, outcome=Outcome.CORRECT
        )

    assert [r.id for r in paged] == [r.id for r in expected]
    assert all(r.outcome == Outcome.CORRECT.value for r in paged)
