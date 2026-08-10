"""Tests for feature extraction.

Every test here runs without downloading a vision-language model. That is the
point of injecting the encoder: a real CLIP checkpoint is several hundred
megabytes, so a suite that required one would be untestable in practice and
would quietly stop being run.

The tests cover region extraction (the seam mask support will use), the
database pass, and the failure modes that must not abort a batch.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import storage
from app.diagnosis import DatasetDiagnosis, diagnose_image
from app.features import (
    MIN_REGION_PIXELS,
    ExtractionReport,
    FeatureExtractionError,
    extract_run_embeddings,
    region_from_box,
)
from utils.annotations import ObjectAnnotation


class FakeBackend:
    """Deterministic stand-in for a real encoder.

    Produces a vector derived from the crop's size, so tests can assert that
    the *right region* reached the encoder rather than merely that something
    did.
    """

    def __init__(self, dimensions: int = 4) -> None:
        """Record the vector width and prepare the seen-crop log."""
        self._dimensions = dimensions
        self.seen: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        """Return the encoder identifier."""
        return "fake/test"

    @property
    def dimensions(self) -> int:
        """Return the vector width."""
        return self._dimensions

    def encode(self, images):
        """Return one size-derived vector per image."""
        vectors = []
        for image in images:
            self.seen.append(image.size)
            width, height = image.size
            padding = [0.0] * (self._dimensions - 2)
            vectors.append([float(width), float(height), *padding])
        return vectors


class Row(dict):
    """Minimal stand-in for a sqlite3.Row, which supports mapping access."""


def _row(**overrides) -> Row:
    """Build a finding row with sensible defaults."""
    row = Row(
        finding_id=1,
        outcome="false_negative",
        class_name="door",
        pred_x1=None, pred_y1=None, pred_x2=None, pred_y2=None,
        truth_x1=10.0, truth_y1=10.0, truth_x2=110.0, truth_y2=110.0,
        truth_polygon=None,
        path="x.jpg",
        width=640,
        height=480,
    )
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Region extraction — the seam mask support will reuse
# ---------------------------------------------------------------------------
def test_region_prefers_ground_truth() -> None:
    """A finding with both boxes encodes the ground-truth region.

    That matches how a finding attributes its class, so per-class grouping and
    per-class embeddings describe the same thing.
    """
    row = _row(pred_x1=200.0, pred_y1=200.0, pred_x2=300.0, pred_y2=300.0)
    x1, y1, x2, y2 = region_from_box(row, padding=0.0)
    assert (x1, y1, x2, y2) == (10, 10, 110, 110)


def test_region_falls_back_to_prediction() -> None:
    """A false positive has no ground truth, so it encodes what was predicted."""
    row = _row(
        outcome="false_positive",
        truth_x1=None, truth_y1=None, truth_x2=None, truth_y2=None,
        pred_x1=20.0, pred_y1=30.0, pred_x2=120.0, pred_y2=130.0,
    )
    assert region_from_box(row, padding=0.0) == (20, 30, 120, 130)


def test_region_adds_padding() -> None:
    """Context around the object is included, because context explains failures."""
    region = region_from_box(_row(), padding=0.1)
    assert region == (0, 0, 120, 120)


def test_region_is_clamped_to_the_image() -> None:
    """Padding past the frame clips rather than producing invalid coordinates.

    Most objects in this project's data touch an edge, so clipping is the
    normal path, not an error case.
    """
    row = _row(truth_x1=0.0, truth_y1=0.0, truth_x2=640.0, truth_y2=480.0)
    x1, y1, x2, y2 = region_from_box(row, padding=0.5)
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (640, 480)


def test_region_rejects_a_finding_with_no_geometry() -> None:
    """Nothing to crop means nothing to encode."""
    row = _row(
        truth_x1=None, truth_y1=None, truth_x2=None, truth_y2=None,
        pred_x1=None, pred_y1=None, pred_x2=None, pred_y2=None,
    )
    assert region_from_box(row) is None


def test_region_rejects_a_region_that_is_too_small() -> None:
    """A few pixels carry no recoverable signal and would only add noise."""
    tiny = MIN_REGION_PIXELS - 2
    row = _row(truth_x1=0.0, truth_y1=0.0, truth_x2=float(tiny), truth_y2=float(tiny))
    assert region_from_box(row, padding=0.0) is None


def test_region_rejects_unknown_image_dimensions() -> None:
    """An image that failed to process has no size, so it cannot be cropped."""
    assert region_from_box(_row(width=None, height=None)) is None


# ---------------------------------------------------------------------------
# The database pass
# ---------------------------------------------------------------------------
def _seed(tmp_path: Path, images: int = 2) -> tuple[Path, Path, int]:
    """Create a database with a saved run over real image files."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    diagnoses = []
    for index in range(images):
        path = image_dir / f"frame_{index}.jpg"
        Image.new("RGB", (200, 200), (120, 130, 140)).save(path)
        diagnoses.append(
            diagnose_image(
                path,
                [
                    # Matches the first truth exactly -> correct.
                    ObjectAnnotation.from_box(
                        0, "door", 10, 10, 60, 60, confidence=0.9
                    ),
                    # Overlaps nothing -> false positive.
                    ObjectAnnotation.from_box(
                        0, "door", 150, 150, 190, 190, confidence=0.8
                    ),
                ],
                [
                    ObjectAnnotation.from_box(0, "door", 10, 10, 60, 60),
                    # Nothing predicts this -> false negative.
                    ObjectAnnotation.from_box(0, "door", 100, 20, 140, 60),
                ],
                image_width=200,
                image_height=200,
            )
        )

    db = tmp_path / "d.db"
    context = storage.RunContext(
        model_path="m.pt", model_sha256="a" * 64, dataset_yaml="d.yaml",
        split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
        localization_iou_floor=0.1, image_size=640,
    )
    with storage.connect(db) as conn:
        run_id = storage.save_dataset_diagnosis(
            conn, context, DatasetDiagnosis(diagnoses=diagnoses)
        )
    return db, image_dir, run_id


def test_extraction_embeds_failures_and_stores_them(tmp_path: Path) -> None:
    """A saved run round-trips into stored vectors."""
    db, _, run_id = _seed(tmp_path)
    backend = FakeBackend()

    with storage.connect(db) as conn:
        report = extract_run_embeddings(conn, run_id, backend)
        stored = storage.load_embeddings(conn, run_id)

    assert isinstance(report, ExtractionReport)
    assert report.embedded > 0
    assert len(stored) == report.embedded
    assert all(row.model_name == "fake/test" for row in stored)
    assert all(len(row.vector) == backend.dimensions for row in stored)


def test_extraction_skips_correct_findings_by_default(tmp_path: Path) -> None:
    """The roadmap asks for embeddings of *failures*."""
    db, _, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
        stored = storage.load_embeddings(conn, run_id)
    assert stored
    assert all(row.outcome != "correct" for row in stored)


def test_extraction_can_include_correct_findings(tmp_path: Path) -> None:
    """A baseline of correct regions is available when wanted."""
    db, _, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        only_failures = extract_run_embeddings(conn, run_id, FakeBackend())
        everything = extract_run_embeddings(
            conn, run_id, FakeBackend(), failures_only=False
        )
    assert everything.considered > only_failures.considered


def test_embeddings_carry_their_findings_outcome_and_class(tmp_path: Path) -> None:
    """Clustering can label vectors without a second query."""
    db, _, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
        stored = storage.load_embeddings(conn, run_id)
    assert all(row.class_name == "door" for row in stored)
    assert all(row.outcome for row in stored)


def test_re_extraction_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """Re-running replaces vectors rather than accumulating duplicates.

    Embeddings are derived data: recomputing one is a correction, not a second
    observation, which is the opposite of how runs behave (D-021).
    """
    db, _, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
        first = len(storage.load_embeddings(conn, run_id))
    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
        second = len(storage.load_embeddings(conn, run_id))
    assert first == second


def test_two_encoders_coexist_without_mixing(tmp_path: Path) -> None:
    """Vectors from different encoders are not comparable and stay separable."""
    db, _, run_id = _seed(tmp_path)

    class OtherBackend(FakeBackend):
        @property
        def name(self) -> str:
            return "other/test"

    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
        extract_run_embeddings(conn, run_id, OtherBackend())
        first = storage.load_embeddings(conn, run_id, model_name="fake/test")
        second = storage.load_embeddings(conn, run_id, model_name="other/test")
        both = storage.load_embeddings(conn, run_id)

    assert len(first) == len(second)
    assert len(both) == len(first) + len(second)


def test_unreadable_image_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One missing file must not abandon the whole pass."""
    db, image_dir, run_id = _seed(tmp_path, images=2)
    next(image_dir.glob("*.jpg")).unlink()

    with storage.connect(db) as conn:
        report = extract_run_embeddings(conn, run_id, FakeBackend())

    assert report.skipped_unreadable > 0
    assert report.embedded > 0


def test_run_with_nothing_to_embed_raises_clearly(tmp_path: Path) -> None:
    """An empty run says what to do rather than silently producing nothing."""
    db = tmp_path / "d.db"
    with storage.connect(db) as conn:
        run_id = storage.save_run(
            conn,
            storage.RunContext(
                model_path="m.pt", model_sha256="a" * 64, dataset_yaml="d.yaml",
                split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
                localization_iou_floor=0.1, image_size=640,
            ),
        )
        with pytest.raises(FeatureExtractionError, match="no findings"):
            extract_run_embeddings(conn, run_id, FakeBackend())


def test_region_extractor_is_a_parameter(tmp_path: Path) -> None:
    """The seam mask support will use is exercised, so it cannot rot.

    A custom extractor changes which pixels reach the encoder without any
    change to the extraction pipeline itself.
    """
    db, _, run_id = _seed(tmp_path)
    backend = FakeBackend()

    def fixed_region(_row):
        """Ignore the finding and always crop the same rectangle."""
        return (0, 0, 32, 16)

    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, backend, region=fixed_region)

    assert backend.seen
    assert all(size == (32, 16) for size in backend.seen)


def test_batching_covers_every_finding(tmp_path: Path) -> None:
    """A batch size smaller than the workload still encodes everything."""
    db, _, run_id = _seed(tmp_path, images=5)
    backend = FakeBackend()
    with storage.connect(db) as conn:
        report = extract_run_embeddings(conn, run_id, backend, batch_size=2)
    assert len(backend.seen) == report.embedded


def test_embeddings_cascade_on_run_delete(tmp_path: Path) -> None:
    """Deleting a run leaves no orphaned vectors."""
    db, _, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        extract_run_embeddings(conn, run_id, FakeBackend())
    with storage.connect(db) as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    with storage.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"] == 0


def test_schema_upgrades_from_version_one(tmp_path: Path) -> None:
    """An existing v1 database gains the new table without losing data.

    Jawad may already have a v1 database. Opening it must not fail and must not
    discard his runs.
    """
    db = tmp_path / "legacy.db"
    connection = sqlite3.connect(db)
    connection.executescript(
        """
        CREATE TABLE schema_info (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
        INSERT INTO schema_info VALUES (1, '2026-01-01T00:00:00+00:00');
        """
    )
    connection.commit()
    connection.close()

    with storage.connect(db) as conn:
        version = conn.execute("SELECT version FROM schema_info").fetchone()["version"]
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert version == storage.SCHEMA_VERSION == 2
    assert "embeddings" in tables
    assert {"runs", "images", "findings"} <= tables
