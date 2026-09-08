"""Tests for outline overlap and mask-level diagnosis.

The distinction that matters throughout: an outline that overlapped nothing
scores ``0.0``, and an outline that was never produced scores ``None``. Losing
that difference would turn every unsegmented prediction into a perfect failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.mask_diagnosis import (
    MASK_CORRECT,
    MASK_NO_OVERLAP,
    MASK_POOR,
    classify_mask,
)
from model_doctor.app.storage import RunContext
from model_doctor.utils.masks import annotation_mask_iou, mask_iou, polygon_bounds

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
SHIFTED = [[5, 0], [15, 0], [15, 10], [5, 10]]
FAR = [[100, 100], [110, 100], [110, 110], [100, 110]]


# ---------------------------------------------------------------------------
# Outline overlap
# ---------------------------------------------------------------------------
def test_identical_outlines_overlap_completely() -> None:
    """The anchor for the scale."""
    assert mask_iou(SQUARE, SQUARE) == pytest.approx(1.0, abs=0.02)


def test_disjoint_outlines_score_zero_not_none() -> None:
    """Zero is a measurement: these were compared and share nothing."""
    assert mask_iou(SQUARE, FAR) == 0.0


def test_half_overlapping_squares_score_about_a_third() -> None:
    """Two 10x10 squares offset by 5 share 50 of 150 pixels."""
    assert mask_iou(SQUARE, SHIFTED) == pytest.approx(1 / 3, abs=0.05)


def test_a_missing_outline_is_unmeasured_not_zero() -> None:
    """The distinction the whole module rests on."""
    assert mask_iou(None, SQUARE) is None
    assert mask_iou(SQUARE, None) is None
    assert mask_iou(None, None) is None


def test_an_outline_with_too_few_vertices_encloses_nothing() -> None:
    """Two points are a line. A line has no area to compare."""
    assert mask_iou([[0, 0], [1, 1]], SQUARE) is None


def test_a_thin_annulus_scores_far_below_its_bounding_box() -> None:
    """The door_frame case, in miniature.

    An outer ring and its slightly-shifted twin share almost no pixels, while
    their bounding boxes overlap almost entirely. This is why box IoU reports
    thin structures as healthier than they are (D-022).
    """
    ring = [[0, 0], [100, 0], [100, 4], [0, 4]]  # a thin horizontal bar
    shifted_ring = [[0, 6], [100, 6], [100, 10], [0, 10]]

    from model_doctor.utils.geometry import BoxGeometryMixin, box_iou

    class Box(BoxGeometryMixin):
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    boxes = box_iou(Box(0, 0, 100, 4), Box(0, 6, 100, 10))
    outlines = mask_iou(ring, shifted_ring)

    assert outlines == 0.0
    assert boxes == 0.0
    # And when they *do* share a bounding region, the outlines still disagree:
    overlapping = mask_iou(ring, [[0, 2], [100, 2], [100, 6], [0, 6]])
    assert 0.0 < overlapping < 0.5


def test_bounds_of_a_degenerate_outline_are_undefined() -> None:
    """Fewer than three vertices bound no region."""
    assert polygon_bounds([[1, 1]]) is None
    assert polygon_bounds(SQUARE) == (0.0, 0.0, 10.0, 10.0)


def test_the_matcher_adapter_scores_unmeasurable_pairs_as_zero() -> None:
    """`match_annotations` needs a float, so the adapter collapses None to 0.

    Reporting code must use `mask_iou` directly and keep the None — a pair that
    could not be compared is not a pair that failed to overlap.
    """

    class Annotation:
        def __init__(self, polygon):
            self.polygon = polygon

    assert annotation_mask_iou(Annotation(SQUARE), Annotation(SQUARE)) == pytest.approx(
        1.0, abs=0.02
    )
    assert annotation_mask_iou(Annotation(None), Annotation(SQUARE)) == 0.0


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def test_verdicts_use_the_same_thresholds_as_box_diagnosis() -> None:
    """So "correct" means the same strength of agreement in both passes."""
    assert classify_mask(config.MATCH_IOU_THRESHOLD) == MASK_CORRECT
    assert classify_mask(config.MATCH_IOU_THRESHOLD - 0.01) == MASK_POOR
    assert classify_mask(config.LOCALIZATION_IOU_FLOOR) == MASK_POOR
    assert classify_mask(config.LOCALIZATION_IOU_FLOOR - 0.01) == MASK_NO_OVERLAP


def test_an_unmeasured_outline_has_no_verdict() -> None:
    """None is the absence of a measurement, not a failing one."""
    assert classify_mask(None) is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _seed(database: Path) -> tuple[int, list[int]]:
    """Create a run with one paired finding and one unpaired one."""
    with storage.connect(database) as connection:
        run_id = storage.save_run(
            connection,
            RunContext(
                model_path="m.pt",
                model_sha256="sha",
                dataset_yaml="d.yaml",
                split="test",
                confidence_threshold=0.25,
                match_iou_threshold=0.5,
                localization_iou_floor=0.1,
                image_size=640,
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?, 'a.jpg', 'a.jpg', 50, 50, 1, 1)",
            (run_id,),
        )
        image_id = connection.execute("SELECT id FROM images").fetchone()["id"]
        ids = []
        for outcome in ("correct", "false_negative"):
            cursor = connection.execute(
                "INSERT INTO findings (run_id, image_id, outcome, class_name, iou) "
                "VALUES (?, ?, ?, 'door_frame', 0.9)",
                (run_id, image_id, outcome),
            )
            ids.append(int(cursor.lastrowid))
    return run_id, ids


def test_mask_results_round_trip_with_their_polygon(tmp_path: Path) -> None:
    """A stored outline must come back as the same points."""
    database = tmp_path / "masks.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(
            connection, run_id, [(ids[0], 0.42, MASK_POOR, SQUARE)]
        )
        rows = storage.load_mask_findings(connection, run_id)

    assert len(rows) == 1
    assert rows[0].mask_iou == pytest.approx(0.42)
    assert rows[0].mask_outcome == MASK_POOR
    assert rows[0].pred_polygon == SQUARE


def test_an_unmeasurable_finding_stores_null_not_zero(tmp_path: Path) -> None:
    """A false negative has no predicted outline. That is not zero overlap."""
    database = tmp_path / "null.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(connection, run_id, [(ids[1], None, None, None)])
        row = storage.load_mask_findings(connection, run_id)[0]

    assert row.mask_iou is None
    assert row.mask_outcome is None
    assert row.pred_polygon is None


def test_disagreements_filter_finds_box_correct_outline_wrong(tmp_path: Path) -> None:
    """The set this milestone exists to surface."""
    database = tmp_path / "disagree.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(
            connection,
            run_id,
            [(ids[0], 0.2, MASK_POOR, SQUARE), (ids[1], 0.9, MASK_CORRECT, SQUARE)],
        )
        disagreements = storage.load_mask_findings(
            connection, run_id, disagreements_only=True
        )

    assert [row.finding_id for row in disagreements] == [ids[0]]


def test_rerunning_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    """Derived data: recomputing corrects, it does not append."""
    database = tmp_path / "replace.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(connection, run_id, [(ids[0], 0.1, MASK_POOR, None)])
        storage.save_mask_findings(
            connection, run_id, [(ids[0], 0.9, MASK_CORRECT, None)]
        )
        rows = storage.load_mask_findings(connection, run_id)

    assert len(rows) == 1
    assert rows[0].mask_iou == pytest.approx(0.9)


def test_mask_findings_cascade_when_the_run_is_deleted(tmp_path: Path) -> None:
    """Derived data must not outlive the run it describes."""
    database = tmp_path / "cascade.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(connection, run_id, [(ids[0], 0.5, MASK_POOR, None)])
        connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

        assert storage.load_mask_findings(connection, run_id) == []


def test_findings_keeps_its_published_shape(tmp_path: Path) -> None:
    """D-020: mask diagnosis adds a table and alters nothing (D-022)."""
    database = tmp_path / "frozen.db"
    _seed(database)

    with storage.connect(database) as connection:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(findings)").fetchall()
        ]

    assert columns == [
        "id",
        "run_id",
        "image_id",
        "outcome",
        "class_id",
        "class_name",
        "confidence",
        "iou",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "truth_x1",
        "truth_y1",
        "truth_x2",
        "truth_y2",
        "truth_polygon",
    ]


def test_a_stored_polygon_is_valid_json(tmp_path: Path) -> None:
    """The dashboard parses this column, so its format is part of the contract."""
    database = tmp_path / "json.db"
    run_id, ids = _seed(database)

    with storage.connect(database) as connection:
        storage.save_mask_findings(
            connection, run_id, [(ids[0], 0.5, MASK_POOR, SQUARE)]
        )
        raw = connection.execute(
            "SELECT pred_polygon FROM mask_findings WHERE finding_id = ?", (ids[0],)
        ).fetchone()["pred_polygon"]

    assert json.loads(raw) == SQUARE
