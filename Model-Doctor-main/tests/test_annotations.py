"""Tests for the generic object-annotation model.

These cover the three dataset shapes the model must support — detection-only,
segmentation-only, and both mixed together — plus the invariant that a bounding
box always exists and always agrees with its polygon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import Detection
from utils.annotations import ObjectAnnotation, polygon_to_bbox
from utils.dataset import load_ground_truth


# ---------------------------------------------------------------------------
# Polygon -> box derivation
# ---------------------------------------------------------------------------
def test_polygon_to_bbox_is_the_outline_extent() -> None:
    """The box spans the outline's minimum and maximum coordinates."""
    polygon = ((10.0, 20.0), (50.0, 15.0), (40.0, 60.0), (5.0, 45.0))
    assert polygon_to_bbox(polygon) == (5.0, 15.0, 50.0, 60.0)


def test_polygon_to_bbox_handles_a_triangle() -> None:
    """Three vertices is the minimum that encloses area, and is valid."""
    assert polygon_to_bbox(((0.0, 0.0), (10.0, 0.0), (5.0, 8.0))) == (
        0.0,
        0.0,
        10.0,
        8.0,
    )


def test_polygon_to_bbox_rejects_an_empty_polygon() -> None:
    """An outline with no vertices has no location, so it raises.

    Substituting a zero box would hide a corrupt label rather than surface it.
    """
    with pytest.raises(ValueError, match="empty polygon"):
        polygon_to_bbox(())


# ---------------------------------------------------------------------------
# The annotation model
# ---------------------------------------------------------------------------
def test_from_polygon_derives_a_matching_box() -> None:
    """A segmentation annotation carries both the outline and its box."""
    polygon = ((10.0, 20.0), (50.0, 15.0), (40.0, 60.0))
    ann = ObjectAnnotation.from_polygon(1, "door_frame", polygon)

    assert ann.has_polygon
    assert ann.polygon == polygon
    assert ann.xyxy == (10.0, 15.0, 50.0, 60.0)
    assert ann.class_name == "door_frame"


def test_from_box_has_no_polygon() -> None:
    """A detection annotation is complete without an outline."""
    ann = ObjectAnnotation.from_box(0, "door", 10.0, 20.0, 110.0, 70.0)
    assert not ann.has_polygon
    assert ann.polygon is None
    assert ann.area == 5000.0


def test_box_always_exists_for_segmentation_data() -> None:
    """Box-based analysis needs no special-casing on a polygon dataset.

    This is the core promise of the model: whatever the source format, the
    consumer can read a bounding box.
    """
    ann = ObjectAnnotation.from_polygon(0, "door", ((1.0, 2.0), (3.0, 8.0), (7.0, 4.0)))
    assert ann.width > 0 and ann.height > 0 and ann.area > 0


def test_ground_truth_has_no_confidence() -> None:
    """Confidence is what distinguishes a prediction from an assertion."""
    ann = ObjectAnnotation.from_box(0, "door", 0.0, 0.0, 1.0, 1.0)
    assert ann.confidence is None
    assert not ann.is_prediction


def test_prediction_is_identified_by_its_confidence() -> None:
    """An annotation carrying certainty reports itself as a prediction."""
    det = Detection(
        class_id=0, class_name="door", x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=0.8
    )
    assert det.is_prediction
    assert det.confidence == 0.8


def test_detection_requires_a_confidence() -> None:
    """Building a prediction without certainty fails immediately."""
    with pytest.raises(ValueError, match="requires a confidence"):
        Detection(class_id=0, class_name="door", x1=0.0, y1=0.0, x2=1.0, y2=1.0)


def test_detection_is_an_object_annotation() -> None:
    """Predictions and ground truth are the same kind of thing."""
    assert issubclass(Detection, ObjectAnnotation)


def test_annotations_are_immutable() -> None:
    """Observations are frozen once read."""
    ann = ObjectAnnotation.from_box(0, "door", 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(AttributeError):
        ann.class_id = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Label parsing — the three supported dataset shapes
# ---------------------------------------------------------------------------
def _write_label(tmp_path: Path, content: str) -> Path:
    """Create an image/label pair and return the image path."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    image = images_dir / "frame.jpg"
    image.touch()
    (labels_dir / "frame.txt").write_text(content, encoding="utf-8")
    return image


def test_detection_only_dataset(tmp_path: Path) -> None:
    """Shape 1: every line is a 5-field box."""
    image = _write_label(tmp_path, "0 0.5 0.5 0.5 0.5\n")
    anns = load_ground_truth(image, 640, 480, {0: "door"})

    assert len(anns) == 1
    assert not anns[0].has_polygon
    assert anns[0].class_name == "door"
    assert anns[0].xyxy == pytest.approx((160.0, 120.0, 480.0, 360.0))


def test_segmentation_only_dataset(tmp_path: Path) -> None:
    """Shape 2: every line is a polygon, denormalised to pixels."""
    # A right triangle: (0,0) (0.5,0) (0.5,0.5) on a 200x100 image.
    image = _write_label(tmp_path, "1 0.0 0.0 0.5 0.0 0.5 0.5\n")
    anns = load_ground_truth(image, 200, 100, {1: "door_frame"})

    assert len(anns) == 1
    ann = anns[0]
    assert ann.has_polygon
    assert ann.polygon == ((0.0, 0.0), (100.0, 0.0), (100.0, 50.0))
    assert ann.xyxy == (0.0, 0.0, 100.0, 50.0)
    assert ann.class_name == "door_frame"


def test_mixed_dataset_in_one_file(tmp_path: Path) -> None:
    """Shape 3: boxes and polygons in the same label file.

    The form is decided per line, not per dataset, so a future dataset
    containing both needs no configuration and no migration.
    """
    image = _write_label(
        tmp_path,
        "0 0.5 0.5 0.5 0.5\n"  # box
        "1 0.0 0.0 0.5 0.0 0.5 0.5\n",  # polygon
    )
    anns = load_ground_truth(image, 200, 100, {0: "door", 1: "door_frame"})

    assert len(anns) == 2
    assert [a.has_polygon for a in anns] == [False, True]
    assert [a.class_name for a in anns] == ["door", "door_frame"]
    # Both expose a usable box regardless of source form.
    assert all(a.area > 0 for a in anns)


def test_many_vertex_polygon_parses(tmp_path: Path) -> None:
    """Real exports reach a hundred vertices; field count is not capped."""
    coords = " ".join(f"{i / 200:.4f} {i / 200:.4f}" for i in range(1, 108))
    image = _write_label(tmp_path, f"0 {coords}\n")
    anns = load_ground_truth(image, 100, 100, {0: "door"})

    assert len(anns) == 1
    assert len(anns[0].polygon) == 107


def test_even_field_count_is_rejected_as_truncated(tmp_path: Path) -> None:
    """A missing coordinate must not be reinterpreted as a shorter polygon."""
    image = _write_label(tmp_path, "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4\n")
    assert load_ground_truth(image, 100, 100, {0: "door"}) == []


def test_six_fields_is_rejected(tmp_path: Path) -> None:
    """Six fields is neither a box nor a polygon with three vertices."""
    image = _write_label(tmp_path, "0 0.1 0.1 0.2 0.2 0.3\n")
    assert load_ground_truth(image, 100, 100, {0: "door"}) == []


def test_one_bad_line_does_not_discard_the_file(tmp_path: Path) -> None:
    """A single malformed row must not make a whole image unusable."""
    image = _write_label(
        tmp_path,
        "0 0.5 0.5 0.5 0.5\n"
        "1 0.1 0.1 0.2 0.2 0.3 0.3 0.4\n"  # truncated
        "not-a-number 0.1 0.1 0.2 0.2\n"
        "\n"
        "1 0.0 0.0 0.5 0.0 0.5 0.5\n",
    )
    anns = load_ground_truth(image, 100, 100, {0: "door", 1: "door_frame"})
    assert [a.class_id for a in anns] == [0, 1]


def test_missing_class_names_fall_back_visibly(tmp_path: Path) -> None:
    """Parsing succeeds without a descriptor; unknown names show as id:<n>."""
    image = _write_label(tmp_path, "7 0.5 0.5 0.5 0.5\n")
    anns = load_ground_truth(image, 100, 100)
    assert anns[0].class_name == "id:7"
