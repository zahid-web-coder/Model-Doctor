"""Tests for shared bounding-box geometry.

The point of :mod:`utils.geometry` is that predictions and ground truth compute
geometry *identically*. These tests assert that equivalence directly, because a
silent divergence between the two would corrupt every failure classification
the project is being built to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import Detection
from utils.annotations import ObjectAnnotation
from utils.geometry import BoxGeometryMixin


def _prediction(x1: float, y1: float, x2: float, y2: float) -> Detection:
    """Build a prediction with the given corners."""
    return Detection(
        class_id=0, class_name="any", x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9
    )


def _ground_truth(x1: float, y1: float, x2: float, y2: float) -> ObjectAnnotation:
    """Build a ground-truth annotation with the given corners."""
    return ObjectAnnotation.from_box(
        class_id=0, class_name="any", x1=x1, y1=y1, x2=x2, y2=y2
    )


def test_derived_geometry_is_correct() -> None:
    """Width, height, area, and centre follow from the corners."""
    box = _prediction(10.0, 20.0, 110.0, 70.0)
    assert box.width == 100.0
    assert box.height == 50.0
    assert box.area == 5000.0
    assert box.center == (60.0, 45.0)
    assert box.xyxy == (10.0, 20.0, 110.0, 70.0)


def test_inverted_box_clamps_to_zero_area() -> None:
    """A degenerate box yields 0.0, never a negative area.

    A negative area would survive an average and quietly skew any size-based
    analysis, which is far worse than an obvious zero.
    """
    box = _prediction(100.0, 100.0, 10.0, 10.0)
    assert box.width == 0.0
    assert box.height == 0.0
    assert box.area == 0.0


def test_prediction_and_ground_truth_agree_on_geometry() -> None:
    """The same corners give the same geometry for both annotation kinds.

    This is the whole reason the mixin exists. If it ever fails, predictions
    and labels have drifted apart and overlap results become meaningless.
    """
    corners = (12.5, 33.0, 200.0, 145.5)
    prediction = _prediction(*corners)
    ground_truth = _ground_truth(*corners)

    assert prediction.xyxy == ground_truth.xyxy
    assert prediction.area == ground_truth.area
    assert prediction.width == ground_truth.width
    assert prediction.height == ground_truth.height
    assert prediction.center == ground_truth.center


def test_both_annotation_kinds_share_one_implementation() -> None:
    """Neither class re-implements geometry locally.

    Guards against someone "helpfully" adding an `area` property back onto one
    of the dataclasses, which would reintroduce the duplication this module
    removed.
    """
    for box_type in (Detection, ObjectAnnotation):
        assert issubclass(box_type, BoxGeometryMixin)
        for name in ("xyxy", "width", "height", "area", "center"):
            assert name not in vars(box_type), (
                f"{box_type.__name__} redefines '{name}' instead of inheriting it"
            )


def test_geometry_is_derived_not_stored() -> None:
    """Frozen dataclasses expose geometry as read-only properties."""
    box = _prediction(0.0, 0.0, 10.0, 10.0)
    with pytest.raises(AttributeError):
        box.area = 999.0  # type: ignore[misc]
