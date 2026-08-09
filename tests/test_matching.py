"""Tests for box IoU and the generic matching engine.

IoU is the numeric foundation every failure classification rests on, and
matching is the step that decides which prediction describes which object. A
defect in either produces confident, wrong diagnoses rather than an error, so
both are tested against hand-computed values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.annotations import ObjectAnnotation
from utils.geometry import box_iou
from utils.matching import match_annotations


def _box(x1: float, y1: float, x2: float, y2: float, cls: int = 0) -> ObjectAnnotation:
    """Build a ground-truth annotation with the given corners."""
    return ObjectAnnotation.from_box(cls, f"c{cls}", x1, y1, x2, y2)


def _pred(
    x1: float, y1: float, x2: float, y2: float, cls: int = 0, conf: float = 0.9
) -> ObjectAnnotation:
    """Build a prediction with the given corners and confidence."""
    return ObjectAnnotation.from_box(cls, f"c{cls}", x1, y1, x2, y2, confidence=conf)


# ---------------------------------------------------------------------------
# Box IoU
# ---------------------------------------------------------------------------
def test_iou_perfect_overlap() -> None:
    """Identical boxes overlap completely."""
    assert box_iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == 1.0


def test_iou_no_overlap() -> None:
    """Disjoint boxes share nothing."""
    assert box_iou(_box(0, 0, 10, 10), _box(20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap() -> None:
    """A half-shifted square overlaps by a computable amount.

    Two 10x10 boxes offset by 5 in x share a 5x10 = 50 region, over a union of
    100 + 100 - 50 = 150. So 50/150 = 1/3.
    """
    assert box_iou(_box(0, 0, 10, 10), _box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_contained_box() -> None:
    """A box wholly inside another gives its own area over the larger area.

    A 5x5 box inside a 10x10 box: intersection 25, union 100. So 0.25.
    """
    assert box_iou(_box(0, 0, 10, 10), _box(0, 0, 5, 5)) == pytest.approx(0.25)


def test_iou_touching_edges_is_zero() -> None:
    """Boxes sharing only an edge have zero area in common.

    This is the boundary case that a naive implementation gets wrong by
    reporting a sliver of overlap where there is none.
    """
    assert box_iou(_box(0, 0, 10, 10), _box(10, 0, 20, 10)) == 0.0
    assert box_iou(_box(0, 0, 10, 10), _box(0, 10, 10, 20)) == 0.0


def test_iou_touching_corners_is_zero() -> None:
    """Boxes meeting at a single corner do not overlap."""
    assert box_iou(_box(0, 0, 10, 10), _box(10, 10, 20, 20)) == 0.0


def test_iou_is_symmetric() -> None:
    """Order of arguments cannot change the answer.

    This is why IoU is a free function rather than a method (D-014).
    """
    a, b = _box(0, 0, 10, 10), _box(3, 4, 12, 14)
    assert box_iou(a, b) == box_iou(b, a)


def test_iou_of_degenerate_boxes_is_zero() -> None:
    """Zero-area boxes yield 0.0 rather than a division error."""
    assert box_iou(_box(5, 5, 5, 5), _box(5, 5, 5, 5)) == 0.0


def test_iou_never_leaves_the_unit_interval() -> None:
    """Overlap is a ratio; it cannot exceed 1 or fall below 0."""
    cases = [
        (_box(0, 0, 10, 10), _box(0, 0, 10, 10)),
        (_box(0, 0, 10, 10), _box(-5, -5, 15, 15)),
        (_box(0, 0, 1, 1), _box(0, 0, 1000, 1000)),
        (_box(0, 0, 10, 10), _box(9.9, 9.9, 20, 20)),
    ]
    for a, b in cases:
        assert 0.0 <= box_iou(a, b) <= 1.0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def test_matching_pairs_the_obvious_case() -> None:
    """One prediction on one object pairs with it."""
    result = match_annotations([_pred(0, 0, 10, 10)], [_box(0, 0, 10, 10)], 0.5)
    assert len(result.matches) == 1
    assert result.matches[0].similarity == 1.0
    assert result.matches[0].same_class


def test_matching_is_one_to_one() -> None:
    """Two predictions on one object leave one unmatched.

    Without this, a duplicate detection would be counted as a second success
    rather than as the spurious prediction it is.
    """
    result = match_annotations(
        [_pred(0, 0, 10, 10), _pred(0, 0, 9, 9)], [_box(0, 0, 10, 10)], 0.5
    )
    assert len(result.matches) == 1
    assert len(result.unmatched_prediction_indices) == 1


def test_matching_prefers_the_strongest_overlap() -> None:
    """When several predictions compete, the best geometric fit wins.

    The weaker prediction is listed first and is more confident, so a
    confidence-ordered or first-come matcher would pick it.
    """
    good, poor = _pred(0, 0, 10, 10, conf=0.5), _pred(0, 0, 6, 6, conf=0.99)
    result = match_annotations([poor, good], [_box(0, 0, 10, 10)], 0.3)
    assert result.matches[0].prediction is good


def test_matching_respects_the_threshold() -> None:
    """A pair below the threshold does not match at all."""
    result = match_annotations([_pred(0, 0, 10, 10)], [_box(8, 8, 18, 18)], 0.5)
    assert result.matches == []
    assert result.unmatched_prediction_indices == [0]
    assert result.unmatched_truth_indices == [0]


def test_matching_ignores_class_by_default() -> None:
    """Cross-class pairs form, so a wrong-class error can be detected.

    If classes had to agree, a confidently mislabelled prediction would be
    reported as an unrelated false positive plus a miss, losing the fact that
    the model found the object and named it wrong.
    """
    result = match_annotations(
        [_pred(0, 0, 10, 10, cls=1)], [_box(0, 0, 10, 10, cls=0)], 0.5
    )
    assert len(result.matches) == 1
    assert not result.matches[0].same_class


def test_matching_can_require_same_class() -> None:
    """With ignore_class off, cross-class pairs are refused."""
    result = match_annotations(
        [_pred(0, 0, 10, 10, cls=1)],
        [_box(0, 0, 10, 10, cls=0)],
        0.5,
        ignore_class=False,
    )
    assert result.matches == []


def test_matching_handles_empty_inputs() -> None:
    """No predictions and no labels is a valid, empty result."""
    assert match_annotations([], [], 0.5).matches == []
    only_pred = match_annotations([_pred(0, 0, 1, 1)], [], 0.5)
    assert only_pred.unmatched_prediction_indices == [0]
    only_truth = match_annotations([], [_box(0, 0, 1, 1)], 0.5)
    assert only_truth.unmatched_truth_indices == [0]


def test_matching_is_deterministic_on_ties() -> None:
    """Identical input always produces identical pairings.

    Two predictions overlap the label equally; the tie must be broken the same
    way every run, or diagnoses would not be reproducible.
    """
    truths = [_box(0, 0, 10, 10)]
    predictions = [_pred(0, 0, 10, 10, conf=0.4), _pred(0, 0, 10, 10, conf=0.8)]
    first = match_annotations(predictions, truths, 0.5)
    for _ in range(5):
        again = match_annotations(predictions, truths, 0.5)
        assert again.matches[0].prediction_index == first.matches[0].prediction_index
    # Higher confidence breaks the tie.
    assert first.matches[0].prediction_index == 1


def test_matching_accepts_a_custom_similarity_function() -> None:
    """The comparison is a parameter — the seam mask IoU will use.

    Nothing in the engine assumes boxes; substituting a different measure
    changes the pairing without touching the algorithm.
    """
    result = match_annotations(
        [_pred(0, 0, 1, 1)],
        [_box(100, 100, 101, 101)],
        0.5,
        similarity=lambda _a, _b: 1.0,
    )
    assert len(result.matches) == 1


def test_mean_similarity_reports_none_without_matches() -> None:
    """An empty result has no average to report."""
    assert match_annotations([], [], 0.5).mean_similarity is None


def test_matching_assigns_multiple_objects_correctly() -> None:
    """Two objects and two predictions pair with their own counterparts."""
    truths = [_box(0, 0, 10, 10), _box(100, 100, 110, 110)]
    predictions = [_pred(99, 99, 109, 109), _pred(1, 1, 11, 11)]
    result = match_annotations(predictions, truths, 0.5)

    assert len(result.matches) == 2
    pairs = {(m.prediction_index, m.truth_index) for m in result.matches}
    assert pairs == {(0, 1), (1, 0)}
