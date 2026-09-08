"""Shared bounding-box geometry.

Predictions (:class:`~model_doctor.app.inference.Detection`) and ground truth
(:class:`~utils.dataset.GroundTruthBox`) are different things with different
fields, but they describe the same shape: a rectangle in absolute pixel
coordinates. Their derived geometry was previously implemented twice, which is
duplication that drifts.

This module holds that logic once.

**Why a mixin rather than a base dataclass.** A base dataclass owning
``x1..y2`` would force those fields to come *first* in every subclass
constructor, so ``Detection(class_id, class_name, confidence, x1, ...)`` would
have to become ``Detection(x1, y1, x2, y2, class_id, ...)``. That reorders
arguments for no benefit and reads worse. A mixin contributes behaviour without
contributing fields, so each class keeps the constructor signature that suits
it while sharing one implementation of the maths.

The contract is structural: any class mixing this in must expose ``x1``,
``y1``, ``x2``, and ``y2`` as floats in absolute pixels.
"""

from __future__ import annotations


class BoxGeometryMixin:
    """Derived geometry for a rectangle given by two opposite corners.

    Every value is computed on access rather than stored. Storing width
    alongside ``x1``/``x2`` would create two sources of truth that can disagree
    after any edit; deriving it means they cannot.

    Coordinates are assumed to be absolute pixels in ``xyxy`` form, with
    ``(x1, y1)`` the top-left corner. Keeping predictions and ground truth in
    this one convention is what allows them to be compared directly.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as an ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        """Return box width in pixels.

        Clamped at zero. A degenerate or inverted box is a data problem, but it
        should surface as an area of ``0.0`` rather than a negative number that
        silently corrupts a downstream average.
        """
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return box height in pixels, clamped at zero."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Return box area in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return the box centre as ``(x, y)`` in pixels."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def box_iou(first: BoxGeometryMixin, second: BoxGeometryMixin) -> float:
    """Return the Intersection over Union of two boxes.

    IoU is the standard measure of "are these two boxes describing the same
    thing": the area they share, divided by the area they jointly cover. It is
    ``1.0`` for identical boxes, ``0.0`` for disjoint ones, and scale-free —
    two boxes overlapping by half give ``0.33`` whether they are 10 pixels wide
    or 1000.

    Written as a free function rather than a method because the operation is
    *symmetric*: ``box_iou(a, b) == box_iou(b, a)``, and neither argument is
    privileged. Writing ``a.iou(b)`` would imply an asymmetry that does not
    exist. This is the rule set out in DECISIONS D-014.

    Args:
        first: Any object exposing pixel corners.
        second: Any object exposing pixel corners.

    Returns:
        Overlap ratio in ``[0.0, 1.0]``.
    """
    # The intersection rectangle is the overlap along each axis independently.
    # When the boxes miss on an axis, the "overlap" comes out negative, so it
    # is clamped — otherwise two disjoint boxes would multiply two negatives
    # into a positive area and report overlap where there is none.
    overlap_width = min(first.x2, second.x2) - max(first.x1, second.x1)
    overlap_height = min(first.y2, second.y2) - max(first.y1, second.y1)
    if overlap_width <= 0.0 or overlap_height <= 0.0:
        return 0.0

    intersection = overlap_width * overlap_height
    # Union counts the shared region once, not twice.
    union = first.area + second.area - intersection
    if union <= 0.0:
        # Both boxes are degenerate. Undefined rather than infinite; reporting
        # zero keeps callers from having to special-case a NaN.
        return 0.0
    return intersection / union
