"""Shared bounding-box geometry.

Predictions (:class:`~app.inference.Detection`) and ground truth
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
