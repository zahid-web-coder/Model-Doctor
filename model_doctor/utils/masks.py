"""Overlap between segmentation outlines, measured by rasterisation.

Separate from :mod:`utils.geometry`, which has no imports at all and is the
foundation every other module rests on. Mask comparison needs OpenCV, and a
dependency-free core is worth more than one fewer file.

**Why rasterise rather than intersect polygons analytically.** A predicted
outline traced from a mask is routinely self-intersecting — it follows pixel
boundaries, doubles back, and occasionally encloses zero area. Analytic
polygon intersection is undefined or wrong on such input, and the libraries
that implement it either raise or silently repair the shape into something the
model did not predict. Rasterising asks a simpler question: which pixels are
inside? That is well defined for any vertex list, and it is what the mask
metric the model itself reports is computed from.

**Why this matters here.** A door frame is a thin rectangular annulus. Its
bounding box is easy and its outline is not, so box IoU and mask IoU disagree
sharply — 0.877 mean box IoU against a mask mAP50-95 of 0.246 on the reference
model (D-022). Measuring only boxes reports that class as failing no worse than
a solid one.
"""

from __future__ import annotations

from collections.abc import Sequence

Point = Sequence[float]
Polygon = Sequence[Point]

# Below this many vertices an outline encloses no area and cannot be filled.
MINIMUM_VERTICES: int = 3

# Rasterisation happens inside the two outlines' shared bounding box rather
# than the whole image: identical result, and the canvas is a few hundred
# pixels rather than a few million. A tiny margin avoids clipping a boundary
# pixel at the extreme edge.
_MARGIN: int = 2


def polygon_bounds(polygon: Polygon) -> tuple[float, float, float, float] | None:
    """Return the axis-aligned extent of an outline.

    Args:
        polygon: Vertices in absolute pixels.

    Returns:
        ``(x1, y1, x2, y2)``, or ``None`` when the outline has too few vertices
        to enclose anything.
    """
    if polygon is None or len(polygon) < MINIMUM_VERTICES:
        return None
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def mask_iou(first: Polygon | None, second: Polygon | None) -> float | None:
    """Return intersection-over-union of two outlines.

    Args:
        first: An outline in absolute pixels, or ``None``.
        second: An outline in absolute pixels, or ``None``.

    Returns:
        Overlap in ``[0, 1]``, or **``None`` when it cannot be measured** —
        either outline missing, or too few vertices to enclose area.

        ``None`` rather than ``0.0`` is the important part. Zero means "these
        outlines do not overlap", which is a measurement. A missing outline
        means nothing was measured, and reporting that as perfect disagreement
        would invent a failure. The caller decides what to do with the
        difference.
    """
    import cv2
    import numpy as np

    first_bounds = polygon_bounds(first) if first is not None else None
    second_bounds = polygon_bounds(second) if second is not None else None
    if first_bounds is None or second_bounds is None:
        return None

    left = int(min(first_bounds[0], second_bounds[0])) - _MARGIN
    top = int(min(first_bounds[1], second_bounds[1])) - _MARGIN
    right = int(max(first_bounds[2], second_bounds[2])) + _MARGIN
    bottom = int(max(first_bounds[3], second_bounds[3])) + _MARGIN

    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None

    def rasterise(polygon: Polygon):
        """Fill one outline into the shared canvas, offset to its origin."""
        canvas = np.zeros((height, width), dtype=np.uint8)
        points = np.array(
            [[round(float(p[0]) - left), round(float(p[1]) - top)] for p in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [points], color=1)
        return canvas

    first_mask = rasterise(first)
    second_mask = rasterise(second)

    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    if union == 0:
        # Both outlines enclosed no pixels. Undefined, not zero overlap.
        return None
    return intersection / union


def annotation_mask_iou(first: object, second: object) -> float:
    """Compare two annotations by outline, for use as a matcher similarity.

    Adapts :func:`mask_iou` to the signature
    :func:`utils.matching.match_annotations` expects (D-018), which requires a
    float. A pair that cannot be compared scores ``0.0`` here so it simply does
    not match — but callers that need to *report* on outlines should use
    :func:`mask_iou` directly and keep the ``None``, because "no outline" and
    "no overlap" are different findings.

    Args:
        first: Anything carrying a ``polygon`` attribute.
        second: Likewise.

    Returns:
        Overlap in ``[0, 1]``; ``0.0`` when either outline is absent.
    """
    score = mask_iou(
        getattr(first, "polygon", None), getattr(second, "polygon", None)
    )
    return 0.0 if score is None else score
