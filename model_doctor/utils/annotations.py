"""The generic object-annotation data model.

One object in one image, whether it came from a human annotator or from a
model. This is the type every later milestone consumes, so its shape is the
most consequential decision in the project.

**Why this module exists.** Detection and segmentation datasets describe the
same thing — an object of some class, somewhere in an image — but record it
differently: a box, or a polygon outline. Modelling those as two unrelated
types would fork every downstream consumer into "the box path" and "the
polygon path". Modelling them as one type with an optional polygon means
analysis code is written once and works on both.

**Why it is not part of** :mod:`utils.geometry`. That module is pure rectangle
arithmetic; it knows nothing about classes, names, or models. An annotation is
a *semantic* object. Merging the two would make the geometry layer depend on
domain concepts it has no reason to know about, and geometry is the piece most
likely to be reused unchanged.

**The central rule: a bounding box always exists.** When a polygon is present,
the box is *derived* from it rather than stored independently. That guarantees
box-based analysis works on a segmentation dataset with no special-casing, and
removes any possibility of the box and polygon disagreeing.

Three dataset shapes are supported by construction:

============================  ==========================================
Detection-only               ``polygon`` is ``None``; the box is provided.
Segmentation-only            ``polygon`` is set; the box is derived.
Mixed, within one dataset    Decided per annotation, not per dataset.
============================  ==========================================

Coordinates are absolute pixels throughout — boxes *and* polygons — matching
:class:`~utils.geometry.BoxGeometryMixin` and DECISIONS D-006. Mixing
normalised and pixel units is the most common source of silent errors in
detection code, so the conversion happens once, at read time.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_doctor.utils.geometry import BoxGeometryMixin

# A closed polygon outline as absolute pixel vertices: ((x, y), (x, y), ...).
# A tuple rather than a list so annotations stay immutable and hashable.
Polygon = tuple[tuple[float, float], ...]


def polygon_to_bbox(polygon: Polygon) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box enclosing ``polygon``.

    The box is the extent of the outline: ``(min x, min y, max x, max y)``.
    This matches how the detection ecosystem converts segments to boxes, so a
    segmentation dataset yields the same boxes here as it does inside a
    training pipeline — which matters, because Model Doctor's job is to explain
    a model's behaviour, not to disagree with its own evaluator about where the
    ground truth is.

    Args:
        polygon: At least one ``(x, y)`` vertex, in absolute pixels.

    Returns:
        ``(x1, y1, x2, y2)`` in absolute pixels.

    Raises:
        ValueError: If ``polygon`` is empty. An annotation with no vertices has
            no meaningful location, and silently substituting a zero box would
            hide a corrupt label rather than surface it.
    """
    if not polygon:
        raise ValueError("Cannot derive a bounding box from an empty polygon.")

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class ObjectAnnotation(BoxGeometryMixin):
    """One annotated object, from ground truth or from a model.

    Frozen because annotations are observations: once read from a label file or
    a model output, nothing should edit them in place. Immutability also makes
    them safe to share across the analysis passes of later milestones.

    Attributes:
        class_id: Zero-based class index.
        class_name: Human-readable name. Resolved from the model's own mapping
            for predictions, or from ``data.yaml`` for ground truth — never
            hardcoded.
        x1, y1, x2, y2: Bounding box corners in absolute pixels. Always
            present, even for a segmentation annotation, where it is derived
            from the polygon.
        polygon: Outline vertices in absolute pixels, or ``None`` for a
            detection-only annotation.
        confidence: The model's certainty in ``[0, 1]``. ``None`` for ground
            truth, which is asserted rather than predicted. This field is what
            distinguishes the two kinds of annotation.
    """

    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: Polygon | None = None
    confidence: float | None = None

    # Geometry (xyxy, width, height, area, center) comes from BoxGeometryMixin,
    # so predictions and ground truth cannot compute it differently.

    @property
    def has_polygon(self) -> bool:
        """Return whether an outline is available for this annotation.

        Analysis code branches on this rather than on a dataset-level flag,
        because a single dataset may legitimately mix the two.
        """
        return self.polygon is not None

    @property
    def is_prediction(self) -> bool:
        """Return whether this came from a model rather than a label file."""
        return self.confidence is not None

    @classmethod
    def from_polygon(
        cls,
        class_id: int,
        class_name: str,
        polygon: Polygon,
        confidence: float | None = None,
    ) -> ObjectAnnotation:
        """Build an annotation from an outline, deriving its bounding box.

        This is the constructor segmentation data should use. Deriving the box
        here — rather than letting callers pass one — is what makes "the box
        and the polygon always agree" a property of the type instead of a rule
        someone has to remember.

        Args:
            class_id: Zero-based class index.
            class_name: Human-readable class name.
            polygon: Outline vertices in absolute pixels.
            confidence: Model certainty, or ``None`` for ground truth.

        Returns:
            An annotation carrying both the outline and its derived box.

        Raises:
            ValueError: If ``polygon`` has no vertices.
        """
        x1, y1, x2, y2 = polygon_to_bbox(polygon)
        return cls(
            class_id=class_id,
            class_name=class_name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            polygon=tuple(polygon),
            confidence=confidence,
        )

    @classmethod
    def from_box(
        cls,
        class_id: int,
        class_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float | None = None,
    ) -> ObjectAnnotation:
        """Build an annotation from a bounding box, with no outline.

        The constructor detection data should use. It exists alongside
        :meth:`from_polygon` so that call sites read as a statement of which
        kind of source they are handling.

        Args:
            class_id: Zero-based class index.
            class_name: Human-readable class name.
            x1: Left edge in absolute pixels.
            y1: Top edge in absolute pixels.
            x2: Right edge in absolute pixels.
            y2: Bottom edge in absolute pixels.
            confidence: Model certainty, or ``None`` for ground truth.

        Returns:
            An annotation with ``polygon`` set to ``None``.
        """
        return cls(
            class_id=class_id,
            class_name=class_name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            polygon=None,
            confidence=confidence,
        )
