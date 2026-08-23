"""Choosing a detector, in one place.

Model Doctor supports two detector families. Everything downstream — matching,
the outcome taxonomy, storage, root-cause analysis, the schema — consumes
:class:`~app.inference.Detection` objects and is indifferent to which one
produced them, so the choice is confined to this module and nothing else has to
branch on it.

**YOLO is the default and stays the default.** It is the only one of the two
that is viable outside a GPU: on this project's columns test set, CPU-only,
YOLO26 @672 runs at 20.9 FPS in 390MB while RF-DETR @480 manages 1.7 FPS in
2.8GB. RF-DETR is the more accurate model on that set and is offered for GPU
and batch work, not as a replacement.

``rfdetr`` is an optional dependency. It is imported only when actually asked
for, so a checkout without it installed keeps working exactly as before — the
import cost and the failure mode both belong to the family that needs them.
"""

from __future__ import annotations

from typing import Any

import config
from app.inference import Detector
from utils.exceptions import ModelLoadError

#: Families this project can load. Keys are what ``MD_DETECTOR`` accepts.
SUPPORTED_FAMILIES: tuple[str, ...] = ("yolo", "rfdetr")


def default_image_size(family: str) -> int:
    """Return the configured input resolution for a family.

    The two models want different values — YOLO's checkpoint was trained at one
    size and RF-DETR is swept to another — so they are configured separately.
    Sharing a single setting would silently mis-size whichever model was not
    being tuned at the time, and mis-sizing degrades accuracy without erroring.
    """
    return config.RFDETR_IMAGE_SIZE if family == "rfdetr" else config.IMAGE_SIZE


def build_detector(
    family: str | None = None,
    *,
    model_path: str | None = None,
    image_size: int | None = None,
    **kwargs: Any,
) -> Detector:
    """Return an unloaded detector of the requested family.

    Args:
        family: ``"yolo"`` or ``"rfdetr"``. Defaults to
            :data:`config.DETECTOR_FAMILY`.
        model_path: Explicit weights path. When omitted, each family falls back
            to its own discovery — YOLO scans the models directory, RF-DETR
            requires an explicit path because its checkpoints are not
            interchangeable with YOLO's despite sharing the ``.pt`` suffix.
        image_size: Input resolution. Defaults to the family's configured size.
        **kwargs: Passed through to the detector, e.g. ``confidence``.

    Returns:
        A :class:`~app.inference.Detector`, not yet loaded. Callers that want
        eager failure should call ``load()``.

    Raises:
        ModelLoadError: The family is not one this project supports, or the
            optional dependency for it is missing.
    """
    resolved = (family or config.DETECTOR_FAMILY).strip().lower()
    if resolved not in SUPPORTED_FAMILIES:
        raise ModelLoadError(
            f"Unknown detector family {resolved!r}. "
            f"Supported: {', '.join(SUPPORTED_FAMILIES)}."
        )

    size = image_size if image_size is not None else default_image_size(resolved)

    if resolved == "rfdetr":
        # Imported here, not at module scope: rfdetr is optional, and a
        # checkout that only uses YOLO must not pay for it or fail without it.
        try:
            from app.rfdetr_adapter import RFDetrDetector
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelLoadError(
                "The rfdetr detector was requested but its dependency is not "
                "installed. Run: pip install -r requirements-rfdetr.txt"
            ) from exc
        return RFDetrDetector(model_path=model_path, image_size=size, **kwargs)

    return Detector(model_path=model_path, image_size=size, **kwargs)


__all__ = ["SUPPORTED_FAMILIES", "build_detector", "default_image_size"]
