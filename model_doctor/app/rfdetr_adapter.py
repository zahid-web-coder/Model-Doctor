"""RF-DETR behind Model Doctor's detector interface — detection and segmentation.

Implements :class:`~model_doctor.app.inference.Detector` for RF-DETR.

The point of this module is that **nothing downstream changes**. Matching,
the outcome taxonomy, storage, root-cause analysis and the schema all consume
:class:`~model_doctor.app.inference.Detection` objects and have no idea which library
produced them — which is exactly what ``app/inference.py``'s docstring promised
when it said swapping detectors "means reimplementing this file and nothing
else". This is that reimplementation, as a subclass rather than an edit, so the
YOLO path is not touched at all.

Three things genuinely differ from Ultralytics and are handled here:

* **Loading.** RF-DETR ships a Lightning checkpoint and its own
  ``from_checkpoint`` constructor rather than a callable model class. Which
  variant a checkpoint is — Nano or Medium, detection or segmentation — is
  recorded inside the file, so it is read from there rather than assumed. The
  library's own resolution order is mirrored in :func:`resolve_variant`.
* **Result shape.** It returns a ``supervision.Detections`` with boolean raster
  masks for a segmentation variant and none at all for a detection one, where
  Ultralytics returns polygons. Model Doctor's annotation model is
  polygon-based, so the raster is traced back to a contour here — see
  :func:`_mask_to_polygon` for why that is lossy in one specific way.
* **Evaluation.** RF-DETR has no ``.val()``. :meth:`validate` therefore refuses
  rather than inventing something, because a metric computed by a different
  evaluator than YOLO's is not comparable to YOLO's, and a silently
  incomparable number is worse than no number.

Class ids are read from the checkpoint's own ``class_names``, never hardcoded,
exactly as the YOLO path reads ``model.names``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from model_doctor.app.inference import (
    Detection,
    Detector,
    ImagePrediction,
    ValidationMetrics,
)
from model_doctor.utils.exceptions import ModelLoadError, ResourceNotFoundError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# A contour needs three points to bound an area. Below that the "polygon" is a
# point or a line segment, which has no interior and would make every mask IoU
# against it zero — worse than admitting there is no usable outline.
MIN_POLYGON_POINTS = 3

# RF-DETR requires a square resolution divisible by `patch_size * num_windows`.
# That product differs per variant — 12 for Seg Nano, 24 for the other
# segmentation sizes, 32 for the detection sizes — so it is read from the
# variant's own configuration in :func:`resolution_multiple`. This value is the
# fallback used only when the checkpoint cannot be identified, and it is the
# smallest of them so an unidentifiable file is not rejected for a rule that
# might not apply to it.
RESOLUTION_MULTIPLE = 12


def _variant_classes() -> dict[str, Any]:
    """Return every ``RFDETR*`` class the installed library exposes, by name.

    Read from the module rather than listed here, so a library release that
    adds a size needs no edit. Raises through to the caller when the library is
    absent, which is the same failure loading would produce anyway.
    """
    import rfdetr

    return {
        name: getattr(rfdetr, name)
        for name in dir(rfdetr)
        if name.startswith("RFDETR") and isinstance(getattr(rfdetr, name), type)
    }


def resolve_variant(weights: Path) -> tuple[str | None, str]:
    """Identify which RF-DETR class a checkpoint is, without executing it.

    Mirrors the order ``RFDETR.from_checkpoint`` itself uses, because the point
    is to agree with the library about what is being loaded — not to invent a
    second opinion:

    1. the ``model_name`` key the training stack writes,
    2. the ``pretrain_weights`` file name recorded in the checkpoint's ``args``,
    3. this file's own name.

    Read with ``weights_only=True``, so identifying a checkpoint never runs the
    code inside it — the same rule :mod:`model_doctor.app.validation` follows.

    Returns:
        ``(class name, explanation)``. The name is ``None`` when the file
        cannot be identified, which is a state the caller handles rather than
        an error: the library is still given the chance to resolve it.
    """
    try:
        import torch

        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - unidentifiable is not fatal here
        return None, (
            f"The checkpoint could not be read as data ({type(error).__name__})."
        )

    try:
        classes = _variant_classes()
    except ImportError:
        return None, "rfdetr is not installed, so its variants cannot be listed."

    if isinstance(checkpoint, dict):
        recorded = str(checkpoint.get("model_name", "")).strip()
        if recorded in classes:
            return recorded, f"The checkpoint records model_name={recorded!r}."

        args = checkpoint.get("args")
        weights_name = ""
        if isinstance(args, dict):
            weights_name = str(args.get("pretrain_weights", "")).strip()
        elif args is not None:
            weights_name = str(getattr(args, "pretrain_weights", "")).strip()
        # "none"/"null"/"" are the library's own unset sentinels, not a file.
        if weights_name.lower() not in ("", "none", "null"):
            match = _class_for_weight_file(classes, Path(weights_name).name)
            if match:
                return match, f"Its args name the {weights_name!r} starter weights."

    match = _class_for_weight_file(classes, weights.name)
    if match:
        return match, f"Inferred from the file name {weights.name!r}."
    return None, (
        "The checkpoint names no variant this library defines; the library will "
        "be asked to resolve it."
    )


def _class_for_weight_file(classes: dict[str, Any], file_name: str) -> str | None:
    """Match a starter-weights file name to the class that declares it."""
    wanted = file_name.strip().lower()
    for name, cls in classes.items():
        config_class = getattr(cls, "_model_config_class", None)
        if config_class is None:
            continue
        try:
            declared = str(config_class().pretrain_weights or "").strip().lower()
        except Exception:  # noqa: BLE001 - a config that will not build is no match
            continue
        if declared and Path(declared).name == wanted:
            return name
    return None


def resolution_multiple(variant: str | None) -> int:
    """Return the resolution stride a variant requires.

    RF-DETR's backbone splits the image into ``patch_size`` tiles across
    ``num_windows`` windows, so the input side must be divisible by their
    product. Both are fields on the variant's own configuration; reading them
    is what lets one adapter serve Nano at 12 and Medium at 24 without a table
    here that would go stale.
    """
    if variant is None:
        return RESOLUTION_MULTIPLE
    try:
        config_class = _variant_classes()[variant]._model_config_class
        model_config = config_class()
        product = int(model_config.patch_size) * int(model_config.num_windows)
    except Exception:  # noqa: BLE001 - fall back rather than refuse to load
        return RESOLUTION_MULTIPLE
    return product if product > 0 else RESOLUTION_MULTIPLE


def snap_resolution(requested: int, multiple: int) -> int:
    """Round a requested resolution up to the nearest legal one.

    Snapping rather than refusing, because the value that arrives here is
    usually the size a trainer recorded before *it* snapped: Ramanujan's
    segmentation trainer rounds to the same multiple at training time and keeps
    the adjusted value only in memory. Refusing would reject the very models
    this adapter exists to read, over a discrepancy neither the operator nor
    the checkpoint created.
    """
    if multiple <= 0 or requested % multiple == 0:
        return requested
    return max(multiple, -(-requested // multiple) * multiple)


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """Trace a boolean raster mask back to a single polygon outline.

    Model Doctor's annotation model carries one polygon per object, so a mask
    that resolves into several disconnected blobs — a column split by an
    occluding beam — cannot be represented faithfully. **The largest component
    is kept and the rest are dropped**, which is the same simplification the
    dataset's own YOLO export made, so predictions and ground truth are at
    least lossy in the same direction.

    Returns ``None`` rather than an empty list when nothing traceable is found,
    preserving the distinction the codebase already draws between "this model
    does not segment" and "it segmented nothing here".
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv is a hard dependency
        raise ModelLoadError("opencv-python is required to trace masks") from exc

    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)
    if len(points) < MIN_POLYGON_POINTS:
        return None
    return [[float(x), float(y)] for x, y in points]


class RFDetrDetector(Detector):
    """Loads an RF-DETR segmentation checkpoint and runs inference with it.

    Substitutable for :class:`~model_doctor.app.inference.Detector` wherever a detector
    is
    expected. Callers see the same ``load`` / ``class_names`` / ``predict_image``
    / ``predict_many`` surface and the same :class:`Detection` objects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept the base detector's arguments unchanged."""
        super().__init__(*args, **kwargs)
        self._class_names: dict[int, str] = {}

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        """Load the RF-DETR weights into memory.

        Idempotent, like the base implementation.

        Raises:
            ResourceNotFoundError: The checkpoint path does not exist.
            ModelLoadError: ``rfdetr`` is missing, or the file is not an
                RF-DETR checkpoint.
        """
        if self._model is not None:
            return

        weights = Path(self._explicit_model_path) if self._explicit_model_path else None

        # The variant decides the resolution rule, so it is resolved first —
        # from the checkpoint when there is one to read, and not at all when
        # there is not. An unidentifiable file keeps the documented fallback,
        # which is what makes a bad resolution still report itself here rather
        # than deep inside the backbone as a tensor shape mismatch that never
        # mentions resolution.
        variant: str | None = None
        why = "No checkpoint to read a variant from."
        if weights is not None and weights.is_file():
            variant, why = resolve_variant(weights)
        multiple = resolution_multiple(variant)
        logger.info("RF-DETR variant: %s — %s", variant or "unidentified", why)

        if self.image_size and self.image_size % multiple:
            if variant is None:
                # Nothing said which rule applies, so the size is refused rather
                # than adjusted towards a stride that may not be this model's.
                raise ModelLoadError(
                    f"RF-DETR resolution must be a multiple of "
                    f"{multiple}; got {self.image_size}."
                )
            snapped = snap_resolution(int(self.image_size), multiple)
            logger.warning(
                "%s requires a resolution divisible by %d; snapping %d to %d. "
                "This mirrors what the trainer does, which records the size it "
                "was asked for rather than the one it used.",
                variant,
                multiple,
                self.image_size,
                snapped,
            )
            self.image_size = snapped

        if weights is None or not weights.is_file():
            raise ResourceNotFoundError(
                f"RF-DETR checkpoint not found: {weights}. Pass an explicit path."
            )

        # Imported lazily for the same reason the YOLO path does it: this
        # module must stay importable where the heavy stack is absent.
        #
        # The variant's own class is used when it was identified, and the base
        # class otherwise — `RFDETR.from_checkpoint` runs the library's own
        # resolution, which is the authority on its own file format and covers
        # detection and segmentation alike.
        try:
            import rfdetr

            loader = getattr(rfdetr, variant, None) if variant else None
            if loader is None:
                loader = getattr(rfdetr, "RFDETR", None)
            if loader is None:
                raise ModelLoadError(
                    "The installed rfdetr exposes neither "
                    f"{variant or 'a matching variant'} nor its base RFDETR "
                    "class, so this checkpoint cannot be loaded."
                )
        except ImportError as exc:
            raise ModelLoadError(
                "rfdetr is not installed. Run: pip install rfdetr==1.8.3"
            ) from exc

        # The checkpoint's training resolution is not binding at inference —
        # RF-DETR interpolates its positional encodings to whatever is asked
        # for. Validity was already checked above.
        kwargs: dict[str, Any] = {}
        if self.image_size:
            kwargs["resolution"] = int(self.image_size)

        # **The device has to be passed at construction.** Unlike Ultralytics,
        # which takes `device=` on every `predict` call, RF-DETR binds its
        # device when the model is built and its `predict` accepts no such
        # argument. Omitting it here does not fall back to `config.DEVICE`; it
        # lets RF-DETR choose for itself, which on this hardware means MPS
        # whatever the caller asked for. That is how a run measured on the CPU
        # came back at 162 ms instead of its true 601 ms — a figure recorded
        # under the wrong device is worse than no figure, because it will be
        # compared with something.
        if self.device:
            kwargs["device"] = str(self.device)

        logger.info(
            "Loading RF-DETR checkpoint %s as %s at resolution %s on device %s",
            weights.name,
            variant or "the library's own choice of class",
            kwargs.get("resolution", "native"),
            kwargs.get("device", "the library's own choice"),
        )
        started = time.perf_counter()
        try:
            self._model = loader.from_checkpoint(str(weights), **kwargs)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load {weights} as an RF-DETR checkpoint: {exc}"
            ) from exc

        # The checkpoint is the authority on what its own class ids mean, the
        # same rule the YOLO path follows. RF-DETR exposes an ordered list
        # where Ultralytics exposes a mapping; index is the class id.
        names = list(getattr(self._model, "class_names", []) or [])
        self._class_names = {index: str(name) for index, name in enumerate(names)}

        self._model_path = weights
        logger.info(
            "RF-DETR ready in %.2fs — %d classes",
            time.perf_counter() - started,
            len(self._class_names),
        )

    @property
    def class_names(self) -> dict[int, str]:
        """Return the checkpoint's own ``{class_id: name}`` mapping."""
        if self._model is None:
            raise ModelLoadError("Model is not loaded — call load() first.")
        return dict(self._class_names)

    # -- inference ----------------------------------------------------------
    def predict_image(
        self, image_path: Path, save_annotated: bool = False
    ) -> ImagePrediction:
        """Run RF-DETR on a single image.

        Mirrors the base contract exactly: per-image failures are captured on
        the returned :class:`ImagePrediction` rather than raised, so one bad
        file cannot abort a batch.

        ``save_annotated`` defaults to ``False`` here because rendering is
        Ultralytics' ``result.plot()`` in the base class and RF-DETR has no
        equivalent; passing ``True`` is accepted and ignored rather than
        pretending to write a file that will not exist.
        """
        # Accepted and ignored: rendering in the base class is Ultralytics'
        # `result.plot()`, which RF-DETR has no equivalent of. Honouring the
        # base signature keeps the two detectors substitutable; pretending to
        # write a file that will not exist would not.
        del save_annotated

        image_path = Path(image_path)
        prediction = ImagePrediction(image_path=image_path)

        if not image_path.is_file():
            prediction.error = f"File not found: {image_path}"
            return prediction

        try:
            self.load()
        except (ResourceNotFoundError, ModelLoadError) as exc:
            prediction.error = str(exc)
            return prediction

        try:
            from PIL import Image

            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                prediction.image_width, prediction.image_height = image.size

                started = time.perf_counter()
                detections = self._model.predict(image, threshold=self.confidence)
                prediction.inference_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            prediction.error = f"Inference failed: {exc}"
            logger.error("Inference failed on %s: %s", image_path.name, exc)
            return prediction

        prediction.detections = self._extract_detections(detections)
        return prediction

    def _extract_detections(self, result: Any) -> list[Detection]:
        """Convert one ``supervision.Detections`` into :class:`Detection`s.

        The parallel arrays ``xyxy`` / ``confidence`` / ``class_id`` describe
        the same detection at row *i*, and ``mask[i]`` is that detection's
        boolean raster at full image resolution. Boxes are already in original
        image pixels, the same space as ground truth, so nothing is rescaled.
        """
        boxes = getattr(result, "xyxy", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = self.class_names
        confidences = getattr(result, "confidence", None)
        class_ids = getattr(result, "class_id", None)
        masks = getattr(result, "mask", None)

        if confidences is None or class_ids is None:
            logger.warning(
                "RF-DETR returned boxes without confidence or class; skipped"
            )
            return []

        detections: list[Detection] = []
        for index, (box, conf, class_id) in enumerate(
            zip(boxes, confidences, class_ids, strict=True)
        ):
            outline = None
            rle = None
            if masks is not None and index < len(masks):
                outline = _mask_to_polygon(masks[index])
                # The raster goes to evaluation untouched, while the polygon
                # above keeps only its largest component. Both are correct for
                # their purpose: the polygon is what Model Doctor stores and
                # draws, and reducing a multi-part instance to one blob there
                # is a deliberate simplification — but scoring against it would
                # charge the model for that simplification rather than for its
                # prediction. RF-DETR's mask is already at full image
                # resolution, so nothing is rescaled.
                if self.keep_raw_masks:
                    from model_doctor.app.evaluation import encode_mask

                    rle = encode_mask(np.asarray(masks[index], dtype=bool))

            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), f"id:{int(class_id)}"),
                    confidence=float(conf),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    polygon=outline,
                    mask_rle=rle,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # -- evaluation ---------------------------------------------------------
    def validate(
        self, data_yaml: Path | None = None, split: str = "val"
    ) -> ValidationMetrics:
        """Refuse to evaluate, and say why.

        The base implementation delegates to Ultralytics' ``model.val()``.
        RF-DETR has no equivalent, and wiring in a *different* evaluator here
        would produce numbers that look like YOLO's and are not comparable to
        them — different matching rules, different NMS, different area
        thresholds. That is precisely the mistake this comparison exists to
        avoid, so it fails loudly instead.

        Both models are scored together, by one evaluator, in
        ``scripts/compare_detectors.py``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RF-DETR has no built-in validator. Evaluate both models with the "
            "shared evaluator in scripts/compare_detectors.py so the numbers "
            "are comparable."
        )


__all__ = [
    "RFDetrDetector",
    "resolution_multiple",
    "resolve_variant",
    "snap_resolution",
]
