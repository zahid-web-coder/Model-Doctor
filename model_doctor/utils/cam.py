"""Gradient-weighted class activation mapping, and the image plumbing it needs.

Detector-agnostic by construction rather than by discipline. This module lives
in ``utils`` precisely because ``utils`` may not import ``app``: the layering
rule makes it structurally impossible for the CAM engine to acquire a
dependency on any particular detector.

The split mirrors :mod:`utils.matching` and :mod:`model_doctor.app.diagnosis` — the
generic
algorithm lives here, and the domain interpretation of its output lives in the
application layer. :class:`CamAdapter` is the seam between them: it answers the
only two architecture-specific questions, which layers to visualise and how to
obtain a differentiable scalar.

Nothing here knows what a detection is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from model_doctor.utils.exceptions import ExplainabilityError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Blend weight for the heatmap over the original image. High enough to read the
# activation, low enough that the underlying object stays identifiable — a
# heatmap you cannot locate in the image explains nothing.
DEFAULT_OVERLAY_ALPHA: float = 0.45


class CamAdapter(Protocol):
    """Everything Grad-CAM needs to know about a particular detector.

    Two questions, both architecture-specific: which layers carry the features
    worth visualising, and how to obtain a differentiable scalar representing
    "this class, at this location". Everything else is generic.
    """

    @property
    def name(self) -> str:
        """Return an identifier recorded alongside each heatmap."""
        ...

    def target_layers(self, model: Any) -> list[Any]:
        """Return the modules whose activations should be visualised."""
        ...

    def scalar_for(
        self, output: Any, class_id: int, box: tuple[float, float, float, float]
    ) -> Any:
        """Return a differentiable scalar for ``class_id`` inside ``box``."""
        ...



class GradCAM:
    """Gradient-weighted class activation mapping over arbitrary layers.

    Entirely detector-agnostic: it hooks modules, weights their activations by
    the gradients of a scalar the caller chooses, and returns a normalised map.
    Which layers and which scalar are the adapter's business.
    """

    def __init__(self, model: Any, target_layers: Sequence[Any]) -> None:
        """Prepare hooks without running anything.

        Args:
            model: The network to explain.
            target_layers: Modules whose activations are visualised.
        """
        self.model = model
        self.target_layers = list(target_layers)
        if not self.target_layers:
            raise ExplainabilityError("At least one target layer is required.")

    def __call__(self, image: Any, scalar_fn: Any) -> Any:
        """Return a normalised activation map for one input.

        Args:
            image: Input tensor shaped ``(1, 3, H, W)``.
            scalar_fn: Callable taking the model output and returning a scalar
                tensor to differentiate.

        Returns:
            A ``(H, W)`` array in ``[0, 1]``.

        Raises:
            ExplainabilityError: If no gradient reached any target layer, which
                means the chosen scalar is disconnected from them.
        """
        import torch

        activations: dict[int, Any] = {}
        gradients: dict[int, Any] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module, _inputs, output):
                # A forward hook that returns a value *replaces* the layer's
                # output, so this must return None.
                if torch.is_tensor(output) and output.requires_grad:
                    activations[index] = output
                    output.register_hook(
                        lambda grad: gradients.__setitem__(index, grad.detach())
                    )

            return hook

        for index, layer in enumerate(self.target_layers):
            handles.append(layer.register_forward_hook(make_hook(index)))

        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                output = self.model(image)
                scalar = scalar_fn(output)
                scalar.backward()
        finally:
            for handle in handles:
                handle.remove()

        maps = []
        received_gradient = False
        height, width = image.shape[-2:]
        for index in range(len(self.target_layers)):
            activation = activations.get(index)
            gradient = gradients.get(index)
            if activation is None or gradient is None or gradient.abs().sum() == 0:
                continue
            received_gradient = True
            # Channel importance is the spatially-averaged gradient; the map is
            # the positively-contributing part of the weighted activation sum.
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation.detach()).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=(height, width), mode="bilinear", align_corners=False
            )
            peak = cam.max()
            # Normalise per level before fusing, so a level with large raw
            # magnitudes cannot drown out a better-localised one. A level whose
            # weights are entirely negative relus to zero — that is a real
            # result ("no positive evidence here"), not an error, so it is kept
            # rather than dropped.
            maps.append((cam / peak)[0, 0] if peak > 0 else cam[0, 0])

        if not maps:
            if received_gradient:
                raise ExplainabilityError(
                    "Gradients reached the target layers but produced no usable "
                    "map. This should not happen; the layer set is likely wrong."
                )
            raise ExplainabilityError(
                "No gradient reached any target layer. The scalar is probably "
                "disconnected from them, or has saturated."
            )

        fused = torch.stack(maps).mean(dim=0)
        span = fused.max() - fused.min()
        if span > 0:
            fused = (fused - fused.min()) / span
        return fused.detach().cpu().numpy()



@dataclass(frozen=True)
class Letterbox:
    """A resize-and-pad transform, and the numbers needed to invert it.

    The model sees a square, padded image. Boxes recorded against the original
    must be mapped into that space or the heatmap would be computed for the
    wrong region — a failure that produces a plausible-looking wrong answer
    rather than an error.
    """

    scale: float
    pad_x: float
    pad_y: float
    size: int

    def map_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map a box from original pixels into letterboxed pixels."""
        x1, y1, x2, y2 = box
        return (
            x1 * self.scale + self.pad_x,
            y1 * self.scale + self.pad_y,
            x2 * self.scale + self.pad_x,
            y2 * self.scale + self.pad_y,
        )


def prepare_image(path: Path, size: int) -> tuple[Any, Letterbox, Any]:
    """Load an image and letterbox it to the model's input size.

    Aspect ratio is preserved by padding rather than stretching, matching what
    the detector saw at inference. A stretched image would shift every feature
    and quietly invalidate the explanation.

    Args:
        path: Image to load.
        size: Square input size.

    Returns:
        ``(tensor, letterbox, original_rgb_array)``.

    Raises:
        ExplainabilityError: If the image cannot be read.
    """
    import numpy as np
    import torch
    from PIL import Image

    try:
        with Image.open(path) as handle:
            original = handle.convert("RGB")
            array = np.array(original)
    except Exception as exc:
        raise ExplainabilityError(f"Could not read {path}: {exc}") from exc

    height, width = array.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0

    resized = original.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (int(pad_x), int(pad_y)))

    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), Letterbox(scale, pad_x, pad_y, size), array


def render_overlay(
    original: Any, cam: Any, letterbox: Letterbox, alpha: float = DEFAULT_OVERLAY_ALPHA
) -> Any:
    """Blend a heatmap over the original image, undoing the letterbox.

    The map is computed in padded space; the padding is cropped and the result
    resized back, so the returned image lines up with the coordinates stored in
    the database and with whatever the dashboard draws on top.

    Args:
        original: Original RGB array.
        cam: Normalised map over the letterboxed input.
        letterbox: The transform that was applied.
        alpha: Heatmap opacity.

    Returns:
        An RGB array the same size as ``original``.
    """
    import cv2
    import numpy as np

    height, width = original.shape[:2]
    top = int(letterbox.pad_y)
    left = int(letterbox.pad_x)
    bottom = letterbox.size - top
    right = letterbox.size - left
    cropped = cam[top:bottom, left:right]
    if cropped.size == 0:
        cropped = cam

    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    coloured = cv2.applyColorMap(
        np.uint8(255 * np.clip(resized, 0.0, 1.0)), cv2.COLORMAP_JET
    )
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * coloured + (1 - alpha) * original)
