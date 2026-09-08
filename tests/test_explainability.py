"""Tests for Grad-CAM explanation.

None of these load a detector. The CAM engine is detector-agnostic by design,
so it is tested against a tiny purpose-built network whose correct answer is
known in advance — which is a stronger check than running the real model and
eyeballing the output.

The YOLO-specific adapter is tested against synthetic head output, so its
anchor arithmetic and scalar selection are verified without a forward pass.

The split mirrors the modules: the generic engine lives in ``utils.cam``, the
detector-specific adapter in ``model_doctor.app.explainability``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app.explainability import Yolo26SegAdapter, _anchor_centres
from model_doctor.utils.cam import GradCAM, Letterbox, prepare_image, render_overlay
from model_doctor.utils.exceptions import ExplainabilityError


class TinyNet(nn.Module):
    """A two-layer CNN whose spatial response is easy to reason about."""

    def __init__(self) -> None:
        """Build the convolutional stack.

        The convolution carries a positive bias so its ReLU cannot zero an
        entire region. Without it a test using a dark input would produce no
        gradient anywhere and fail for a reason unrelated to what it checks.
        """
        super().__init__()
        conv = nn.Conv2d(3, 8, 3, padding=1)
        # Positive weights make activation monotonic in input brightness, and a
        # positive bias stops the ReLU switching a region off entirely. Both
        # matter because Grad-CAM pools gradients *globally*: its map tracks
        # activation magnitude, not where the gradient came from. With the
        # default random weights a bright region produces no reliably larger
        # activation, and a localisation test becomes a coin flip.
        conv.weight.data.abs_()
        nn.init.constant_(conv.bias, 0.5)
        self.features = nn.Sequential(conv, nn.ReLU())
        # Positive head weights keep the channel weights positive, so the map
        # cannot relu to zero for reasons unrelated to what is being tested.
        self.head = nn.Conv2d(8, 2, 1)
        nn.init.constant_(self.head.weight, 0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """Return per-pixel class logits."""
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# The generic CAM engine
# ---------------------------------------------------------------------------
def test_cam_returns_a_normalised_map_of_input_size() -> None:
    """The map covers the whole input and spans [0, 1]."""
    net = TinyNet().eval()
    cam = GradCAM(net, [net.features])
    image = torch.rand(1, 3, 64, 64)

    heat = cam(image, lambda out: out[0, 0].max())

    assert heat.shape == (64, 64)
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    assert np.isfinite(heat).all()


def test_cam_localises_where_the_signal_is() -> None:
    """A scalar taken from one corner produces a map peaking near it.

    This is the property that makes a heatmap meaningful. A CAM that is merely
    normalised and finite could still be noise; this asserts it actually tracks
    the region the scalar came from.
    """
    torch.manual_seed(0)
    net = TinyNet().eval()
    cam = GradCAM(net, [net.features])
    # A dim background rather than zeros, so every position has a live
    # activation and the comparison below measures localisation rather than
    # an accident of which regions the ReLU happened to switch off.
    image = torch.rand(1, 3, 32, 32) * 0.1
    image[..., :8, :8] = 1.0  # a bright top-left patch

    # The head is a 1x1 convolution, so a scalar taken from the top-left can
    # only receive gradient through top-left activations.
    heat = cam(image, lambda out: out[0, 0, :8, :8].mean())

    top_left = heat[:8, :8].mean()
    bottom_right = heat[-8:, -8:].mean()
    assert top_left > bottom_right


def test_cam_requires_at_least_one_target_layer() -> None:
    """An empty layer list is a programming error, not a silent no-op."""
    with pytest.raises(ExplainabilityError, match="target layer"):
        GradCAM(TinyNet(), [])


def test_cam_reports_when_no_gradient_reaches_the_layer() -> None:
    """A disconnected scalar fails loudly rather than returning noise.

    This is the failure that motivated the whole investigation: the decoded
    YOLO score produces exactly zero gradient, and a silent zero map would look
    like a valid explanation.
    """
    net = TinyNet().eval()
    cam = GradCAM(net, [net.features])
    image = torch.rand(1, 3, 32, 32)

    # A constant is differentiable but carries no gradient to any layer.
    detached = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(ExplainabilityError, match="No gradient"):
        cam(image, lambda _out: detached * 1.0)


def test_cam_fuses_multiple_layers() -> None:
    """Several target layers combine into one map rather than erroring."""
    net = TinyNet().eval()
    cam = GradCAM(net, [net.features, net.head])
    heat = cam(torch.rand(1, 3, 32, 32), lambda out: out[0, 1].max())
    assert heat.shape == (32, 32)


def test_cam_hooks_do_not_alter_the_forward_output() -> None:
    """Hooks must return None; returning a value replaces the layer's output.

    That mistake produces a confusing type error deep inside the network, so it
    is worth pinning down.
    """
    net = TinyNet().eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        before = net(image).clone()

    GradCAM(net, [net.features])(image, lambda out: out[0, 0].max())

    with torch.no_grad():
        after = net(image)
    assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# Letterboxing — a wrong mapping silently explains the wrong region
# ---------------------------------------------------------------------------
def test_letterbox_maps_a_box_into_padded_space() -> None:
    """Scale then offset, matching how the image was placed on the canvas."""
    box = Letterbox(scale=0.5, pad_x=10.0, pad_y=20.0, size=640).map_box(
        (100.0, 200.0, 300.0, 400.0)
    )
    assert box == (60.0, 120.0, 160.0, 220.0)


def test_prepare_image_preserves_aspect_ratio(tmp_path: Path) -> None:
    """Padding, not stretching. Stretching would shift every feature."""
    path = tmp_path / "wide.jpg"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(path)

    tensor, letterbox, original = prepare_image(path, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert original.shape == (200, 400, 3)
    assert letterbox.scale == pytest.approx(1.6)
    # A 400x200 image scaled by 1.6 is 640x320, so it is padded vertically only.
    assert letterbox.pad_x == pytest.approx(0.0)
    assert letterbox.pad_y == pytest.approx(160.0)


def test_prepare_image_reports_an_unreadable_file(tmp_path: Path) -> None:
    """A missing image says so instead of raising something opaque."""
    with pytest.raises(ExplainabilityError, match="Could not read"):
        prepare_image(tmp_path / "absent.jpg", 640)


def test_render_overlay_matches_the_original_size() -> None:
    """The overlay lines up with the coordinates stored in the database."""
    original = np.zeros((200, 400, 3), dtype=np.uint8)
    cam = np.linspace(0, 1, 640 * 640, dtype=np.float32).reshape(640, 640)

    out = render_overlay(original, cam, Letterbox(1.6, 0.0, 160.0, 640))

    assert out.shape == original.shape
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# The YOLO adapter
# ---------------------------------------------------------------------------
def test_anchor_centres_match_the_pyramid_layout() -> None:
    """9261 anchors at 672 input is 84x84 + 42x42 + 21x21."""
    total = 84 * 84 + 42 * 42 + 21 * 21
    centres = _anchor_centres(total, torch.device("cpu"))

    assert centres.shape == (total, 2)
    # First anchor sits at the centre of the finest grid's first cell.
    assert centres[0].tolist() == pytest.approx([4.0, 4.0])
    assert centres[:, 0].max() <= 672 and centres[:, 1].max() <= 672


def test_anchor_centres_rejects_an_unknown_layout() -> None:
    """An unrecognised anchor count is reported, never silently guessed."""
    with pytest.raises(ExplainabilityError, match="anchor grid layout"):
        _anchor_centres(12345, torch.device("cpu"))


def _fake_output(scores: torch.Tensor):
    """Build head output shaped like the real model's second return value."""
    return (None, {"one2one": {"scores": scores}})


def test_adapter_selects_the_strongest_logit_inside_the_box() -> None:
    """The scalar answers "what fired *here*", not "what fires anywhere"."""
    total = 84 * 84 + 42 * 42 + 21 * 21
    scores = torch.full((1, 2, total), -10.0, requires_grad=True)
    centres = _anchor_centres(total, torch.device("cpu"))

    inside = int(
        (
            (centres[:, 0] > 100)
            & (centres[:, 0] < 200)
            & (centres[:, 1] > 100)
            & (centres[:, 1] < 200)
        )
        .nonzero()[0]
        .item()
    )
    outside = int((centres[:, 0] > 600).nonzero()[0].item())
    boosted = scores.detach().clone()
    boosted[0, 0, inside] = 5.0
    boosted[0, 0, outside] = 99.0  # much stronger, but far away
    boosted.requires_grad_(True)

    scalar = Yolo26SegAdapter().scalar_for(
        _fake_output(boosted), 0, (100.0, 100.0, 200.0, 200.0)
    )
    assert scalar.item() == pytest.approx(5.0)


def test_adapter_falls_back_to_the_nearest_anchor() -> None:
    """A box too small to contain any anchor centre still yields a scalar.

    Small objects are precisely the ones worth explaining, so this must not
    fail on them.
    """
    total = 84 * 84 + 42 * 42 + 21 * 21
    scores = torch.rand(1, 2, total, requires_grad=True)
    scalar = Yolo26SegAdapter().scalar_for(
        _fake_output(scores), 1, (100.0, 100.0, 100.5, 100.5)
    )
    assert scalar.requires_grad


def test_adapter_rejects_an_out_of_range_class() -> None:
    """A class index the model does not have is a caller error."""
    total = 84 * 84 + 42 * 42 + 21 * 21
    scores = torch.rand(1, 2, total)
    with pytest.raises(ExplainabilityError, match="outside the model's"):
        Yolo26SegAdapter().scalar_for(_fake_output(scores), 7, (0.0, 0.0, 50.0, 50.0))


def test_adapter_rejects_output_without_raw_scores() -> None:
    """A model whose head does not expose raw logits is reported clearly."""
    adapter = Yolo26SegAdapter()
    with pytest.raises(ExplainabilityError, match="raw head predictions"):
        adapter.scalar_for(torch.rand(1, 300, 38), 0, (0.0, 0.0, 10.0, 10.0))
    with pytest.raises(ExplainabilityError, match="no 'scores'"):
        adapter.scalar_for((None, {"one2one": {}}), 0, (0.0, 0.0, 10.0, 10.0))


def test_adapter_name_records_the_layers_used() -> None:
    """Stored heatmaps carry which layers produced them."""
    assert Yolo26SegAdapter().name == "yolo26-seg:16,19,22"
    assert Yolo26SegAdapter(layer_indices=(22,)).name == "yolo26-seg:22"
