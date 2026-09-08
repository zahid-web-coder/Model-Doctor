"""Tests for graceful behaviour when resources are absent.

These matter more than usual on this project. The model and dataset are not
available yet, so "behaves correctly with nothing installed" is the *primary*
contract right now — not an edge case. Each test below pins down one promise
made in the module docstrings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor import config
from model_doctor.app.inference import (
    Detection,
    ImagePrediction,
    build_parser,
    format_detections,
)
from model_doctor.utils import resources
from model_doctor.utils.dataset import (
    label_path_for_image,
    load_dataset_config,
    load_ground_truth,
)
from model_doctor.utils.exceptions import DatasetConfigError


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------
def test_find_images_on_missing_directory_returns_empty(tmp_path: Path) -> None:
    """A non-existent directory yields [] instead of raising."""
    assert resources.find_images(tmp_path / "does-not-exist") == []


def test_find_images_is_case_insensitive(tmp_path: Path) -> None:
    """Uppercase extensions must not be silently skipped."""
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.JPG").touch()
    (tmp_path / "c.PNG").touch()
    (tmp_path / "notes.txt").touch()

    found = {p.name for p in resources.find_images(tmp_path)}
    assert found == {"a.jpg", "b.JPG", "c.PNG"}


def test_find_images_is_sorted(tmp_path: Path) -> None:
    """Ordering is deterministic so batch runs are reproducible."""
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        (tmp_path / name).touch()
    found = [p.name for p in resources.find_images(tmp_path)]
    assert found == ["a.jpg", "b.jpg", "c.jpg"]


def test_discover_model_returns_none_when_absent(tmp_path: Path, monkeypatch) -> None:
    """No weights anywhere is reported as None, not an exception."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "best.pt")
    assert resources.discover_model() is None


def test_discover_model_falls_back_to_any_pt(tmp_path: Path, monkeypatch) -> None:
    """Any .pt file is found even when it is not named best.pt."""
    weights = tmp_path / "yolov8s_run12.pt"
    weights.write_bytes(b"fake-weights")

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "best.pt")
    assert resources.discover_model() == weights


def test_check_model_treats_empty_file_as_missing(tmp_path: Path, monkeypatch) -> None:
    """A 0-byte file is a failed download, not a usable model."""
    stub = tmp_path / "best.pt"
    stub.touch()

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", stub)

    status = resources.check_model()
    assert not status.available
    assert "0 bytes" in status.detail


def test_verify_all_never_raises() -> None:
    """The health check must survive any resource state."""
    statuses = resources.verify_all()
    assert len(statuses) == 4
    assert all(isinstance(s, resources.ResourceStatus) for s in statuses)


def test_format_report_is_renderable() -> None:
    """The report renders regardless of what is missing."""
    report = resources.format_report(resources.verify_all())
    assert "Model Doctor" in report


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------
def test_load_dataset_config_missing_file_raises_clear_error(tmp_path: Path) -> None:
    """A missing data.yaml produces our error type, not a bare OSError."""
    with pytest.raises(DatasetConfigError, match="not found"):
        load_dataset_config(tmp_path / "data.yaml")


def test_load_dataset_config_accepts_list_names(tmp_path: Path) -> None:
    """The list form of `names:` maps index -> name."""
    (tmp_path / "test" / "images").mkdir(parents=True)
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        "test: test/images\nnc: 2\nnames: ['crack', 'spall']\n", encoding="utf-8"
    )

    dataset = load_dataset_config(yaml_file)
    assert dataset.class_names == {0: "crack", 1: "spall"}
    assert dataset.num_classes == 2
    assert dataset.splits["test"] == (tmp_path / "test" / "images").resolve()


def test_load_dataset_config_accepts_dict_names(tmp_path: Path) -> None:
    """The dict form of `names:` is equally valid."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names:\n  0: crack\n  1: spall\n", encoding="utf-8")
    assert load_dataset_config(yaml_file).class_names == {0: "crack", 1: "spall"}


def test_name_for_unknown_id_does_not_raise(tmp_path: Path) -> None:
    """An out-of-range class id degrades visibly instead of crashing."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names: ['crack']\n", encoding="utf-8")
    assert load_dataset_config(yaml_file).name_for(99) == "id:99"


# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------
def test_label_path_swaps_last_images_segment() -> None:
    """images/ -> labels/ substitution targets the correct segment."""
    image = Path("/data/images/test/images/frame_001.jpg")
    assert label_path_for_image(image) == Path("/data/images/test/labels/frame_001.txt")


def test_load_ground_truth_denormalises_correctly(tmp_path: Path) -> None:
    """A centred, half-size box maps to the expected pixel corners."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = images_dir / "frame.jpg"
    image.touch()
    # class 0, centre (0.5, 0.5), size 0.5 x 0.5 on a 640x480 image.
    (labels_dir / "frame.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    boxes = load_ground_truth(image, 640, 480)
    assert len(boxes) == 1
    assert boxes[0].class_id == 0
    assert boxes[0].xyxy == pytest.approx((160.0, 120.0, 480.0, 360.0))
    assert boxes[0].area == pytest.approx(320.0 * 240.0)


def test_load_ground_truth_missing_label_is_a_negative_sample(tmp_path: Path) -> None:
    """An image with no label file is legitimate background, not an error."""
    image = tmp_path / "images" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    assert load_ground_truth(image, 640, 480) == []


def test_load_ground_truth_skips_malformed_lines(tmp_path: Path) -> None:
    """Truncated and non-numeric lines are skipped; valid lines still parse.

    The 8-field line has an even count, meaning a coordinate is missing. That
    is a truncated polygon, not a valid one, and must not be reinterpreted.
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = images_dir / "frame.jpg"
    image.touch()
    (labels_dir / "frame.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n"
        "1 0.1 0.1 0.2 0.2 0.3 0.3 0.4\n"  # 8 fields, even — truncated polygon
        "not-a-number 0.1 0.1 0.2 0.2\n"  # non-numeric
        "\n"  # blank
        "1 0.25 0.25 0.1 0.1\n",
        encoding="utf-8",
    )

    boxes = load_ground_truth(image, 100, 100)
    assert [b.class_id for b in boxes] == [0, 1]


# ---------------------------------------------------------------------------
# Inference containers and CLI
# ---------------------------------------------------------------------------
def test_detection_geometry() -> None:
    """Derived geometry properties are computed, never stored twice."""
    det = Detection(
        class_id=0,
        class_name="crack",
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=70.0,
        confidence=0.9,
    )
    assert det.width == 100.0
    assert det.height == 50.0
    assert det.area == 5000.0


def test_image_prediction_error_state() -> None:
    """An errored prediction reports itself as failed and renders safely."""
    prediction = ImagePrediction(image_path=Path("missing.jpg"), error="File not found")
    assert not prediction.succeeded
    assert "ERROR" in format_detections(prediction)


def test_format_detections_handles_zero_detections() -> None:
    """An empty result renders a clear message rather than an empty table."""
    prediction = ImagePrediction(
        image_path=Path("frame.jpg"), image_width=640, image_height=480
    )
    assert "No detections" in format_detections(prediction)


def test_cli_defaults() -> None:
    """CLI parses with no arguments and defaults to the test split."""
    args = build_parser().parse_args([])
    assert args.split == "test"
    assert args.check is False
