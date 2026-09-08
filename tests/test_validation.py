"""Validation must refuse clearly rather than let a bad input fail later.

The value of this module is entirely in its negative cases. Anyone can accept
a good dataset; the question is whether an operator who uploads the wrong thing
is told which thing was wrong, before a detector spends twenty minutes finding
out for them.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from model_doctor.app import validation


def _dataset(root: Path, *, labels: str = "0 0.5 0.5 0.2 0.2\n", split: str = "test",
             classes: str = "names:\n  0: column\n") -> Path:
    """Build a minimal but genuine YOLO dataset on disk."""
    from PIL import Image

    images = root / split / "images"
    label_dir = root / split / "labels"
    images.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "a.jpg")
    (label_dir / "a.txt").write_text(labels)
    (root / "data.yaml").write_text(f"{split}: {split}/images\n{classes}")
    return root


class TestDatasetValidation:
    """A dataset is accepted only when every part the pipeline needs is there."""

    def test_accepts_a_well_formed_dataset(self, tmp_path: Path) -> None:
        """Accepts a well formed dataset."""
        result = validation.validate_dataset(
            _dataset(tmp_path / "ds"), require_split="test"
        )
        assert result.ok, [c.detail for c in result.failures()]
        assert result.facts["class_names"] == {0: "column"}
        assert result.facts["image_count"] == 1

    def test_refuses_an_upload_with_no_descriptor(self, tmp_path: Path) -> None:
        """Refuses an upload with no descriptor."""
        (tmp_path / "loose").mkdir()
        (tmp_path / "loose" / "a.jpg").write_bytes(b"x")
        result = validation.validate_dataset(tmp_path / "loose")
        assert not result.ok
        assert "data.yaml" in result.failures()[0].detail

    def test_refuses_a_dataset_with_no_labels(self, tmp_path: Path) -> None:
        """Ground truth is the thing failures are diagnosed against."""
        root = _dataset(tmp_path / "ds")
        for label in (root / "test" / "labels").iterdir():
            label.unlink()
        (root / "test" / "labels").rmdir()
        result = validation.validate_dataset(root, require_split="test")
        assert not result.ok
        assert any("Ground truth is" in c.detail for c in result.failures())

    def test_reports_a_split_that_does_not_exist(self, tmp_path: Path) -> None:
        """Reports a split that does not exist."""
        result = validation.validate_dataset(
            _dataset(tmp_path / "ds"), require_split="train"
        )
        assert not result.ok

    def test_rejects_annotations_that_are_not_normalised(self, tmp_path: Path) -> None:
        """Pixel coordinates in a YOLO label file are a silent disaster later."""
        root = _dataset(tmp_path / "ds", labels="0 640 480 100 100\n")
        result = validation.validate_dataset(root, require_split="test")
        assert not result.ok
        assert any("normalised" in c.detail for c in result.failures())

    def test_rejects_a_class_id_the_dataset_does_not_declare(
        self, tmp_path: Path
    ) -> None:
        """Rejects a class id the dataset does not declare."""
        root = _dataset(tmp_path / "ds", labels="7 0.5 0.5 0.2 0.2\n")
        result = validation.validate_dataset(root, require_split="test")
        assert not result.ok
        assert any("class id 7" in c.detail for c in result.failures())

    def test_rejects_a_truncated_annotation_line(self, tmp_path: Path) -> None:
        """Rejects a truncated annotation line."""
        root = _dataset(tmp_path / "ds", labels="0 0.5 0.5\n")
        result = validation.validate_dataset(root, require_split="test")
        assert not result.ok
        assert any("at least 5" in c.detail for c in result.failures())

    def test_accepts_a_polygon_annotation(self, tmp_path: Path) -> None:
        """Segmentation labels are longer than five fields and are still valid."""
        root = _dataset(tmp_path / "ds", labels="0 0.1 0.1 0.5 0.1 0.3 0.6\n")
        assert validation.validate_dataset(root, require_split="test").ok

    def test_finds_a_descriptor_one_level_down(self, tmp_path: Path) -> None:
        """A dataset zipped inside its own folder is the common case."""
        outer = tmp_path / "upload"
        outer.mkdir()
        _dataset(outer / "mydata")
        result = validation.validate_dataset(outer, require_split="test")
        assert result.ok, [c.detail for c in result.failures()]

    def test_does_not_hardcode_the_project_dataset(self, tmp_path: Path) -> None:
        """Any compatible dataset is acceptable, not only the demo one."""
        root = _dataset(
            tmp_path / "other",
            classes="names:\n  0: crack\n  1: spall\n",
            labels="1 0.5 0.5 0.2 0.2\n",
        )
        result = validation.validate_dataset(root, require_split="test")
        assert result.ok, [c.detail for c in result.failures()]
        assert result.facts["class_names"] == {0: "crack", 1: "spall"}


class TestModelValidation:
    """A checkpoint is identified without being executed."""

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        """Reports a missing file."""
        result = validation.validate_model(tmp_path / "nope.pt")
        assert not result.ok

    def test_refuses_an_unsupported_declared_family(self, tmp_path: Path) -> None:
        """Refuses an unsupported declared family."""
        weights = tmp_path / "m.pt"
        weights.write_bytes(b"not really a checkpoint")
        result = validation.validate_model(weights, declared_family="detectron")
        assert not result.ok
        assert any("no adapter" in c.detail for c in result.failures())

    def test_a_file_that_is_not_a_checkpoint_is_not_guessed_at(
        self, tmp_path: Path
    ) -> None:
        """Refusing beats inventing a family for an unreadable file."""
        weights = tmp_path / "m.pt"
        weights.write_bytes(b"\x00\x01\x02 not a torch file")
        family, explanation = validation.detect_model_family(weights)
        assert family is None
        assert "supported" in explanation.lower()

    def test_identification_never_unpickles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The byte scan must answer without torch.load being reached at all.

        This is the security property, not a performance one: a checkpoint that
        serialises a model object executes code when unpickled, so identifying
        one must not require unpickling it.
        """
        weights = tmp_path / "yolo_like.pt"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr(
                "m/data.pkl", b"\x80\x04ultralytics.nn.tasks\nSegmentationModel"
            )
        weights.write_bytes(buffer.getvalue())

        def explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("torch.load must not be called to identify a file")

        import torch

        monkeypatch.setattr(torch, "load", explode)
        family, _ = validation.detect_model_family(weights)
        assert family == "yolo"

    def test_a_declared_family_that_contradicts_the_file_is_reported(
        self, tmp_path: Path
    ) -> None:
        """Running an RF-DETR checkpoint as YOLO produces nonsense, not an error."""
        weights = tmp_path / "m.pt"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("m/data.pkl", b"rfdetr.models")
        weights.write_bytes(buffer.getvalue())

        result = validation.validate_model(weights, declared_family="yolo")
        assert not result.ok
        assert any("looks like 'rfdetr'" in c.detail for c in result.failures())


class TestUnidentifiableCheckpoints:
    """A file that cannot be identified must not report itself as validated."""

    def test_an_unidentifiable_file_fails_validation(self, tmp_path: Path) -> None:
        """Found by driving the UI: every check showed a green tick.

        With a declared family, an unidentifiable checkpoint fell through to
        the branch that records a *pass*, so the panel read "Declared family
        matches the file" beside an explanation saying it could not be
        identified at all. `ok` was therefore true, and once a valid dataset
        was added the Run button would enable and start a run that could only
        fail during inference.
        """
        weights = tmp_path / "notamodel.pt"
        weights.write_bytes(b"this is not a checkpoint")

        result = validation.validate_model(weights, declared_family="yolo")

        assert not result.ok, "an unidentifiable checkpoint reported as valid"
        failed = [c.name for c in result.failures()]
        assert "Declared family matches the file" in failed

    def test_an_identified_matching_file_still_passes(self, tmp_path: Path) -> None:
        """The fix must not make every declared family fail."""
        weights = tmp_path / "m.pt"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("m/data.pkl", b"ultralytics.nn.tasks")
        weights.write_bytes(buffer.getvalue())

        result = validation.validate_model(weights, declared_family="yolo")
        assert result.ok, [c.detail for c in result.failures()]

    def test_the_same_file_without_a_declared_family_also_fails(
        self, tmp_path: Path
    ) -> None:
        """Both paths through this function must agree about an unknown file."""
        weights = tmp_path / "notamodel.pt"
        weights.write_bytes(b"this is not a checkpoint")
        assert not validation.validate_model(weights).ok
