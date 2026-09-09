"""Reading ground truth in whichever format the dataset is actually in.

The point of this layer is that a COCO dataset is read *as COCO* — never
converted, never rewritten, never quietly interpreted as YOLO — and that both
formats arrive downstream as the same ``ObjectAnnotation``. These tests hold
that line at the two places it can break: the mapping from arbitrary COCO
category ids to a model's contiguous class ids, and the treatment of a split
whose annotations cannot be found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.utils.dataset import load_dataset_config
from model_doctor.utils.exceptions import DatasetConfigError
from model_doctor.utils.ground_truth import (
    FORMAT_COCO,
    FORMAT_YOLO,
    detect_format,
    open_ground_truth,
    read_coco_categories,
    verify_class_names,
)


# ---------------------------------------------------------------------------
# Fixtures: the two layouts, built the way the exporters actually write them
# ---------------------------------------------------------------------------
def _coco_split(
    root: Path,
    *,
    categories=((1, "column"), (2, "beam")),
    annotations=None,
    images=("a.jpg",),
    size=(100, 50),
    file_name_prefix="",
) -> Path:
    """A Roboflow-shaped split: images beside one annotation file."""
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(images, start=1):
        Image.new("RGB", size).save(root / name)
        records.append({
            "id": index * 10,  # ids that are neither 0- nor 1-based
            "file_name": f"{file_name_prefix}{name}",
            "width": size[0],
            "height": size[1],
        })
    payload = {
        "images": records,
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for cid, name in categories
        ],
        "annotations": list(annotations or []),
    }
    (root / "_annotations.coco.json").write_text(json.dumps(payload))
    return root


def _yolo_split(root: Path, *, labels="0 0.5 0.5 0.2 0.2\n") -> Path:
    """The original layout: images/ and labels/ side by side."""
    images = root / "images"
    label_dir = root / "labels"
    images.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(images / "a.jpg")
    (label_dir / "a.txt").write_text(labels)
    return images


def _box(annotation_id, image_id, category_id, bbox=(10, 10, 20, 20), **extra):
    entry = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": float(bbox[2] * bbox[3]),
        "iscrowd": 0,
    }
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Format detection — explicit, never guessed
# ---------------------------------------------------------------------------
class TestFormatDetection:
    def test_a_yolo_split_is_recognised(self, tmp_path):
        assert detect_format(_yolo_split(tmp_path / "ds")) == FORMAT_YOLO

    def test_a_coco_split_is_recognised(self, tmp_path):
        assert detect_format(_coco_split(tmp_path / "test")) == FORMAT_COCO

    def test_a_split_with_neither_is_reported_as_neither(self, tmp_path):
        """Not "empty": a split whose annotations cannot be found is unknown."""
        bare = tmp_path / "bare"
        bare.mkdir()
        Image.new("RGB", (10, 10)).save(bare / "a.jpg")
        assert detect_format(bare) is None

    def test_labels_win_when_a_split_somehow_has_both(self, tmp_path):
        """The format that works today keeps working, unchanged."""
        images = _yolo_split(tmp_path / "ds")
        (images.parent / "_annotations.coco.json").write_text("{}")
        assert detect_format(images) == FORMAT_YOLO

    def test_an_unreadable_split_refuses_rather_than_reads_as_empty(self, tmp_path):
        """The protection that matters most.

        Reading an unrecognised split as one with no annotations would mark
        every prediction spurious, every image empty, and report success.
        """
        bare = tmp_path / "bare"
        bare.mkdir()
        with pytest.raises(DatasetConfigError, match="neither layout"):
            open_ground_truth(bare, {0: "column"})


# ---------------------------------------------------------------------------
# Category ids — arbitrary in COCO, contiguous in a model
# ---------------------------------------------------------------------------
class TestCategoryMapping:
    def test_ids_map_by_sorted_position(self, tmp_path):
        """The same rule rfdetr applies with remap_category_ids=True."""
        annotation = _coco_split(
            tmp_path / "test", categories=((7, "column"), (3, "beam"))
        ) / "_annotations.coco.json"
        names, mapping = read_coco_categories(annotation)
        assert names == {0: "beam", 1: "column"}, "sorted by id, not by file order"
        assert mapping == {3: 0, 7: 1}

    def test_non_contiguous_ids_are_not_assumed_away(self, tmp_path):
        """Ids of 1 and 900 must still become labels 0 and 1."""
        split = _coco_split(
            tmp_path / "test",
            categories=((1, "column"), (900, "beam")),
            annotations=[_box(1, 10, 900)],
        )
        source = open_ground_truth(split)
        assert source.class_names == {0: "column", 1: "beam"}
        found = source.annotations_for(split / "a.jpg", 100, 50)
        assert [a.class_id for a in found] == [1]
        assert [a.class_name for a in found] == ["beam"]

    def test_an_undeclared_category_is_skipped_not_guessed(self, tmp_path):
        split = _coco_split(
            tmp_path / "test",
            categories=((1, "column"),),
            annotations=[_box(1, 10, 1), _box(2, 10, 99)],
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert len(found) == 1, "the unknown category is dropped, not relabelled"

    def test_a_file_with_no_categories_is_refused(self, tmp_path):
        split = _coco_split(tmp_path / "test", categories=())
        with pytest.raises(DatasetConfigError, match="no categories"):
            open_ground_truth(split)


# ---------------------------------------------------------------------------
# Geometry — boxes, polygons, RLE
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_a_box_is_read_in_absolute_pixels(self, tmp_path):
        """COCO is already absolute; nothing is denormalised."""
        split = _coco_split(
            tmp_path / "test", annotations=[_box(1, 10, 1, bbox=(10, 20, 30, 40))]
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert (found[0].x1, found[0].y1) == (10.0, 20.0)
        assert (found[0].x2, found[0].y2) == (40.0, 60.0)

    def test_a_polygon_is_preserved_not_reduced_to_its_box(self, tmp_path):
        split = _coco_split(
            tmp_path / "test",
            annotations=[
                _box(1, 10, 1, segmentation=[[0, 0, 40, 0, 40, 30, 0, 30]])
            ],
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert found[0].polygon is not None
        assert len(found[0].polygon) == 4
        assert found[0].has_polygon

    def test_the_largest_piece_wins_for_a_split_object(self, tmp_path):
        """One outline per object, the same reduction the prediction side makes."""
        split = _coco_split(
            tmp_path / "test",
            annotations=[_box(1, 10, 1, segmentation=[
                [0, 0, 2, 0, 2, 2, 0, 2],              # area 4
                [10, 10, 40, 10, 40, 40, 10, 40],      # area 900
            ])],
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        xs = [point[0] for point in found[0].polygon]
        assert min(xs) == 10.0, "kept the larger ring"

    def test_an_rle_mask_becomes_an_outline(self, tmp_path):
        """RLE is decoded rather than discarded, so a mask still yields geometry."""
        pytest.importorskip("pycocotools")
        pytest.importorskip("cv2")
        import numpy as np
        from pycocotools import mask as coco_mask

        raster = np.zeros((50, 100), dtype=np.uint8)
        raster[10:40, 20:60] = 1
        rle = coco_mask.encode(np.asfortranarray(raster))
        rle["counts"] = rle["counts"].decode("utf-8")

        split = _coco_split(
            tmp_path / "test", annotations=[_box(1, 10, 1, segmentation=rle)]
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert found[0].polygon is not None
        xs = [p[0] for p in found[0].polygon]
        ys = [p[1] for p in found[0].polygon]
        assert 19 <= min(xs) <= 21 and 58 <= max(xs) <= 60
        assert 9 <= min(ys) <= 11 and 38 <= max(ys) <= 40

    def test_a_malformed_annotation_does_not_discard_the_file(self, tmp_path):
        split = _coco_split(
            tmp_path / "test",
            annotations=[{"id": 1, "image_id": 10, "category_id": 1}, _box(2, 10, 1)],
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert len(found) == 1, "the good annotation survives the bad one"


# ---------------------------------------------------------------------------
# Image identity — the file name is the only key both halves share
# ---------------------------------------------------------------------------
class TestImageMapping:
    def test_annotations_reach_the_right_image(self, tmp_path):
        split = _coco_split(
            tmp_path / "test",
            images=("a.jpg", "b.jpg"),
            annotations=[_box(1, 10, 1), _box(2, 20, 2), _box(3, 20, 2)],
        )
        source = open_ground_truth(split)
        assert len(source.annotations_for(split / "a.jpg", 100, 50)) == 1
        assert len(source.annotations_for(split / "b.jpg", 100, 50)) == 2

    def test_a_directory_prefix_in_file_name_is_tolerated(self, tmp_path):
        """Some exporters write "images/a.jpg"; a prediction knows only "a.jpg"."""
        split = _coco_split(
            tmp_path / "test",
            annotations=[_box(1, 10, 1)],
            file_name_prefix="images/",
        )
        found = open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50)
        assert len(found) == 1

    def test_an_unannotated_image_is_a_negative_sample(self, tmp_path):
        """Exactly as a missing .txt is on the YOLO side — absent, not an error."""
        split = _coco_split(tmp_path / "test", images=("a.jpg", "b.jpg"),
                            annotations=[_box(1, 10, 1)])
        assert open_ground_truth(split).annotations_for(split / "b.jpg", 100, 50) == []

    def test_empty_annotations_are_read_as_empty_not_refused(self, tmp_path):
        split = _coco_split(tmp_path / "test", annotations=[])
        assert open_ground_truth(split).annotations_for(split / "a.jpg", 100, 50) == []


# ---------------------------------------------------------------------------
# The evaluator's view
# ---------------------------------------------------------------------------
class TestCocoGroundTruthForTheEvaluator:
    def test_categories_are_the_models_label_space(self, tmp_path):
        """A prediction's class id must mean the same as an annotation's."""
        split = _coco_split(
            tmp_path / "test",
            categories=((5, "column"), (9, "beam")),
            annotations=[_box(1, 10, 9)],
        )
        gt = open_ground_truth(split).coco_ground_truth(split)
        assert [c["id"] for c in gt["categories"]] == [0, 1]
        assert gt["annotations"][0]["category_id"] == 1

    def test_segmentation_is_carried_through_untouched(self, tmp_path):
        """The evaluator scores every component, not the one kept for display."""
        rings = [[0, 0, 2, 0, 2, 2], [10, 10, 40, 10, 40, 40]]
        split = _coco_split(
            tmp_path / "test", annotations=[_box(1, 10, 1, segmentation=rings)]
        )
        gt = open_ground_truth(split).coco_ground_truth(split)
        assert gt["annotations"][0]["segmentation"] == rings

    def test_only_images_on_disk_are_scored_against(self, tmp_path):
        split = _coco_split(tmp_path / "test", images=("a.jpg",),
                            annotations=[_box(1, 10, 1)])
        payload = json.loads((split / "_annotations.coco.json").read_text())
        payload["images"].append(
            {"id": 99, "file_name": "gone.jpg", "width": 10, "height": 10}
        )
        (split / "_annotations.coco.json").write_text(json.dumps(payload))

        gt = open_ground_truth(split).coco_ground_truth(split)
        assert [i["file_name"] for i in gt["images"]] == ["a.jpg"]

    def test_annotations_naming_no_present_image_are_refused(self, tmp_path):
        """Silence here would look like a model that finds nothing."""
        from model_doctor.app.evaluation import EvaluationError

        split = _coco_split(tmp_path / "test", images=("a.jpg",))
        payload = json.loads((split / "_annotations.coco.json").read_text())
        payload["images"] = [
            {"id": 99, "file_name": "elsewhere.jpg", "width": 10, "height": 10}
        ]
        (split / "_annotations.coco.json").write_text(json.dumps(payload))

        with pytest.raises(EvaluationError, match="none of the images"):
            open_ground_truth(split).coco_ground_truth(split)


# ---------------------------------------------------------------------------
# Class names must be the model's class names
# ---------------------------------------------------------------------------
class TestClassNameAgreement:
    def _source(self, tmp_path, categories):
        return open_ground_truth(_coco_split(tmp_path / "test", categories=categories))

    def test_matching_names_pass(self, tmp_path):
        source = self._source(tmp_path, ((1, "column"), (2, "beam")))
        verify_class_names(source, {0: "column", 1: "beam"})

    def test_a_mismatch_is_refused_with_both_names(self, tmp_path):
        source = self._source(tmp_path, ((1, "column"), (2, "beam")))
        with pytest.raises(DatasetConfigError) as error:
            verify_class_names(source, {0: "column", 1: "slab"})
        assert "beam" in str(error.value) and "slab" in str(error.value)

    def test_a_subset_is_a_warning_not_a_refusal(self, tmp_path):
        """A model with more classes than the split uses is legitimate."""
        source = self._source(tmp_path, ((1, "column"),))
        verify_class_names(source, {0: "column", 1: "beam"})

    def test_an_unknown_side_is_not_second_guessed(self, tmp_path):
        source = self._source(tmp_path, ((1, "column"),))
        verify_class_names(source, {})


# ---------------------------------------------------------------------------
# The descriptor: a generated placeholder must not outrank real categories
# ---------------------------------------------------------------------------
class TestDescriptorClassNames:
    def test_coco_categories_beat_a_placeholder_data_yaml(self, tmp_path):
        """Ramanujan generates ``names: {0: object}`` for a COCO upload."""
        root = tmp_path / "ds"
        _coco_split(root / "test", categories=((1, "column"), (2, "beam")))
        (root / "data.yaml").write_text("names:\n  0: object\ntest: test\n")

        dataset = load_dataset_config(root / "data.yaml")
        assert dataset.class_names == {0: "column", 1: "beam"}

    def test_a_yolo_descriptor_is_left_exactly_as_it_was(self, tmp_path):
        """The working path does not move."""
        root = tmp_path / "ds"
        _yolo_split(root / "test")
        (root / "data.yaml").write_text(
            "names:\n  0: column\n  1: beam\ntest: test/images\n"
        )
        dataset = load_dataset_config(root / "data.yaml")
        assert dataset.class_names == {0: "column", 1: "beam"}


# ---------------------------------------------------------------------------
# Split selection
# ---------------------------------------------------------------------------
class TestSplitSelection:
    def test_each_split_is_read_from_its_own_annotation_file(self, tmp_path):
        root = tmp_path / "ds"
        _coco_split(root / "test", categories=((1, "column"),),
                    annotations=[_box(1, 10, 1)])
        _coco_split(root / "valid", categories=((1, "column"),),
                    images=("a.jpg", "b.jpg"),
                    annotations=[_box(1, 10, 1), _box(2, 20, 1)])
        (root / "data.yaml").write_text(
            "names:\n  0: column\ntest: test\nval: valid\n"
        )
        dataset = load_dataset_config(root / "data.yaml")

        assert set(dataset.splits) == {"test", "val"}
        test = open_ground_truth(dataset.splits["test"])
        val = open_ground_truth(dataset.splits["val"])
        assert len(test.annotations_for(dataset.splits["test"] / "a.jpg", 100, 50)) == 1
        assert len(val.annotations_for(dataset.splits["val"] / "b.jpg", 100, 50)) == 1


# ---------------------------------------------------------------------------
# The YOLO path, unchanged
# ---------------------------------------------------------------------------
class TestYoloIsUntouched:
    def test_a_yolo_box_reads_as_it_always_did(self, tmp_path):
        images = _yolo_split(tmp_path / "ds")
        source = open_ground_truth(images, {0: "column"})
        assert source.format == FORMAT_YOLO
        found = source.annotations_for(images / "a.jpg", 100, 50)
        assert len(found) == 1
        assert found[0].class_name == "column"
        assert (found[0].x1, found[0].x2) == (40.0, 60.0)

    def test_a_yolo_polygon_reads_as_it_always_did(self, tmp_path):
        images = _yolo_split(tmp_path / "ds", labels="0 0.1 0.1 0.5 0.1 0.3 0.6\n")
        found = open_ground_truth(images, {0: "column"}).annotations_for(
            images / "a.jpg", 100, 50
        )
        assert found[0].polygon is not None
        assert len(found[0].polygon) == 3

    def test_a_missing_label_file_is_still_a_negative_sample(self, tmp_path):
        images = _yolo_split(tmp_path / "ds")
        (images.parent / "labels" / "a.txt").unlink()
        assert open_ground_truth(images, {0: "column"}).annotations_for(
            images / "a.jpg", 100, 50
        ) == []

    def test_the_evaluator_view_still_comes_from_the_labels(self, tmp_path):
        images = _yolo_split(tmp_path / "ds")
        gt = open_ground_truth(images, {0: "column"}).coco_ground_truth(images)
        assert len(gt["images"]) == 1
        assert len(gt["annotations"]) == 1
        assert gt["categories"][0]["name"] == "column"
