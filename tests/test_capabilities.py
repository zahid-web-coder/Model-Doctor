"""The feasibility gate: can this knob move the output at all?

The question these tests exist for is one Model Doctor previously got wrong.
An NMS experiment was proposed for a checkpoint whose head has no suppression
stage; had it been run, "we varied it and nothing changed" would have read as a
finding about the model rather than about the plumbing. So the acceptance case
here is that exact question, asked of the real checkpoint, and the required
answer is a refusal with a reason.

Two disciplines are pinned throughout:

* **``unknown`` is never a synonym for ``inert``.** An observed absence of
  effect, with no architectural reason to expect one, stays ``unknown``. Three
  agreeing images are not a proof of inertness, and a reader must not be able
  to spend one as if it were.
* **The gate writes nothing.** Not to the database, not to the weights, and in
  particular never back-fills a ``NULL`` provenance column from a re-probe: the
  probe describes the checkpoint as it is now, while the column means what a
  run passed, and conflating them would manufacture history.

Checkpoint-dependent tests are skipped, never failed, when the weights are
absent — and they verify the file's hash before trusting it, because a probe of
a different file describes a different model.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app import capabilities, storage
from app.capabilities import (
    ACTUATES,
    BOTH,
    EMPIRICAL,
    INERT,
    KNOBS,
    MAX_PROBE_IMAGES,
    NO_BASIS,
    STRUCTURAL,
    UNKNOWN,
    CapabilityError,
    aggregate,
    experiment_feasibility,
    inspect_checkpoint,
    knob_is_forwarded,
    knob_is_supported,
    select_probe_images,
    verdict_for_run,
)
from app.storage import RunContext

# The reference checkpoint. Its identity is asserted, not assumed: these tests
# make claims about a specific architecture, and a same-named file with
# different weights would make those claims false while still passing.
YOLO26 = Path(
    "/Users/user_/model-doctor-workspace/mr_QSJWOFvJhM38C658zFw/model/best__3_.pt"
)
YOLO26_SHA_PREFIX = "546ee2af"


def _checkpoint_available() -> bool:
    if not YOLO26.is_file():
        return False
    return storage.file_sha256(YOLO26).startswith(YOLO26_SHA_PREFIX)


needs_checkpoint = pytest.mark.skipif(
    not _checkpoint_available(),
    reason="The reference YOLO26 checkpoint is not present on this machine.",
)


def _context(**overrides) -> RunContext:
    base = {
        "model_path": "missing.pt",
        "model_sha256": "a" * 64,
        "dataset_yaml": "d.yaml",
        "split": "test",
        "confidence_threshold": 0.25,
        "match_iou_threshold": 0.5,
        "localization_iou_floor": 0.1,
        "image_size": 448,
    }
    base.update(overrides)
    return RunContext(**base)


@pytest.fixture()
def database(tmp_path: Path) -> Path:
    """A database holding one run with three images of varying density."""
    path = tmp_path / "capabilities.db"
    with storage.connect(path) as connection:
        run_id = storage.save_run(connection, _context())
        for count, name in [(1, "one.jpg"), (3, "three.jpg"), (2, "two.jpg")]:
            connection.execute(
                "INSERT INTO images (run_id, path, filename, width, height, "
                "prediction_count, truth_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, f"/nowhere/{name}", name, 100, 100, count, 1),
            )
        connection.commit()
    return path


# ---------------------------------------------------------------------------
# 1-4. The real YOLO26 acceptance case
# ---------------------------------------------------------------------------
@needs_checkpoint
def test_nms_iou_is_inert_on_the_reference_checkpoint(tmp_path: Path) -> None:
    """The question that was got wrong: it must now be refused, with a reason."""
    path = tmp_path / "real.db"
    with storage.connect(path) as connection:
        run_id = storage.save_run(
            connection,
            _context(
                model_path=str(YOLO26), model_sha256=storage.file_sha256(YOLO26)
            ),
        )
        run = storage.load_run(connection, run_id)
        verdict = verdict_for_run(connection, run, "nms_iou", probe=False)

    assert verdict.actuation == INERT
    assert verdict.basis == STRUCTURAL
    assert verdict.supported is True
    assert verdict.forwarded is True
    # Phrased narrowly on purpose: where the knob does not reach, and nothing
    # about post-processing in general.
    assert (
        "NMS-threshold actuation is not available on this checkpoint's "
        "inference path" in verdict.reason
    )
    assert "end2end=True" in verdict.reason


@needs_checkpoint
def test_nms_iou_probe_finds_identical_output_across_thresholds() -> None:
    """The empirical half, independent of any database."""
    checkpoint = inspect_checkpoint(YOLO26)
    images = sorted(
        Path(
            "/Users/user_/model-doctor-workspace/mr_QSJWOFvJhM38C658zFw/dataset/"
            "unpacked/staircase_merged_1280_yolo/test/images"
        ).glob("*.jpg")
    )
    if not images:
        pytest.skip("The reference dataset split is not present.")
    changed, detail = capabilities.probe_actuation(
        checkpoint, "nms_iou", [str(images[0])], confidence=0.25
    )
    assert changed is False
    assert "Identical output" in detail


@needs_checkpoint
def test_image_size_actuates_on_the_reference_checkpoint() -> None:
    """The positive control. If this fails, no other verdict is worth reading."""
    checkpoint = inspect_checkpoint(YOLO26)
    images = sorted(
        Path(
            "/Users/user_/model-doctor-workspace/mr_QSJWOFvJhM38C658zFw/dataset/"
            "unpacked/staircase_merged_1280_yolo/test/images"
        ).glob("*.jpg")
    )
    if not images:
        pytest.skip("The reference dataset split is not present.")
    changed, _ = capabilities.probe_actuation(
        checkpoint, "image_size", [str(images[0])], confidence=0.25
    )
    assert changed is True


@needs_checkpoint
def test_checkpoint_reports_its_native_training_resolution() -> None:
    """448 is what it was trained at, which is what makes 640 off-nominal."""
    checkpoint = inspect_checkpoint(YOLO26)
    assert checkpoint.exists is True
    assert checkpoint.family == "yolo"
    assert checkpoint.end2end is True
    assert checkpoint.native_image_size == 448
    assert checkpoint.trained_with_nms is False


# ---------------------------------------------------------------------------
# 5. supported is not forwarded
# ---------------------------------------------------------------------------
def test_rfdetr_supports_iou_but_never_forwards_it() -> None:
    """Same verdict, different reason — and the difference must survive.

    ``RFDetrDetector`` inherits ``iou`` from its base, so it accepts one; its
    predict call passes ``threshold`` alone, so the value is dropped. Asserted
    against the adapter rather than a live model, so this runs without the
    optional dependency installed.
    """
    assert knob_is_supported("rfdetr", "nms_iou") is True
    assert knob_is_forwarded("rfdetr", "nms_iou") is False
    assert knob_is_forwarded("yolo", "nms_iou") is True
    assert knob_is_forwarded("rfdetr", "image_size") is True


# ---------------------------------------------------------------------------
# 6-7. Vocabulary discipline
# ---------------------------------------------------------------------------
def test_no_observed_effect_without_structural_basis_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The central discipline: absence of an effect is not proof of inertness."""
    weights = tmp_path / "plain.pt"
    weights.write_bytes(b"not really a checkpoint")
    path = tmp_path / "u.db"
    with storage.connect(path) as connection:
        run_id = storage.save_run(
            connection,
            _context(
                model_path=str(weights), model_sha256=storage.file_sha256(weights)
            ),
        )
        connection.execute(
            "INSERT INTO images (run_id, path, filename, width, height, "
            "prediction_count, truth_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, str(weights), "x.jpg", 10, 10, 2, 1),
        )
        connection.commit()
        # A model with no end2end head, whose output happens not to move.
        monkeypatch.setattr(
            capabilities,
            "inspect_checkpoint",
            lambda _: capabilities.Checkpoint(
                path=weights, exists=True, family="yolo", head_class="Detect",
                end2end=False, native_image_size=640,
            ),
        )
        def unmoved(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            return False, "no change"

        monkeypatch.setattr(capabilities, "probe_actuation", unmoved)
        run = storage.load_run(connection, run_id)
        verdict = verdict_for_run(connection, run, "nms_iou")

    assert verdict.actuation == UNKNOWN, "no-effect without a reason is not inert"
    assert verdict.basis == EMPIRICAL
    assert "not evidence that the knob is inert" in verdict.reason


def test_absent_checkpoint_reports_unknown_rather_than_raising(
    database: Path,
) -> None:
    """'What is knowable here?' has 'nothing' as a valid answer."""
    with storage.connect(database) as connection:
        run = storage.load_run(connection, 1)
        verdict = verdict_for_run(connection, run, "nms_iou")
    assert verdict.actuation == UNKNOWN
    assert verdict.basis == NO_BASIS
    assert verdict.supported is None and verdict.forwarded is None
    assert "not at" in verdict.reason


# ---------------------------------------------------------------------------
# 8-9. Provenance
# ---------------------------------------------------------------------------
def test_null_provenance_stays_null_and_a_probe_never_backfills_it(
    database: Path,
) -> None:
    """The probe describes the checkpoint; the column means what a run passed."""
    with storage.connect(database) as connection:
        before = dict(connection.execute("SELECT * FROM runs WHERE id=1").fetchone())
        assert before["nms_iou_threshold"] is None

        run = storage.load_run(connection, 1)
        assert run.nms_iou_threshold is None
        verdict = verdict_for_run(connection, run, "nms_iou")
        assert verdict.recorded_value is None

        after = dict(connection.execute("SELECT * FROM runs WHERE id=1").fetchone())
    assert after == before, "a feasibility probe must never write provenance"


def test_a_moved_or_changed_checkpoint_is_not_probed(tmp_path: Path) -> None:
    """A probe of a different file would describe a different model."""
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"contents")
    path = tmp_path / "m.db"
    with storage.connect(path) as connection:
        run_id = storage.save_run(
            connection,
            _context(model_path=str(weights), model_sha256="b" * 64),
        )
        run = storage.load_run(connection, run_id)
        verdict = verdict_for_run(connection, run, "nms_iou")
    assert verdict.actuation == UNKNOWN
    assert "not the checkpoint this run used" in verdict.reason


# ---------------------------------------------------------------------------
# 10. Aggregation
# ---------------------------------------------------------------------------
def test_inert_dominates_unknown_in_the_aggregate() -> None:
    """One dead arm makes a one-knob design impossible across the set."""

    def verdict(actuation: str) -> capabilities.KnobVerdict:
        return capabilities.KnobVerdict(
            run_id=1, knob="nms_iou", supported=True, forwarded=True,
            actuation=actuation, basis=NO_BASIS, reason="",
        )

    assert aggregate([verdict(ACTUATES), verdict(ACTUATES)]) == ACTUATES
    assert aggregate([verdict(ACTUATES), verdict(UNKNOWN)]) == UNKNOWN
    assert aggregate([verdict(ACTUATES), verdict(INERT)]) == INERT
    assert aggregate([verdict(UNKNOWN), verdict(INERT)]) == INERT
    assert aggregate([]) == UNKNOWN


# ---------------------------------------------------------------------------
# 11-12. Invariants
# ---------------------------------------------------------------------------
def test_feasibility_leaves_the_database_byte_identical(
    database: Path, tmp_path: Path
) -> None:
    """The gate reads. It does not write, anywhere, ever."""
    reference = tmp_path / "before.db"
    shutil.copy(database, reference)
    with storage.connect(database) as connection:
        experiment_feasibility(connection, "nms_iou", [1])
    assert database.read_bytes() == reference.read_bytes()


def test_two_identical_probes_return_identical_verdicts(database: Path) -> None:
    """Determinism, so a verdict can be quoted rather than re-derived."""
    with storage.connect(database) as connection:
        first = experiment_feasibility(connection, "nms_iou", [1])
        second = experiment_feasibility(connection, "nms_iou", [1])
    assert first == second


# ---------------------------------------------------------------------------
# Supporting behaviour
# ---------------------------------------------------------------------------
def test_probe_images_prefer_two_predictions_and_respect_the_cap(
    database: Path,
) -> None:
    """A suppression knob cannot be seen on an image with one prediction."""
    with storage.connect(database) as connection:
        rows = select_probe_images(connection, 1, limit=99)
        assert [r["prediction_count"] for r in rows] == [3, 2]
        assert len(rows) <= MAX_PROBE_IMAGES
        assert len(select_probe_images(connection, 1, limit=1)) == 1


def test_unknown_knob_and_unknown_run_are_refused(database: Path) -> None:
    """Both are caller errors, and neither should read as a verdict."""
    with storage.connect(database) as connection:
        with pytest.raises(CapabilityError, match="Unknown knob"):
            experiment_feasibility(connection, "temperature", [1])
        with pytest.raises(CapabilityError, match="No such run"):
            experiment_feasibility(connection, "nms_iou", [999])
        with pytest.raises(CapabilityError, match="At least one run"):
            experiment_feasibility(connection, "nms_iou", [])


def test_only_two_knobs_are_gated_in_this_version() -> None:
    """Scope is part of the contract; widening it is a decision, not a drift."""
    assert KNOBS == ("nms_iou", "image_size")


def test_schema_15_adds_only_the_three_unrecoverable_columns(
    tmp_path: Path,
) -> None:
    """Everything a checkpoint can answer about itself stays out of the schema."""
    path = tmp_path / "s.db"
    with storage.connect(path) as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert storage.SCHEMA_VERSION == 15
    assert {
        "nms_iou_threshold",
        "inference_max_detections",
        "inference_library_version",
    } <= columns
    for derivable in ("detector_family", "head_class", "end2end", "native_image_size"):
        assert derivable not in columns, (
            f"{derivable} is recoverable from the checkpoint; storing it "
            "creates a second version of a fact that can disagree"
        )


def test_new_runs_record_provenance_and_old_ones_keep_null(tmp_path: Path) -> None:
    """The migration adds columns; it does not invent values for existing rows."""
    path = tmp_path / "p.db"
    with storage.connect(path) as connection:
        old = storage.save_run(connection, _context())
        new = storage.save_run(
            connection,
            _context(
                nms_iou_threshold=0.45,
                inference_max_detections=300,
                inference_library_version="ultralytics 8.4.115",
            ),
        )
        assert storage.load_run(connection, old).nms_iou_threshold is None
        recorded = storage.load_run(connection, new)
        assert recorded.nms_iou_threshold == 0.45
        assert recorded.inference_max_detections == 300
        assert recorded.inference_library_version == "ultralytics 8.4.115"


def test_fingerprint_is_unchanged_by_the_new_provenance(tmp_path: Path) -> None:
    """Adding an inert knob to the fingerprint would manufacture independence."""
    from app import comparison

    path = tmp_path / "f.db"
    with storage.connect(path) as connection:
        a = storage.save_run(connection, _context())
        b = storage.save_run(connection, _context(nms_iou_threshold=0.9))
        first = storage.load_run(connection, a)
        second = storage.load_run(connection, b)
    assert comparison.run_fingerprint(first) == comparison.run_fingerprint(second)


def test_heterogeneous_checkpoints_are_reported_not_judged(tmp_path: Path) -> None:
    """Whether that invalidates a comparison is app.comparison's existing job."""
    path = tmp_path / "h.db"
    with storage.connect(path) as connection:
        storage.save_run(connection, _context(model_sha256="a" * 64))
        storage.save_run(connection, _context(model_sha256="c" * 64))
        report = experiment_feasibility(connection, "nms_iou", [1, 2])
    assert report["distinct_checkpoints"] == 2
    assert report["single_experiment"] is False
    assert any("distinct checkpoints" in c for c in report["caveats"])


def test_report_shape_is_stable(database: Path) -> None:
    """The MCP contract, pinned."""
    with storage.connect(database) as connection:
        report = experiment_feasibility(connection, "nms_iou", [1])
    assert set(report) == {
        "knob",
        "aggregate",
        "single_experiment",
        "blocking",
        "distinct_checkpoints",
        "distinct_families",
        "runs",
        "caveats",
    }
    assert set(report["runs"][0]) == {
        "run_id",
        "knob",
        "supported",
        "forwarded",
        "actuation",
        "basis",
        "reason",
        "recorded_value",
        "checkpoint_native_image_size",
        "probe_images",
    }
    assert report["aggregate"] in (ACTUATES, INERT, UNKNOWN)


def test_mcp_exposes_the_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Present on the read-only surface, and annotated as read-only."""
    from app import mcp_server

    path = tmp_path / "mcp.db"
    with storage.connect(path) as connection:
        storage.save_run(connection, _context())
    monkeypatch.setattr("config.DB_PATH", path)
    server = mcp_server.create_server()
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert "experiment_feasibility" in names


def test_basis_constants_are_distinct() -> None:
    """`structural+empirical` must not collide with either half."""
    assert len({STRUCTURAL, EMPIRICAL, BOTH, NO_BASIS}) == 4
    assert BOTH not in (STRUCTURAL, EMPIRICAL)


def test_probe_can_be_skipped_entirely(database: Path) -> None:
    """Structural-only answers must load no images at all."""
    with storage.connect(database) as connection:
        run = storage.load_run(connection, 1)
        verdict = verdict_for_run(connection, run, "nms_iou", probe=False)
    assert verdict.probe_images == []


def test_sqlite_row_factory_is_untouched(database: Path) -> None:
    """The gate borrows the caller's connection and must not reconfigure it."""
    with storage.connect(database) as connection:
        factory = connection.row_factory
        experiment_feasibility(connection, "nms_iou", [1])
        assert connection.row_factory is factory
        assert isinstance(
            connection.execute("SELECT 1 AS x").fetchone(), sqlite3.Row
        )
