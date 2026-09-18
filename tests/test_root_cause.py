"""Tests for root-cause attribution.

Every factor is tested twice: once on input that should trigger it, once on
input that should not. A detector that fires on everything explains nothing,
so the negative cases matter as much as the positive ones.

Images are synthesised rather than loaded, so the suite stays fast and needs no
dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor import config
from model_doctor.app import storage
from model_doctor.app.diagnosis import DatasetDiagnosis, diagnose_image
from model_doctor.app.root_cause import (
    BLUR,
    CLASS_IMBALANCE,
    CROWDING,
    EDGE_TRUNCATION,
    LOW_LIGHT,
    RECURRING_MISCLASSIFICATION,
    SMALL_OBJECT,
    THIN_STRUCTURE,
    BlurFactor,
    ClassImbalanceFactor,
    CrowdingFactor,
    EdgeTruncationFactor,
    FindingContext,
    LowLightFactor,
    RecurringMisclassificationFactor,
    RootCauseError,
    SmallObjectFactor,
    ThinStructureFactor,
    analyse_run,
    build_contexts,
    calibrate_finding_factors,
    measure_factor_rates,
)
from model_doctor.utils.annotations import ObjectAnnotation


def _context(**overrides) -> FindingContext:
    """Build a finding context with sensible defaults."""
    defaults = {
        "finding_id": 1,
        "outcome": "false_negative",
        "class_name": "door",
        "box": (100.0, 100.0, 300.0, 300.0),
        "image_width": 1000,
        "image_height": 1000,
        "region": None,
        "neighbours": (),
    }
    defaults.update(overrides)
    return FindingContext(**defaults)


def _sharp(size: int = 64) -> np.ndarray:
    """A high-contrast checkerboard: maximal high-frequency detail."""
    grid = np.indices((size, size)).sum(axis=0) % 2
    return (grid * 255).astype(np.uint8)


def _flat(value: int = 128, size: int = 64) -> np.ndarray:
    """A uniform region: no detail at all."""
    return np.full((size, size), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------
def test_blur_fires_on_a_flat_region() -> None:
    """A region with no high-frequency content is blurred by definition."""
    evidence = BlurFactor().detect(_context(region=_flat()))
    assert evidence is not None
    assert evidence.factor == BLUR
    assert 0.0 <= evidence.score <= 1.0
    assert "Laplacian" in evidence.evidence


def test_blur_stays_silent_on_a_sharp_region() -> None:
    """A detector that fires on sharp images would explain nothing."""
    assert BlurFactor().detect(_context(region=_sharp())) is None


def test_blur_needs_pixels_and_says_so() -> None:
    """The runner uses this flag to skip detectors when images are unavailable."""
    assert BlurFactor().needs_pixels is True
    assert BlurFactor().detect(_context(region=None)) is None


# ---------------------------------------------------------------------------
# Low light
# ---------------------------------------------------------------------------
def test_low_light_fires_on_a_dark_region() -> None:
    """Mean luminance below the threshold is reported with its measurement."""
    evidence = LowLightFactor().detect(_context(region=_flat(20)))
    assert evidence is not None
    assert evidence.factor == LOW_LIGHT
    assert "20.0/255" in evidence.evidence


def test_low_light_stays_silent_on_a_bright_region() -> None:
    """A well-exposed region is not flagged."""
    assert LowLightFactor().detect(_context(region=_flat(200))) is None


def test_low_light_score_rises_as_the_region_darkens() -> None:
    """Severity is comparable within the factor, so the worst can be ranked."""
    darker = LowLightFactor().detect(_context(region=_flat(5)))
    lighter = LowLightFactor().detect(_context(region=_flat(50)))
    assert darker is not None and lighter is not None
    assert darker.score > lighter.score


# ---------------------------------------------------------------------------
# Small object
# ---------------------------------------------------------------------------
def test_small_object_fires_on_a_tiny_box() -> None:
    """A box covering a fraction of a percent of the frame is small."""
    evidence = SmallObjectFactor().detect(
        _context(box=(0.0, 0.0, 20.0, 20.0))  # 400px of 1,000,000
    )
    assert evidence is not None
    assert evidence.factor == SMALL_OBJECT


def test_small_object_stays_silent_on_a_large_box() -> None:
    """Most objects in a close-up dataset are large; those are not the problem."""
    assert SmallObjectFactor().detect(_context(box=(0.0, 0.0, 500.0, 500.0))) is None


def test_small_object_is_measured_relative_to_the_image() -> None:
    """The same pixel box is small in a large frame and not in a small one."""
    box = (0.0, 0.0, 30.0, 30.0)
    in_large_frame = SmallObjectFactor().detect(
        _context(box=box, image_width=4000, image_height=4000)
    )
    in_small_frame = SmallObjectFactor().detect(
        _context(box=box, image_width=100, image_height=100)
    )
    assert in_large_frame is not None
    assert in_small_frame is None


def test_small_object_needs_image_dimensions() -> None:
    """An image that failed to process has no size, so no fraction exists."""
    assert (
        SmallObjectFactor().detect(_context(image_width=None, image_height=None))
        is None
    )


# ---------------------------------------------------------------------------
# Edge truncation
# ---------------------------------------------------------------------------
def test_edge_truncation_fires_and_names_the_sides() -> None:
    """Knowing which edges are touched is what makes the finding actionable."""
    evidence = EdgeTruncationFactor().detect(
        _context(box=(0.0, 0.0, 500.0, 500.0), image_width=1000, image_height=1000)
    )
    assert evidence is not None
    assert evidence.factor == EDGE_TRUNCATION
    assert "left" in evidence.evidence and "top" in evidence.evidence


def test_edge_truncation_stays_silent_in_the_middle() -> None:
    """A box well inside the frame is not truncated."""
    assert (
        EdgeTruncationFactor().detect(_context(box=(400.0, 400.0, 600.0, 600.0)))
        is None
    )


def test_edge_truncation_severity_rises_with_more_sides() -> None:
    """An object running off all four edges is more truncated than one corner."""
    one = EdgeTruncationFactor().detect(_context(box=(0.0, 400.0, 500.0, 600.0)))
    all_four = EdgeTruncationFactor().detect(_context(box=(0.0, 0.0, 1000.0, 1000.0)))
    assert one is not None and all_four is not None
    assert all_four.score > one.score


# ---------------------------------------------------------------------------
# Crowding
# ---------------------------------------------------------------------------
def test_crowding_fires_on_heavy_overlap() -> None:
    """Neighbouring annotations that overlap heavily are reported."""
    evidence = CrowdingFactor().detect(
        _context(
            box=(100.0, 100.0, 300.0, 300.0),
            neighbours=[(110.0, 110.0, 310.0, 310.0)],
        )
    )
    assert evidence is not None
    assert evidence.factor == CROWDING
    assert "IoU" in evidence.evidence


def test_crowding_stays_silent_on_distant_neighbours() -> None:
    """An object with space around it is not crowded."""
    assert (
        CrowdingFactor().detect(
            _context(neighbours=[(900.0, 900.0, 950.0, 950.0)])
        )
        is None
    )


def test_crowding_needs_neighbours() -> None:
    """A lone object cannot be crowded."""
    assert CrowdingFactor().detect(_context(neighbours=())) is None


# ---------------------------------------------------------------------------
# Run-level factors
# ---------------------------------------------------------------------------
def test_class_imbalance_fires_on_a_rare_class() -> None:
    """Findings of an under-represented class carry the factor."""
    contexts = [_context(finding_id=i, class_name="door") for i in range(20)]
    contexts.append(_context(finding_id=99, class_name="handle"))

    results = ClassImbalanceFactor().detect(contexts)

    assert 99 in results
    assert results[99].factor == CLASS_IMBALANCE
    assert all(c.finding_id not in results for c in contexts if c.class_name == "door")


def test_class_imbalance_stays_silent_when_balanced() -> None:
    """An even split is not an imbalance."""
    contexts = [
        _context(finding_id=i, class_name="door" if i % 2 else "door_frame")
        for i in range(20)
    ]
    assert ClassImbalanceFactor().detect(contexts) == {}


def test_class_imbalance_needs_more_than_one_class() -> None:
    """A single-class dataset cannot be imbalanced."""
    contexts = [_context(finding_id=i, class_name="door") for i in range(10)]
    assert ClassImbalanceFactor().detect(contexts) == {}


def test_recurring_misclassification_fires_above_the_threshold() -> None:
    """A class named wrongly several times is a pattern, not an accident."""
    contexts = [
        _context(finding_id=i, outcome="wrong_class", class_name="door_frame")
        for i in range(4)
    ]
    results = RecurringMisclassificationFactor().detect(contexts)

    assert len(results) == 4
    assert all(e.factor == RECURRING_MISCLASSIFICATION for e in results.values())


def test_recurring_misclassification_ignores_isolated_cases() -> None:
    """One or two mistakes are not yet a recurring pattern."""
    contexts = [
        _context(finding_id=i, outcome="wrong_class", class_name="door")
        for i in range(2)
    ]
    assert RecurringMisclassificationFactor().detect(contexts) == {}


def test_recurring_misclassification_ignores_other_outcomes() -> None:
    """A missed object was not misidentified."""
    contexts = [
        _context(finding_id=i, outcome="false_negative", class_name="door")
        for i in range(10)
    ]
    assert RecurringMisclassificationFactor().detect(contexts) == {}


# ---------------------------------------------------------------------------
# Context building and the full pass
# ---------------------------------------------------------------------------
def _seed(tmp_path: Path, dark: bool = False) -> tuple[Path, int]:
    """Create a database with a saved run over real image files."""
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    shade = 10 if dark else 200
    diagnoses = []
    for index in range(2):
        path = image_dir / f"frame_{index}.jpg"
        Image.new("RGB", (400, 400), (shade, shade, shade)).save(path)
        diagnoses.append(
            diagnose_image(
                path,
                [
                    ObjectAnnotation.from_box(
                        0, "door", 350, 350, 395, 395, confidence=0.8
                    )
                ],
                [
                    ObjectAnnotation.from_box(0, "door", 0, 0, 200, 200),
                    ObjectAnnotation.from_box(1, "door_frame", 10, 10, 210, 210),
                ],
                image_width=400,
                image_height=400,
            )
        )

    db = tmp_path / "d.db"
    context = storage.RunContext(
        model_path="m.pt", model_sha256="a" * 64, dataset_yaml="d.yaml",
        split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
        localization_iou_floor=0.1, image_size=640,
    )
    with storage.connect(db) as conn:
        run_id = storage.save_dataset_diagnosis(
            conn, context, DatasetDiagnosis(diagnoses=diagnoses)
        )
    return db, run_id


def test_contexts_group_by_image_and_carry_neighbours(tmp_path: Path) -> None:
    """Crowding is only measurable if a finding knows what else is nearby."""
    db, run_id = _seed(tmp_path)
    with storage.connect(db) as conn:
        rows = storage.load_findings_for_embedding(conn, run_id, failures_only=True)
    contexts, unreadable = build_contexts(rows)

    assert unreadable == 0
    assert contexts
    assert any(context.neighbours for context in contexts)
    assert any(context.region is not None for context in contexts)


def test_analysis_attributes_factors_and_stores_them(tmp_path: Path) -> None:
    """A saved run round-trips into stored attributions."""
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        report = analyse_run(conn, run_id)
        stored = storage.load_root_causes(conn, run_id)

    assert report.attributed > 0
    assert len(stored) == report.attributed
    assert {row.factor for row in stored} & {LOW_LIGHT, BLUR, EDGE_TRUNCATION}
    assert all(0.0 <= row.score <= 1.0 for row in stored)
    assert all(row.evidence for row in stored)


def test_analysis_is_repeatable_without_duplicating(tmp_path: Path) -> None:
    """Attribution is derived data: recomputing corrects, never accumulates."""
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
        first = len(storage.load_root_causes(conn, run_id))
    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
        second = len(storage.load_root_causes(conn, run_id))
    assert first == second


def test_analysis_without_pixels_still_uses_geometry(tmp_path: Path) -> None:
    """Skipping image reads disables only the factors that need them."""
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        analyse_run(conn, run_id, load_pixels=False)
        factors = {row.factor for row in storage.load_root_causes(conn, run_id)}

    assert BLUR not in factors and LOW_LIGHT not in factors
    assert EDGE_TRUNCATION in factors or CROWDING in factors


def test_analysis_survives_an_unreadable_image(tmp_path: Path) -> None:
    """A missing file disables its pixel factors, not the whole pass."""
    db, run_id = _seed(tmp_path, dark=True)
    next((tmp_path / "images").glob("*.jpg")).unlink()

    with storage.connect(db) as conn:
        report = analyse_run(conn, run_id)

    assert report.images_unreadable > 0
    assert report.attributed > 0


def test_analysis_rejects_a_run_with_no_failures(tmp_path: Path) -> None:
    """An empty run says what to do rather than reporting nothing found."""
    db = tmp_path / "empty.db"
    with storage.connect(db) as conn:
        run_id = storage.save_run(
            conn,
            storage.RunContext(
                model_path="m.pt", model_sha256="a" * 64, dataset_yaml="d.yaml",
                split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
                localization_iou_floor=0.1, image_size=640,
            ),
        )
        with pytest.raises(RootCauseError, match="no failures"):
            analyse_run(conn, run_id)


def test_factors_are_injectable(tmp_path: Path) -> None:
    """The detector sets are parameters, so a caller can narrow the analysis."""
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        analyse_run(
            conn, run_id, finding_factors=[LowLightFactor()], run_factors=[]
        )
        factors = {row.factor for row in storage.load_root_causes(conn, run_id)}
    assert factors == {LOW_LIGHT}


def test_root_causes_cascade_on_run_delete(tmp_path: Path) -> None:
    """Deleting a run leaves no orphaned attributions."""
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
    with storage.connect(db) as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    with storage.connect(db) as conn:
        remaining = conn.execute("SELECT COUNT(*) AS n FROM root_causes").fetchone()
    assert remaining["n"] == 0


def test_attribution_is_ready_for_cluster_summaries(tmp_path: Path) -> None:
    """Factors attach to findings, so grouping later is a join and a GROUP BY.

    This is the property that makes per-cluster summaries free when the
    clustering milestone lands: no schema change and no second pipeline.
    """
    db, run_id = _seed(tmp_path, dark=True)
    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
        # Stand in for the future clusters table to prove the join works.
        conn.execute(
            "CREATE TEMP TABLE fake_clusters AS "
            "SELECT id AS finding_id, (id % 2) AS cluster_id FROM findings "
            "WHERE run_id = ?",
            (run_id,),
        )
        summary = conn.execute(
            """
            SELECT c.cluster_id, rc.factor, COUNT(*) AS n
            FROM root_causes rc JOIN fake_clusters c ON c.finding_id = rc.finding_id
            WHERE rc.run_id = ?
            GROUP BY c.cluster_id, rc.factor
            """,
            (run_id,),
        ).fetchall()

    assert summary
    assert {"cluster_id", "factor", "n"} <= set(summary[0].keys())


# ---------------------------------------------------------------------------
# Neighbours include correct detections (D-031)
# ---------------------------------------------------------------------------
def _seed_one_correct_one_missed(tmp_path: Path) -> tuple[Path, int]:
    """Create a run where a missed object overlaps a correctly-detected one.

    The two boxes overlap at IoU 0.82, far above the crowding threshold. The
    only annotation near the miss is the *correct* detection, so this is the
    case that silently produced no crowding evidence when contexts were built
    from failures alone.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    path = image_dir / "frame.jpg"
    Image.new("RGB", (400, 400), (200, 200, 200)).save(path)

    diagnosis = diagnose_image(
        path,
        [ObjectAnnotation.from_box(0, "door", 0, 0, 200, 200, confidence=0.9)],
        [
            ObjectAnnotation.from_box(0, "door", 0, 0, 200, 200),
            ObjectAnnotation.from_box(1, "door_frame", 10, 10, 210, 210),
        ],
        image_width=400,
        image_height=400,
    )

    db = tmp_path / "neighbours.db"
    context = storage.RunContext(
        model_path="m.pt", model_sha256="a" * 64, dataset_yaml="d.yaml",
        split="test", confidence_threshold=0.25, match_iou_threshold=0.5,
        localization_iou_floor=0.1, image_size=640,
    )
    with storage.connect(db) as conn:
        run_id = storage.save_dataset_diagnosis(
            conn, context, DatasetDiagnosis(diagnoses=[diagnosis])
        )
    return db, run_id


def test_crowding_sees_correct_detections_as_neighbours(tmp_path: Path) -> None:
    """A failure beside a correct detection is crowded, and must be recorded.

    Crowding is a property of what is physically nearby. Whether the neighbour
    happened to be detected correctly is irrelevant to whether this object was
    occluded, so excluding correct findings under-reported it.
    """
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
        stored = storage.load_root_causes(conn, run_id)

    crowding = [row for row in stored if row.factor == CROWDING]
    assert crowding, "the missed object overlaps a correct detection at IoU 0.82"
    assert all(row.outcome != "correct" for row in stored)


def test_correct_findings_never_receive_root_cause_rows(tmp_path: Path) -> None:
    """`root_causes` is documented as covering failures. That must stay true."""
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        analyse_run(conn, run_id)
        measure_factor_rates(conn, run_id, load_pixels=False)
        outcomes = {row.outcome for row in storage.load_root_causes(conn, run_id)}

    assert "correct" not in outcomes


# ---------------------------------------------------------------------------
# Base rates
# ---------------------------------------------------------------------------
def test_base_rates_measure_both_groups_and_store_lift(tmp_path: Path) -> None:
    """A factor's count is only interpretable against its control rate."""
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        report = measure_factor_rates(conn, run_id, load_pixels=False)
        stored = storage.load_factor_rates(conn, run_id)

    assert report.failure_total >= 1
    assert report.correct_total >= 1
    assert stored, "geometric factors run without pixels"
    for row in stored:
        assert row.failure_total > 0
        assert row.correct_total > 0
        assert 0.0 <= row.p_value <= 1.0
        assert row.failure_count <= row.failure_total
        assert row.correct_count <= row.correct_total


def test_base_rates_are_replaced_not_accumulated(tmp_path: Path) -> None:
    """Recomputing corrects a measurement; it does not add a second one."""
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        measure_factor_rates(conn, run_id, load_pixels=False)
        first = storage.load_factor_rates(conn, run_id)
        measure_factor_rates(conn, run_id, load_pixels=False)
        second = storage.load_factor_rates(conn, run_id)

    assert len(first) == len(second)
    assert [row.factor for row in first] == [row.factor for row in second]


def test_base_rates_skip_run_level_factors(tmp_path: Path) -> None:
    """Run factors are defined in terms of mistakes and have no control rate.

    Giving `recurring_misclassification` a base rate would invent a comparison
    rather than report one, since a correct finding cannot be a repeated
    misidentification.
    """
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        measure_factor_rates(conn, run_id, load_pixels=False)
        factors = {row.factor for row in storage.load_factor_rates(conn, run_id)}

    assert RECURRING_MISCLASSIFICATION not in factors
    assert CLASS_IMBALANCE not in factors


def test_base_rates_need_both_groups(tmp_path: Path) -> None:
    """With no correct findings there is no control, and nothing is stored."""
    db, run_id = _seed(tmp_path)

    with storage.connect(db) as conn:
        conn.execute(
            "DELETE FROM findings WHERE run_id = ? AND outcome = 'correct'", (run_id,)
        )
        report = measure_factor_rates(conn, run_id, load_pixels=False)

        assert report.correct_total == 0
        assert storage.load_factor_rates(conn, run_id) == []


def test_factor_rates_cascade_on_run_delete(tmp_path: Path) -> None:
    """Derived data must not outlive the run it describes."""
    db, run_id = _seed_one_correct_one_missed(tmp_path)

    with storage.connect(db) as conn:
        measure_factor_rates(conn, run_id, load_pixels=False)
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

        assert storage.load_factor_rates(conn, run_id) == []


# ---------------------------------------------------------------------------
# Thin structures (D-032)
# ---------------------------------------------------------------------------
def test_thin_structure_fires_on_a_long_narrow_box() -> None:
    """A 10:1 box is emphatically thin and must be flagged."""
    factor = ThinStructureFactor(ratio=4.0)

    evidence = factor.detect(_context(box=(0.0, 0.0, 20.0, 200.0)))

    assert evidence is not None
    assert evidence.factor == THIN_STRUCTURE
    assert "10.0:1" in evidence.evidence
    assert "tall" in evidence.evidence


def test_thin_structure_ignores_a_square_box() -> None:
    """A detector that fires on everything explains nothing."""
    factor = ThinStructureFactor(ratio=4.0)

    assert factor.detect(_context(box=(0.0, 0.0, 100.0, 100.0))) is None


def test_thin_structure_is_orientation_agnostic() -> None:
    """A wide sliver is as thin as a tall one; only the wording differs."""
    factor = ThinStructureFactor(ratio=4.0)

    tall = factor.detect(_context(box=(0.0, 0.0, 20.0, 200.0)))
    wide = factor.detect(_context(box=(0.0, 0.0, 200.0, 20.0)))

    assert tall is not None and wide is not None
    assert tall.score == wide.score
    assert "wide" in wide.evidence


def test_thin_structure_ignores_degenerate_boxes() -> None:
    """A zero-width box would divide by zero, not be infinitely thin."""
    factor = ThinStructureFactor(ratio=4.0)

    assert factor.detect(_context(box=(10.0, 10.0, 10.0, 200.0))) is None
    assert factor.detect(_context(box=None)) is None


def test_thin_structure_score_rises_with_the_ratio() -> None:
    """Severity must rank instances within the factor."""
    factor = ThinStructureFactor(ratio=4.0)

    mild = factor.detect(_context(box=(0.0, 0.0, 10.0, 45.0)))
    severe = factor.detect(_context(box=(0.0, 0.0, 10.0, 200.0)))

    assert mild is not None and severe is not None
    assert severe.score > mild.score


# ---------------------------------------------------------------------------
# Calibration (D-032)
# ---------------------------------------------------------------------------
def test_calibration_derives_thresholds_from_the_data() -> None:
    """"Small" must mean small for this dataset, not against a constant.

    The configured fallback is 0.12% of image area. These objects span 1% to
    10%, so a calibrated threshold must land inside that range — a constant
    would never fire.
    """
    contexts = [
        _context(finding_id=i, box=(0.0, 0.0, float(side), float(side)))
        for i, side in enumerate((100, 150, 200, 250, 300), start=1)
    ]

    factors = {f.name: f for f in calibrate_finding_factors(contexts)}

    small = factors[SMALL_OBJECT]
    assert small.fraction is not None
    assert 0.01 <= small.fraction <= 0.10
    assert small.fraction > config.SMALL_OBJECT_AREA_FRACTION


def test_calibration_falls_back_when_there_is_no_geometry() -> None:
    """With nothing to learn from, the configured defaults must still apply."""
    factors = {f.name: f for f in calibrate_finding_factors([])}

    assert factors[SMALL_OBJECT].fraction == config.SMALL_OBJECT_AREA_FRACTION
    assert factors[THIN_STRUCTURE].ratio == config.THIN_STRUCTURE_RATIO


def test_calibration_leaves_absolute_factors_alone() -> None:
    """A dark region is dark regardless of how dark the rest of the run is."""
    contexts = [_context(box=(0.0, 0.0, 100.0, 100.0))]

    factors = {f.name: f for f in calibrate_finding_factors(contexts)}

    assert factors[BLUR].threshold == config.BLUR_VARIANCE_THRESHOLD
    assert factors[LOW_LIGHT].threshold == config.LOW_LIGHT_THRESHOLD
    assert factors[EDGE_TRUNCATION].margin == config.EDGE_TRUNCATION_MARGIN


def test_calibration_uses_every_finding_not_only_failures() -> None:
    """Calibrating on failures alone would make the reference the measurement.

    The threshold derived from a mixed set must differ from one derived from
    the failures alone, or the distinction is not being made.
    """
    # Two small failures against eight large correct findings. The lower
    # quartile of the mixed set falls among the large ones; of the failures
    # alone it falls among the small ones. If calibration used only failures,
    # the two thresholds would be identical.
    small_failures = [
        _context(finding_id=i, outcome="false_negative", box=(0.0, 0.0, 50.0, 50.0))
        for i in range(1, 3)
    ]
    large_correct = [
        _context(finding_id=i, outcome="correct", box=(0.0, 0.0, 800.0, 800.0))
        for i in range(3, 11)
    ]

    mixed = {f.name: f for f in calibrate_finding_factors(
        small_failures + large_correct
    )}[SMALL_OBJECT]
    failures_only = {f.name: f for f in calibrate_finding_factors(small_failures)}[
        SMALL_OBJECT
    ]

    assert mixed.fraction != failures_only.fraction


def test_calibrated_factors_include_the_full_standard_set() -> None:
    """Calibration must not silently drop a detector."""
    names = {f.name for f in calibrate_finding_factors([])}

    assert names == {
        BLUR, LOW_LIGHT, SMALL_OBJECT, THIN_STRUCTURE, EDGE_TRUNCATION, CROWDING
    }
