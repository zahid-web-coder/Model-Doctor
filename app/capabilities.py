"""Whether an inference knob can move a model's output at all.

Model Doctor can compare runs, but it could not ask the question that has to
come *before* an experiment: **if I vary this parameter, will anything happen?**
Without that, a knob that is silently ignored looks exactly like a knob that
made no difference, and "we varied it and nothing changed" reads as a finding
about the model rather than a finding about the plumbing.

That is not hypothetical. The reference checkpoint's head declares
``end2end = True``; its own training arguments record ``nms = False``; and
:mod:`app.inference` passes ``iou`` to ``predict()`` on every call. Varying that
value from 0.45 to 0.10 leaves the output byte-identical, including a pair of
predictions whose boxes overlap at 0.974. **NMS-threshold actuation is not
available on this checkpoint's inference path.** Stated exactly that narrowly:
it says where the knob does not reach, and nothing about whether
post-processing of some other kind could be involved in anything.

**Three questions, not one.** They are routinely conflated and they have
different answers in this codebase today:

* ``supported`` — does the detector accept the parameter at all?
* ``forwarded`` — does Model Doctor's own path pass it to the model?
* ``actuation`` — does changing it change the output?

:class:`~app.rfdetr_adapter.RFDetrDetector` inherits ``iou`` from its base and
so *supports* it, but its predict call passes only ``threshold``, so it never
*forwards* it. The YOLO path forwards it and the head ignores it. Same verdict,
different reason, and a reader who cannot tell them apart cannot tell a dead
architecture from a wiring bug.

**Nothing here writes.** The database is opened read-only, weights are read but
never saved, and a probe that cannot reach a checkpoint reports ``unknown``
rather than guessing. Run it directly::

    python -m app.capabilities --knob nms_iou --runs 4,5,12
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app import storage
from app.detectors import detect_family
from utils.exceptions import ModelDoctorError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class CapabilityError(ModelDoctorError):
    """A feasibility question could not be answered, with a reason to act on."""


#: Knobs this version will answer for.
#:
#: Two, deliberately. ``nms_iou`` is the question that motivated the gate;
#: ``image_size`` is the knob the next experiment varies and doubles as the
#: probe's positive control — if a probe cannot see *that* move, the probe is
#: broken and every other verdict it returns is worthless.
#:
#: ``confidence`` is absent because ``image_size`` already serves as the
#: control. ``max_det`` is absent because its inertness would be *data*
#: dependent — 300 against an observed peak of 5 — which is a different claim
#: from "the head ignores this", and mixing the two would blur the vocabulary
#: in the first version that establishes it.
KNOBS: tuple[str, ...] = ("nms_iou", "image_size")

#: The knob changes the output.
ACTUATES: str = "actuates"
#: The knob cannot change the output, and there is a reason to believe so.
INERT: str = "inert"
#: Not established. **Never a synonym for inert.**
UNKNOWN: str = "unknown"

ACTUATION: tuple[str, ...] = (ACTUATES, INERT, UNKNOWN)

#: Read from the architecture: a head that declares one-to-one matching has no
#: suppression stage to tune, and that is true before any image is seen.
STRUCTURAL: str = "structural"
#: Observed: the same images produced the same output at two knob values.
EMPIRICAL: str = "empirical"
BOTH: str = "structural+empirical"
NO_BASIS: str = "none"

#: Most images the empirical phase may touch. A cap, not a target.
MAX_PROBE_IMAGES: int = 5
#: How many it uses when nothing constrains it.
DEFAULT_PROBE_IMAGES: int = 3

#: Values a knob is varied between. Chosen to straddle anything a real
#: suppression stage would react to: a pair surviving 0.10 survives everything.
_PROBE_VALUES: dict[str, tuple[Any, ...]] = {
    "nms_iou": (0.45, 0.10),
    "image_size": (448, 640),
}


@dataclass(frozen=True)
class KnobVerdict:
    """What is known about one knob on one run.

    Every field is separately answerable, and ``actuation`` is deliberately not
    derivable from the other two: a knob can be supported, forwarded and still
    inert.
    """

    run_id: int
    knob: str
    supported: bool | None
    forwarded: bool | None
    actuation: str
    basis: str
    reason: str
    #: What the run passed, or ``None`` when it was never recorded. ``None``
    #: means *not recorded* and is never replaced by a default: today's
    #: configuration says nothing about what an environment held months ago.
    recorded_value: float | int | None = None
    checkpoint_native_image_size: int | None = None
    probe_images: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Structural inspection — reads the checkpoint, never the dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Checkpoint:
    """What a weights file says about itself."""

    path: Path
    exists: bool
    family: str | None = None
    head_class: str | None = None
    end2end: bool | None = None
    native_image_size: int | None = None
    trained_with_nms: bool | None = None
    error: str | None = None


def inspect_checkpoint(weights: str | Path) -> Checkpoint:
    """Read a checkpoint's architecture without running it on anything.

    The weights are loaded into memory to reach the head, and never written
    back. A file that cannot be read is reported as such rather than raising:
    the caller's question is "what is knowable here", and "nothing" is a valid
    answer to it.

    Args:
        weights: Path to a ``.pt`` checkpoint.

    Returns:
        A :class:`Checkpoint`. ``exists=False`` when the path does not resolve;
        every architectural field is ``None`` when it could not be determined.
    """
    path = Path(weights)
    if not path.is_file():
        return Checkpoint(path=path, exists=False, error="Checkpoint not found.")

    family = detect_family(path)
    if family != "yolo":
        # RF-DETR is DETR-derived and carries no suppression stage, but this
        # module does not assert that from the family name alone — the adapter
        # never forwards the parameter anyway, which is the fact that decides
        # the verdict and is checkable without the optional dependency.
        return Checkpoint(path=path, exists=True, family=family)

    native: int | None = None
    trained_nms: bool | None = None
    try:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
        train_args = raw.get("train_args") if isinstance(raw, dict) else None
        if isinstance(train_args, dict):
            size = train_args.get("imgsz")
            native = int(size) if isinstance(size, (int, float)) else None
            if "nms" in train_args:
                trained_nms = bool(train_args["nms"])
    except Exception as error:  # pragma: no cover - depends on the file
        logger.debug("Could not read train_args from %s: %s", path, error)

    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        head = model.model.model[-1]
        return Checkpoint(
            path=path,
            exists=True,
            family=family,
            head_class=type(head).__name__,
            end2end=bool(getattr(head, "end2end", False)),
            native_image_size=native,
            trained_with_nms=trained_nms,
        )
    except Exception as error:
        return Checkpoint(
            path=path,
            exists=True,
            family=family,
            native_image_size=native,
            trained_with_nms=trained_nms,
            error=f"Could not load the model to read its head: {error}",
        )


def knob_is_supported(family: str | None, knob: str) -> bool | None:
    """Whether a family's detector accepts the parameter at all.

    Support is about the constructor signature and nothing else. Both families
    build through ``Detector.__init__``, which takes ``iou`` and
    ``image_size``, so both accept both — whether the value then survives as
    far as the model is :func:`knob_is_forwarded`, and whether the model reads
    it is the probe's question.
    """
    if family is None:
        return None
    return knob in KNOBS


def knob_is_forwarded(family: str | None, knob: str) -> bool | None:
    """Whether Model Doctor's own path passes the parameter to the model.

    This is a fact about *this repository*, not about the model, and it is the
    one a wiring bug hides in. ``RFDetrDetector.predict`` calls the underlying
    model with ``threshold`` alone, so ``iou`` is accepted, stored on the
    object, and then dropped.
    """
    if family is None:
        return None
    if knob == "image_size":
        return True
    if knob == "nms_iou":
        return family != "rfdetr"
    return None


# ---------------------------------------------------------------------------
# Probe-image selection — from stored evidence, not arbitrary
# ---------------------------------------------------------------------------
def select_probe_images(
    connection: sqlite3.Connection, run_id: int, limit: int = DEFAULT_PROBE_IMAGES
) -> list[sqlite3.Row]:
    """Choose images on which the knob has a chance of biting.

    A suppression threshold cannot be observed on an image carrying one
    prediction; there is nothing for it to suppress. So images with at least
    two predictions are preferred, and the database already knows which those
    are. Falling back to the busiest images when none qualify keeps the probe
    answerable rather than empty, and the caller is told which happened.

    Args:
        connection: An open connection, read-only is sufficient.
        run_id: Run whose stored images are drawn from.
        limit: How many to return, capped at :data:`MAX_PROBE_IMAGES`.

    Returns:
        Rows with ``id``, ``path`` and ``prediction_count``, busiest first.
    """
    capped = max(1, min(int(limit), MAX_PROBE_IMAGES))
    rows = connection.execute(
        """
        SELECT id, path, prediction_count
        FROM images
        WHERE run_id = ? AND error IS NULL AND prediction_count >= 2
        ORDER BY prediction_count DESC, id ASC
        LIMIT ?
        """,
        (run_id, capped),
    ).fetchall()
    if rows:
        return list(rows)
    return list(
        connection.execute(
            """
            SELECT id, path, prediction_count
            FROM images
            WHERE run_id = ? AND error IS NULL
            ORDER BY prediction_count DESC, id ASC
            LIMIT ?
            """,
            (run_id, capped),
        ).fetchall()
    )


def _signature(result: Any) -> tuple:
    """Reduce one prediction result to something comparable and stable."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()
    xyxy = boxes.xyxy.cpu().numpy().round(2)
    conf = boxes.conf.cpu().numpy().round(4)
    return tuple(
        (float(a), float(b), float(c), float(d), float(e))
        for (a, b, c, d), e in zip(xyxy.tolist(), conf.tolist(), strict=True)
    )


def probe_actuation(
    checkpoint: Checkpoint, knob: str, images: Sequence[str], confidence: float
) -> tuple[bool | None, str]:
    """Vary one knob over a few images and report whether anything moved.

    Args:
        checkpoint: An inspected, existing checkpoint.
        knob: One of :data:`KNOBS`.
        images: Image paths, already capped by the caller.
        confidence: Held fixed across both arms.

    Returns:
        ``(changed, detail)``. ``changed`` is ``None`` when the probe could not
        run at all, which is not the same as ``False``.
    """
    if not images or not checkpoint.exists or checkpoint.family != "yolo":
        return None, "No probe was run."
    low, high = _PROBE_VALUES[knob]
    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint.path))
        for image in images:
            arms = []
            for value in (low, high):
                kwargs: dict[str, Any] = {
                    "source": image,
                    "conf": confidence,
                    "max_det": config.MAX_DETECTIONS,
                    "verbose": False,
                }
                if knob == "nms_iou":
                    kwargs["iou"] = value
                    kwargs["imgsz"] = checkpoint.native_image_size or config.IMAGE_SIZE
                else:
                    kwargs["imgsz"] = value
                    kwargs["iou"] = config.NMS_IOU_THRESHOLD
                arms.append(_signature(model.predict(**kwargs)[0]))
            if arms[0] != arms[1]:
                return True, (
                    f"Output differs between {knob}={low} and {knob}={high} "
                    f"on {Path(image).name}."
                )
    except Exception as error:
        return None, f"Probe could not run: {error}"
    return False, (
        f"Identical output at {knob}={low} and {knob}={high} on "
        f"{len(images)} probe image(s)."
    )


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def _recorded_value(run: storage.RunRecord, knob: str) -> float | int | None:
    if knob == "image_size":
        return run.image_size
    return run.nms_iou_threshold


def verdict_for_run(
    connection: sqlite3.Connection,
    run: storage.RunRecord,
    knob: str,
    *,
    probe: bool = True,
    probe_images: int = DEFAULT_PROBE_IMAGES,
) -> KnobVerdict:
    """Answer all three questions for one knob on one run.

    A checkpoint that has moved or changed is not re-probed: ``model_sha256``
    must still match, because a probe of a *different* file describes a
    different model. When the run predates schema 15 its ``recorded_value``
    stays ``None`` — capability is recoverable from the weights, but the value
    a run passed is not recoverable from anything, and inventing it would be
    worse than admitting it is gone.
    """
    if knob not in KNOBS:
        raise CapabilityError(
            f"Unknown knob {knob!r}. Supported: {', '.join(KNOBS)}."
        )

    recorded = _recorded_value(run, knob)
    checkpoint = inspect_checkpoint(run.model_path)
    notes: list[str] = []

    if not checkpoint.exists:
        return KnobVerdict(
            run_id=run.id,
            knob=knob,
            supported=None,
            forwarded=None,
            actuation=UNKNOWN,
            basis=NO_BASIS,
            reason=(
                f"The checkpoint recorded for this run is not at "
                f"{run.model_path}; nothing about it can be read."
            ),
            recorded_value=recorded,
        )

    digest = storage.file_sha256(checkpoint.path)
    if digest != run.model_sha256:
        return KnobVerdict(
            run_id=run.id,
            knob=knob,
            supported=None,
            forwarded=None,
            actuation=UNKNOWN,
            basis=NO_BASIS,
            reason=(
                "The file at the recorded path is not the checkpoint this run "
                f"used (sha256 {digest[:8]} against {run.model_sha256[:8]}), "
                "so probing it would describe a different model."
            ),
            recorded_value=recorded,
        )

    supported = knob_is_supported(checkpoint.family, knob)
    forwarded = knob_is_forwarded(checkpoint.family, knob)

    structural: str | None = None
    if knob == "nms_iou":
        if forwarded is False:
            structural = (
                f"The {checkpoint.family} adapter accepts iou but its predict "
                "call never passes it to the model."
            )
        elif checkpoint.end2end:
            trained = (
                f" and the checkpoint's train_args record "
                f"nms={checkpoint.trained_with_nms}"
                if checkpoint.trained_with_nms is not None
                else ""
            )
            # Phrased exactly this narrowly on purpose. It names where the
            # knob does not reach, and claims nothing about whether
            # post-processing of any other kind bears on anything.
            structural = (
                f"{checkpoint.head_class} declares end2end=True{trained}; "
                "NMS-threshold actuation is not available on this "
                "checkpoint's inference path."
            )

    changed: bool | None = None
    detail = "No probe was requested."
    used: list[int] = []
    if probe:
        rows = select_probe_images(connection, run.id, probe_images)
        used = [int(row["id"]) for row in rows]
        paths = [str(row["path"]) for row in rows if Path(str(row["path"])).is_file()]
        if not rows:
            detail = "This run stored no images to probe."
        elif not paths:
            detail = "None of this run's stored images are on disk."
        else:
            if all(int(row["prediction_count"]) < 2 for row in rows):
                notes.append(
                    "No stored image carries two predictions, so the probe is "
                    "less sensitive than it would otherwise be."
                )
            changed, detail = probe_actuation(
                checkpoint, knob, paths, run.confidence_threshold
            )

    if changed is True:
        actuation, basis = ACTUATES, EMPIRICAL
    elif structural and changed is False:
        actuation, basis = INERT, BOTH
    elif structural:
        actuation, basis = INERT, STRUCTURAL
    elif changed is False:
        # No effect seen, and no architectural reason to expect none. Three
        # images agreeing is not inertness, and calling it that would let a
        # future reader spend the finding as if it were.
        actuation, basis = UNKNOWN, EMPIRICAL
        notes.append(
            "No structural basis; absence of an observed effect is not "
            "evidence that the knob is inert."
        )
    else:
        actuation, basis = UNKNOWN, NO_BASIS

    reason = " ".join(part for part in ([structural] if structural else []) + [detail])
    if notes:
        reason = f"{reason} {' '.join(notes)}"

    return KnobVerdict(
        run_id=run.id,
        knob=knob,
        supported=supported,
        forwarded=forwarded,
        actuation=actuation,
        basis=basis,
        reason=reason.strip(),
        recorded_value=recorded,
        checkpoint_native_image_size=checkpoint.native_image_size,
        probe_images=used,
    )


def aggregate(verdicts: Sequence[KnobVerdict]) -> str:
    """Combine per-run verdicts, pessimistically and on purpose.

    ``inert`` dominates ``unknown``: it is positive evidence that spending runs
    on this knob is wasted, which is the only thing this gate exists to say. A
    single dead arm also makes a one-knob design impossible across the set, so
    the aggregate reports the blocking fact rather than the majority one.
    """
    if not verdicts:
        return UNKNOWN
    if any(v.actuation == INERT for v in verdicts):
        return INERT
    if all(v.actuation == ACTUATES for v in verdicts):
        return ACTUATES
    return UNKNOWN


def experiment_feasibility(
    connection: sqlite3.Connection,
    knob: str,
    run_ids: Sequence[int],
    *,
    probe: bool = True,
    probe_images: int = DEFAULT_PROBE_IMAGES,
) -> dict[str, Any]:
    """Report whether varying one knob across these runs could do anything.

    Describes; does not prescribe. Whether differing checkpoints invalidate a
    comparison is :mod:`app.comparison`'s existing judgement, and a second
    opinion here could only disagree with the first.
    """
    if knob not in KNOBS:
        raise CapabilityError(
            f"Unknown knob {knob!r}. Supported: {', '.join(KNOBS)}."
        )
    if not run_ids:
        raise CapabilityError("At least one run id is required.")

    runs: list[storage.RunRecord] = []
    missing: list[int] = []
    for run_id in run_ids:
        run = storage.load_run(connection, int(run_id))
        if run is None:
            missing.append(int(run_id))
        else:
            runs.append(run)
    if missing:
        raise CapabilityError(
            f"No such run: {', '.join(str(m) for m in missing)}."
        )

    verdicts = [
        verdict_for_run(
            connection, run, knob, probe=probe, probe_images=probe_images
        )
        for run in runs
    ]
    blocking = [
        f"run {v.run_id}: {v.reason}" for v in verdicts if v.actuation == INERT
    ]
    families = sorted({detect_family(r.model_path) for r in runs})
    checkpoints = {r.model_sha256 for r in runs}
    caveats: list[str] = []
    if knob == "nms_iou" and any(v.recorded_value is None for v in verdicts):
        unrecorded = [str(v.run_id) for v in verdicts if v.recorded_value is None]
        caveats.append(
            f"runs {', '.join(unrecorded)} predate schema 15; the value passed "
            "at inference is unrecoverable and is reported as null, never as a "
            "default."
        )
    if len(checkpoints) > 1:
        caveats.append(
            f"{len(checkpoints)} distinct checkpoints; these runs are not arms "
            "of one controlled experiment."
        )

    return {
        "knob": knob,
        "aggregate": aggregate(verdicts),
        "single_experiment": len(checkpoints) == 1 and not blocking,
        "blocking": blocking,
        "distinct_checkpoints": len(checkpoints),
        "distinct_families": families,
        "runs": [asdict(v) for v in verdicts],
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.capabilities",
        description="Ask whether an inference knob can move a model's output.",
    )
    parser.add_argument(
        "--knob", required=True, choices=list(KNOBS), help="Knob to interrogate."
    )
    parser.add_argument(
        "--runs", required=True, help="Comma-separated run ids, e.g. 4,5,12."
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Structural inspection only; run no inference.",
    )
    parser.add_argument(
        "--probe-images",
        type=int,
        default=DEFAULT_PROBE_IMAGES,
        help=f"Images to probe, capped at {MAX_PROBE_IMAGES}.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser.parse_args(argv)


def format_feasibility(report: dict[str, Any]) -> str:
    """Render a feasibility report for a terminal."""
    lines = [
        "",
        f"Feasibility — knob {report['knob']}",
        "-" * 78,
        f"  aggregate          : {report['aggregate']}",
        f"  single experiment  : {report['single_experiment']}",
        f"  checkpoints        : {report['distinct_checkpoints']}"
        f"   families: {', '.join(report['distinct_families'])}",
        "",
    ]
    for run in report["runs"]:
        lines.append(
            f"  run {run['run_id']:<4} {run['actuation']:<9} "
            f"basis={run['basis']:<21} supported={run['supported']} "
            f"forwarded={run['forwarded']}"
        )
        lines.append(f"      recorded value: {run['recorded_value']}")
        lines.append(f"      {run['reason']}")
    if report["blocking"]:
        lines += ["", "  Blocking:"] + [f"    - {b}" for b in report["blocking"]]
    if report["caveats"]:
        lines += ["", "  Caveats:"] + [f"    - {c}" for c in report["caveats"]]
    lines.append("-" * 78)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m app.capabilities``."""
    args = _parse_args(argv)
    try:
        ids = [int(part) for part in args.runs.split(",") if part.strip()]
    except ValueError:
        print("--runs must be comma-separated integers, e.g. 4,5,12")
        return 2
    try:
        with storage.connect_read_only() as connection:
            report = experiment_feasibility(
                connection,
                args.knob,
                ids,
                probe=not args.no_probe,
                probe_images=args.probe_images,
            )
    except (CapabilityError, FileNotFoundError, sqlite3.Error) as error:
        print(f"Error: {error}")
        return 1
    print(json.dumps(report, indent=2) if args.json else format_feasibility(report))
    return 0


__all__ = [
    "ACTUATES",
    "ACTUATION",
    "BOTH",
    "Checkpoint",
    "CapabilityError",
    "DEFAULT_PROBE_IMAGES",
    "EMPIRICAL",
    "INERT",
    "KNOBS",
    "KnobVerdict",
    "MAX_PROBE_IMAGES",
    "NO_BASIS",
    "STRUCTURAL",
    "UNKNOWN",
    "aggregate",
    "experiment_feasibility",
    "format_feasibility",
    "inspect_checkpoint",
    "knob_is_forwarded",
    "knob_is_supported",
    "main",
    "probe_actuation",
    "select_probe_images",
    "verdict_for_run",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
