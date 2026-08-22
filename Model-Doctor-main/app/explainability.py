"""Grad-CAM — showing where the model looked.

A finding says a prediction was wrong. This module answers the next question:
*what in the image drove it?* It produces, per finding, a heatmap over the
original image showing which regions most influenced the model's score for that
class at that location.

**The generic part knows nothing about YOLO.** :class:`GradCAM` hooks a set of
layers, weights their activations by the gradients of a caller-supplied scalar,
and returns a normalised map. Supporting another detector means writing another
:class:`CamAdapter` — the same seam pattern as ``SimilarityFn`` (D-018),
``RegionExtractor`` (D-023), and ``EmbeddingBackend``.

**What was hard, and why the obvious approach fails.** Backpropagating a
YOLO26 detection's *decoded* class score yields exactly zero gradient at every
layer. The decoded output has passed through a sigmoid that saturates. The raw
pre-sigmoid logits survive on the head's secondary return value, and gradients
do flow through them in eval mode, so that is what this module targets. See
DECISIONS D-024.

Heatmaps are written as image files and recorded in the ``heatmaps`` table, so
a dashboard can find them with a single left join rather than probing the
filesystem.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app import storage
from utils.cam import CamAdapter, GradCAM, prepare_image, render_overlay
from utils.exceptions import ExplainabilityError
from utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class Yolo26SegAdapter:
    """Adapter for Ultralytics YOLO26 segmentation models.

    **Target layers** are the three feature-pyramid tensors the head consumes,
    at strides 8, 16 and 32. Fusing all three is the default: the coarsest
    level carries the strongest gradient but only a few percent of it is
    non-zero and a 21x21 map upscaled to full resolution is unreadable, while
    the finest level supplies the spatial detail. Betting on one level would
    make heatmap quality depend on object size.

    **The scalar** is a raw class logit, deliberately not the decoded score.
    The decoded score has been through a sigmoid which saturates, and
    backpropagating it produces gradients that are exactly zero — measured, not
    assumed. The head returns its raw predictions as a second value, and in
    evaluation mode gradients flow through them.
    """

    layer_indices: tuple[int, ...] = (16, 19, 22)

    @property
    def name(self) -> str:
        """Return the adapter identifier stored with each heatmap."""
        return f"yolo26-seg:{','.join(str(i) for i in self.layer_indices)}"

    def target_layers(self, model: Any) -> list[Any]:
        """Return the pyramid-level modules feeding the detection head."""
        backbone = model.model
        return [backbone[i] for i in self.layer_indices]

    def _raw_scores(self, output: Any) -> Any:
        """Return the head's raw pre-sigmoid class logits.

        Raises:
            ExplainabilityError: If the model did not return them, which means
                the architecture differs from what this adapter targets.
        """
        if not (isinstance(output, tuple) and len(output) == 2):
            raise ExplainabilityError(
                "Model did not return raw head predictions; this adapter "
                "targets an end-to-end YOLO26 head."
            )
        preds = output[1]
        branch = preds.get("one2one") or preds.get("one2many")
        if not isinstance(branch, dict) or "scores" not in branch:
            raise ExplainabilityError("Head predictions contain no 'scores'.")
        return branch["scores"]

    def scalar_for(
        self, output: Any, class_id: int, box: tuple[float, float, float, float]
    ) -> Any:
        """Return the strongest raw logit for ``class_id`` within ``box``.

        Anchors are laid out as the concatenation of the pyramid levels, so an
        anchor's index determines its level and its position in that level's
        grid. Selecting only the anchors whose centre falls inside the region
        makes the resulting heatmap answer "what made the model respond *here*"
        rather than "what does this class look like in general".

        Args:
            output: The model's forward output.
            class_id: Class whose logit should be maximised.
            box: Region of interest, in letterboxed input pixels.

        Returns:
            A scalar tensor suitable for ``backward()``.

        Raises:
            ExplainabilityError: If the class index is out of range.
        """
        import torch

        scores = self._raw_scores(output)  # (1, nc, A)
        num_classes = scores.shape[1]
        if not 0 <= class_id < num_classes:
            raise ExplainabilityError(
                f"class_id {class_id} outside the model's {num_classes} classes."
            )

        centres = _anchor_centres(scores.shape[2], scores.device)
        x1, y1, x2, y2 = box
        inside = (
            (centres[:, 0] >= x1)
            & (centres[:, 0] <= x2)
            & (centres[:, 1] >= y1)
            & (centres[:, 1] <= y2)
        )
        row = scores[0, class_id]
        if bool(inside.any()):
            return row[inside].max()

        # No anchor centre lands inside a very small box. Falling back to the
        # nearest anchor keeps a heatmap available rather than failing on
        # precisely the small objects most worth explaining.
        centre = torch.tensor(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0], device=scores.device
        )
        nearest = int((centres - centre).pow(2).sum(dim=1).argmin())
        return row[nearest]


def _anchor_centres(total: int, device: Any) -> Any:
    """Return every anchor's centre in input pixels, in head order.

    The head concatenates its pyramid levels finest-first, so the grid sizes
    are derived from the total rather than hardcoded — a different input size
    or a different number of levels stays correct without configuration.

    Args:
        total: Number of anchors the head produced.
        device: Device the result should live on.

    Returns:
        A ``(total, 2)`` tensor of ``(x, y)`` centres.

    Raises:
        ExplainabilityError: If the anchor count matches no known layout.
    """
    import torch

    for strides in ((8, 16, 32), (8, 16, 32, 64)):
        for size in (config.IMAGE_SIZE, 640, 672, 1280):
            grids = [size // s for s in strides]
            if sum(g * g for g in grids) != total:
                continue
            points = []
            for grid, stride in zip(grids, strides, strict=True):
                ys, xs = torch.meshgrid(
                    torch.arange(grid, device=device),
                    torch.arange(grid, device=device),
                    indexing="ij",
                )
                # +0.5 puts the point at the cell centre, matching how the head
                # decodes positions.
                points.append(
                    torch.stack(
                        ((xs.flatten() + 0.5) * stride, (ys.flatten() + 0.5) * stride),
                        dim=1,
                    ).float()
                )
            return torch.cat(points, dim=0)

    raise ExplainabilityError(
        f"Could not infer the anchor grid layout for {total} anchors."
    )


# ---------------------------------------------------------------------------
# Generic Grad-CAM
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeatmapReport:
    """What one explanation pass did."""

    run_id: int
    method: str
    adapter: str
    considered: int = 0
    generated: int = 0
    skipped_no_geometry: int = 0
    skipped_unreadable: int = 0
    skipped_no_gradient: int = 0

    def describe(self) -> str:
        """Render the report for console output."""
        lines = ["", "Grad-CAM explanation", "-" * 60]
        lines.append(f"  Run                : {self.run_id}")
        lines.append(f"  Method             : {self.method}")
        lines.append(f"  Adapter            : {self.adapter}")
        lines.append(f"  Findings considered: {self.considered}")
        lines.append(f"  Heatmaps generated : {self.generated}")
        for label, value in (
            ("Skipped (geometry) ", self.skipped_no_geometry),
            ("Skipped (image)    ", self.skipped_unreadable),
            ("Skipped (gradient) ", self.skipped_no_gradient),
        ):
            if value:
                lines.append(f"  {label}: {value}")
        lines.append("-" * 60)
        lines.append("")
        return "\n".join(lines)


def explain_run(
    connection: Any,
    run_id: int,
    model: Any,
    adapter: CamAdapter,
    output_dir: Path | None = None,
    failures_only: bool = True,
    limit: int | None = None,
    image_size: int | None = None,
) -> HeatmapReport:
    """Generate a heatmap per finding for a saved run.

    Reads findings back from the database, exactly as feature extraction does
    (D-023): the write path stays untouched and a run saved earlier can be
    explained without re-running inference.

    A finding's prediction box is explained when it has one; otherwise its
    ground-truth box is used. That means a false negative is explained too —
    the map shows what the model attended to in a region where it found
    nothing, which is often the more useful question.

    Args:
        connection: An open database connection.
        run_id: Run to explain. Its findings must already be saved.
        model: A loaded detector network in evaluation mode.
        adapter: Supplies target layers and the differentiable scalar.
        output_dir: Where images are written. Defaults to
            ``results/heatmaps/run_<id>``.
        failures_only: Explain only findings that represent mistakes.
        limit: Explain at most this many findings.
        image_size: Model input size. Defaults to :data:`config.IMAGE_SIZE`.

    Returns:
        A description of what was produced and what was skipped.

    Raises:
        ExplainabilityError: If the run has no findings to explain.
    """
    rows = storage.load_findings_for_embedding(connection, run_id, failures_only)
    if not rows:
        raise ExplainabilityError(
            f"Run {run_id} has no findings to explain. Diagnose and save it first."
        )
    if limit:
        rows = rows[:limit]

    size = image_size or config.IMAGE_SIZE
    destination = output_dir or (config.RESULTS_DIR / "heatmaps" / f"run_{run_id}")
    destination.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, adapter.target_layers(model))
    no_geometry = unreadable = no_gradient = 0
    written: list[tuple[int, str]] = []

    for row in rows:
        box = _finding_box(row)
        if box is None:
            no_geometry += 1
            continue
        try:
            tensor, letterbox, original = prepare_image(Path(row["path"]), size)
        except ExplainabilityError as exc:
            unreadable += 1
            logger.warning("%s", exc)
            continue

        class_id = row["class_id"] if row["class_id"] is not None else 0
        mapped = letterbox.map_box(box)
        try:
            # Loop variables are bound as defaults rather than captured by
            # reference. Safe today because the call is immediate, but a late
            # binding here would silently explain the wrong region if this ever
            # became deferred.
            heat = cam(
                tensor,
                lambda out, _c=class_id, _b=mapped: adapter.scalar_for(out, _c, _b),
            )
        except ExplainabilityError as exc:
            # One finding whose scalar produced no gradient must not abandon
            # the pass — the same isolation principle as per-image inference.
            no_gradient += 1
            logger.warning("Finding %s: %s", row["finding_id"], exc)
            continue

        overlay = render_overlay(original, heat, letterbox)
        path = destination / f"finding_{row['finding_id']}.png"
        _write_image(path, overlay)
        written.append((row["finding_id"], str(path)))

    storage.save_heatmaps(connection, run_id, "grad-cam", adapter.name, written)
    logger.info("Wrote %d heatmap(s) for run %d", len(written), run_id)

    return HeatmapReport(
        run_id=run_id,
        method="grad-cam",
        adapter=adapter.name,
        considered=len(rows),
        generated=len(written),
        skipped_no_geometry=no_geometry,
        skipped_unreadable=unreadable,
        skipped_no_gradient=no_gradient,
    )


def _finding_box(row: Any) -> tuple[float, float, float, float] | None:
    """Return the region a finding should be explained over.

    Prediction first, then ground truth — the reverse of feature extraction's
    preference. Here the question is "what drove the model", so where the model
    actually responded is the more faithful region; where it responded to
    nothing, its ground-truth location is the next best question.
    """
    for prefix in ("pred", "truth"):
        values = [row[f"{prefix}_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
        if all(value is not None for value in values):
            return (values[0], values[1], values[2], values[3])
    return None


def _write_image(path: Path, array: Any) -> None:
    """Write an RGB array to disk as a PNG."""
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def load_explainable_model(weights: Path | None = None) -> Any:
    """Load a detector network prepared for gradient computation.

    Ultralytics runs inference under ``no_grad`` and may hold half-precision
    weights; neither is compatible with Grad-CAM. The network is therefore
    unwrapped, cast to float, put in evaluation mode, and has gradients enabled
    on its parameters.
    """
    from ultralytics import YOLO

    from utils.resources import check_model

    status = check_model(weights)
    if not status.available or status.path is None:
        raise ExplainabilityError(status.detail)

    net = YOLO(str(status.path)).model.float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    logger.info("Loaded %s for explanation", status.path.name)
    return net


def build_parser() -> argparse.ArgumentParser:
    """Construct the explainability command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-explain",
        description="Generate Grad-CAM heatmaps for a saved run's findings.",
    )
    parser.add_argument("--run", type=int, help="Run id. Defaults to the newest.")
    parser.add_argument("--db", type=str, help="Database path override.")
    parser.add_argument("--model", type=str, help="Weights path override.")
    parser.add_argument("--imgsz", type=int, help="Model input size.")
    parser.add_argument("--limit", type=int, help="Explain at most N findings.")
    parser.add_argument(
        "--all-findings", action="store_true", help="Explain correct findings too."
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated layer indices to visualise (default: 16,19,22).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to explain.
    """
    args = build_parser().parse_args(argv)
    config.ensure_output_dirs()

    indices = (
        tuple(int(v) for v in args.layers.split(",")) if args.layers else (16, 19, 22)
    )
    adapter = Yolo26SegAdapter(layer_indices=indices)

    with storage.connect(Path(args.db) if args.db else None) as connection:
        run_id = args.run
        if run_id is None:
            runs = storage.list_runs(connection)
            if not runs:
                logger.error(
                    "No runs found. Run: python -m app.diagnosis --split test --save"
                )
                return 1
            run_id = runs[0].id
            logger.info("Using newest run %d", run_id)

        try:
            model = load_explainable_model(
                Path(args.model) if args.model else None
            )
            report = explain_run(
                connection,
                run_id,
                model,
                adapter,
                failures_only=not args.all_findings,
                limit=args.limit,
                image_size=args.imgsz,
            )
        except ExplainabilityError as exc:
            logger.error("%s", exc)
            return 1

    print(report.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
