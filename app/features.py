"""Feature extraction — turning failures into vectors.

A failure record says *what* went wrong. To find out whether many failures went
wrong for the *same reason*, they have to be comparable to each other, and
pixel crops are not comparable in any useful sense. This module encodes each
failed region into a vector where semantic similarity becomes geometric
closeness, which is what the failure-grouping milestone will cluster.

Nothing here clusters, explains, or recommends. It produces vectors.

**It runs as a pass over the database, not over in-memory diagnoses.** Findings
are read back after they were saved, which means:

* the write path in :mod:`app.storage` needed no changes at all;
* findings already carry their primary keys, so vectors attach cleanly;
* it can run over a run saved days ago, with no re-inference.

**Two seams, both with a reason today rather than a hoped-for one.**

*Region extraction* is a parameter. Today :func:`region_from_box` crops a
finding's bounding box. When mask-level diagnosis lands (DECISIONS D-022), a
polygon-based extractor replaces it at the call site — and the polygons are
already persisted, so no schema change is needed for that step.

*The encoder* is injected. This is not speculative: a real CLIP model is
several hundred megabytes, and the test suite must never download one. An
injectable backend is the only way to test this pipeline at all, and it keeps
the third-party dependency at one boundary (the same isolation principle as
DECISIONS D-005).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app import storage
from utils.exceptions import ModelDoctorError
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# A crop rectangle in absolute pixels: (x1, y1, x2, y2).
Region = tuple[int, int, int, int]

# Given a finding row, return the region to encode, or None to skip it.
RegionExtractor = Callable[[Any], Region | None]

# Default context added around a region before cropping, as a fraction of its
# size. A bounding box tight against an object discards the surroundings, and
# the surroundings are often the explanation — a missed object may be missed
# *because* of what is next to it. Small enough not to swamp the object itself.
DEFAULT_PADDING: float = 0.15

# Smallest crop worth encoding. Below this the region carries no recoverable
# signal and would only add noise to a cluster.
MIN_REGION_PIXELS: int = 8


class FeatureExtractionError(ModelDoctorError):
    """Feature extraction could not run."""


# ---------------------------------------------------------------------------
# Region extraction — the seam mask support will use
# ---------------------------------------------------------------------------
def region_from_box(
    row: Any, padding: float = DEFAULT_PADDING
) -> Region | None:
    """Return the crop rectangle for a finding, from its bounding boxes.

    Ground truth is preferred when present, falling back to the prediction.
    That matches how a finding attributes its class, so each finding encodes
    the thing it is actually about: a false positive encodes what the model
    *thought* it saw, a false negative encodes what it *missed*.

    Args:
        row: A finding row from
            :func:`~app.storage.load_findings_for_embedding`.
        padding: Context to add around the region, as a fraction of its size.

    Returns:
        A clamped pixel rectangle, or ``None`` if the finding has no usable
        geometry or the region is too small to carry signal.
    """
    box = _first_present(row, ("truth", "pred"))
    if box is None:
        return None

    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return _clamp(
        x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, row["width"], row["height"]
    )


def _first_present(
    row: Any, prefixes: Sequence[str]
) -> tuple[float, float, float, float] | None:
    """Return the first fully-populated box among ``prefixes``."""
    for prefix in prefixes:
        values = [row[f"{prefix}_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
        if all(value is not None for value in values):
            return (values[0], values[1], values[2], values[3])
    return None


def _clamp(
    x1: float, y1: float, x2: float, y2: float, width: Any, height: Any
) -> Region | None:
    """Clip a rectangle to the image and reject degenerate results.

    Padding routinely pushes a region past the frame — most objects in this
    project's data touch an edge — so clipping is the normal path, not an
    error case.
    """
    if not width or not height:
        return None
    left = max(0, int(round(x1)))
    top = max(0, int(round(y1)))
    right = min(int(width), int(round(x2)))
    bottom = min(int(height), int(round(y2)))
    if right - left < MIN_REGION_PIXELS or bottom - top < MIN_REGION_PIXELS:
        return None
    return (left, top, right, bottom)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EmbeddingBackend(Protocol):
    """Anything that turns cropped images into vectors.

    A protocol rather than a base class so a test double needs only to match
    the shape, and so no part of this module imports a vision-language library
    to be usable.
    """

    @property
    def name(self) -> str:
        """Return an identifier stored alongside every vector."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the length of the vectors produced."""
        ...

    def encode(self, images: Sequence[Any]) -> list[list[float]]:
        """Return one vector per input image, in the same order."""
        ...


class ClipBackend:
    """CLIP image encoder, loaded lazily.

    Vectors are L2-normalised, so cosine similarity reduces to a dot product
    and Euclidean distance becomes a monotone function of it. That makes the
    output usable by clustering algorithms that assume either metric, without
    the caller needing to know which was intended.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ) -> None:
        """Configure the encoder without loading any weights.

        Args:
            model_name: OpenCLIP architecture name.
            pretrained: Pretrained tag for that architecture.
            device: Compute device. Defaults to :data:`config.DEVICE`.
        """
        self._model_name = model_name or config.CLIP_MODEL
        self._pretrained = pretrained or config.CLIP_PRETRAINED
        self._device = device or config.DEVICE
        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        """Return the architecture and pretrained tag as one identifier."""
        return f"{self._model_name}/{self._pretrained}"

    @property
    def dimensions(self) -> int:
        """Return the embedding width, loading the model if necessary."""
        if self._dimensions is None:
            self._load()
        assert self._dimensions is not None
        return self._dimensions

    def _load(self) -> None:
        """Load weights and the matching preprocessing pipeline.

        Deferred until first use so that importing this module — which the CLI
        and the tests both do — never downloads several hundred megabytes.
        """
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise FeatureExtractionError(
                "open_clip_torch is not installed. Run: "
                "pip install -r requirements.txt"
            ) from exc

        logger.info(
            "Loading CLIP %s (%s) on device '%s'",
            self._model_name,
            self._pretrained,
            self._device,
        )
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained
            )
        except Exception as exc:
            raise FeatureExtractionError(
                f"Could not load CLIP {self._model_name}/{self._pretrained}: {exc}"
            ) from exc

        model = model.to(self._device).eval()
        self._model = model
        self._preprocess = preprocess
        with torch.no_grad():
            probe = torch.zeros(1, 3, 224, 224, device=self._device)
            self._dimensions = int(model.encode_image(probe).shape[-1])
        logger.info("CLIP ready — %d dimensions", self._dimensions)

    def encode(self, images: Sequence[Any]) -> list[list[float]]:
        """Encode a batch of PIL images into normalised vectors.

        Args:
            images: Cropped regions to encode.

        Returns:
            One vector per image, in the same order.
        """
        if not images:
            return []
        self._load()
        import torch

        batch = torch.stack(
            [self._preprocess(image) for image in images]  # type: ignore[misc]
        ).to(self._device)
        with torch.no_grad():
            vectors = self._model.encode_image(batch)  # type: ignore[union-attr]
            # Normalising here rather than at the consumer means every stored
            # vector is directly comparable, whatever reads it later.
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.cpu().float().tolist()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionReport:
    """What one extraction pass did.

    Attributes:
        run_id: Run the embeddings belong to.
        model_name: Encoder identifier stored with every vector.
        considered: Findings examined.
        embedded: Findings that produced a vector.
        skipped_no_region: Findings with no usable geometry.
        skipped_unreadable: Findings whose image could not be opened.
    """

    run_id: int
    model_name: str
    considered: int = 0
    embedded: int = 0
    skipped_no_region: int = 0
    skipped_unreadable: int = 0

    def describe(self) -> str:
        """Render the report for console output."""
        lines = ["", "Feature extraction", "-" * 60]
        lines.append(f"  Run                : {self.run_id}")
        lines.append(f"  Encoder            : {self.model_name}")
        lines.append(f"  Findings considered: {self.considered}")
        lines.append(f"  Embedded           : {self.embedded}")
        if self.skipped_no_region:
            lines.append(f"  Skipped (geometry) : {self.skipped_no_region}")
        if self.skipped_unreadable:
            lines.append(f"  Skipped (image)    : {self.skipped_unreadable}")
        lines.append("-" * 60)
        lines.append("")
        return "\n".join(lines)


def extract_run_embeddings(
    connection: sqlite3.Connection,
    run_id: int,
    backend: EmbeddingBackend,
    region: RegionExtractor = region_from_box,
    failures_only: bool = True,
    batch_size: int = 32,
) -> ExtractionReport:
    """Embed a saved run's findings and store the vectors.

    Args:
        connection: An open database connection.
        run_id: Run to process. Its findings must already be saved.
        backend: Encoder to use.
        region: How to derive a crop from a finding. Defaults to its bounding
            box; a polygon-based extractor slots in here for mask support
            without touching anything else.
        failures_only: Embed only findings that represent mistakes, which is
            what the roadmap asks for.
        batch_size: Crops encoded per forward pass.

    Returns:
        A description of what was processed and what was skipped.

    Raises:
        FeatureExtractionError: If the run has no findings to embed.
    """
    from PIL import Image

    rows = storage.load_findings_for_embedding(connection, run_id, failures_only)
    if not rows:
        raise FeatureExtractionError(
            f"Run {run_id} has no findings to embed. Diagnose and save it first."
        )

    considered = len(rows)
    no_region = 0
    unreadable = 0
    embedded: list[tuple[int, Sequence[float]]] = []

    pending_ids: list[int] = []
    pending_crops: list[Any] = []

    def flush() -> None:
        """Encode whatever has accumulated and clear the buffer."""
        if not pending_crops:
            return
        vectors = backend.encode(pending_crops)
        embedded.extend(zip(pending_ids, vectors, strict=True))
        pending_ids.clear()
        pending_crops.clear()

    for row in rows:
        box = region(row)
        if box is None:
            no_region += 1
            continue
        try:
            with Image.open(row["path"]) as image:
                crop = image.convert("RGB").crop(box)
        except Exception as exc:
            # One unreadable image must not abandon the pass. The same
            # principle as per-image error isolation during inference.
            unreadable += 1
            logger.warning("Could not read %s: %s", row["path"], exc)
            continue

        pending_ids.append(row["finding_id"])
        pending_crops.append(crop)
        if len(pending_crops) >= batch_size:
            flush()

    flush()

    written = storage.save_embeddings(connection, run_id, backend.name, embedded)
    logger.info("Stored %d embedding(s) for run %d", written, run_id)

    return ExtractionReport(
        run_id=run_id,
        model_name=backend.name,
        considered=considered,
        embedded=written,
        skipped_no_region=no_region,
        skipped_unreadable=unreadable,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the feature-extraction command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-features",
        description="Encode a saved run's failures into vectors.",
    )
    parser.add_argument(
        "--run", type=int, help="Run id to process. Defaults to the newest run."
    )
    parser.add_argument("--db", type=str, help="Database path override.")
    parser.add_argument("--model", type=str, help="OpenCLIP architecture name.")
    parser.add_argument("--pretrained", type=str, help="OpenCLIP pretrained tag.")
    parser.add_argument(
        "--all-findings",
        action="store_true",
        help="Embed correct findings too, not only failures.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Crops per forward pass."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        ``0`` on success, ``1`` when there is nothing to process.
    """
    args = build_parser().parse_args(argv)
    db_path = Path(args.db) if args.db else None

    with storage.connect(db_path) as connection:
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

        backend = ClipBackend(model_name=args.model, pretrained=args.pretrained)
        try:
            report = extract_run_embeddings(
                connection,
                run_id,
                backend,
                failures_only=not args.all_findings,
                batch_size=args.batch_size,
            )
        except FeatureExtractionError as exc:
            logger.error("%s", exc)
            return 1

    print(report.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
