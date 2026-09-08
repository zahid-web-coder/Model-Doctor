"""Explain *why* two detectors differ, using Model Doctor's own analysis.

The mAP table says which model scores higher. This says what each one gets
wrong, and on what kind of object — which is the question Model Doctor exists
to answer.

Nothing here is a new metric. It drives the project's existing
:func:`model_doctor.app.diagnosis.diagnose_image` for the outcome taxonomy and the
existing
:mod:`model_doctor.app.root_cause` factor detectors for the conditions, then reports
each
factor's rate on failures against its rate on correct detections with the
project's own :func:`utils.statistics.lift` and Fisher test. A factor that is
common on failures *and* common on correct detections explains nothing, and the
lift is what shows that.

Both models are read from the same stored predictions, diagnosed against the
same ground truth, at the same operating threshold.

Usage::

    python scripts/analyse_failures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app.diagnosis import diagnose_image
from model_doctor.app.root_cause import (
    BlurFactor,
    CrowdingFactor,
    EdgeTruncationFactor,
    FindingContext,
    LowLightFactor,
    SmallObjectFactor,
    ThinStructureFactor,
)
from model_doctor.utils.annotations import ObjectAnnotation
from model_doctor.utils.logging_utils import get_logger
from model_doctor.utils.statistics import fisher_exact_two_sided, lift

logger = get_logger(__name__)

OPERATING_CONFIDENCE = 0.25


def to_annotations(
    records: list[dict[str, Any]], with_confidence: bool
) -> list[ObjectAnnotation]:
    """Build annotations from stored detection dicts."""
    out: list[ObjectAnnotation] = []
    for det in records:
        x1, y1, x2, y2 = det["bbox"]
        out.append(ObjectAnnotation(
            class_id=int(det["class_id"]),
            class_name=str(det["class_name"]),
            confidence=float(det["confidence"]) if with_confidence else None,
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            polygon=det.get("polygon"),
        ))
    return out


def truths_for(gt: dict[str, Any]) -> dict[str, list[ObjectAnnotation]]:
    """Group the canonical ground truth by file name."""
    cat_names = {c["id"]: c["name"] for c in gt.get("categories", [])}
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt["annotations"]:
        by_id[ann["image_id"]].append(ann)

    out: dict[str, list[ObjectAnnotation]] = {}
    for image in gt["images"]:
        anns = []
        for a in by_id.get(image["id"], []):
            x, y, w, h = a["bbox"]
            seg = a["segmentation"][0]
            polygon = [[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)]
            anns.append(ObjectAnnotation(
                class_id=int(a["category_id"]),
                class_name=cat_names.get(
                    int(a["category_id"]), f"id:{a['category_id']}"
                ),
                confidence=None,
                x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
                polygon=polygon,
            ))
        out[image["file_name"]] = anns
    return out


def diagnose_model(payload: dict[str, Any], truths: dict[str, list[ObjectAnnotation]],
                   images_dir: Path) -> dict[str, Any]:
    """Run the outcome taxonomy and the factor detectors for one model."""
    factors = [
        SmallObjectFactor(), ThinStructureFactor(), EdgeTruncationFactor(),
        CrowdingFactor(), BlurFactor(), LowLightFactor(),
    ]
    needs_pixels = any(f.needs_pixels for f in factors)

    outcomes: Counter[str] = Counter()
    # factor -> outcome-kind -> count
    factor_hits: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    per_image_failures: list[tuple[str, int]] = []
    ious: list[float] = []

    import numpy as np
    from PIL import Image

    finding_id = 0
    for record in payload["images"]:
        name = record["file_name"]
        truth = truths.get(name, [])
        preds = [
            d for d in record["detections"]
            if d["confidence"] >= OPERATING_CONFIDENCE
        ]
        diagnosis = diagnose_image(
            image_path=images_dir / name,
            predictions=to_annotations(preds, with_confidence=True),
            truths=truth,
            image_width=record["width"],
            image_height=record["height"],
        )

        grey = None
        if needs_pixels:
            try:
                with Image.open(images_dir / name) as handle:
                    grey = np.asarray(handle.convert("L"))
            except Exception:
                grey = None

        boxes_on_image = [(t.x1, t.y1, t.x2, t.y2) for t in truth]
        failures_here = 0
        for finding in diagnosis.findings:
            finding_id += 1
            kind = "failure" if finding.outcome.is_failure else "correct"
            outcomes[finding.outcome.value] += 1
            totals[kind] += 1
            if kind == "failure":
                failures_here += 1
            if finding.iou is not None:
                ious.append(float(finding.iou))

            ann = finding.truth or finding.prediction
            if ann is None:
                continue
            box = (ann.x1, ann.y1, ann.x2, ann.y2)
            region = None
            if grey is not None:
                x1, y1, x2, y2 = (int(max(0, v)) for v in box)
                if x2 > x1 and y2 > y1:
                    region = grey[y1:y2, x1:x2]

            context = FindingContext(
                finding_id=finding_id,
                outcome=finding.outcome.value,
                class_name=ann.class_name,
                box=box,
                image_width=record["width"],
                image_height=record["height"],
                region=region,
                neighbours=[b for b in boxes_on_image if b != box],
            )
            for factor in factors:
                try:
                    if factor.detect(context) is not None:
                        factor_hits[factor.name][kind] += 1
                except Exception:
                    continue

        per_image_failures.append((name, failures_here))

    rows = []
    for name, hits in sorted(factor_hits.items()):
        f, c = hits["failure"], hits["correct"]
        ft, ct = totals["failure"], totals["correct"]
        rows.append({
            "factor": name,
            "failure_count": f, "failure_total": ft,
            "correct_count": c, "correct_total": ct,
            "lift": lift(f, ft, c, ct),
            "p_value": fisher_exact_two_sided(f, ft, c, ct),
        })

    return {
        "outcomes": dict(outcomes),
        "totals": dict(totals),
        "factors": rows,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "worst_images": sorted(per_image_failures, key=lambda p: -p[1])[:10],
    }


def main(argv: list[str] | None = None) -> int:
    """Diagnose every model and write the comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/comparison")
    parser.add_argument(
        "--images", default="datasets/columns_all_1280_yolo/test/images"
    )
    args = parser.parse_args(argv)

    root = Path(args.results)
    gt = json.loads((root / "ground_truth.json").read_text())
    truths = truths_for(gt)

    report: dict[str, Any] = {}
    keys = (
        "yolo26", "rfdetr_312", "rfdetr_384", "rfdetr_480",
        "rfdetr_576", "rfdetr_672", "rfdetr_768",
    )
    for key in keys:
        payload = json.loads((root / f"{key}.json").read_text())
        logger.info("diagnosing %s", key)
        report[key] = diagnose_model(payload, truths, Path(args.images))
        report[key]["label"] = payload["model"]["label"]

    (root / "failure_analysis.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", root / "failure_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
