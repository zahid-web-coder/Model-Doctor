import type { Finding, Outcome } from "@/lib/api/rows";
import type { Prediction, Shape } from "@/components/compare/Overlay";

/**
 * Assembling the geometry the Compare overlay draws, for one finding.
 *
 * **The overlay is reused exactly as Compare uses it, with no new props.**
 * `Overlay` already decides what to draw from the geometry it is handed —
 * `isBox` and `isPolygon` guard every shape — so a layer is hidden by passing
 * `null` for that geometry rather than by threading another flag through a
 * component two screens share. Adding per-layer flags would have meant editing
 * a renderer the Compare page depends on, to serve a toggle only Failures has.
 *
 * That is also why this lives outside the table component: the table is
 * `"use client"`, and keeping the mapping in a neutral module leaves it
 * testable on its own terms, the way `lib/rootCauses.ts` is.
 *
 * Nothing here changes what a finding *is*. It reads stored geometry and
 * decides how to present it.
 */

/** Which layers the reader has asked to see. */
export interface Layers {
  truthMask: boolean;
  predictionMask: boolean;
  boxes: boolean;
}

export const ALL_LAYERS: Layers = {
  truthMask: true,
  predictionMask: true,
  boxes: true,
};

/**
 * Colour role for a prediction's outline.
 *
 * Matches the legend the overlay already publishes: a matched prediction is
 * "good", anything else is "failed or extra". `poor_localization` counts as
 * failed — the box landed on the object but not well enough to match, which is
 * precisely what the reader is being shown.
 */
export function toneFor(outcome: Outcome): "good" | "bad" {
  return outcome === "correct" ? "good" : "bad";
}

/**
 * A short label for the prediction's tag.
 *
 * Confidence is included because it is the number a reader checks first on a
 * false positive. It is omitted rather than shown as zero when absent — a
 * false negative has no prediction to be confident about.
 */
export function labelFor(finding: Finding): string {
  const name = finding.class_name ?? "prediction";
  return finding.confidence === null
    ? name
    : `${name} ${(finding.confidence * 100).toFixed(1)}%`;
}

/**
 * Ground-truth geometry for a finding, or null when there is none to draw.
 *
 * A false positive matched no annotated object, so it has neither box nor
 * polygon — that absence is the finding, and drawing nothing is correct rather
 * than a gap. Verified against run 5: all 29 false positives store no
 * `truth_polygon`, and every other outcome stores one.
 */
export function truthShapeOf(finding: Finding, layers: Layers): Shape | null {
  const box = layers.boxes ? finding.truth_box : null;
  const polygon = layers.truthMask ? finding.truth_polygon : null;
  if (!box && !polygon) return null;
  return { box, polygon };
}

/**
 * Predicted geometry for a finding, or null when nothing was predicted.
 *
 * The polygon arrives from `mask_findings`, not from the finding — the two are
 * separate tables and the mask pass covers a run independently. All 33 false
 * negatives in run 5 store no `pred_polygon`, for the same structural reason
 * false positives store no truth.
 */
export function predictionShapeOf(
  finding: Finding,
  predictionPolygon: number[][] | null,
  layers: Layers,
): Prediction | null {
  const box = layers.boxes ? finding.pred_box : null;
  const polygon = layers.predictionMask ? predictionPolygon : null;
  if (!box && !polygon) return null;
  return { box, polygon, tone: toneFor(finding.outcome), label: labelFor(finding) };
}

/**
 * Whether this finding has any outline to show at all.
 *
 * Used to decide between the overlay and the plain image. A run analysed
 * before the mask pass, or a detector that predicts boxes only, legitimately
 * has none — and an empty SVG that looks like a broken viewer is worse than
 * the photograph on its own.
 */
export function hasOutline(
  finding: Finding,
  predictionPolygon: number[][] | null,
): boolean {
  const truth = finding.truth_polygon;
  return (
    (Array.isArray(truth) && truth.length >= 3) ||
    (Array.isArray(predictionPolygon) && predictionPolygon.length >= 3)
  );
}
