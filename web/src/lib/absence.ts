import type { Outcome } from "@/lib/api/rows";

/**
 * Why a cell in the findings table is empty.
 *
 * **Not every empty cell means the same thing, and the table used to render
 * them identically.** A false positive has no ground-truth class because
 * nothing was there to match — that absence *is* the finding. A false negative
 * has no confidence because there was no prediction to be confident about. A
 * `wrong_class` row cannot show the predicted class because the schema stores
 * one class per finding and for that outcome it is the ground truth.
 *
 * Rendering all three as a bare "n/a" invites the reading that data is
 * missing, which for a false positive suggests the *image* is unlabelled —
 * a different thing entirely, and one that would call for a different
 * response. These are two orthogonal facts:
 *
 * * ``outcome = false_positive`` — this prediction matched no ground-truth
 *   object. Says nothing about the rest of the image.
 * * ``images.truth_count = 0`` — the image carries no annotations at all.
 *
 * An unlabelled image produces false positives, but a false positive does not
 * imply an unlabelled image. The wording below is deliberately scoped to *this
 * prediction* so it can never be read as a claim about the image.
 *
 * None of this changes a metric. Every finding is counted exactly as before.
 */

export interface Absence {
  /** Terse enough for a dense table cell. */
  short: string;
  /** A full sentence, for the detail panel and the cell's tooltip. */
  long: string;
  /**
   * True when the value is absent *by definition* of the outcome, rather than
   * merely unmeasured. Structural absences are expected and are not a gap in
   * the data.
   */
  structural: boolean;
}

/** Fields of a finding that can legitimately be empty. */
export type Field = "prediction" | "truth" | "confidence" | "iou";

const NO_TRUTH: Absence = {
  short: "no match",
  long:
    "No matching ground-truth object. This prediction did not land on any " +
    "annotated object, which is what makes it a false positive — the absence " +
    "is the finding, not missing data. Other objects in the same image may " +
    "well be annotated and matched.",
  structural: true,
};

const NO_PREDICTION: Absence = {
  short: "not detected",
  long:
    "No matching prediction. An annotated object is present here and the " +
    "model did not find it, which is what makes it a false negative.",
  structural: true,
};

const NO_CONFIDENCE: Absence = {
  short: "—",
  long:
    "No confidence, because there was no prediction to score. A false " +
    "negative is a miss, so nothing was predicted for this object.",
  structural: true,
};

const NO_IOU_UNMATCHED: Absence = {
  short: "—",
  long:
    "No overlap to measure: IoU compares a prediction with a ground-truth " +
    "object, and this finding has only one of the two.",
  structural: true,
};

const PREDICTED_CLASS_UNKNOWN: Absence = {
  short: "not recorded",
  long:
    "The predicted class is not recoverable for a wrong_class finding. The " +
    "schema stores one class per finding, and for this outcome that is the " +
    "ground truth. It is shown as absent rather than inferred.",
  structural: true,
};

/**
 * Explain why one field of one finding is empty, or return null.
 *
 * Null means the field should carry a value; if it is empty anyway, that is a
 * genuine gap rather than an expected one, and the caller should say so.
 */
export function absence(outcome: Outcome, field: Field): Absence | null {
  switch (outcome) {
    case "false_positive":
      if (field === "truth") return NO_TRUTH;
      if (field === "iou") return NO_IOU_UNMATCHED;
      return null;

    case "false_negative":
      if (field === "prediction") return NO_PREDICTION;
      if (field === "confidence") return NO_CONFIDENCE;
      if (field === "iou") return NO_IOU_UNMATCHED;
      return null;

    case "wrong_class":
      if (field === "prediction") return PREDICTED_CLASS_UNKNOWN;
      return null;

    case "correct":
    case "poor_localization":
      // Both carry a prediction and a matched ground-truth object, so every
      // field should hold a value.
      return null;

    default:
      return null;
  }
}

/**
 * Whether an outcome is one where the model produced a detection at all.
 *
 * `wrong_class` counts: something was predicted, it was simply the wrong
 * class. What cannot be shown is *which* class, which `absence` explains.
 */
export function hasPrediction(outcome: Outcome): boolean {
  return outcome !== "false_negative";
}

/** Whether an outcome has a ground-truth object behind it. */
export function hasTruth(outcome: Outcome): boolean {
  return outcome !== "false_positive";
}
