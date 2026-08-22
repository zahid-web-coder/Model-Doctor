import { CLASSES } from "./classes";

/**
 * Placeholder findings, shaped like the `findings` table.
 *
 * `prediction` is null for a wrong_class finding. SCHEMA.md is explicit that
 * the predicted class is not recoverable for those, so inventing one here
 * would put a value on screen that the API can never supply.
 */
export interface FailureInstance {
  id: string;
  prediction: string | null;
  groundTruth: string | null;
  confidence: number | null;
  failureType: "false_positive" | "false_negative" | "wrong_class" | "poor_localization";
  rootCauseFactor: string;
  iou: number | null;
}

const TYPES = ["false_positive", "false_negative", "wrong_class", "poor_localization"] as const;
const FACTORS = ["small_object", "thin_structure", "crowding", "edge_truncation", "low_contrast"];

export const mockFailures: FailureInstance[] = Array.from({ length: 60 }).map((_, i) => {
  const failureType = TYPES[i % TYPES.length];
  const cls = CLASSES[i % CLASSES.length];
  const other = CLASSES[(i + 1) % CLASSES.length];

  return {
    id: `img-${1000 + i}`,
    // A false positive has no ground truth; a false negative has no prediction;
    // a wrong_class has a truth but an unrecoverable prediction.
    prediction: failureType === "false_negative" || failureType === "wrong_class" ? null : cls,
    groundTruth: failureType === "false_positive" ? null : other,
    confidence: failureType === "false_negative" ? null : 0.5 + ((i * 7) % 45) / 100,
    failureType,
    rootCauseFactor: FACTORS[i % FACTORS.length],
    iou: failureType === "poor_localization" ? 0.1 + ((i * 3) % 35) / 100 : null,
  };
});

export const FAILURE_LABEL: Record<FailureInstance["failureType"], string> = {
  false_positive: "False Positive",
  false_negative: "False Negative",
  wrong_class: "Wrong Class",
  poor_localization: "Poor Localization",
};
