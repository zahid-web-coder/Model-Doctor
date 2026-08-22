/**
 * Factor rates, shaped like the `factor_rates` table.
 *
 * Every factor carries lift and significance, and the two honest outcomes are
 * kept deliberately: a factor that is common in failures but equally common in
 * correct detections (lift ~1.00, not significant), and one with too little
 * data to compute lift at all. Those are the cases the product exists to catch.
 */
export interface RootCauseFactor {
  factor: string;
  rateInFailures: number;
  rateInCorrect: number;
  lift: number | null;
  pValue: number | null;
  failures: number;
  impact: "High" | "Medium" | "Low" | null;
  reading: string;
  evidence: string[];
}

export const mockRootCauses: RootCauseFactor[] = [
  {
    factor: "small_object", rateInFailures: 0.33, rateInCorrect: 0.12, lift: 2.77, pValue: 0.0001,
    failures: 412, impact: "High", reading: "highly over-represented in failures",
    evidence: [
      "Object area below the 25th percentile on 412 failed findings.",
      "Replicated on the val split at 3.45x, p<0.001.",
    ],
  },
  {
    factor: "thin_structure", rateInFailures: 0.28, rateInCorrect: 0.125, lift: 2.14, pValue: 0.0005,
    failures: 351, impact: "High", reading: "over-represented in failures",
    evidence: [
      "Aspect ratio above the 75th percentile on 351 failed findings.",
      "Replicated on the val split at 2.32x, p<0.001.",
    ],
  },
  {
    factor: "crowding", rateInFailures: 0.18, rateInCorrect: 0.104, lift: 1.72, pValue: 0.041,
    failures: 232, impact: "Medium", reading: "significant on test, not replicated on val",
    evidence: [
      "Three or more instances within the same neighbourhood.",
      "Failed to replicate on the val split (p=0.310), so not actionable alone.",
    ],
  },
  {
    factor: "low_contrast", rateInFailures: 0.15, rateInCorrect: 0.105, lift: 1.42, pValue: 0.006,
    failures: 198, impact: "Medium", reading: "slightly over-represented",
    evidence: ["Local contrast below threshold on 198 failed findings."],
  },
  {
    factor: "edge_truncation", rateInFailures: 0.71, rateInCorrect: 0.76, lift: 1.00, pValue: 1.0,
    failures: 892, impact: null, reading: "no significant difference",
    evidence: [
      "Present on 71% of failures — and on 76% of correct detections.",
      "Describes the dataset, not the failures. Not a cause.",
    ],
  },
  {
    factor: "rotation_anomaly", rateInFailures: 0.05, rateInCorrect: 0.05, lift: null, pValue: null,
    failures: 12, impact: null, reading: "insufficient data to calculate lift",
    evidence: ["Only 12 findings carry this factor — too few for a stable estimate."],
  },
];
