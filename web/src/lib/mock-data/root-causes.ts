export interface RootCauseFactor {
  factor: string;
  rateInFailures: number;
  rateInCorrect: number;
  lift: number | null;
  pValue: number | null;
  reading: string;
  evidence: string[];
}

export const mockRootCauses: RootCauseFactor[] = [
  {
    factor: "blur",
    rateInFailures: 0.33,
    rateInCorrect: 0.12,
    lift: 2.71,
    pValue: 0.0001,
    reading: "highly over-represented in failures",
    evidence: [
      "Detected via Laplacian variance (σ² < 100) on 412 failed images.",
      "Most frequent in 'Crack' and 'Deformation' classes.",
    ]
  },
  {
    factor: "low_light",
    rateInFailures: 0.28,
    rateInCorrect: 0.125,
    lift: 2.24,
    pValue: 0.0005,
    reading: "over-represented in failures",
    evidence: [
      "Average brightness < 40 on 351 failed images.",
    ]
  },
  {
    factor: "occlusion",
    rateInFailures: 0.18,
    rateInCorrect: 0.104,
    lift: 1.72,
    pValue: 0.002,
    reading: "over-represented in failures",
    evidence: [
      "Bounding box IoU with known occluders > 0.5 in 232 cases."
    ]
  },
  {
    factor: "small_object",
    rateInFailures: 0.15,
    rateInCorrect: 0.105,
    lift: 1.42,
    pValue: 0.006,
    reading: "slightly over-represented",
    evidence: [
      "Object area < 2% of image total area."
    ]
  },
  {
    factor: "contrast",
    rateInFailures: 0.14,
    rateInCorrect: 0.11,
    lift: 1.27,
    pValue: 0.031,
    reading: "marginally over-represented",
    evidence: [
      "RMS contrast < 20."
    ]
  },
  {
    factor: "background_clutter",
    rateInFailures: 0.45,
    rateInCorrect: 0.43,
    lift: 1.04,
    pValue: 0.45,
    reading: "no significant difference",
    evidence: [
      "High edge density in background regions."
    ]
  },
  {
    factor: "rotation_anomaly",
    rateInFailures: 0.05,
    rateInCorrect: 0.05,
    lift: null,
    pValue: null,
    reading: "insufficient data to calculate lift",
    evidence: [
      "Object rotated > 45 degrees."
    ]
  }
];
