/**
 * What a scanned plate reports in the 360° view.
 *
 * **Measured, not invented** — the same rule as `DIAGNOSIS` in `landing.ts`,
 * and from the same place: run 3 (the `val` split) of `db/model_doctor.db`,
 * whose factor rates are the figures the diagnosis beat already quotes.
 * Held as constants rather than fetched for the same two reasons: the landing
 * page stands up with the API down, and a claim on it should not change
 * because somebody ran a new analysis.
 *
 * **The plates are a stand-in.** Run 3 is a door model — its classes are
 * `door` and `door_frame` — and the belt carries plates because that is the
 * scene's metaphor for "an image going through inspection". So each card names
 * the real class and says the plate stands in for the photograph. Presenting
 * a door frame's measurements as a plate's would be exactly the kind of claim
 * this page argues against.
 *
 * **Chosen by rule, not by hand.** One finding per story, the lowest id in run
 * 3 that fits it, so the set is reproducible rather than curated:
 *
 * - correct:        outcome = 'correct'
 * - missed:         outcome = 'false_negative' carrying `small_object`
 * - spurious:       a failure whose only factor is `edge_truncation`
 * - poorly localised: outcome = 'poor_localization'
 *
 * Factor evidence strings are the engine's own `root_causes.evidence`, and
 * each verdict applies the engine's own bar (`comparison.factor_qualifies`:
 * lift > 1 and p < 0.05) to run 3's `factor_rates`. Re-measure with:
 *
 *   select f.id, f.outcome, f.class_name, f.confidence, f.iou,
 *          rc.factor, rc.evidence
 *   from findings f left join root_causes rc on rc.finding_id = f.id
 *   where f.run_id = 3 and f.id in (330, 344, 341, 337);
 *
 *   select factor, lift, p_value from factor_rates where run_id = 3;
 */

export type FactorVerdict = "cause" | "chance";

export type ScanFactor = {
  factor: string;
  /** The engine's measurement for this one finding, verbatim. */
  evidence: string;
  verdict: FactorVerdict;
  /** Run-level lift and p for this factor, as displayed. */
  lift: string;
  p: string;
};

export type ScanFinding = {
  id: number;
  outcome: "Correct" | "Missed" | "Spurious" | "Poorly localised";
  className: string;
  /** Null when there was no prediction to have a confidence. */
  confidence: string | null;
  iou: string | null;
  factors: ScanFactor[];
  /** One sentence: what this finding shows about the method. */
  reading: string;
};

/** Run 3's factor rates, as the cards display them. */
const RUN3 = {
  small_object: { lift: "3.45", p: "<0.001", verdict: "cause" },
  thin_structure: { lift: "2.32", p: "<0.001", verdict: "cause" },
  blur: { lift: "1.17", p: "0.30", verdict: "chance" },
  edge_truncation: { lift: "1.04", p: "0.54", verdict: "chance" },
} as const;

function factor(name: keyof typeof RUN3, evidence: string): ScanFactor {
  const rate = RUN3[name];
  return { factor: name, evidence, verdict: rate.verdict, lift: rate.lift, p: rate.p };
}

export const SCAN_RUN_LABEL = "Run 3 · val";

export const SCAN_FINDINGS: ScanFinding[] = [
  {
    id: 330,
    outcome: "Correct",
    className: "door_frame",
    confidence: "0.558",
    iou: "0.907",
    factors: [],
    reading: "Found and outlined well. Nothing here needs explaining.",
  },
  {
    id: 344,
    outcome: "Missed",
    className: "door_frame",
    confidence: null,
    iou: null,
    factors: [
      factor("small_object", "0.383% of image < 17.28%"),
      factor("thin_structure", "36.7:1 wide, above 5.1:1"),
    ],
    reading:
      "Tiny and extremely thin — and both conditions are measurably more "
      + "common on this run's failures than on its successes.",
  },
  {
    id: 341,
    outcome: "Spurious",
    className: "door",
    confidence: "0.574",
    iou: null,
    factors: [factor("edge_truncation", "touches top, right, bottom")],
    reading:
      "Cut off by the frame — yet that explains nothing: edge truncation is "
      + "almost as common on the detections the model got right.",
  },
  {
    id: 337,
    outcome: "Poorly localised",
    className: "door_frame",
    confidence: "0.757",
    iou: "0.404",
    factors: [
      factor("small_object", "2.913% of image < 17.28%"),
      factor("blur", "Laplacian variance 69.4 < 100"),
      factor("edge_truncation", "touches right"),
    ],
    reading:
      "Three conditions present, one that matters. Blur and truncation are "
      + "here, but only small size separates failures from successes.",
  },
];

/**
 * Which finding a plate reports. Stable per plate — the same plate always
 * gives the same answer, so a second click does not read as a new result.
 */
export function findingForPlate(plate: number): ScanFinding {
  const n = SCAN_FINDINGS.length;
  return SCAN_FINDINGS[((plate % n) + n) % n];
}
