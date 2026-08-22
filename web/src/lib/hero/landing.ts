/**
 * The landing page's scroll state, its beats, and the camera path.
 *
 * **GSAP writes exactly one number.** `scroll.t` runs 0..1 across the whole
 * page and everything cinematic is a pure function of it — camera, beam,
 * plate travel, station sweep. That keeps §3's ownership rule intact with one
 * writer instead of eight, makes scrubbing correct in both directions, and
 * makes the reduced-motion path a matter of pinning `t` to fixed values.
 *
 * It is a plain mutable object rather than React state on purpose: the value
 * changes every frame, and putting it in a store would re-render the tree
 * sixty times a second to move a matrix React does not own.
 */
export const scroll = { t: 0 };

export type BeatName = "hero" | "inspection" | "diagnosis" | "verification" | "transition";

/** Where each beat starts and ends in `t`. Five equal beats, one screen each. */
export const BEATS: Record<BeatName, [number, number]> = {
  hero: [0.0, 0.2],
  inspection: [0.2, 0.4],
  diagnosis: [0.4, 0.6],
  verification: [0.6, 0.8],
  transition: [0.8, 1.0],
};

export const BEAT_ORDER: BeatName[] = [
  "hero", "inspection", "diagnosis", "verification", "transition",
];

/** 0..1 progress within a beat, clamped outside it. */
export function beatProgress(t: number, beat: BeatName) {
  const [from, to] = BEATS[beat];
  return clamp01((t - from) / (to - from));
}

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

/** Smoothstep. Linear camera moves read as mechanical; this eases both ends. */
export const ease = (k: number) => k * k * (3 - 2 * k);

/**
 * Camera keyframes, in the asset's own world space.
 *
 * Every value is derived from geometry read out of the GLB rather than tuned
 * blind: the conveyor runs x -3..+3 with its plate surface at y 0.7, the arm
 * stands at x -0.86 / z 0.54, and the beam lands on the centreline at
 * x ~ -0.04. The verification station sits downstream at x 2.0.
 */
export type CameraKey = {
  at: number;
  position: [number, number, number];
  target: [number, number, number];
};

export const CAMERA_PATH: CameraKey[] = [
  // HERO — wide, high, the whole line in frame.
  //
  // The hero *holds* its shot. The first version travelled from 10m to 7m
  // across this beat and then to 3m in the next, so by the time the headline
  // had finished fading the camera was already deep in a close-up and the two
  // screens read as unrelated shots rather than one move. Establishing means
  // staying put long enough for the frame to be read.
  { at: 0.0, position: [7.0, 3.8, 7.9], target: [-0.2, 1.05, 0.0] },
  { at: 0.2, position: [6.5, 3.5, 7.3], target: [-0.2, 1.08, 0.0] },
  // INSPECTION — one continuous push down the line to the scanner. Pulled back
  // from the old 2.55 so the pod is the subject without being cropped by it.
  { at: 0.4, position: [3.0, 2.1, 3.35], target: [-0.15, 1.15, 0.05] },
  // DIAGNOSIS — the plate under the aperture, held left of centre.
  //
  // Aimed to the right of the plate rather than straight at it, which is what
  // pushes the subject into the left of the frame and leaves the right clear
  // for the evidence cards. Framed straight on, the copy and the machine fight
  // for the same pixels and neither reads.
  { at: 0.6, position: [1.45, 1.35, 2.15], target: [0.5, 0.82, 0.05] },
  // VERIFICATION — downstream, the station straddling the belt.
  { at: 0.8, position: [3.4, 1.7, 2.6], target: [1.9, 0.98, 0.0] },
  // TRANSITION — pushing into the station's display.
  { at: 1.0, position: [2.4, 1.15, 1.15], target: [2.06, 0.99, 0.0] },
];

/** Where the verification station stands on the belt. */
export const STATION_X = 2.0;

/**
 * The diagnosis beat's figures.
 *
 * **Measured, not invented.** These come from runs 2 and 3 in the project's
 * own database — the same numbers the Root Causes screen renders. They are
 * held as constants rather than fetched because the landing page has to stand
 * up with the API down, and because a marketing claim should not silently
 * change when somebody runs a new analysis.
 *
 * If the shipped database changes, these need re-measuring. The query is
 * `select factor, failure_count, failure_total, correct_count, correct_total,
 * lift, p_value from factor_rates where run_id = 3`.
 */
export const DIAGNOSIS = {
  /** The obvious suspect: most common on failures, and almost as common on
   *  the plates that passed. It explains nothing. */
  ruledOut: {
    factor: "edge_truncation",
    onFailures: "73.6%",
    onCorrect: "70.8%",
    lift: "1.04",
    p: "0.54",
    verdict: "Ruled out",
    why: "Nearly as common on the plates that passed.",
  },
  /** The real one: three times rarer, three times more predictive. */
  cause: {
    factor: "small_object",
    onFailures: "43.3%",
    onCorrect: "12.5%",
    lift: "3.45",
    p: "<0.001",
    verdict: "Likely cause",
    why: "Three times over-represented in failures.",
  },
  /** Verification is replication, not a pass/fail lamp. */
  replication: {
    factor: "small_object",
    runs: [
      { label: "Run 3 · val", findings: "496", lift: "3.45", p: "1.5e-14" },
      { label: "Run 2 · test", findings: "278", lift: "2.77", p: "3.4e-06" },
    ],
  },
} as const;
