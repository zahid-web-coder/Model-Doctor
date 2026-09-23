import { create } from "zustand";
import type { ScanFinding } from "./scanFindings";

/**
 * A plate scan in the 360° view: click a plate, the belt carries it under the
 * scanner, the beam comes up, and the finding it stands for is read out.
 *
 * **An ownership switch, not a second writer.** `InspectionSequence` is still
 * the only thing that writes the plate, beam and arm on the landing page. This
 * store only tells it *which* script to follow: the free-running belt, or a
 * scan. Anything that would write the arm directly — a drag, the nudge
 * buttons, Reset — cancels the scan first, so the arm is never pulled two ways.
 *
 * Phases: `carry` (the belt brings the plate under the aperture while the arm
 * settles to its aim), `scan` (the beam rises and the heatmap with it),
 * `result` (held, with the card shown until it is closed or another plate is
 * clicked).
 */
export type ScanPhase = "idle" | "carry" | "scan" | "result";

export type ScanState = {
  phase: ScanPhase;
  finding: ScanFinding | null;
  /** Clock time the current phase started, from the render loop's clock. */
  since: number;
  /** `objectProgress` when the carry began, and how far forward it travels. */
  from: number;
  travel: number;
  /** The arm pose at the start, eased back to rest during the carry. */
  armFrom: { armYaw: number; headTilt: number; headRotation: number };
};

const IDLE: ScanState = {
  phase: "idle",
  finding: null,
  since: 0,
  from: 0,
  travel: 0,
  armFrom: { armYaw: 0, headTilt: 0, headRotation: 0 },
};

export const useScan = create<ScanState>()(() => ({ ...IDLE }));

/**
 * A scan asked for by finding rather than by plate — the "Scan an example"
 * buttons. The panel lives outside the canvas and cannot see the belt, so it
 * posts the finding here and `PlateScanner`, which knows where every plate
 * is, picks the one that reaches the scanner soonest. `nonce` makes asking
 * for the same finding twice a new request rather than no change.
 */
export type ScanRequest = { finding: ScanFinding; nonce: number };
export const useScanRequest = create<{ request: ScanRequest | null }>()(() => ({
  request: null,
}));

export function requestScan(finding: ScanFinding) {
  useScanRequest.setState((s) => ({
    request: { finding, nonce: (s.request?.nonce ?? 0) + 1 },
  }));
}

/** Whether the pointer is over the verification station's monitor. */
export const useMonitorHover = create<{ hover: boolean }>()(() => ({ hover: false }));

/** Stop any scan and hand the belt back to its free run. */
export function cancelScan() {
  if (useScan.getState().phase !== "idle") useScan.setState({ ...IDLE });
}

