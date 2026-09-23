"use client";

import { useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useHero, HERO_DEFAULTS } from "@/lib/hero/store";
import { scroll, clamp01, lerp, ease } from "@/lib/hero/landing";
import { useScan, cancelScan } from "@/lib/hero/scan";

/**
 * Interpolate a value across keyframes given as `[t, value]` pairs.
 *
 * Small enough that a scan beats any structure, and readable enough that the
 * whole choreography below can be read as a table rather than as code.
 */
function at(keys: readonly (readonly [number, number])[], t: number) {
  if (t <= keys[0][0]) return keys[0][1];
  const last = keys[keys.length - 1];
  if (t >= last[0]) return last[1];
  let i = 0;
  while (i < keys.length - 2 && t > keys[i + 1][0]) i += 1;
  const [ta, va] = keys[i];
  const [tb, vb] = keys[i + 1];
  return lerp(va, vb, ease(clamp01((t - ta) / (tb - ta))));
}

/**
 * The plate's position along the belt, as scroll progress.
 *
 * Derived from geometry, not tuned: the conveyor runs x -3..+3, the beam lands
 * on the centreline at x ~ -0.04, and the station stands at x 2.0. So the
 * plate is under the aperture at (−0.04 + 3) / 6 ~ 0.49 and inside the station
 * at (2.0 + 3) / 6 ~ 0.83.
 */
export const UNDER_BEAM = 0.493;
const AT_STATION = 0.833;

const PROGRESS = [
  [0.00, 0.180],
  [0.20, 0.300],
  [0.40, UNDER_BEAM],   // arrives under the aperture as INSPECTION ends
  [0.60, UNDER_BEAM],   // and is held there for the whole DIAGNOSIS beat
  [0.80, AT_STATION],   // then carries on into the station
  [1.00, 0.905],
] as const;

/**
 * Pilot glow, then the scanner activates, then it hands off downstream.
 *
 * The hero is deliberately near zero. The heatmap's opacity is derived from
 * the beam (§3), so any beam in the hero puts a heatmap on whichever plate
 * happens to be under the aperture — and since the line is running, one always
 * eventually is. At 0.22 that read as a result appearing at random before
 * anything had been inspected.
 *
 * A pilot glow of 0.05 keeps the lens alive without the overlay reaching a
 * visible alpha, and the scanner coming up during INSPECTION is the beat the
 * sequence was supposed to have anyway.
 */
const BEAM = [
  [0.00, 0.05],
  [0.20, 0.06],
  [0.32, 0.62],
  [0.40, 1.00],
  [0.60, 1.00],
  [0.72, 0.30],
  [1.00, 0.16],
] as const;

/** The rollers never stop, but they ease off while a plate is being held. */
const CONVEYOR = [
  [0.00, 0.42],
  [0.38, 0.20],
  [0.58, 0.10],
  [0.72, 0.46],
  [1.00, 0.34],
] as const;

/** A small sweep as the scanner acquires, settling for the diagnosis. */
const ARM_YAW = [
  [0.00, 0.10],
  [0.24, -0.16],
  [0.40, 0.02],
  [0.62, 0.00],
  [1.00, -0.06],
] as const;

const HEAD_TILT = [
  [0.00, 0.00],
  [0.30, 0.05],
  [0.50, 0.00],
  [1.00, -0.03],
] as const;

/**
 * Drives the machine from `scroll.t`.
 *
 * `driveObject` is switched off for the whole landing page, which hands
 * ownership of `objectProgress` from the render loop to the scroll timeline.
 * That switch is exactly what §3's one-owner rule is for — without it the loop
 * would keep advancing the plate while the timeline scrubbed it back, and the
 * plate would stutter in a way that only shows up mid-scroll.
 *
 * The store is module-global and shared with the dashboard hero, and a click
 * through to `/dashboard` is a client-side navigation that does not reload it.
 * So everything written here is put back on unmount.
 */
/**
 * How long a scan's phases last, in seconds. The carry scales with how far the
 * belt has to run, within bounds, so a plate one slot away does not take as
 * long as one that has to go round.
 */
const CARRY_MIN = 0.9;
const CARRY_MAX = 3.2;
const CARRY_PER_LOOP = 4.2;
const SCAN_RISE = 1.1;

/**
 * One frame of a plate scan (see `lib/hero/scan.ts`).
 *
 * Still this component writing, and still only these values — the scan store
 * decides which script is followed, never who writes. Durations collapse to
 * zero under reduced motion, so the scan jumps straight to its result instead
 * of animating to it.
 */
function driveScan(now: number, animate: boolean) {
  const scan = useScan.getState();
  if (scan.since < 0) {
    useScan.setState({ since: now });
    return;
  }
  const elapsed = now - scan.since;

  if (scan.phase === "carry") {
    const duration = animate
      ? Math.min(CARRY_MAX, Math.max(CARRY_MIN, scan.travel * CARRY_PER_LOOP))
      : 0;
    const k = duration > 0 ? clamp01(elapsed / duration) : 1;
    const e = ease(k);
    const next = scan.from + scan.travel * e;
    useHero.setState({
      objectProgress: next - Math.floor(next),
      // The rollers spin with the carry and wind down as the plate arrives.
      conveyorSpeed: lerp(0.95, 0.05, e),
      beamIntensity: 0.12,
      // The arm settles back to its modelled aim, so the beam lands on the
      // belt wherever the visitor had swung it.
      armYaw: lerp(scan.armFrom.armYaw, 0, e),
      headTilt: lerp(scan.armFrom.headTilt, 0, e),
      headRotation: lerp(scan.armFrom.headRotation, 0, e),
    });
    if (k >= 1) useScan.setState({ phase: "scan", since: now });
    return;
  }

  if (scan.phase === "scan") {
    const k = animate ? clamp01(elapsed / SCAN_RISE) : 1;
    // The heatmap's opacity is derived from the beam in `Machine`, so raising
    // the beam is what makes the attention map appear on the plate.
    useHero.setState({ conveyorSpeed: 0, beamIntensity: lerp(0.12, 1, ease(k)) });
    if (k >= 1) useScan.setState({ phase: "result", since: now });
    return;
  }

  // result: held until the card is closed or another plate is clicked.
  useHero.setState({ conveyorSpeed: 0, beamIntensity: 1 });
}

export function InspectionSequence({ animate = true }: { animate?: boolean }) {
  useEffect(() => {
    useHero.setState({ driveObject: false });
    return () => {
      cancelScan();
      useHero.setState({ ...HERO_DEFAULTS });
    };
  }, []);

  useFrame(({ clock }) => {
    const state = useHero.getState();

    // Leaving the 360° view ends any scan: the timeline takes the machine back.
    if (!state.freeOrbit && useScan.getState().phase !== "idle") cancelScan();

    // In orbit mode the visitor owns the arm and the timeline does not.
    // Writing `armYaw` here as well would snap it back sixty times a second
    // and the controls would look broken. The plate and the beam keep running
    // so the machine is still alive to look at.
    //
    // The exception is a scan the visitor asked for by clicking a plate: then
    // this drives the plate, beam and arm until the scan ends. Any direct arm
    // input cancels the scan first, so the two never write at once.
    if (state.freeOrbit && useScan.getState().phase !== "idle") {
      driveScan(clock.elapsedTime, animate);
      return;
    }
    if (state.freeOrbit) {
      useHero.setState({
        objectProgress: (state.objectProgress + 0.0016) % 1,
        beamIntensity: 0.95,
        conveyorSpeed: 0.32,
      });
      return;
    }

    const t = scroll.t;
    useHero.setState({
      objectProgress: at(PROGRESS, t),
      beamIntensity: at(BEAM, t),
      conveyorSpeed: at(CONVEYOR, t),
      armYaw: at(ARM_YAW, t),
      headTilt: at(HEAD_TILT, t),
    });
  });

  return null;
}
