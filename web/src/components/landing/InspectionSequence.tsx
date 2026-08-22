"use client";

import { useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useHero, HERO_DEFAULTS } from "@/lib/hero/store";
import { scroll, clamp01, lerp, ease } from "@/lib/hero/landing";

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
const UNDER_BEAM = 0.493;
const AT_STATION = 0.833;

const PROGRESS = [
  [0.00, 0.180],
  [0.20, 0.300],
  [0.40, UNDER_BEAM],   // arrives under the aperture as INSPECTION ends
  [0.60, UNDER_BEAM],   // and is held there for the whole DIAGNOSIS beat
  [0.80, AT_STATION],   // then carries on into the station
  [1.00, 0.905],
] as const;

/** Idle glow, then full scan, then handed off downstream. */
const BEAM = [
  [0.00, 0.22],
  [0.22, 0.35],
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
export function InspectionSequence() {
  useEffect(() => {
    useHero.setState({ driveObject: false });
    return () => { useHero.setState({ ...HERO_DEFAULTS }); };
  }, []);

  useFrame(() => {
    const state = useHero.getState();

    // In orbit mode the visitor owns the arm and the timeline does not.
    // Writing `armYaw` here as well would snap it back sixty times a second
    // and the controls would look broken. The plate and the beam keep running
    // so the machine is still alive to look at.
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
