import { create } from "zustand";

/**
 * The seven hero parameters from docs/HERO_3D_DESIGN.md §3, and nothing else.
 *
 * Read inside useFrame with `useHero.getState()` rather than by subscribing:
 * a per-frame subscription would re-render React sixty times a second to move
 * a matrix that React does not own.
 */
export type CameraPreset = "idle" | "inspect" | "wide";

export type HeroState = {
  conveyorSpeed: number;
  objectProgress: number;
  armYaw: number;
  headRotation: number;
  headTilt: number;
  beamIntensity: number;
  cameraPreset: CameraPreset;

  /**
   * Rig-test controls. NOT hero parameters — the seven above are the state
   * model. These exist so the shoulder and elbow hinges can be exercised
   * directly; no timeline writes them.
   */
  shoulderSwing: number;
  forearmSwing: number;

  /**
   * Ownership switches, not an eighth and ninth parameter. §3's rule is that
   * exactly one system writes each value; these make the handover explicit
   * rather than letting two systems fight over one matrix.
   *
   * `freeOrbit` stays false in the dashboard: an orbit control inside a
   * scrolling page swallows the wheel, and the hero is a piece of the page
   * rather than a viewer.
   */
  driveObject: boolean;
  freeOrbit: boolean;
};

export const HERO_DEFAULTS: HeroState = {
  conveyorSpeed: 0.35,
  objectProgress: 0.15,
  armYaw: 0,
  headRotation: 0,
  headTilt: 0,
  beamIntensity: 0.9,
  cameraPreset: "idle",
  shoulderSwing: 0,
  forearmSwing: 0,
  driveObject: true,
  freeOrbit: false,
};

export const useHero = create<HeroState>()(() => ({ ...HERO_DEFAULTS }));
