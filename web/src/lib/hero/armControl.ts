import { useHero } from "./store";

/**
 * How a pointer or a button maps onto the arm.
 *
 * **One definition, read by both the drag handler and the nudge buttons.**
 * They were written separately and disagreed: the buttons swung the arm the
 * opposite way to their own arrows, and the drag ran opposite to the buttons.
 * Anything that converts screen direction to arm motion belongs here, because
 * two copies of a sign convention is two chances to get it backwards.
 *
 * ## The sign
 *
 * `armYaw` rotates `ArmBase` about the asset's Z (three's +Y). From the story
 * camera — which sits on +X/+Z looking back at the line — a positive rotation
 * carries the scanner to the *left* of frame. So screen-right is negative yaw,
 * and everything that speaks in screen terms goes through `SCREEN_TO_YAW`.
 *
 * Tilt is the same story: the head pitches up on positive `headTilt`, so
 * dragging *down* has to reduce it or the machine fights the hand holding it.
 */

export const YAW_LIMIT = 1;
export const TILT_LIMIT = 0.4;

/** Screen-right (+1) to arm yaw. Negative: the arm swings the other way. */
export const SCREEN_TO_YAW = -1;
/** Screen-down (+1) to head tilt. Negative: drag down, head goes down. */
export const SCREEN_TO_TILT = -1;

/** One button press. Small enough to aim with, given the buttons repeat. */
export const YAW_STEP = 0.05;
export const TILT_STEP = 0.022;

/** Pixels to radians for a drag. A third of a laptop screen covers the sweep. */
export const YAW_PER_PIXEL = 0.0045;
export const TILT_PER_PIXEL = 0.0022;

export const clampTo = (v: number, limit: number) =>
  Math.max(-limit, Math.min(limit, v));

/** Nudge the arm by one step, in screen terms: `+1` is right / down. */
export function nudgeArm(axis: "yaw" | "tilt", screenDirection: number) {
  useHero.setState((s) =>
    axis === "yaw"
      ? { armYaw: clampTo(s.armYaw + screenDirection * SCREEN_TO_YAW * YAW_STEP, YAW_LIMIT) }
      : { headTilt: clampTo(s.headTilt + screenDirection * SCREEN_TO_TILT * TILT_STEP, TILT_LIMIT) },
  );
}

/** Absolute pose from a drag, given where it started and how far it has moved. */
export function poseFromDrag(
  start: { yaw: number; tilt: number },
  dx: number,
  dy: number,
) {
  return {
    armYaw: clampTo(start.yaw + dx * SCREEN_TO_YAW * YAW_PER_PIXEL, YAW_LIMIT),
    headTilt: clampTo(start.tilt + dy * SCREEN_TO_TILT * TILT_PER_PIXEL, TILT_LIMIT),
  };
}

export function resetArm() {
  useHero.setState({ armYaw: 0, headTilt: 0, headRotation: 0 });
}
