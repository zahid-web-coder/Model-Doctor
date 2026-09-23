import { useHero } from "./store";
import { cancelScan } from "./scan";

/**
 * How a pointer or a button maps onto the arm.
 *
 * **One definition, read by both the drag handler and the nudge buttons.**
 * They were written separately and disagreed: the buttons swung the arm the
 * opposite way to their own arrows, and the drag ran opposite to the buttons.
 *
 * ## Why the direction cannot be a constant
 *
 * `armYaw` rotates `ArmBase` about three's +Y, which carries the scanner along
 * a circle. Whether that circle takes it left or right *on screen* depends
 * entirely on where the camera is standing — and this view lets the visitor
 * put the camera anywhere. From the default hero angle a positive yaw reads as
 * screen-right; orbit round to the far side of the line and the same rotation
 * reads as screen-left.
 *
 * So the first two attempts at this were both wrong in the same way: they
 * picked a constant. The first matched no angle, the second matched the
 * starting angle and inverted the moment you orbited past 90°. The sign has to
 * be measured against the camera that is actually there, which is what
 * `screen.yawSign` is — recomputed every frame by `ArmDragger` from the
 * scanner's tangential direction and the camera's right vector.
 *
 * Tilt has the same problem and gets the same treatment. The head pitches
 * about its own transverse axis, and which way that reads on screen depends on
 * both where the camera is and where the arm has been swung to — so it is
 * measured against the camera too, not assumed.
 */

export const YAW_LIMIT = 1;
export const TILT_LIMIT = 0.4;

/**
 * Live mapping from screen-right to the sign of `armYaw`, kept up to date by
 * the scene. Module state rather than React state because it is written every
 * frame and read inside event handlers — a store would re-render the controls
 * sixty times a second to change a number nothing renders.
 */
export const screen = { yawSign: 1, tiltSign: -1 };

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
  // Direct arm input ends a plate scan: the scan eases the arm to rest, and
  // two writers on one hinge is the jitter the ownership rule exists to stop.
  cancelScan();
  useHero.setState((s) =>
    axis === "yaw"
      ? { armYaw: clampTo(s.armYaw + screenDirection * screen.yawSign * YAW_STEP, YAW_LIMIT) }
      : { headTilt: clampTo(s.headTilt + screenDirection * screen.tiltSign * TILT_STEP, TILT_LIMIT) },
  );
}

/** Absolute pose from a drag, given where it started and how far it has moved. */
export function poseFromDrag(
  start: { yaw: number; tilt: number },
  dx: number,
  dy: number,
) {
  return {
    armYaw: clampTo(start.yaw + dx * screen.yawSign * YAW_PER_PIXEL, YAW_LIMIT),
    headTilt: clampTo(start.tilt + dy * screen.tiltSign * TILT_PER_PIXEL, TILT_LIMIT),
  };
}

export function resetArm() {
  cancelScan();
  useHero.setState({ armYaw: 0, headTilt: 0, headRotation: 0 });
}
