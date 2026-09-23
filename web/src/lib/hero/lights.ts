import { create } from "zustand";

/**
 * The hall lights: on or off, from the switch by the service door or the
 * 360° panel.
 *
 * **One level, many lights.** Every light that dims — the key spot, the fill,
 * the high-bay lamps and their shafts, the image-based lighting — scales by
 * the same `hallLevel.value`, advanced once per frame by `HallLighting`. So
 * the room changes as one, and no light can lag or lead the others.
 *
 * **What stays on when they are off** is what really would: the guard's
 * torch, the verification monitor, the machine's indicator lamps, the CCTV
 * recording lights and the exit sign. A little moonlight-cool ambient keeps
 * the room legible rather than black.
 */
export const useHallLights = create<{ on: boolean }>()(() => ({ on: true }));

export function toggleHallLights() {
  useHallLights.setState((s) => ({ on: !s.on }));
}

/** 0 (off) … 1 (on). Mutable on purpose: read in render loops, not React. */
export const hallLevel = { value: 1 };

/** Seconds for the lights to die away when switched off. */
export const FADE_OUT = 0.18;
/** Seconds of start-up flicker when switched on. */
export const STRIKE = 0.9;

/**
 * High-bay lamps do not fade up; they strike. A couple of stutters, a dim
 * moment, then full output — the flicker is what sells a switch being thrown.
 * `t` is seconds since switching on; returns the level at that moment.
 */
export function strike(t: number): number {
  if (t <= 0) return 0;
  if (t >= STRIKE) return 1;
  if (t < 0.06) return 0.85;
  if (t < 0.14) return 0.05;
  if (t < 0.2) return 0.7;
  if (t < 0.32) return 0.1;
  // The warm-up: from half output to full over the rest of the strike.
  const u = (t - 0.32) / (STRIKE - 0.32);
  return 0.5 + 0.5 * u * u * (3 - 2 * u);
}

/**
 * The level one frame on. Switching off fades out over `FADE_OUT`; switching
 * on follows `strike`, timed from `since` (seconds since the switch was
 * thrown). `dt` is the frame time.
 */
export function stepLevel(level: number, on: boolean, since: number, dt: number): number {
  if (on) return strike(since);
  return Math.max(0, level - dt / FADE_OUT);
}
