"use client";

import dynamic from "next/dynamic";
import { useRef } from "react";
import {
  useWebGLSupport, useReducedMotion, useOnScreen, useWideViewport,
} from "@/lib/hero/capabilities";
import { useHero } from "@/lib/hero/store";

/**
 * The landing scene's boundary. Same reason as `HeroCanvas`: `next/dynamic`
 * with `ssr: false` has to be called from a client component, and three must
 * never reach the server bundle.
 *
 * The dashboard's `HeroModel` is a separate dynamic import, so visiting the
 * landing page never downloads it and vice versa — the two scenes share the
 * GLB and the animation modules, not a chunk.
 */
const LandingScene = dynamic(() => import("./LandingScene"), {
  ssr: false,
  loading: () => <LandingFallback label="Loading scene" />,
});

/**
 * The still the page falls back to: no WebGL, a phone, or while the chunk
 * loads.
 *
 * **A frame of the real scene, not a stand-in.** Phones are deliberately kept
 * off the live scene (see `useWideViewport`: pixel count and thermal budget),
 * and until now they got a warm gradient in its place — the whole landing
 * page without its subject. `landing-poster.webp` is the opening shot
 * captured from the live canvas itself, so a phone sees the actual line for
 * the cost of one 80 KB image, and on desktop the frame shown while the scene
 * loads is the frame the scene opens on: no visible swap.
 *
 * Cropped towards the robot: on a portrait screen `cover` keeps the middle of
 * a landscape frame, and the subject sits right of centre, beside the copy.
 * The gradient stays underneath in case the image fails.
 *
 * **Re-capture it when the scene changes**, or the still and the live opening
 * drift apart. It is the canvas at `scroll.t = 0`, 1440 × 900 CSS pixels,
 * saved as WebP at quality 0.82.
 */
export function LandingFallback({ label }: { label?: string }) {
  return (
    <div
      className="absolute inset-0 bg-[radial-gradient(120%_90%_at_70%_15%,#a2988a_0%,#8a8071_45%,#5f5849_100%)]"
      role="img"
      aria-label="Model Doctor inspection line"
    >
      {/* Decorative: the container carries the accessible name. */}
      {/* eslint-disable-next-line @next/next/no-img-element -- a fixed, pre-sized still; next/image would add a loader for nothing */}
      <img
        src="/landing-poster.webp"
        alt=""
        aria-hidden
        className="absolute inset-0 h-full w-full object-cover"
        style={{ objectPosition: "62% 55%" }}
        decoding="async"
      />
      {label && <span className="sr-only">{label}</span>}
    </div>
  );
}

export function LandingCanvas() {
  const ref = useRef<HTMLDivElement>(null);
  const webgl = useWebGLSupport();
  const reduced = useReducedMotion();
  const wide = useWideViewport();
  const onScreen = useOnScreen(ref);
  const freeOrbit = useHero((s) => s.freeOrbit);

  // Undecided on either question renders the still. Mounting a canvas on a
  // maybe is how a phone ends up briefly running the whole scene.
  const run = webgl === true && wide === true;

  return (
    <div ref={ref} className="absolute inset-0">
      {run ? (
        <LandingScene paused={!onScreen} animate={!reduced} interactive={freeOrbit} />
      ) : (
        <LandingFallback label={webgl === null || wide === null ? "Preparing scene" : undefined} />
      )}
    </div>
  );
}
