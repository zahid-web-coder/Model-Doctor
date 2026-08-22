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
 * The still the page falls back to: no WebGL, mobile, or while the chunk
 * loads. A warm concrete gradient rather than a blank screen, because the
 * canvas is the full height of the viewport and an empty one reads as a
 * failure to load.
 */
export function LandingFallback({ label }: { label?: string }) {
  return (
    <div
      className="absolute inset-0 bg-[radial-gradient(120%_90%_at_70%_15%,#a2988a_0%,#8a8071_45%,#5f5849_100%)]"
      role="img"
      aria-label="Model Doctor inspection line"
    >
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
