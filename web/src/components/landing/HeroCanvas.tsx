"use client";

import dynamic from "next/dynamic";
import { useRef } from "react";
import { useWebGLSupport, useReducedMotion, useOnScreen } from "@/lib/hero/capabilities";

/**
 * The boundary that keeps three out of the server bundle.
 *
 * `next/dynamic` with `ssr: false` has to be called from a client component in
 * the App Router — doing it from a server component is a build error, and the
 * error does not say so plainly. This file exists for that one reason, so the
 * page underneath can stay a server component.
 */
const HeroModel = dynamic(() => import("./HeroModel"), {
  ssr: false,
  loading: () => <HeroFallback label="Loading scene" />,
});

/**
 * What fills the slot when the canvas cannot or should not run: while the
 * chunk loads, and permanently when WebGL is unavailable.
 *
 * Not a blank box. The slot has a fixed height in the Overview grid, so an
 * empty one collapses the layout around it and reads as a broken panel.
 */
function HeroFallback({ label }: { label?: string }) {
  return (
    <div
      className="w-full h-full bg-gradient-to-br from-[#8a8071] via-[#7b7365] to-[#6f6757]"
      role="img"
      aria-label="Model Doctor inspection machine"
    >
      {label && <span className="sr-only">{label}</span>}
    </div>
  );
}

export function HeroCanvas({ className }: { className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const webgl = useWebGLSupport();
  const reduced = useReducedMotion();
  const onScreen = useOnScreen(ref);

  return (
    <div ref={ref} className={className}>
      {webgl === false ? (
        <HeroFallback />
      ) : webgl === null ? (
        // Still deciding. Render the fallback rather than the canvas so a
        // browser without WebGL never briefly mounts one.
        <HeroFallback label="Preparing scene" />
      ) : (
        // Reduced motion renders the real scene and freezes it. A still frame
        // of the actual machine, not a blank space.
        <HeroModel paused={!onScreen} animate={!reduced} />
      )}
    </div>
  );
}
