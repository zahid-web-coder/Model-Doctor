"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";

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

function useWebGLSupport() {
  const [supported, setSupported] = useState<boolean | null>(null);
  useEffect(() => {
    try {
      const canvas = document.createElement("canvas");
      setSupported(
        !!(canvas.getContext("webgl2") || canvas.getContext("webgl")),
      );
    } catch {
      setSupported(false);
    }
  }, []);
  return supported;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * Pause the render loop while the slot is off screen.
 *
 * Verified by scrolling away and watching CPU, not by reading this code —
 * a loop that keeps running looks identical from the outside.
 */
function useOnScreen(ref: React.RefObject<HTMLElement | null>) {
  const [onScreen, setOnScreen] = useState(true);
  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setOnScreen(entry.isIntersecting),
      { rootMargin: "120px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return onScreen;
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
