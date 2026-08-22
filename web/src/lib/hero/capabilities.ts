"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The three questions every canvas on this site has to answer before it
 * mounts: can this browser render it, does the visitor want motion, and is it
 * even on screen.
 *
 * Shared by the dashboard hero and the landing scene. They were written twice
 * first, which is how a fix to the WebGL probe ends up applied to one of them.
 */

/** `null` while undecided — callers must not mount a canvas on a maybe. */
export function useWebGLSupport() {
  const [supported, setSupported] = useState<boolean | null>(null);
  useEffect(() => {
    try {
      const canvas = document.createElement("canvas");
      setSupported(!!(canvas.getContext("webgl2") || canvas.getContext("webgl")));
    } catch {
      setSupported(false);
    }
  }, []);
  return supported;
}

export function useReducedMotion() {
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
 * Whether the element is on screen, so the render loop can be stopped when it
 * is not.
 *
 * Verified by scrolling away and watching CPU, not by reading this code — a
 * loop that keeps running looks identical from the outside.
 */
export function useOnScreen(ref: React.RefObject<HTMLElement | null>) {
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

/**
 * True once the viewport is wide enough to be worth a scroll-driven 3D story.
 *
 * Deliberately a width test rather than a user-agent test: what makes the
 * landing sequence a bad idea on a phone is the pixel count and the thermal
 * budget, and a narrow desktop window has the same problem as a large phone.
 */
export function useWideViewport(minWidth = 900) {
  const [wide, setWide] = useState<boolean | null>(null);
  useEffect(() => {
    const check = () => {
      // A viewport reporting zero width has not been laid out yet — a
      // background tab, a display:none container, an iframe before its first
      // layout. Answering "narrow" there is wrong and, because the answer is
      // then never revisited, permanently disables the scene. Stay undecided
      // and wait for a real measurement.
      const width = window.innerWidth;
      if (width === 0) return;
      setWide(width >= minWidth);
    };
    check();
    // Both, deliberately. matchMedia fires on the threshold crossing, resize
    // fires when a zero-width viewport is finally laid out — and the first
    // measurement can easily be the one that never crosses a threshold.
    const query = window.matchMedia(`(min-width: ${minWidth}px)`);
    query.addEventListener("change", check);
    window.addEventListener("resize", check);
    return () => {
      query.removeEventListener("change", check);
      window.removeEventListener("resize", check);
    };
  }, [minWidth]);
  return wide;
}

/** A stable ref for the observer above, so callers do not have to make one. */
export function useObservedRef<T extends HTMLElement>() {
  return useRef<T>(null);
}
