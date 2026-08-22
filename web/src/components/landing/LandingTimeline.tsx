"use client";

import { useEffect } from "react";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import gsap from "gsap";
import { scroll, BEATS, BEAT_ORDER, clamp01, ease } from "@/lib/hero/landing";

/**
 * Scroll, wired to the scene.
 *
 * **One ScrollTrigger writes one number, and everything else is a function of
 * it** — including the copy. The camera, beam, plate, station and every beat's
 * opacity all read `scroll.t`.
 *
 * The copy used to have its own per-beat triggers, each with its own start and
 * end computed from a separately measured scroll range. That is two
 * measurements of the same thing, and they drifted: the camera would be most
 * of the way into the inspection push while the hero headline was still at
 * full opacity, which is exactly the "these two shots aren't connected"
 * feeling. There is now one measurement and one clock, so the copy cannot
 * disagree with the camera about where the page is.
 *
 * Reduced motion gets no trigger at all: `t` is pinned, every beat's copy is
 * shown at once, and the page becomes an ordinary document.
 */

/** Fade in over the first fifth of a beat, hold, fade out over the last. */
const EDGE = 0.2;

function opacityFor(t: number, index: number, count: number, from: number, to: number) {
  const k = clamp01((t - from) / (to - from));
  // The first beat has no fade-in — at scroll zero a scrubbed fade would leave
  // the headline invisible on arrival. The last has no fade-out; its call to
  // action is what the page is for and there is nothing after it.
  const rising = index === 0 ? 1 : ease(clamp01(k / EDGE));
  const falling = index === count - 1 ? 1 : ease(clamp01((1 - k) / EDGE));
  return Math.min(rising, falling);
}

export function LandingTimeline({ reduced }: { reduced: boolean }) {
  useEffect(() => {
    const beats = Array.from(document.querySelectorAll<HTMLElement>("[data-beat]"));

    if (reduced) {
      scroll.t = BEATS.diagnosis[0];
      beats.forEach((el) => { el.style.opacity = "1"; });
      return;
    }

    gsap.registerPlugin(ScrollTrigger);
    const container = document.querySelector<HTMLElement>("[data-landing-scroll]");
    if (!container) return;

    const ordered = BEAT_ORDER
      .map((name) => ({ name, el: beats.find((b) => b.dataset.beat === name) }))
      .filter((b): b is { name: typeof BEAT_ORDER[number]; el: HTMLElement } => !!b.el);

    const apply = (t: number) => {
      scroll.t = t;
      ordered.forEach(({ name, el }, i) => {
        const [from, to] = BEATS[name];
        const o = opacityFor(t, i, ordered.length, from, to);
        el.style.opacity = String(o);
        // A beat at zero opacity still covers the canvas and would swallow a
        // drag meant for the machine.
        el.style.pointerEvents = o < 0.02 ? "none" : "";
        // A small lift, tied to the same number rather than to its own tween.
        el.style.transform = `translateY(${(1 - o) * 14}px)`;
      });
    };

    const master = ScrollTrigger.create({
      trigger: container,
      start: "top top",
      end: "bottom bottom",
      // Enough to take the step out of a wheel notch, little enough that the
      // camera still feels attached to the scrollbar.
      scrub: 0.6,
      onUpdate: (self) => apply(self.progress),
    });

    apply(0);

    // The layout settles after fonts and the canvas mount; without this the
    // trigger is measured against a page that is about to change height.
    const settle = window.setTimeout(() => ScrollTrigger.refresh(), 200);

    return () => {
      window.clearTimeout(settle);
      master.kill();
    };
  }, [reduced]);

  return null;
}
