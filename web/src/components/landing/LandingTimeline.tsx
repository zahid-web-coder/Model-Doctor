"use client";

import { useEffect } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { scroll, BEAT_ORDER, BEATS } from "@/lib/hero/landing";

/**
 * Scroll, wired to the scene.
 *
 * **One ScrollTrigger writes one number.** `scroll.t` is the only thing the
 * scrub touches; the camera, the beam, the plate and the station all derive
 * from it inside `useFrame`. Driving each of them with its own tween would
 * work until two of them disagreed about where a scrub landed, and that
 * disagreement only shows up mid-scroll, which is the worst place to debug it.
 *
 * The copy is a separate concern and does use its own triggers — but those
 * write DOM opacity, never scene state, so there is still exactly one owner
 * per value.
 *
 * Reduced motion gets no triggers at all: `t` is pinned, every beat's copy is
 * shown at once, and the page becomes an ordinary document.
 */
export function LandingTimeline({ reduced }: { reduced: boolean }) {
  useEffect(() => {
    const beats = Array.from(
      document.querySelectorAll<HTMLElement>("[data-beat]"),
    );

    if (reduced) {
      // A still frame of the real scene, with the whole story readable.
      scroll.t = BEATS.diagnosis[0];
      beats.forEach((el) => { el.style.opacity = "1"; });
      return;
    }

    gsap.registerPlugin(ScrollTrigger);
    const container = document.querySelector<HTMLElement>("[data-landing-scroll]");
    if (!container) return;

    const master = ScrollTrigger.create({
      trigger: container,
      start: "top top",
      end: "bottom bottom",
      scrub: 0.6,
      onUpdate: (self) => { scroll.t = self.progress; },
    });

    // Each beat's copy fades up as its slice of the page arrives and fades out
    // before the next one. Held at full opacity across the middle so there is
    // a moment to actually read it.
    //
    // The first and last beats are the exceptions, and getting them wrong is
    // invisible until you load the page: the hero has no fade-in, because at
    // scroll zero a scrubbed timeline sits at progress zero and the headline
    // would start invisible on arrival. The last beat has no fade-out, because
    // its call to action is the thing the whole page is for and there is
    // nothing after it to make room for.
    const height = container.scrollHeight - window.innerHeight;
    const fades = beats.map((el, index) => {
      const name = el.dataset.beat as (typeof BEAT_ORDER)[number];
      const [from, to] = BEATS[name] ?? [0, 1];
      const isFirst = index === 0;
      const isLast = index === beats.length - 1;

      gsap.set(el, { opacity: isFirst ? 1 : 0, y: isFirst ? 0 : 18 });

      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: container,
          start: `top+=${from * height} top`,
          end: `top+=${to * height} top`,
          scrub: 0.4,
        },
      });
      if (!isFirst) tl.to(el, { opacity: 1, y: 0, duration: 0.25, ease: "power2.out" });
      tl.to(el, { opacity: 1, duration: isFirst || isLast ? 0.75 : 0.5 });
      if (!isLast) tl.to(el, { opacity: 0, y: -14, duration: 0.25, ease: "power2.in" });
      return tl;
    });

    // The layout settles after fonts and the canvas mount; without this the
    // trigger positions are measured against a page that is about to change
    // height, and every beat lands slightly early.
    const settle = window.setTimeout(() => ScrollTrigger.refresh(), 200);

    return () => {
      window.clearTimeout(settle);
      fades.forEach((tl) => {
        tl.scrollTrigger?.kill();
        tl.kill();
      });
      master.kill();
    };
  }, [reduced]);

  return null;
}
