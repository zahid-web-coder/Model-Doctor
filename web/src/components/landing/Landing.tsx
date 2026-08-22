"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { LandingCanvas } from "./LandingCanvas";
import { LandingTimeline } from "./LandingTimeline";
import { DiagnosisMoment, VerificationPanel } from "./DiagnosisMoment";
import { InteractiveControls, ScrollGuide } from "./InteractiveControls";
import { useReducedMotion, useWideViewport } from "@/lib/hero/capabilities";
import { useHero } from "@/lib/hero/store";
import { Logo } from "@/components/shared/Logo";

/**
 * The landing page.
 *
 * One canvas, pinned for the height of the story, with the copy scrolling over
 * it. Five beats, one screen each: factory, inspection, diagnosis,
 * verification, and the hand-off to the dashboard.
 *
 * The canvas is `sticky` rather than `fixed` so it is contained by the story
 * section and stops being painted the moment the section ends. `fixed` would
 * keep it composited underneath everything below it.
 */

const BEAT_COUNT = 5;

function Beat({
  name, align = "left", children,
}: {
  name: string;
  align?: "left" | "right" | "centre";
  children: React.ReactNode;
}) {
  const place =
    align === "right" ? "justify-end" : align === "centre" ? "justify-center" : "justify-start";
  return (
    <div
      data-beat={name}
      className={`absolute inset-0 flex items-center ${place} px-8 md:px-16 lg:px-24 pointer-events-none`}
    >
      <div className="pointer-events-auto">{children}</div>
    </div>
  );
}

export function Landing() {
  const router = useRouter();
  const reduced = useReducedMotion();
  const wide = useWideViewport();
  const [entering, setEntering] = useState(false);

  // One gate, read by the canvas, the copy and the timeline. These used to
  // test `wide === true` and `wide !== false` separately, which left a state —
  // viewport width still undecided — where the beats faded in over a static
  // fallback and the story played with nothing behind it.
  const immersive = wide === true && !reduced;
  const freeOrbit = useHero((s) => s.freeOrbit);
  const [scrolled, setScrolled] = useState(false);

  // Prefetch so the hand-off has nothing to wait for.
  useEffect(() => { router.prefetch("/dashboard"); }, [router]);

  // The scroll hint stays up for the whole story and only retires on the last
  // beat. Hiding it at the first scroll was wrong: there are five beats, each
  // one a full screen, and dropping the cue after the first one leaves a
  // visitor with no signal that four more are below — which is exactly where
  // people stop.
  useEffect(() => {
    if (!immersive) return;
    const onScroll = () => {
      const range = document.body.scrollHeight - window.innerHeight;
      setScrolled(range > 0 && window.scrollY / range > 0.9);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [immersive]);

  // The story and orbit mode cannot both own the camera, so leaving the page
  // in orbit must not carry that into the dashboard hero.
  useEffect(() => () => { useHero.setState({ freeOrbit: false }); }, []);

  /**
   * The hand-off.
   *
   * The camera push finishes, the frame is covered, and only then does the
   * route change. Navigating mid-animation unmounts the canvas while it is
   * still moving, and the white flash that produces is what gives this effect
   * away every time it is done the obvious way.
   */
  const enter = useCallback(() => {
    if (reduced) { router.push("/dashboard"); return; }
    setEntering(true);
    window.setTimeout(() => router.push("/dashboard"), 620);
  }, [router, reduced]);

  return (
    <main className="bg-[#6f6757]">
      {/* Always reachable, never behind the story.

          The bar spans the full width and is `fixed`, so it is transparent to
          the pointer and only its two children take clicks. Left solid, it
          eats every drag that starts in the top strip of the canvas — which is
          where the machine usually is. */}
      <header className="fixed top-0 inset-x-0 z-30 flex items-center justify-between px-6 md:px-10 py-5 pointer-events-none">
        <span className="flex items-center gap-2.5 text-white pointer-events-auto">
          <Logo size={26} title={null} />
          <span className="font-heading text-[15px] tracking-wide">
            Model&nbsp;Doctor
          </span>
        </span>
        <Link
          href="/dashboard"
          onClick={(e) => { e.preventDefault(); enter(); }}
          className="pointer-events-auto rounded-md bg-white/12 hover:bg-white/20 border border-white/20 backdrop-blur-sm px-4 py-2 text-[12px] font-medium text-white transition-colors"
        >
          Launch Dashboard →
        </Link>
      </header>

      <section
        data-landing-scroll
        className="relative"
        style={{ height: immersive ? `${BEAT_COUNT * 100}vh` : "auto" }}
      >
        <div
          className={
            immersive
              ? "sticky top-0 h-screen w-full overflow-hidden"
              : "relative h-[70vh] w-full overflow-hidden"
          }
        >
          <LandingCanvas />

          {/* Scrim. White type on lit warm concrete is not readable without
              one, but a full-frame wash flattens the room the scene exists to
              show. So it is weighted to the floor and to the left, where the
              copy sits, and leaves the upper right — where the machine is —
              nearly clear. */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/65 via-black/15 to-transparent pointer-events-none" />
          {/* Both edges, because the copy alternates sides across the beats —
              hero and inspection sit left, diagnosis right. Weighting only one
              side leaves whichever beat is on the other one unreadable. The
              centre stays clear so the machine is never washed out. */}
          <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-black/5 to-black/55 pointer-events-none" />
          {/* Two soft pools under where the copy actually sits. The linear
              scrims above carry the headings, but body text at 13px lands on
              polished rollers and loses against the specular highlights — and
              darkening the whole frame enough to fix that flattens the room.
              A local pool costs nothing and leaves the machine lit. */}
          <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(46%_38%_at_20%_58%,rgba(0,0,0,0.5),transparent_70%)]" />
          <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(46%_42%_at_78%_52%,rgba(0,0,0,0.45),transparent_70%)]" />

          {/* Beats. Absolutely stacked and cross-faded by GSAP, except under
              reduced motion where they all show at once and stack in flow.
              They clear out of the way in orbit mode: the copy sits over the
              machine you are trying to look at, and its `pointer-events-auto`
              content would swallow the drags meant for the controls. */}
          {immersive && (
            <div
              // `pointer-events-none` here is not enough on its own: each
              // beat's content sets `pointer-events-auto` to stay clickable
              // during the story, and an `auto` child overrides a `none`
              // parent. So orbit mode forces it down the whole subtree —
              // otherwise invisible copy keeps eating drags across the middle
              // and right of the screen.
              className={`absolute inset-0 transition-opacity duration-300 ${
                freeOrbit
                  ? "opacity-0 pointer-events-none [&_*]:pointer-events-none"
                  : "opacity-100"
              }`}
            >
              <Beat name="hero">
                <div className="max-w-[560px]">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-4">
                    Vision failure diagnosis
                  </p>
                  <h1 className="font-heading text-white text-[40px] md:text-[54px] leading-[1.06] mb-5">
                    AI that doesn&rsquo;t just
                    <br />
                    detect failures.
                    <br />
                    <span className="text-[#f0d78a]">It explains them.</span>
                  </h1>
                  <p className="text-[14px] leading-relaxed text-white/65 max-w-[420px]">
                    Every claim carries its evidence: how often a factor appears
                    on failures, how often it appears when the model was right,
                    and whether the difference survives a second run.
                  </p>
                </div>
              </Beat>

              <Beat name="inspection">
                <div className="max-w-[460px]">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-3">
                    Inspection
                  </p>
                  <h2 className="font-heading text-white text-[32px] leading-[1.12] mb-3">
                    Every prediction,
                    <br />
                    against its ground truth.
                  </h2>
                  <p className="text-[13px] leading-relaxed text-white/60">
                    The scanner records what the model saw and where it looked.
                    The heatmap is attention, not a score — the beginning of an
                    explanation rather than the end of one.
                  </p>
                </div>
              </Beat>

              <Beat name="diagnosis" align="right">
                <DiagnosisMoment />
              </Beat>

              <Beat name="verification">
                <VerificationPanel />
              </Beat>

              <Beat name="transition" align="centre">
                <div className="text-center max-w-[520px]">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-3">
                    The dashboard
                  </p>
                  <h2 className="font-heading text-white text-[34px] leading-[1.12] mb-4">
                    Then hand it to the people
                    <br />
                    who have to fix it.
                  </h2>
                  <p className="text-[13px] leading-relaxed text-white/60 mb-7">
                    Failures, clusters, root causes and attention maps — with
                    the arithmetic behind every claim left visible.
                  </p>
                  <button
                    type="button"
                    onClick={enter}
                    className="inline-flex items-center gap-2 rounded-md bg-[#c9a227] hover:bg-[#d8b23a] px-6 py-3 text-[13px] font-medium text-[#1d1a14] transition-colors"
                  >
                    Open the dashboard →
                  </button>
                </div>
              </Beat>
            </div>
          )}
        </div>

        {/* Reduced motion, and anything narrower than a laptop, get the whole
            story as an ordinary document under a still of the scene. Not a
            degraded version — the same words, all readable at once. */}
        {!immersive && (
          <div className="relative bg-[#6f6757] px-6 py-16 flex flex-col gap-16 items-start">
            <div className="max-w-[560px]">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-4">
                Vision failure diagnosis
              </p>
              <h1 className="font-heading text-white text-[34px] leading-[1.08] mb-5">
                AI that doesn&rsquo;t just detect failures.{" "}
                <span className="text-[#f0d78a]">It explains them.</span>
              </h1>
              <p className="text-[14px] leading-relaxed text-white/65">
                Every claim carries its evidence: how often a factor appears on
                failures, how often it appears when the model was right, and
                whether the difference survives a second run.
              </p>
            </div>
            <DiagnosisMoment />
            <VerificationPanel />
            <button
              type="button"
              onClick={enter}
              className="inline-flex items-center gap-2 rounded-md bg-[#c9a227] hover:bg-[#d8b23a] px-6 py-3 text-[13px] font-medium text-[#1d1a14] transition-colors"
            >
              Open the dashboard →
            </button>
          </div>
        )}
      </section>

      <LandingTimeline reduced={!immersive} />

      {/* Operate the machine directly. Only where the canvas is actually
          running — there is nothing to orbit behind a still. */}
      {immersive && <InteractiveControls />}
      {immersive && <ScrollGuide hidden={scrolled || freeOrbit} />}

      {/* The hand-off cover. Painted in the dashboard's own background colour
          so the route change happens between two identical frames. */}
      <div
        aria-hidden
        className={`fixed inset-0 z-40 bg-[#f4efe3] pointer-events-none transition-opacity duration-500 ${
          entering ? "opacity-100" : "opacity-0"
        }`}
      />
    </main>
  );
}
