"use client";

import { useCallback, useEffect, useRef } from "react";
import { useHero } from "@/lib/hero/store";

/**
 * The controls that turn the story into something you can operate.
 *
 * Entering orbit mode is an ownership handover, not a mode flag: the scroll
 * timeline stops writing the camera and the arm, and the pointer starts. §3's
 * rule holds either way — exactly one system writes each value, and which one
 * is explicit rather than emergent.
 *
 * **Dragging the machine is the primary control.** These buttons are the
 * keyboard-and-accessibility path and a hint that the arm moves at all; they
 * repeat while held, because a nudge per click took a dozen clicks to cross
 * the arm's range and was miserable to use.
 *
 * Page scroll is locked while orbiting. Without it the story keeps advancing
 * underneath, so leaving orbit mode drops you somewhere you never scrolled to.
 */

const YAW_STEP = 0.05;
const YAW_LIMIT = 1;
const TILT_STEP = 0.022;
const TILT_LIMIT = 0.4;
const REPEAT_MS = 40;
const HOLD_DELAY_MS = 220;

const clamp = (v: number, limit: number) => Math.max(-limit, Math.min(limit, v));

/**
 * A button that fires once on press and then repeats while held.
 *
 * The initial delay is what makes a single tap still mean "one step" — without
 * it, every press produces a burst and fine adjustment becomes impossible.
 */
function HoldButton({
  onStep, label, children, wide,
}: {
  onStep: () => void;
  label: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  const timers = useRef<{ delay?: number; repeat?: number }>({});

  const stop = useCallback(() => {
    window.clearTimeout(timers.current.delay);
    window.clearInterval(timers.current.repeat);
    timers.current = {};
  }, []);

  const start = useCallback(() => {
    onStep();
    stop();
    timers.current.delay = window.setTimeout(() => {
      timers.current.repeat = window.setInterval(onStep, REPEAT_MS);
    }, HOLD_DELAY_MS);
  }, [onStep, stop]);

  // A pointerup outside the button, or an unmount mid-hold, must not leave an
  // interval running forever.
  useEffect(() => stop, [stop]);

  return (
    <button
      type="button"
      aria-label={label}
      onPointerDown={start}
      onPointerUp={stop}
      onPointerLeave={stop}
      onPointerCancel={stop}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onStep(); }}
      className={`h-9 ${wide ? "px-3" : "w-9"} grid place-items-center rounded-md border border-white/20 bg-black/40 text-white/85 text-[13px] backdrop-blur-sm hover:bg-black/60 hover:border-white/35 active:bg-white/20 transition-colors select-none touch-none`}
    >
      {children}
    </button>
  );
}

export function InteractiveControls() {
  const freeOrbit = useHero((s) => s.freeOrbit);

  useEffect(() => {
    if (!freeOrbit) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [freeOrbit]);

  // Escape leaves orbit mode. A control that traps you is worse than no
  // control, and the exit button can end up behind something.
  useEffect(() => {
    if (!freeOrbit) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") useHero.setState({ freeOrbit: false });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [freeOrbit]);

  const nudgeYaw = (direction: number) => () =>
    useHero.setState((s) => ({ armYaw: clamp(s.armYaw + direction * YAW_STEP, YAW_LIMIT) }));
  const nudgeTilt = (direction: number) => () =>
    useHero.setState((s) => ({ headTilt: clamp(s.headTilt + direction * TILT_STEP, TILT_LIMIT) }));

  if (!freeOrbit) {
    return (
      <button
        type="button"
        onClick={() => useHero.setState({ freeOrbit: true })}
        className="fixed bottom-6 right-6 z-30 inline-flex items-center gap-2 rounded-md border border-white/20 bg-black/40 backdrop-blur-sm px-4 py-2.5 text-[12px] font-medium text-white hover:bg-black/60 hover:border-white/35 transition-colors"
      >
        <span aria-hidden>⟳</span> 360° view
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-30 flex flex-col items-end gap-3">
      <div className="rounded-lg border border-white/15 bg-black/45 backdrop-blur-sm p-3 w-[212px]">
        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/45 mb-2">
          Operate the arm
        </p>
        {/* The drag is the control; the buttons below are the fallback. Saying
            so is the whole affordance, since the grab volume is invisible. */}
        <p className="text-[11px] leading-relaxed text-white/60 mb-3">
          Drag the machine to swing and tilt it. Drag anywhere else to orbit.
        </p>

        <div className="flex items-center gap-2 mb-2">
          <span className="text-[11px] text-white/50 w-9">Swing</span>
          <HoldButton onStep={nudgeYaw(-1)} label="Swing arm left">←</HoldButton>
          <HoldButton onStep={nudgeYaw(1)} label="Swing arm right">→</HoldButton>
        </div>

        <div className="flex items-center gap-2 mb-3">
          <span className="text-[11px] text-white/50 w-9">Tilt</span>
          <HoldButton onStep={nudgeTilt(1)} label="Tilt scanner up">↑</HoldButton>
          <HoldButton onStep={nudgeTilt(-1)} label="Tilt scanner down">↓</HoldButton>
        </div>

        <HoldButton
          wide
          onStep={() => useHero.setState({ armYaw: 0, headTilt: 0, headRotation: 0 })}
          label="Reset the arm to its modelled pose"
        >
          Reset
        </HoldButton>

        <p className="text-[10px] leading-relaxed text-white/35 mt-3">
          Scroll to zoom · Esc to exit
        </p>
      </div>

      <button
        type="button"
        onClick={() => useHero.setState({ freeOrbit: false })}
        className="inline-flex items-center gap-2 rounded-md border border-white/25 bg-[#c9a227] px-4 py-2.5 text-[12px] font-medium text-[#1d1a14] hover:bg-[#d8b23a] transition-colors"
      >
        Exit 360° view
      </button>
    </div>
  );
}

/**
 * A small persistent hint that the page is scroll-driven.
 *
 * A pinned canvas with no visible scrollbar movement looks like a static
 * image, and the entire story is behind the scroll — so without this the most
 * likely outcome is that a visitor looks at beat one and leaves.
 *
 * The arrow is the part that carries it. The first version was a mouse-wheel
 * outline with a travelling dot, which is a convention people read only if
 * they already know it; a chevron pointing down is not ambiguous.
 */
export function ScrollGuide({ hidden }: { hidden?: boolean }) {
  return (
    <div
      aria-hidden
      className={`fixed left-1/2 -translate-x-1/2 bottom-6 z-20 flex flex-col items-center gap-1.5 transition-opacity duration-500 ${
        hidden ? "opacity-0" : "opacity-100"
      }`}
    >
      <span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-white/50">
        Scroll
      </span>
      <svg
        width="22" height="30" viewBox="0 0 22 30" fill="none"
        className="animate-[scrollhint_1.8s_ease-in-out_infinite]"
      >
        <path
          d="M4 9 L11 16 L18 9" stroke="rgba(255,255,255,0.85)" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
        />
        <path
          d="M4 18 L11 25 L18 18" stroke="rgba(255,255,255,0.4)" strokeWidth="2"
          strokeLinecap="round" strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}
