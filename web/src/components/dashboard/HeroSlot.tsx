"use client";

import { HeroCanvas } from "@/components/landing/HeroCanvas";
import { useRun } from "@/lib/run-context";

/**
 * The hero.
 *
 * The slot keeps its exact box — aspect, radius, overlays, and the copy that
 * sits on top — and only what fills it changes. The R3F scene replaced the
 * video here; `HeroCanvas` is behind `next/dynamic` with `ssr: false`, so
 * three never reaches the server bundle, and it carries its own no-WebGL and
 * reduced-motion paths.
 *
 * Client component because the call to action follows the selected run. It
 * used to be hardcoded to `/runs/1287/failures` — a run that does not exist,
 * so the Overview's primary action 500'd and ignored run selection entirely.
 */
export function HeroSlot() {
  const { runId } = useRun();

  return (
    <div className="relative rounded-xl overflow-hidden border border-border/40 bg-panel-dark min-h-[320px]">
      <div className="absolute inset-0">
        <HeroCanvas className="w-full h-full" />
      </div>

      {/* Overlays live outside the canvas so the fill can be swapped without
          touching the copy that sits on it. */}
      <div className="absolute inset-0 bg-gradient-to-r from-canvas via-canvas/70 to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-t from-canvas/60 to-transparent pointer-events-none" />

      <div className="relative h-full flex flex-col justify-center p-8 max-w-[460px]">
        <h1 className="font-heading text-ink leading-[1.05] text-[42px] mb-4">
          Diagnose.<br />Understand.<br />
          <span className="text-slate/70">Improve.</span>
        </h1>
        <p className="text-[14px] leading-relaxed text-slate mb-6 max-w-[330px]">
          Find why a vision model fails, not just how often — with the evidence
          for every claim.
        </p>
        <div className="flex items-center gap-3">
          <a
            href={`/runs/${runId}/failures`}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-brass/90 text-white text-[13px] font-medium hover:bg-brass transition-colors"
          >
            Explore Failures →
          </a>
          <a
            href="/runs"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md border border-border/60 bg-card/80 text-ink text-[13px] font-medium hover:border-brass/50 transition-colors"
          >
            Run History
          </a>
        </div>
      </div>
    </div>
  );
}
