"use client";

import { useEffect } from "react";
import { useMonitorHover } from "@/lib/hero/scan";
import { DIAGNOSIS } from "@/lib/hero/landing";

/**
 * The verification monitor's figures, at a size that can be read.
 *
 * On the machine the screen is a few pixels tall at any orbit distance. In the
 * 360° view, hovering it shows the same figures here — from `DIAGNOSIS`, the
 * same measured constants the screen texture and the verification beat draw,
 * so the three can never disagree.
 *
 * A readout, not a control: it ignores the pointer, so it can never sit
 * between the visitor and the scene they are orbiting.
 */
export function MonitorCard() {
  const hover = useMonitorHover((s) => s.hover);

  // Leaving the 360° view unmounts this; clear the flag so the next visit does
  // not open with a stale readout before the pointer has moved.
  useEffect(() => () => { useMonitorHover.setState({ hover: false }); }, []);

  if (!hover) return null;
  const { factor, runs } = DIAGNOSIS.replication;

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed bottom-6 left-6 z-30 w-[300px] rounded-lg border border-white/15 bg-black/55 backdrop-blur-md p-4 text-white"
    >
      <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/45 mb-2">
        Verification monitor
      </p>
      <p className="text-[13px] text-white/85 mb-3">
        Does <span className="font-mono text-white">{factor}</span> hold on a
        second, independent run?
      </p>
      <table className="w-full text-[12px] mb-3">
        <thead>
          <tr className="text-white/40 text-left">
            <th className="font-normal pb-1">Run</th>
            <th className="font-normal pb-1 text-right">Findings</th>
            <th className="font-normal pb-1 text-right">Lift</th>
            <th className="font-normal pb-1 text-right">p</th>
          </tr>
        </thead>
        <tbody className="font-mono text-white/90">
          {runs.map((run) => (
            <tr key={run.label}>
              <td className="py-0.5 font-sans">{run.label}</td>
              <td className="py-0.5 text-right">{run.findings}</td>
              <td className="py-0.5 text-right">{run.lift}×</td>
              <td className="py-0.5 text-right">{run.p}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[12px] text-[#9fdcae]">
        Replicated — same direction, significant on both.
      </p>
    </div>
  );
}
