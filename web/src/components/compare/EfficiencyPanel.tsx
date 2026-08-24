"use client";

import type { Run } from "@/lib/api/rows";
import { REASONS } from "@/lib/compare/provenance";

/**
 * Accuracy against compute — the half of that trade-off Model Doctor can
 * currently answer, and an honest account of the half it cannot.
 *
 * Only input resolution is genuinely available per run. Latency, throughput
 * and memory were measured during the RF-DETR investigation, but those numbers
 * live in script artefacts keyed by configuration name rather than by run id,
 * so attributing them to a run here would be a guess dressed as a measurement.
 * They are listed as unavailable with the reason instead — a visible gap is
 * more useful than a plausible number.
 */

function Line({
  label,
  a,
  b,
  reason,
}: {
  label: string;
  a?: string | null;
  b?: string | null;
  reason?: string;
}) {
  const missing = reason !== undefined;
  return (
    <tr className="border-b border-border/20 last:border-0">
      <td className="py-2.5 px-3 align-top">
        <span className="text-[13px] text-ink">{label}</span>
        {missing && (
          <span className="block text-[10px] text-slate/70 leading-snug mt-0.5 max-w-[380px]">
            {reason}
          </span>
        )}
      </td>
      {[a, b].map((value, index) => (
        <td key={index} className="py-2.5 px-3 text-right align-top">
          {missing || value == null ? (
            <span className="text-slate/70 text-[12px] italic">Not available</span>
          ) : (
            <span className="font-mono text-[13px] text-ink">{value}</span>
          )}
        </td>
      ))}
    </tr>
  );
}

export function EfficiencyPanel({
  labelA,
  labelB,
  runA,
  runB,
}: {
  labelA: string;
  labelB: string;
  runA: Run;
  runB: Run;
}) {
  return (
    <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
      <header className="px-3 py-3 border-b border-border/40">
        <h3 className="text-[13px] font-medium text-ink">Efficiency</h3>
        <p className="text-[11px] text-slate mt-0.5">
          Model Doctor does not currently record timing or memory per run. What it
          does record is shown; the rest says why not.
        </p>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-slate border-b border-border/40 bg-canvas/40">
              <th className="font-medium text-[11px] py-2 px-3">Measurement</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right">{labelA}</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right">{labelB}</th>
            </tr>
          </thead>
          <tbody>
            <Line
              label="Input resolution"
              a={`${runA.image_size} px`}
              b={`${runB.image_size} px`}
            />
            <Line label="Checkpoint size" reason={REASONS.CHECKPOINT} />
            <Line label="Inference latency" reason={REASONS.TIMING} />
            <Line label="Throughput (FPS)" reason={REASONS.TIMING} />
            <Line label="CPU memory" reason={REASONS.MEMORY} />
            <Line label="GPU memory" reason={REASONS.MEMORY} />
          </tbody>
        </table>
      </div>
    </section>
  );
}
