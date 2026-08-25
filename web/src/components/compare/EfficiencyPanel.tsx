"use client";

import type { Benchmark, Run } from "@/lib/api/rows";
import { REASONS } from "@/lib/compare/provenance";

/**
 * Accuracy against compute — the other half of the trade-off.
 *
 * **Every figure here is device-specific, and that governs the whole panel.**
 * A run records no device: it is resolved at run time, so the same checkpoint
 * on the same data yields a different latency on CPU and on MPS. Each
 * benchmark therefore carries the device it ran on, and this panel refuses to
 * put two devices in the same comparison — it picks a device both runs were
 * measured on, and says so plainly. Showing one model's GPU number beside
 * another's CPU number would be a benchmark of two machines wearing the
 * clothes of a model comparison.
 *
 * These are controlled measurements, not the `inference_ms` a diagnosis run
 * happens to record. That figure includes the first image's lazy kernel
 * compilation and whatever device the run used; this one warms up first,
 * measures a fixed count, and names its device. They are never presented as
 * the same measurement.
 */

const mb = (value: number) =>
  value >= 1024 ? `${(value / 1024).toFixed(2)} GB` : `${Math.round(value)} MB`;

const bytes = (value: number) => mb(value / (1024 * 1024));

/** Devices both runs were benchmarked on, best first. */
export function sharedDevices(a: Benchmark[], b: Benchmark[]): string[] {
  const inB = new Set(b.map((row) => row.device));
  return a
    .map((row) => row.device)
    .filter((device) => inB.has(device))
    .sort();
}

function Row({
  label,
  a,
  b,
  reason,
  lowerIsBetter,
  format = (value) => value.toFixed(1),
  neutral,
}: {
  label: string;
  a?: number | null;
  b?: number | null;
  reason?: string;
  lowerIsBetter?: boolean;
  format?: (value: number) => string;
  /** A setting rather than a measurement: shown, never scored. */
  neutral?: boolean;
}) {
  const comparable =
    !neutral && !reason && typeof a === "number" && typeof b === "number" && a !== b;
  const winner = comparable ? (lowerIsBetter ? (a < b ? "a" : "b") : a > b ? "a" : "b") : null;

  return (
    <tr className="border-b border-border/20 last:border-0">
      <td className="py-2.5 px-3 align-top">
        <span className="text-[13px] text-ink">{label}</span>
        {reason && (
          <span className="block text-[10px] text-slate/70 leading-snug mt-0.5 max-w-[380px]">
            {reason}
          </span>
        )}
      </td>
      {([a, b] as const).map((value, index) => {
        const isWinner = winner === (index === 0 ? "a" : "b");
        return (
          <td key={index} className="py-2.5 px-3 text-right align-top">
            {reason || value == null ? (
              <span className="text-slate/70 text-[12px] italic">Not available</span>
            ) : (
              <>
                <span
                  className={`font-mono text-[13px] ${
                    isWinner ? "text-brass font-semibold" : "text-ink"
                  }`}
                >
                  {format(value)}
                </span>
                {isWinner && <span className="text-brass text-[11px] ml-1.5">✓</span>}
              </>
            )}
          </td>
        );
      })}
    </tr>
  );
}

export function EfficiencyPanel({
  labelA,
  labelB,
  runA,
  runB,
  benchmarksA,
  benchmarksB,
}: {
  labelA: string;
  labelB: string;
  runA: Run;
  runB: Run;
  benchmarksA: Benchmark[];
  benchmarksB: Benchmark[];
}) {
  const shared = sharedDevices(benchmarksA, benchmarksB);
  const device = shared[0] ?? null;
  const a = device ? benchmarksA.find((row) => row.device === device) : undefined;
  const b = device ? benchmarksB.find((row) => row.device === device) : undefined;

  const num = (row: Benchmark | undefined, key: string): number | null => {
    const value = row?.measurements?.[key];
    return typeof value === "number" ? value : null;
  };

  // Three distinct situations, and they must not be collapsed into one
  // "unavailable": nobody measured, only one side measured, or both measured
  // but never on the same device.
  const neither = benchmarksA.length === 0 || benchmarksB.length === 0;
  const missing = neither
    ? REASONS.NOT_BENCHMARKED
    : device === null
      ? REASONS.DEVICE_MISMATCH
      : undefined;

  const devicesOf = (rows: Benchmark[]) =>
    rows.length ? rows.map((row) => row.device).join(", ") : "none";

  return (
    <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
      <header className="px-3 py-3 border-b border-border/40">
        <h3 className="text-[13px] font-medium text-ink">Efficiency</h3>
        <p className="text-[11px] text-slate mt-0.5">
          {device ? (
            <>
              Measured on <span className="font-mono text-ink">{device}</span>, the
              same device for both. Latency and memory are properties of a model on
              a device, so only like is compared with like.
            </>
          ) : neither ? (
            "Not benchmarked. Input resolution is a run setting and is shown; everything else needs a controlled measurement."
          ) : (
            <>
              No shared device: {labelA} was measured on{" "}
              <span className="font-mono">{devicesOf(benchmarksA)}</span> and {labelB}{" "}
              on <span className="font-mono">{devicesOf(benchmarksB)}</span>. Those
              figures are not comparable, so none is declared a winner.
            </>
          )}
        </p>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-slate border-b border-border/40 bg-canvas/40">
              <th className="font-medium text-[11px] py-2 px-3">Measurement</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right max-w-[180px] truncate">
                {labelA}
              </th>
              <th className="font-medium text-[11px] py-2 px-3 text-right max-w-[180px] truncate">
                {labelB}
              </th>
            </tr>
          </thead>
          <tbody>
            {/* A run setting, not a measurement. Shown for context and never
                scored: a higher resolution is a different configuration, not a
                better model, and marking one as the winner would say otherwise. */}
            <Row
              label="Input resolution"
              a={runA.image_size}
              b={runB.image_size}
              format={(value) => `${value} px`}
              neutral
            />
            <Row
              label="Checkpoint size"
              a={num(a, "checkpoint_bytes")}
              b={num(b, "checkpoint_bytes")}
              reason={missing ? REASONS.CHECKPOINT : undefined}
              format={bytes}
              lowerIsBetter
            />
            <Row
              label="Inference latency (ms)"
              a={num(a, "latency_mean_ms")}
              b={num(b, "latency_mean_ms")}
              reason={missing}
              lowerIsBetter
            />
            <Row
              label="Median latency (ms)"
              a={num(a, "latency_median_ms")}
              b={num(b, "latency_median_ms")}
              reason={missing}
              lowerIsBetter
            />
            <Row
              label="Throughput (FPS)"
              a={num(a, "fps")}
              b={num(b, "fps")}
              reason={missing}
              format={(value) => value.toFixed(2)}
            />
            <Row
              label="Peak process memory"
              a={num(a, "rss_peak_mb")}
              b={num(b, "rss_peak_mb")}
              reason={missing}
              format={mb}
              lowerIsBetter
            />
            <Row
              label="GPU memory"
              a={num(a, "gpu_driver_mb")}
              b={num(b, "gpu_driver_mb")}
              reason={
                missing ??
                (num(a, "gpu_driver_mb") === null && num(b, "gpu_driver_mb") === null
                  ? REASONS.NO_GPU_COUNTER
                  : undefined)
              }
              format={mb}
              lowerIsBetter
            />
            <Row
              label="Cold start (s)"
              a={num(a, "cold_start_s")}
              b={num(b, "cold_start_s")}
              reason={missing}
              lowerIsBetter
            />
          </tbody>
        </table>
      </div>
      {shared.length > 1 && (
        <p className="px-3 pb-3 text-[10px] text-slate/70">
          Also benchmarked on {shared.slice(1).join(", ")}.
        </p>
      )}
    </section>
  );
}
