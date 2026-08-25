"use client";

import type { RunMetrics } from "@/lib/compare/metrics";
import { winner } from "@/lib/compare/metrics";
import type { Benchmark, FactorRate } from "@/lib/api/rows";
import { sharedDevices } from "./EfficiencyPanel";
import { isAvailable } from "@/lib/compare/provenance";

/**
 * Which model to ship, argued per dimension.
 *
 * **Deliberately not a single score.** Combining accuracy, failure profile and
 * robustness into one number requires weights, and there is no defensible
 * weighting independent of the deployment: a model that finds more objects but
 * hallucinates more is the right choice in one setting and the wrong one in
 * another. A composite would hide exactly the trade-off the reader came here
 * to make.
 *
 * So each dimension is decided separately from stored evidence, and the closing
 * sentence states the trade-off rather than resolving it. Dimensions Model
 * Doctor cannot currently measure are listed as undecided, not omitted — their
 * absence is part of the answer.
 */

const SIGNIFICANCE = 0.05;

interface Dimension {
  name: string;
  winner: "a" | "b" | null;
  evidence: string;
  undecided?: boolean;
}

function robustness(
  a: FactorRate[],
  b: FactorRate[],
): Dimension[] {
  const mapA = new Map(a.map((r) => [r.factor, r]));
  const mapB = new Map(b.map((r) => [r.factor, r]));
  const shared = [...mapA.keys()].filter((k) => mapB.has(k)).sort();

  const out: Dimension[] = [];
  for (const factor of shared) {
    const ra = mapA.get(factor)!;
    const rb = mapB.get(factor)!;
    if (ra.lift === null || rb.lift === null) continue;

    const aSig = ra.p_value !== null && ra.p_value < SIGNIFICANCE;
    const bSig = rb.p_value !== null && rb.p_value < SIGNIFICANCE;
    if (!aSig && !bSig) continue; // neither shows an association worth calling

    const lower = Math.min(ra.lift, rb.lift);
    const higher = Math.max(ra.lift, rb.lift);
    if (lower <= 0.01 || higher / lower < 1.5) continue; // not materially apart

    out.push({
      name: `Robustness — ${factor.replace(/_/g, " ")}`,
      winner: ra.lift < rb.lift ? "a" : "b",
      evidence: `lift ${ra.lift.toFixed(2)}× against ${rb.lift.toFixed(2)}×; lower is better`,
    });
  }
  return out;
}

/**
 * Compute cost, decided only when both runs were measured on the same device.
 *
 * A run stores no device, so latency and memory could otherwise be compared
 * across machines and reported as a property of the models. Where there is no
 * shared device the dimension stays undecided and says why — an unanswered
 * question, not an answered one with a caveat.
 */
function compute(a: Benchmark[], b: Benchmark[]): Dimension[] {
  if (!a.length || !b.length) {
    return [
      {
        name: "Compute cost",
        winner: null,
        undecided: true,
        evidence: "not benchmarked on both sides",
      },
    ];
  }

  const device = sharedDevices(a, b)[0];
  if (!device) {
    return [
      {
        name: "Compute cost",
        winner: null,
        undecided: true,
        evidence: "no device both runs were measured on",
      },
    ];
  }

  const rowA = a.find((row) => row.device === device);
  const rowB = b.find((row) => row.device === device);
  const value = (row: Benchmark | undefined, key: string): number | null => {
    const raw = row?.measurements?.[key];
    return typeof raw === "number" ? raw : null;
  };

  const out: Dimension[] = [];
  const add = (
    name: string,
    key: string,
    lowerIsBetter: boolean,
    render: (value: number) => string,
  ) => {
    const x = value(rowA, key);
    const y = value(rowB, key);
    if (x === null || y === null) return;
    out.push({
      name,
      winner: x === y ? null : (lowerIsBetter ? x < y : x > y) ? "a" : "b",
      evidence: `${render(x)} against ${render(y)} on ${device}`,
    });
  };

  add("Speed — latency", "latency_mean_ms", true, (v) => `${v.toFixed(1)} ms`);
  add("Memory footprint", "rss_peak_mb", true, (v) =>
    v >= 1024 ? `${(v / 1024).toFixed(2)} GB` : `${Math.round(v)} MB`,
  );
  add("Checkpoint size", "checkpoint_bytes", true, (v) => {
    const megabytes = v / (1024 * 1024);
    return megabytes >= 1024
      ? `${(megabytes / 1024).toFixed(2)} GB`
      : `${Math.round(megabytes)} MB`;
  });

  return out.length
    ? out
    : [
        {
          name: "Compute cost",
          winner: null,
          undecided: true,
          evidence: `benchmarked on ${device}, but no shared measurement`,
        },
      ];
}

export function Verdict({
  labelA,
  labelB,
  a,
  b,
  factorsA,
  factorsB,
  benchmarksA,
  benchmarksB,
}: {
  labelA: string;
  labelB: string;
  a: RunMetrics;
  b: RunMetrics;
  factorsA: FactorRate[];
  factorsB: FactorRate[];
  benchmarksA: Benchmark[];
  benchmarksB: Benchmark[];
}) {
  const value = (m: RunMetrics, key: keyof RunMetrics) => {
    const v = m[key];
    return isAvailable(v) && typeof v.value === "number" ? v.value : null;
  };

  const dimensions: Dimension[] = [
    {
      name: "Detection accuracy (F1)",
      winner: winner(a.f1, b.f1, true),
      evidence: (() => {
        const x = value(a, "f1");
        const y = value(b, "f1");
        return x !== null && y !== null
          ? `${(x * 100).toFixed(1)}% against ${(y * 100).toFixed(1)}%`
          : "not computable";
      })(),
    },
    {
      name: "Recall — finding the objects",
      winner: winner(a.recall, b.recall, true),
      evidence: (() => {
        const x = value(a, "recall");
        const y = value(b, "recall");
        return x !== null && y !== null
          ? `${(x * 100).toFixed(1)}% against ${(y * 100).toFixed(1)}%`
          : "not computable";
      })(),
    },
    {
      name: "Precision — avoiding false alarms",
      winner: winner(a.precision, b.precision, true),
      evidence: (() => {
        const x = value(a, "precision");
        const y = value(b, "precision");
        return x !== null && y !== null
          ? `${(x * 100).toFixed(1)}% against ${(y * 100).toFixed(1)}%`
          : "not computable";
      })(),
    },
    {
      name: "Outline quality (mean mask IoU)",
      winner: winner(a.meanMaskIou, b.meanMaskIou, true),
      evidence: (() => {
        const x = value(a, "meanMaskIou");
        const y = value(b, "meanMaskIou");
        return x !== null && y !== null
          ? `${x.toFixed(3)} against ${y.toFixed(3)}`
          : "not measured for both runs";
      })(),
    },
    ...robustness(factorsA, factorsB),
    ...compute(benchmarksA, benchmarksB),
  ];

  const decided = dimensions.filter((d) => !d.undecided && d.winner);
  const aWins = decided.filter((d) => d.winner === "a").length;
  const bWins = decided.filter((d) => d.winner === "b").length;

  // Only claim compute is unsettled when it actually is. Once both runs are
  // benchmarked on one device this stops being a caveat and starts being
  // evidence, and the closing sentence has to stop saying otherwise.
  const undecidedNote = dimensions.some((d) => d.undecided)
    ? " Compute cost is undecided, so a deployment constrained by latency or memory is not settled by this comparison."
    : "";

  const closing = (() => {
    if (decided.length === 0) {
      return "No dimension is decided by the stored data for these two runs.";
    }
    if (aWins === bWins) {
      return `${labelA} and ${labelB} each lead on ${aWins} of the ${decided.length} decided dimensions. The choice depends on which of them your deployment weights more heavily.`;
    }
    const [lead, trail, leadCount, trailCount] =
      aWins > bWins ? [labelA, labelB, aWins, bWins] : [labelB, labelA, bWins, aWins];
    const tail =
      trailCount > 0
        ? ` ${trail} still leads on ${trailCount}, so this is a trade-off rather than a clean win.`
        : "";
    return `${lead} leads on ${leadCount} of the ${decided.length} decided dimensions.${tail}${undecidedNote}`;
  })();

  return (
    <section className="bg-card border border-border/40 rounded-lg p-4">
      <h3 className="text-[13px] font-medium text-ink">Which model should I use?</h3>
      <p className="text-[11px] text-slate mt-0.5 mb-3">
        Decided per dimension from stored data. There is deliberately no single
        score — the weighting belongs to your deployment, not to this page.
      </p>

      <div className="flex flex-col gap-1.5 mb-4">
        {/* The winning model never truncates and the evidence yields space to
            it: this column is the page's actual output, and a checkpoint name
            cut to "rfdetr_nano_seg.pt · ru…" fails at the one job the row has.
            The tick leads for the same reason — it survives even if a very long
            checkpoint name eventually has to clip. Rows wrap rather than
            compress on narrow viewports. */}
        {dimensions.map((dimension) => (
          <div
            key={dimension.name}
            className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 py-1.5 border-b border-border/20 last:border-0"
          >
            {/* Single line only once there is genuinely room for all three
                parts. Below that the name takes its own line so the evidence
                keeps enough width to stay readable rather than clipping to
                "81.3% agai…". */}
            <span className="text-[12px] text-ink basis-full xl:basis-auto shrink-0">
              {dimension.name}
            </span>
            <span className="text-[10px] text-slate/70 flex-1 min-w-0 text-right truncate">
              {dimension.evidence}
            </span>
            <span className="text-[12px] font-medium shrink-0 text-right">
              {dimension.undecided ? (
                <span className="text-slate/70 italic">Not available</span>
              ) : dimension.winner === "a" ? (
                <span className="text-brass">✓ {labelA}</span>
              ) : dimension.winner === "b" ? (
                <span className="text-brass">✓ {labelB}</span>
              ) : (
                <span className="text-slate">tie</span>
              )}
            </span>
          </div>
        ))}
      </div>

      <p className="text-[12px] leading-relaxed text-ink bg-canvas/60 border border-border/30 rounded-md p-3">
        {closing}
      </p>
    </section>
  );
}
