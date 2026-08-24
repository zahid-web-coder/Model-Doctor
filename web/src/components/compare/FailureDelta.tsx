"use client";

import type { Outcomes } from "@/lib/api/rows";

/**
 * Where the two runs' failures differ, as opposed bars on a shared scale.
 *
 * A shared scale is the whole point: drawn per row, a difference of three would
 * look like a difference of thirty. Every value here is a stored count from the
 * selected runs.
 */

const FAILURES: { key: keyof Outcomes; label: string }[] = [
  { key: "false_negative", label: "False negative" },
  { key: "false_positive", label: "False positive" },
  { key: "poor_localization", label: "Poor localisation" },
  { key: "wrong_class", label: "Wrong class" },
];

export function FailureDelta({
  labelA,
  labelB,
  a,
  b,
}: {
  labelA: string;
  labelB: string;
  a: Outcomes;
  b: Outcomes;
}) {
  const scale = Math.max(
    1,
    ...FAILURES.flatMap(({ key }) => [a[key], b[key]]),
  );

  return (
    <section className="bg-card border border-border/40 rounded-lg p-3">
      <h3 className="text-[13px] font-medium text-ink">Failure profile</h3>
      <p className="text-[11px] text-slate mt-0.5 mb-3">
        Fewer is better in every row. Bars share one scale.
      </p>

      <div className="flex items-center gap-3 mb-3 text-[10px] text-slate">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-slate/60" />
          <span className="truncate max-w-[150px]">{labelA}</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-brass" />
          <span className="truncate max-w-[150px]">{labelB}</span>
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {FAILURES.map(({ key, label }) => {
          const left = a[key];
          const right = b[key];
          const delta = right - left;
          return (
            <div key={key}>
              <div className="flex items-baseline justify-between mb-1">
                <span className="text-[12px] text-ink">{label}</span>
                <span
                  className={`font-mono text-[11px] ${
                    delta === 0
                      ? "text-slate"
                      : delta < 0
                        ? "text-brass"
                        : "text-[#B3452F]"
                  }`}
                >
                  {delta === 0 ? "no change" : `${delta > 0 ? "+" : ""}${delta}`}
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 rounded-full bg-black/5 overflow-hidden">
                    <div
                      className="h-full bg-slate/60 rounded-full"
                      style={{ width: `${(left / scale) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-slate w-8 text-right">
                    {left}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 rounded-full bg-black/5 overflow-hidden">
                    <div
                      className="h-full bg-brass rounded-full"
                      style={{ width: `${(right / scale) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-ink w-8 text-right">
                    {right}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
