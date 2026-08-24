"use client";

import type { FactorRate } from "@/lib/api/rows";

/**
 * How each run's failures associate with the same factors.
 *
 * Lift is how many times more common a factor is among failures than among
 * correct detections. A lift near 1.0 means the factor separates nothing,
 * however common it is — which is the single most useful thing this table
 * says, and the reason a bare frequency column would mislead.
 *
 * **The reading beneath each row is generated from the stored numbers, not
 * written about them.** It states only what lift and p-value support: no
 * causal language, no narrative, nothing an LLM supplied. Where significance
 * is absent it says so rather than implying a finding.
 */

const SIGNIFICANCE = 0.05;

/** A materially different association, rather than sampling noise. */
const MEANINGFUL_RATIO = 1.5;

function reading(a: FactorRate | undefined, b: FactorRate | undefined,
                 labelA: string, labelB: string): string {
  if (!a || !b) return "Not measured for both runs.";
  if (a.lift === null || b.lift === null) return "Lift not measurable for both runs.";

  const aSig = a.p_value !== null && a.p_value < SIGNIFICANCE;
  const bSig = b.p_value !== null && b.p_value < SIGNIFICANCE;

  if (!aSig && !bSig) {
    return "Neither run shows a statistically significant association.";
  }

  const [higher, lower, higherName, lowerName] =
    a.lift >= b.lift ? [a.lift, b.lift, labelA, labelB] : [b.lift, a.lift, labelB, labelA];

  // Guard the ratio: a near-zero denominator makes any difference look huge.
  const ratio = lower > 0.01 ? higher / lower : Infinity;
  if (ratio < MEANINGFUL_RATIO) {
    return `Both runs associate failures with this factor to a similar degree (${a.lift.toFixed(2)}× against ${b.lift.toFixed(2)}×).`;
  }

  const onlyOne = aSig !== bSig ? ` Significant for ${aSig ? labelA : labelB} only.` : "";
  return `${lowerName} shows a substantially lower association (${lower.toFixed(2)}× against ${higher.toFixed(2)}× for ${higherName}).${onlyOne}`;
}

function LiftCell({ rate }: { rate: FactorRate | undefined }) {
  if (!rate || rate.lift === null) {
    return (
      <td className="py-2.5 px-3 text-right align-top">
        <span className="text-slate/70 text-[12px] italic">Not available</span>
      </td>
    );
  }
  const significant = rate.p_value !== null && rate.p_value < SIGNIFICANCE;
  return (
    <td className="py-2.5 px-3 text-right align-top">
      <span className="font-mono text-[13px] text-ink">{rate.lift.toFixed(2)}×</span>
      <span
        className={`block font-mono text-[10px] mt-0.5 ${
          significant ? "text-brass" : "text-slate/70"
        }`}
      >
        p {rate.p_value === null ? "n/a" : rate.p_value < 0.001 ? "<0.001" : rate.p_value.toFixed(3)}
        {significant ? " *" : ""}
      </span>
    </td>
  );
}

export function RootCauseCompare({
  labelA,
  labelB,
  a,
  b,
}: {
  labelA: string;
  labelB: string;
  a: FactorRate[];
  b: FactorRate[];
}) {
  const byFactor = (rates: FactorRate[]) =>
    new Map(rates.map((rate) => [rate.factor, rate]));
  const mapA = byFactor(a);
  const mapB = byFactor(b);
  const factors = [...new Set([...mapA.keys(), ...mapB.keys()])].sort();

  if (factors.length === 0) {
    return (
      <section className="bg-card border border-border/40 rounded-lg p-4">
        <h3 className="text-[13px] font-medium text-ink">Root causes</h3>
        <p className="text-[12px] text-slate mt-1">
          Neither run has factor rates. Run <code>app.root_cause</code> for both to
          populate this.
        </p>
      </section>
    );
  }

  return (
    <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
      <header className="px-3 py-3 border-b border-border/40">
        <h3 className="text-[13px] font-medium text-ink">Root causes</h3>
        <p className="text-[11px] text-slate mt-0.5">
          Lift with its p-value, from each selected run. A lift near 1.0 separates
          nothing. <span className="text-brass">*</span> marks p &lt; 0.05.
        </p>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-left text-slate border-b border-border/40 bg-canvas/40">
              <th className="font-medium text-[11px] py-2 px-3">Factor</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right">{labelA}</th>
              <th className="font-medium text-[11px] py-2 px-3 text-right">{labelB}</th>
            </tr>
          </thead>
          <tbody>
            {factors.map((factor) => (
              <tr key={factor} className="border-b border-border/20 last:border-0">
                <td className="py-2.5 px-3 align-top">
                  <span className="font-mono text-[12px] text-ink">{factor}</span>
                  <span className="block text-[10px] text-slate/70 leading-snug mt-0.5 max-w-[420px]">
                    {reading(mapA.get(factor), mapB.get(factor), labelA, labelB)}
                  </span>
                </td>
                <LiftCell rate={mapA.get(factor)} />
                <LiftCell rate={mapB.get(factor)} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
