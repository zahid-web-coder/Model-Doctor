"use client";

import { DIAGNOSIS } from "@/lib/hero/landing";

/**
 * The diagnosis beat.
 *
 * This is the one that has to land. Everything before it is inspection, which
 * every vision vendor can show; what makes this product different is that the
 * obvious cause is usually wrong, and that it can prove it.
 *
 * **Rendered as DOM, not as 3D text.** The numbers are the argument, so they
 * have to be legible at every viewport, selectable, screen-readable, and
 * styled with the same type as the rest of the site. Extruded text in the
 * scene would be none of those things and would cost a font to download.
 *
 * The figures come from the project's own database — see `DIAGNOSIS` in
 * `lib/hero/landing.ts` for provenance.
 */

function FactorCard({
  factor, onFailures, onCorrect, lift, p, verdict, why, ruledOut,
}: {
  factor: string; onFailures: string; onCorrect: string; lift: string;
  p: string; verdict: string; why: string; ruledOut?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 backdrop-blur-sm transition-colors ${
        ruledOut
          ? "border-white/10 bg-black/25 text-white/45"
          : "border-[#c9a227]/40 bg-black/35 text-white"
      }`}
    >
      <div className="flex items-baseline justify-between gap-3 mb-3">
        <span
          className={`font-mono text-[13px] ${ruledOut ? "line-through decoration-white/40" : ""}`}
        >
          {factor}
        </span>
        <span
          className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded ${
            ruledOut ? "bg-white/10 text-white/50" : "bg-[#c9a227]/25 text-[#f0d78a]"
          }`}
        >
          {verdict}
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
        <dt className="text-white/45">On failures</dt>
        <dd className="font-mono text-right">{onFailures}</dd>
        <dt className="text-white/45">On correct</dt>
        <dd className="font-mono text-right">{onCorrect}</dd>
        <dt className="text-white/45">Lift</dt>
        <dd className={`font-mono text-right ${ruledOut ? "" : "text-[#f0d78a]"}`}>{lift}×</dd>
        <dt className="text-white/45">p</dt>
        <dd className="font-mono text-right">{p}</dd>
      </dl>

      <p className="text-[11px] leading-relaxed mt-3 text-white/55">{why}</p>
    </div>
  );
}

export function DiagnosisMoment() {
  const { ruledOut, cause } = DIAGNOSIS;
  return (
    <div className="max-w-[520px]">
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-3">
        Diagnosis
      </p>
      <h2 className="font-heading text-white text-[32px] leading-[1.12] mb-3">
        The most common cause
        <br />
        explained nothing.
      </h2>
      <p className="text-[13px] leading-relaxed text-white/60 mb-5">
        Edge truncation was on nearly three quarters of the failures. It was
        also on nearly three quarters of the plates that passed — so it
        separates nothing. The real cause is three times rarer and three times
        more predictive.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FactorCard {...ruledOut} ruledOut />
        <FactorCard {...cause} />
      </div>
    </div>
  );
}

/**
 * Verification, as replication rather than a lamp.
 *
 * A green light would say we grade parts. The claim that actually matters is
 * that the finding survives on a second, independent split — so that is what
 * the station reports.
 */
export function VerificationPanel() {
  const { replication } = DIAGNOSIS;
  return (
    <div className="max-w-[480px]">
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#f0d78a]/80 mb-3">
        Verification
      </p>
      <h2 className="font-heading text-white text-[32px] leading-[1.12] mb-3">
        And it holds on
        <br />
        a second split.
      </h2>
      <p className="text-[13px] leading-relaxed text-white/60 mb-5">
        A finding that appears once is a coincidence. {replication.factor} was
        measured again on an independent run, and it replicated — same
        direction, same significance.
      </p>

      <div className="rounded-lg border border-white/10 bg-black/30 backdrop-blur-sm overflow-hidden">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-white/40 border-b border-white/10">
              <th className="font-medium text-left px-4 py-2">Run</th>
              <th className="font-medium text-right px-4 py-2">Findings</th>
              <th className="font-medium text-right px-4 py-2">Lift</th>
              <th className="font-medium text-right px-4 py-2">p</th>
            </tr>
          </thead>
          <tbody>
            {replication.runs.map((r) => (
              <tr key={r.label} className="border-b border-white/5 last:border-0">
                <td className="px-4 py-2 text-white/75">{r.label}</td>
                <td className="px-4 py-2 text-right font-mono text-white/60">{r.findings}</td>
                <td className="px-4 py-2 text-right font-mono text-[#f0d78a]">{r.lift}×</td>
                <td className="px-4 py-2 text-right font-mono text-white/60">{r.p}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
