"use client";

import { useScan, cancelScan } from "@/lib/hero/scan";
import { SCAN_RUN_LABEL, type ScanFactor } from "@/lib/hero/scanFindings";

/**
 * What a scanned plate turned out to be, shown beside the machine.
 *
 * Styled as a sibling of the "Operate the arm" panel — same glass, same type —
 * because it is the same mode's output, not a new surface.
 *
 * **The verdict is the point of the card.** A finding's factors are listed
 * with the run-level test beside each one, so the card shows the method rather
 * than asserting a cause: a factor can be present on a failure and still be
 * chance. That is the whole argument of the page, made on one object.
 *
 * A live region, so the result is announced when it arrives rather than
 * appearing silently for someone who cannot see it land.
 */
export function ScanCard() {
  const phase = useScan((s) => s.phase);
  const finding = useScan((s) => s.finding);

  if (phase === "idle" || !finding) return null;

  if (phase !== "result") {
    return (
      <div
        role="status"
        aria-live="polite"
        className="fixed top-28 left-6 z-30 rounded-md border border-white/15 bg-black/45 backdrop-blur-sm px-4 py-2.5 text-[12px] text-white/80"
      >
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#9fd3ff] mr-2 align-middle animate-pulse" />
        {phase === "carry" ? "Bringing the plate to the scanner…" : "Scanning…"}
      </div>
    );
  }

  const failed = finding.outcome !== "Correct";

  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed top-28 left-6 z-30 w-[320px] rounded-lg border border-white/15 bg-black/55 backdrop-blur-md p-4 text-white"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/45">
          Scan result · {SCAN_RUN_LABEL}
        </p>
        <button
          type="button"
          onClick={cancelScan}
          aria-label="Close the scan result and resume the line"
          className="-mt-1 -mr-1 rounded px-1.5 text-[16px] leading-none text-white/50 hover:text-white hover:bg-white/10 transition-colors"
        >
          ×
        </button>
      </div>

      <h3
        className={`font-heading text-[26px] leading-tight mb-1 ${
          failed ? "text-[#e8b86a]" : "text-[#9fdcae]"
        }`}
      >
        {finding.outcome}
      </h3>
      <p className="text-[12px] text-white/65 mb-3">
        <span className="font-mono text-white/85">{finding.className}</span>
        {finding.confidence && <> · confidence {finding.confidence}</>}
        {finding.iou && <> · IoU {finding.iou}</>}
        <span className="text-white/35"> · finding #{finding.id}</span>
      </p>

      {finding.factors.length > 0 && (
        <ul className="space-y-2 mb-3">
          {finding.factors.map((f) => <FactorRow key={f.factor} factor={f} />)}
        </ul>
      )}

      <p className="text-[12px] leading-relaxed text-white/75 mb-3">{finding.reading}</p>

      <p className="text-[10px] leading-relaxed text-white/35">
        A real finding from the project&apos;s own analysis. The plate on the
        belt stands in for the photograph it came from.
      </p>
    </div>
  );
}

function FactorRow({ factor }: { factor: ScanFactor }) {
  const cause = factor.verdict === "cause";
  return (
    <li className="rounded-md border border-white/10 bg-white/[0.04] px-2.5 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[12px] text-white/90">{factor.factor}</span>
        <span
          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
            cause ? "bg-[#c9a227]/25 text-[#f0cf6a]" : "bg-white/10 text-white/55"
          }`}
        >
          {cause ? "Likely cause" : "Chance"}
        </span>
      </div>
      <p className="text-[11px] text-white/50 mt-0.5">{factor.evidence}</p>
      <p className="text-[11px] text-white/40 mt-0.5">
        Across the run: {factor.lift}×, p {factor.p}
        {!cause && " — not distinguishable from chance"}
      </p>
    </li>
  );
}
