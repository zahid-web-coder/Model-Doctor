"use client";

import { use, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { mockFailures, FAILURE_LABEL, FailureInstance } from "@/lib/mock-data/failures";
import { CLASSES } from "@/lib/mock-data/classes";
import { StatCard } from "@/components/shared/StatCard";
import { Pagination } from "@/components/shared/Pagination";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { pct, predictedClass, NOT_MEASURED } from "@/lib/format";

const PAGE_SIZE = 10;

/**
 * Browse and analyse failed predictions.
 *
 * Matches the design: export action, three filters, the five-tile summary
 * strip, and a paginated table. Two columns render absence rather than a
 * guess — a wrong_class finding has no recoverable predicted class, and a
 * false negative has no confidence, so both show n/a.
 */
export default function FailuresPage({ params }: { params: Promise<{ runId: string }> }) {
  use(params);

  const [classFilter, setClassFilter] = useState("All Classes");
  const [factorFilter, setFactorFilter] = useState("All Root Causes");
  const [typeFilter, setTypeFilter] = useState("All Types");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<FailureInstance | null>(mockFailures[0]);

  const factors = useMemo(
    () => ["All Root Causes", ...Array.from(new Set(mockFailures.map((f) => f.rootCauseFactor)))],
    []
  );

  const rows = useMemo(() => {
    return mockFailures.filter((f) => {
      if (classFilter !== "All Classes" && f.groundTruth !== classFilter) return false;
      if (factorFilter !== "All Root Causes" && f.rootCauseFactor !== factorFilter) return false;
      if (typeFilter !== "All Types" && FAILURE_LABEL[f.failureType] !== typeFilter) return false;
      return true;
    });
  }, [classFilter, factorFilter, typeFilter]);

  const counts = useMemo(() => {
    const by = (t: FailureInstance["failureType"]) =>
      mockFailures.filter((f) => f.failureType === t).length;
    return {
      total: mockFailures.length,
      fn: by("false_negative"),
      wc: by("wrong_class"),
      pl: by("poor_localization"),
      fp: by("false_positive"),
    };
  }, []);

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const current = Math.min(page, pageCount);
  const visible = rows.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE);
  const share = (n: number) => `${((n / counts.total) * 100).toFixed(1)}%`;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Failures</h3>
          <p className="text-[13px] text-slate">Browse and analyse failed predictions.</p>
        </div>
        <button
          type="button"
          className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors"
        >
          <Download size={13} className="text-slate" />
          Export
        </button>
      </div>

      <div className="flex items-center gap-3 mb-4 shrink-0">
        <span className="text-[12px] text-slate">Filter:</span>
        <FilterSelect value={classFilter} options={["All Classes", ...CLASSES]} onChange={(v) => { setClassFilter(v); setPage(1); }} />
        <FilterSelect value={factorFilter} options={factors} onChange={(v) => { setFactorFilter(v); setPage(1); }} width="w-[170px]" />
        <FilterSelect value={typeFilter} options={["All Types", ...Object.values(FAILURE_LABEL)]} onChange={(v) => { setTypeFilter(v); setPage(1); }} width="w-[170px]" />
      </div>

      <div className="flex gap-3 mb-5 shrink-0">
        <StatCard label="Total Failures" value={counts.total.toLocaleString()} tone="brass" />
        <StatCard label="False Negative" value={counts.fn.toString()} sub={share(counts.fn)} />
        <StatCard label="Wrong Class" value={counts.wc.toString()} sub={share(counts.wc)} />
        <StatCard label="Poor Localization" value={counts.pl.toString()} sub={share(counts.pl)} />
        <StatCard label="False Positive" value={counts.fp.toString()} sub={share(counts.fp)} />
      </div>

      <div className="flex-1 flex gap-4 overflow-hidden">
        <div className="flex-1 flex flex-col overflow-hidden bg-card border border-border/40 rounded-lg">
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 bg-card border-b border-border/40">
                <tr className="text-left text-slate">
                  <th className="font-medium px-4 py-3 w-[70px]">Image</th>
                  <th className="font-medium px-3 py-3">ID</th>
                  <th className="font-medium px-3 py-3">Prediction</th>
                  <th className="font-medium px-3 py-3">Ground Truth</th>
                  <th className="font-medium px-3 py-3 text-right">Confidence</th>
                  <th className="font-medium px-3 py-3">Failure Type</th>
                  <th className="font-medium px-3 py-3">Root Cause</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((f) => (
                  <tr
                    key={f.id}
                    onClick={() => setSelected(f)}
                    className={`border-b border-border/25 cursor-pointer transition-colors ${
                      selected?.id === f.id ? "bg-[#EBE6D8]/50" : "hover:bg-black/[0.02]"
                    }`}
                  >
                    <td className="px-4 py-2.5">
                      <div className="w-9 h-9 rounded bg-gradient-to-tr from-slate-700 to-slate-500" />
                    </td>
                    <td className="px-3 py-2.5 font-mono text-[12px] text-ink">{f.id}</td>
                    <td className={`px-3 py-2.5 ${f.prediction ? "text-ink" : "text-slate italic"}`}>
                      {predictedClass(f.prediction)}
                    </td>
                    <td className={`px-3 py-2.5 ${f.groundTruth ? "text-ink" : "text-slate italic"}`}>
                      {predictedClass(f.groundTruth)}
                    </td>
                    <td className={`px-3 py-2.5 text-right font-mono text-[12px] ${f.confidence === null ? "text-slate italic" : "text-ink"}`}>
                      {pct(f.confidence)}
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="text-[11px] px-2 py-0.5 rounded bg-black/5 text-slate whitespace-nowrap">
                        {FAILURE_LABEL[f.failureType]}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-[12px] text-slate">{f.rootCauseFactor}</td>
                  </tr>
                ))}
                {visible.length === 0 && (
                  <tr><td colSpan={7} className="px-4 py-10 text-center text-slate">No findings match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="px-4 border-t border-border/40">
            <Pagination
              page={current} pageCount={pageCount} total={rows.length}
              pageSize={PAGE_SIZE} onChange={setPage}
            />
          </div>
        </div>

        <aside className="w-[260px] shrink-0 bg-card border border-border/40 rounded-lg p-4 overflow-y-auto custom-scrollbar">
          {selected ? (
            <>
              <p className="font-mono text-[13px] text-ink mb-3">{selected.id}</p>
              <div className="w-full aspect-square rounded bg-gradient-to-tr from-slate-700 to-slate-500 mb-4" />
              <Detail label="Failure type" value={FAILURE_LABEL[selected.failureType]} />
              <Detail label="Prediction" value={predictedClass(selected.prediction)} muted={!selected.prediction} />
              <Detail label="Ground truth" value={predictedClass(selected.groundTruth)} muted={!selected.groundTruth} />
              <Detail label="Confidence" value={pct(selected.confidence, 2)} muted={selected.confidence === null} />
              <Detail label="Box IoU" value={selected.iou === null ? NOT_MEASURED : selected.iou.toFixed(3)} muted={selected.iou === null} />
              <Detail label="Correlated root cause" value={selected.rootCauseFactor} mono />
              {selected.failureType === "wrong_class" && (
                <p className="mt-4 text-[11px] leading-relaxed text-slate border-t border-border/40 pt-3">
                  The predicted class is not recoverable for a wrong_class finding — the schema
                  does not store it, so it is shown as n/a rather than inferred.
                </p>
              )}
            </>
          ) : (
            <p className="text-[13px] text-slate">Select a finding.</p>
          )}
        </aside>
      </div>
    </div>
  );
}

function Detail({ label, value, mono, muted }: { label: string; value: string; mono?: boolean; muted?: boolean }) {
  return (
    <div className="mb-3">
      <p className="text-[10px] uppercase tracking-wider text-slate mb-0.5">{label}</p>
      <p className={`text-[13px] ${mono ? "font-mono text-[12px]" : ""} ${muted ? "text-slate italic" : "text-ink"}`}>
        {value}
      </p>
    </div>
  );
}
