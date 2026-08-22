"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Download } from "lucide-react";
import { StatCard } from "@/components/shared/StatCard";
import { Pagination } from "@/components/shared/Pagination";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { api } from "@/lib/api/client";
import {
  OUTCOME_LABEL, FAILURE_OUTCOMES,
  type Finding, type Outcomes, type Page,
} from "@/lib/api/rows";
import { pct, num, NOT_MEASURED } from "@/lib/format";

/**
 * The findings table.
 *
 * Two columns render absence rather than a guess. A false negative has no
 * prediction and therefore no confidence. And `findings` stores one
 * `class_name` per row — for a wrong_class finding that is the ground truth;
 * the predicted class is not recoverable, so it shows n/a. SCHEMA.md is
 * explicit that inferring it is not possible.
 *
 * Paging is in the URL because the API pages server-side: the page is part of
 * what is being viewed, so it survives a reload and can be linked.
 */
export function FailuresTable({
  runId, page, pageSize, findings, outcomes,
}: {
  runId: string;
  page: number;
  pageSize: number;
  findings: Page<Finding>;
  outcomes: Outcomes | null;
}) {
  const router = useRouter();
  const [outcomeFilter, setOutcomeFilter] = useState("All Types");
  const [classFilter, setClassFilter] = useState("All Classes");
  const [selected, setSelected] = useState<Finding | null>(findings.items[0] ?? null);

  const classes = useMemo(
    () => ["All Classes", ...Array.from(new Set(findings.items.map((f) => f.class_name).filter((c): c is string => Boolean(c))))],
    [findings.items]
  );

  // Filtering is client-side over the current page: the endpoint takes limit
  // and offset but no outcome or class parameter, so filtering server-side
  // would mean inventing query parameters the API does not have.
  const rows = findings.items.filter((f) => {
    if (outcomeFilter !== "All Types" && (OUTCOME_LABEL[f.outcome] ?? f.outcome) !== outcomeFilter) return false;
    if (classFilter !== "All Classes" && f.class_name !== classFilter) return false;
    return true;
  });

  const pageCount = Math.max(1, Math.ceil(findings.total / pageSize));
  const share = (n: number) => {
    const totalFailures = outcomes
      ? FAILURE_OUTCOMES.reduce((sum, k) => sum + outcomes[k], 0)
      : 0;
    return totalFailures ? pct(n / totalFailures) : NOT_MEASURED;
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Failures</h3>
          <p className="text-[13px] text-slate">Browse and analyse failed predictions.</p>
        </div>
        <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
          <Download size={13} className="text-slate" /> Export
        </button>
      </div>

      <div className="flex items-center gap-3 mb-4 shrink-0">
        <span className="text-[12px] text-slate">Filter:</span>
        <FilterSelect value={classFilter} options={classes} onChange={setClassFilter} />
        <FilterSelect
          value={outcomeFilter}
          options={["All Types", ...FAILURE_OUTCOMES.map((o) => OUTCOME_LABEL[o]), OUTCOME_LABEL.correct]}
          onChange={setOutcomeFilter}
          width="w-[175px]"
        />
        <span className="text-[11px] text-slate">filters apply to this page</span>
      </div>

      <div className="flex gap-3 mb-5 shrink-0">
        <StatCard label="Total findings" value={num(findings.total)} tone="brass" />
        {FAILURE_OUTCOMES.map((key) => (
          <StatCard
            key={key}
            label={OUTCOME_LABEL[key]}
            value={outcomes ? num(outcomes[key]) : NOT_MEASURED}
            sub={outcomes ? share(outcomes[key]) : undefined}
          />
        ))}
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
                  <th className="font-medium px-3 py-3 text-right">IoU</th>
                  <th className="font-medium px-3 py-3">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((f) => {
                  // `class_name` is the predicted class only when there is a
                  // prediction to speak of.
                  const hasPrediction = f.outcome === "false_positive" || f.outcome === "correct" || f.outcome === "poor_localization";
                  const hasTruth = f.outcome !== "false_positive";
                  return (
                    <tr
                      key={f.id}
                      onClick={() => setSelected(f)}
                      className={`border-b border-border/25 cursor-pointer transition-colors ${
                        selected?.id === f.id ? "bg-[#EBE6D8]/50" : "hover:bg-black/[0.02]"
                      }`}
                    >
                      <td className="px-4 py-2">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={api.imageUrl(f.image_id)} alt="" className="w-9 h-9 rounded object-cover bg-black/10" loading="lazy" />
                      </td>
                      <td className="px-3 py-2 font-mono text-[12px] text-ink">#{f.id}</td>
                      <td className={`px-3 py-2 ${hasPrediction ? "text-ink" : "text-slate italic"}`}>
                        {hasPrediction ? f.class_name ?? NOT_MEASURED : NOT_MEASURED}
                      </td>
                      <td className={`px-3 py-2 ${hasTruth ? "text-ink" : "text-slate italic"}`}>
                        {hasTruth ? f.class_name ?? NOT_MEASURED : NOT_MEASURED}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono text-[12px] ${f.confidence === null ? "text-slate italic" : "text-ink"}`}>
                        {pct(f.confidence)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono text-[12px] ${f.iou === null ? "text-slate italic" : "text-ink"}`}>
                        {f.iou === null ? NOT_MEASURED : f.iou.toFixed(3)}
                      </td>
                      <td className="px-3 py-2">
                        <span className="text-[11px] px-2 py-0.5 rounded bg-black/5 text-slate whitespace-nowrap">
                          {OUTCOME_LABEL[f.outcome] ?? f.outcome}
                        </span>
                      </td>
                    </tr>
                  );
                })}
                {rows.length === 0 && (
                  <tr><td colSpan={7} className="px-4 py-10 text-center text-slate">No findings match these filters on this page.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="px-4 border-t border-border/40">
            <Pagination
              page={page} pageCount={pageCount} total={findings.total} pageSize={pageSize}
              onChange={(next) => router.push(`/runs/${runId}/failures?page=${next}`)}
            />
          </div>
        </div>

        <aside className="w-[260px] shrink-0 bg-card border border-border/40 rounded-lg p-4 overflow-y-auto custom-scrollbar">
          {selected ? (
            <>
              <p className="font-mono text-[13px] text-ink mb-3">Finding #{selected.id}</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={api.imageUrl(selected.image_id)} alt="" className="w-full aspect-square rounded object-cover bg-black/10 mb-4" />
              <Detail label="Outcome" value={OUTCOME_LABEL[selected.outcome] ?? selected.outcome} />
              <Detail label="Class" value={selected.class_name ?? NOT_MEASURED} mono muted={!selected.class_name} />
              <Detail label="Confidence" value={pct(selected.confidence, 2)} muted={selected.confidence === null} />
              <Detail label="Box IoU" value={selected.iou === null ? NOT_MEASURED : selected.iou.toFixed(4)} muted={selected.iou === null} />
              <Detail label="Image" value={`#${selected.image_id}`} mono />
              {selected.outcome === "wrong_class" && (
                <p className="mt-4 text-[11px] leading-relaxed text-slate border-t border-border/40 pt-3">
                  The predicted class is not recoverable for a wrong_class finding — the
                  schema stores one class per row, and for this outcome that is the ground
                  truth. It is shown as {NOT_MEASURED} rather than inferred.
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
      <p className={`text-[13px] ${mono ? "font-mono text-[12px]" : ""} ${muted ? "text-slate italic" : "text-ink"}`}>{value}</p>
    </div>
  );
}
