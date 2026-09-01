"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { StatCard } from "@/components/shared/StatCard";
import { Pagination } from "@/components/shared/Pagination";
import { ExportButton } from "@/components/shared/ExportButton";
import { Lightbox } from "@/components/shared/Lightbox";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { api } from "@/lib/api/client";
import {
  OUTCOME_LABEL, FAILURE_OUTCOMES,
  type Finding, type Outcomes, type Page,
} from "@/lib/api/rows";
import { pct, num, NOT_MEASURED } from "@/lib/format";
import { absence, hasPrediction, hasTruth } from "@/lib/absence";

/**
 * The findings table.
 *
 * **An empty cell says why it is empty.** A false positive has no ground-truth
 * class because nothing was there to match; a false negative has no confidence
 * because nothing was predicted; a wrong_class finding cannot show its
 * predicted class because the schema stores one class per row and for that
 * outcome it is the ground truth. Rendering all three as a bare "n/a" invited
 * the reading that data was missing — and for a false positive, that the image
 * had never been annotated, which is a different thing entirely. See
 * `lib/absence.ts`, where the distinction lives and is tested.
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
  const [zoomed, setZoomed] = useState<Finding | null>(null);

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
        <ExportButton
          rows={rows}
          columns={["id", "image_id", "outcome", "class_name", "confidence", "iou"]}
          filename={`run-${runId}-findings-page-${page}`}
          scope={`the ${rows.length} rows on this page`}
        />
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
                  // Each empty cell says *why* it is empty. A bare "n/a" reads
                  // as missing data, which for a false positive wrongly
                  // suggests the image was never annotated.
                  const noPrediction = absence(f.outcome, "prediction");
                  const noTruth = absence(f.outcome, "truth");
                  const noConfidence = absence(f.outcome, "confidence");
                  const noIou = absence(f.outcome, "iou");
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
                      <td className={`px-3 py-2 ${noPrediction ? "text-slate italic" : "text-ink"}`}>
                        {noPrediction ? (
                          <span title={noPrediction.long}>{noPrediction.short}</span>
                        ) : (
                          (hasPrediction(f.outcome) && f.class_name) || NOT_MEASURED
                        )}
                      </td>
                      <td className={`px-3 py-2 ${noTruth ? "text-slate italic" : "text-ink"}`}>
                        {noTruth ? (
                          <span title={noTruth.long}>{noTruth.short}</span>
                        ) : (
                          (hasTruth(f.outcome) && f.class_name) || NOT_MEASURED
                        )}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono text-[12px] ${f.confidence === null ? "text-slate italic" : "text-ink"}`}>
                        {noConfidence ? (
                          <span title={noConfidence.long}>{noConfidence.short}</span>
                        ) : (
                          pct(f.confidence)
                        )}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono text-[12px] ${f.iou === null ? "text-slate italic" : "text-ink"}`}>
                        {noIou ? (
                          <span title={noIou.long}>{noIou.short}</span>
                        ) : f.iou === null ? (
                          NOT_MEASURED
                        ) : (
                          f.iou.toFixed(3)
                        )}
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
              onChange={(next) => router.push(`/runs/${runId}/failures?page=${next}&size=${pageSize}`)}
              onPageSizeChange={(size) => router.push(`/runs/${runId}/failures?page=1&size=${size}`)}
            />
          </div>
        </div>

        <aside className="w-[260px] shrink-0 bg-card border border-border/40 rounded-lg p-4 overflow-y-auto custom-scrollbar">
          {selected ? (
            <>
              <p className="font-mono text-[13px] text-ink mb-3">Finding #{selected.id}</p>
              <button
                type="button"
                onClick={() => setZoomed(selected)}
                title="Open full size"
                className="block w-full mb-4 rounded overflow-hidden ring-offset-2 hover:ring-2 hover:ring-brass/50 transition-shadow cursor-zoom-in"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={api.imageUrl(selected.image_id)} alt="" className="w-full aspect-square rounded object-cover bg-black/10" />
              </button>
              <Detail label="Outcome" value={OUTCOME_LABEL[selected.outcome] ?? selected.outcome} />
              <Detail
                label={hasTruth(selected.outcome) ? "Ground-truth class" : "Predicted class"}
                value={selected.class_name ?? NOT_MEASURED}
                mono
                muted={!selected.class_name}
              />
              <Detail
                label="Confidence"
                value={
                  absence(selected.outcome, "confidence")?.short ??
                  pct(selected.confidence, 2)
                }
                muted={selected.confidence === null}
              />
              <Detail
                label="Box IoU"
                value={
                  absence(selected.outcome, "iou")?.short ??
                  (selected.iou === null ? NOT_MEASURED : selected.iou.toFixed(4))
                }
                muted={selected.iou === null}
              />
              <Detail label="Image" value={`#${selected.image_id}`} mono />

              {/* The full reason an outcome has empty fields, rather than the
                  terse form the table cell has room for. Shown for every
                  outcome that has one, so a reader never has to infer whether
                  a blank means "expected" or "missing". */}
              {(["prediction", "truth", "confidence", "iou"] as const)
                .map((field) => absence(selected.outcome, field))
                .filter((entry, index, all) => entry && all.indexOf(entry) === index)
                .map((entry) => (
                  <p
                    key={entry!.long}
                    className="mt-4 text-[11px] leading-relaxed text-slate border-t border-border/40 pt-3"
                  >
                    {entry!.long}
                  </p>
                ))}
            </>
          ) : (
            <p className="text-[13px] text-slate">Select a finding.</p>
          )}
        </aside>
      </div>

      <Lightbox
        open={zoomed !== null}
        onClose={() => setZoomed(null)}
        title={zoomed ? `Finding #${zoomed.id}` : ""}
        subtitle={
          zoomed
            ? `${zoomed.class_name ?? NOT_MEASURED} • ${OUTCOME_LABEL[zoomed.outcome] ?? zoomed.outcome}`
            : undefined
        }
      >
        {zoomed && (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={api.imageUrl(zoomed.image_id)}
            alt={`Source image for finding ${zoomed.id}`}
            className="max-w-full max-h-[75vh] object-contain rounded"
          />
        )}
      </Lightbox>
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
