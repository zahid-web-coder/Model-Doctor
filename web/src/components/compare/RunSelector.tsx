"use client";

import type { Run } from "@/lib/api/rows";
import type { Compatibility } from "@/lib/compare/align";
import { modelName, datasetName } from "@/lib/derive";

/**
 * Picking the two runs, and stating plainly whether they can be compared.
 *
 * The compatibility banner is not decoration. Two runs over different datasets,
 * different splits, or different ground truth produce a page where every number
 * is answering a different question — so a mismatch blocks rather than
 * footnotes, and the reason is spelled out rather than reduced to an icon.
 */

function Picker({
  label,
  runs,
  value,
  onChange,
}: {
  label: string;
  runs: Run[];
  value: string;
  onChange: (id: string) => void;
}) {
  const run = runs.find((r) => String(r.id) === value);
  return (
    <div className="flex-1 min-w-0">
      <label className="block text-[10px] font-semibold uppercase tracking-wider text-slate mb-1.5">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-card border border-border/60 rounded-md px-3 py-2 text-[13px] text-ink font-mono hover:border-brass/50 focus:border-brass focus:outline-none transition-colors"
      >
        {runs.map((option) => (
          <option key={option.id} value={String(option.id)}>
            Run #{option.id} — {modelName(option) ?? "n/a"} · {option.split}
          </option>
        ))}
      </select>
      {run && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
          <dt className="text-slate">Checkpoint</dt>
          <dd className="text-ink truncate font-mono">{modelName(run) ?? "n/a"}</dd>
          <dt className="text-slate">Dataset</dt>
          <dd className="text-ink truncate">{datasetName(run) ?? "n/a"}</dd>
          <dt className="text-slate">Split</dt>
          <dd className="text-ink">{run.split}</dd>
          <dt className="text-slate">Resolution</dt>
          <dd className="text-ink font-mono">{run.image_size} px</dd>
          <dt className="text-slate">Confidence</dt>
          <dd className="text-ink font-mono">{run.confidence_threshold}</dd>
        </dl>
      )}
    </div>
  );
}

export function RunSelector({
  runs,
  idA,
  idB,
  onChangeA,
  onChangeB,
  compatibility,
}: {
  runs: Run[];
  idA: string;
  idB: string;
  onChangeA: (id: string) => void;
  onChangeB: (id: string) => void;
  compatibility: Compatibility | null;
}) {
  const tone =
    compatibility?.level === "blocked"
      ? "border-[#B3452F]/50 bg-[#B3452F]/5"
      : compatibility?.level === "warning"
        ? "border-brass/50 bg-brass/5"
        : "border-border/40 bg-canvas/40";

  return (
    <section className="bg-card border border-border/40 rounded-lg p-4">
      {/* Side by side only when each picker still has room for a checkpoint
          name. Sharing a tablet-width row between them clipped the very values
          that identify the two runs. */}
      <div className="flex flex-col lg:flex-row gap-5 items-start">
        <Picker label="Model A" runs={runs} value={idA} onChange={onChangeA} />
        <span className="hidden lg:block text-slate text-[12px] pt-7 shrink-0">vs</span>
        <Picker label="Model B" runs={runs} value={idB} onChange={onChangeB} />
      </div>

      {compatibility && (
        <div className={`mt-4 rounded-md border p-3 ${tone}`}>
          <p
            className={`text-[12px] font-medium ${
              compatibility.level === "blocked" ? "text-[#B3452F]" : "text-ink"
            }`}
          >
            {compatibility.headline}
          </p>
          {compatibility.details.length > 0 && (
            <ul className="mt-1.5 flex flex-col gap-1">
              {compatibility.details.map((detail) => (
                <li key={detail} className="text-[11px] text-slate leading-relaxed">
                  · {detail}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
