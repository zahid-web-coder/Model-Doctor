"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { ExportButton } from "@/components/shared/ExportButton";
import { StatCard } from "@/components/shared/StatCard";
import type { ImageDiagnosis } from "@/lib/api/rows";
import { num } from "@/lib/format";

/**
 * Every image in a run, with the shape of the mistake made on it.
 *
 * **`empty` is never counted as clean.** An image with nothing to find and
 * nothing found is correct behaviour, but folding it into the clean rate would
 * let a test set of blank photographs score perfectly. The rate is over images
 * that had something to find or something predicted.
 *
 * **The thresholds are shown, not hidden.** Merge and split counts move with
 * the coverage threshold, so the value that produced these verdicts is stated
 * beneath the table rather than left implicit.
 */

/** What each verdict means, in one line, for a reader who has not read the docs. */
const MEANING: Record<string, string> = {
  clean: "every object found, nothing extra",
  empty: "nothing to find and nothing predicted",
  zero_prediction: "objects present, no prediction at all",
  merged: "one prediction stretched across several objects",
  split: "several predictions on one object",
  merged_and_split: "both merged and split in the same image",
  partly_missed: "some objects found, others untouched",
  spurious: "found the objects, plus something that is not one",
  partly_missed_and_spurious: "both missed and spurious",
  partial_coverage: "an object was reached but not taken — neither found nor missed",
};

const LABEL: Record<string, string> = {
  clean: "Clean",
  empty: "Empty",
  zero_prediction: "Nothing detected",
  merged: "Merged",
  split: "Split",
  merged_and_split: "Merged + split",
  partly_missed: "Partly missed",
  spurious: "Spurious",
  partly_missed_and_spurious: "Missed + spurious",
  partial_coverage: "Partial coverage",
};

const TONE: Record<string, string> = {
  clean: "bg-[#66805A]/12 text-[#66805A]",
  empty: "bg-black/5 text-slate",
};

export function ImagesTable({
  runId,
  rows,
  filenames,
}: {
  runId: string;
  rows: ImageDiagnosis[];
  filenames: Record<number, string>;
}) {
  const [verdict, setVerdict] = useState("All verdicts");
  const [size, setSize] = useState("All images");

  const verdicts = useMemo(
    () => ["All verdicts", ...Array.from(new Set(rows.map((r) => r.verdict)))],
    [rows],
  );

  const visible = rows.filter((r) => {
    if (verdict !== "All verdicts" && r.verdict !== verdict) return false;
    if (size === "Single-object" && r.gt_count !== 1) return false;
    if (size === "Multi-object" && r.gt_count <= 1) return false;
    return true;
  });

  // Deliberately excludes `empty` from the denominator — see the docstring.
  const scored = rows.filter((r) => r.verdict !== "empty");
  const clean = scored.filter((r) => r.verdict === "clean").length;
  const merged = rows.filter((r) => r.merged).length;
  const split = rows.filter((r) => r.split).length;

  if (rows.length === 0) {
    return (
      <div className="flex flex-col h-full">
        <h3 className="text-[17px] font-heading text-ink">Images</h3>
        <p className="text-[13px] text-slate mb-6">
          What shape the model&rsquo;s mistake took, per photograph.
        </p>
        <div className="flex-1 grid place-items-center">
          <p className="text-[13px] text-slate max-w-[440px] text-center leading-relaxed">
            Image-level diagnosis has not been measured for this run. It is a
            separate pass, and it needs outline measurements because box overlap
            misreports diagonal objects. Produce it with{" "}
            <span className="font-mono text-[12px] text-ink">
              python -m app.image_diagnosis --run {runId}
            </span>
            .
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Images</h3>
          <p className="text-[13px] text-slate">
            What shape the model&rsquo;s mistake took, per photograph.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <FilterSelect
            value={verdict}
            options={verdicts.map((v) => (v === "All verdicts" ? v : LABEL[v] ?? v))}
            onChange={(chosen) =>
              setVerdict(
                chosen === "All verdicts"
                  ? chosen
                  : (Object.keys(LABEL).find((k) => LABEL[k] === chosen) ?? chosen),
              )
            }
            width="w-[170px]"
          />
          <FilterSelect
            value={size}
            options={["All images", "Single-object", "Multi-object"]}
            onChange={setSize}
            width="w-[150px]"
          />
          <ExportButton
            rows={visible.map((r) => ({
              image_id: r.image_id,
              filename: filenames[r.image_id] ?? "",
              verdict: r.verdict,
              gt_count: r.gt_count,
              pred_count: r.pred_count,
              correct: r.outcomes.correct ?? 0,
              false_negative: r.outcomes.false_negative ?? 0,
              false_positive: r.outcomes.false_positive ?? 0,
              poor_localization: r.outcomes.poor_localization ?? 0,
              wrong_class: r.outcomes.wrong_class ?? 0,
              merged: r.merged,
              split: r.split,
              objects_untouched: r.objects_untouched,
              predictions_on_nothing: r.predictions_on_nothing,
              cover_hit: r.cover_hit,
              cover_miss: r.cover_miss,
            }))}
            columns={[
              "image_id", "filename", "verdict", "gt_count", "pred_count",
              "correct", "false_negative", "false_positive", "poor_localization",
              "wrong_class", "merged", "split", "objects_untouched",
              "predictions_on_nothing", "cover_hit", "cover_miss",
            ]}
            filename={`run-${runId}-image-diagnoses`}
            scope={`the ${visible.length} images shown`}
          />
        </div>
      </div>

      <div className="flex gap-3 mb-5 shrink-0">
        <StatCard
          label="Clean images"
          value={num(clean)}
          sub={`of ${num(scored.length)} with something to find`}
          tone="brass"
        />
        <StatCard label="Affected" value={num(scored.length - clean)} />
        <StatCard
          label="Merged"
          value={num(merged)}
          sub="one prediction, several objects"
        />
        <StatCard
          label="Split"
          value={num(split)}
          sub="several predictions, one object"
        />
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
        <table className="w-full text-[13px]">
          <thead className="sticky top-0 bg-canvas border-b border-border/40">
            <tr className="text-left text-slate">
              <th className="font-medium px-3 py-2.5">Image</th>
              <th className="font-medium px-3 py-2.5 text-right">Objects</th>
              <th className="font-medium px-3 py-2.5 text-right">Predictions</th>
              <th className="font-medium px-3 py-2.5">Outcomes</th>
              <th className="font-medium px-3 py-2.5">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr
                key={r.image_id}
                className="border-b border-border/25 hover:bg-black/[0.02] transition-colors"
              >
                <td className="px-3 py-2.5">
                  <Link
                    href={`/runs/${runId}/failures`}
                    className="font-mono text-[12px] text-ink hover:text-brass transition-colors"
                    title={filenames[r.image_id]}
                  >
                    #{r.image_id}
                  </Link>
                  <span className="text-[11px] text-slate ml-2 truncate inline-block max-w-[220px] align-bottom">
                    {filenames[r.image_id] ?? ""}
                  </span>
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-[12px] text-ink">
                  {r.gt_count}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-[12px] text-slate">
                  {r.pred_count}
                </td>
                <td className="px-3 py-2.5 font-mono text-[11px] text-slate">
                  {Object.entries(r.outcomes)
                    .filter(([, n]) => n > 0)
                    .map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`)
                    .join(" · ") || "—"}
                </td>
                <td className="px-3 py-2.5">
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded whitespace-nowrap ${
                      TONE[r.verdict] ?? "bg-[#A65C48]/10 text-[#A65C48]"
                    }`}
                    title={MEANING[r.verdict] ?? r.verdict}
                  >
                    {LABEL[r.verdict] ?? r.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {visible.length === 0 && (
          <p className="py-10 text-center text-slate text-[13px]">
            No images match these filters.
          </p>
        )}
      </div>

      <p className="text-[11px] text-slate mt-3 shrink-0 leading-relaxed border-t border-border/40 pt-3">
        Measured on outlines, not boxes: a box around a diagonal object sweeps
        across its neighbours. A prediction counts as finding an object at{" "}
        <span className="font-mono">{rows[0].cover_hit.toFixed(2)}</span> mask
        coverage and as missing it below{" "}
        <span className="font-mono">{rows[0].cover_miss.toFixed(2)}</span>. Merge
        and split counts move with the first of those, so every pairwise
        measurement is stored and can be re-read at a different threshold.
        Finding-level outcomes are unchanged.
      </p>
    </div>
  );
}
