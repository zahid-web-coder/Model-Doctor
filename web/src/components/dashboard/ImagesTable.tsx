"use client";

import { useEffect, useMemo, useState } from "react";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { ExportButton } from "@/components/shared/ExportButton";
import { StatCard } from "@/components/shared/StatCard";
import { LayerToggle } from "@/components/shared/LayerToggle";
import { Lightbox } from "@/components/shared/Lightbox";
import { Overlay, OverlayLegend } from "@/components/compare/Overlay";
import { api } from "@/lib/api/client";
import { fitViewport } from "@/lib/compare/geometry";
import {
  ALL_LAYERS, NO_LAYERS, defaultLayers, predictionShapeOf, truthShapeOf,
  type Layers,
} from "@/lib/findingShapes";
import type { Finding, ImageDiagnosis, Outcome } from "@/lib/api/rows";
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

/** Outcome filter labels, mapped back to the stored keys. */
const OUTCOME_KEYS: Record<string, Outcome> = {
  Correct: "correct",
  "False Negative": "false_negative",
  "False Positive": "false_positive",
  "Poor Localization": "poor_localization",
  "Wrong Class": "wrong_class",
};

const TONE: Record<string, string> = {
  clean: "bg-[#66805A]/12 text-[#66805A]",
  empty: "bg-black/5 text-slate",
};

/** Qualified clean: green, because the objects were all found, but not the plain one. */
const QUALIFIED = "bg-[#66805A]/12 text-[#66805A] ring-1 ring-inset ring-[#A65C48]/35";

/**
 * How one image's verdict reads on screen.
 *
 * **A clean image can hold a loose box, and the badge has to say so.** This
 * pass measures mask coverage; the finding-level outcome measures box overlap,
 * so an outline covering 99% of an object whose box scored 0.28 IoU is
 * legitimately `clean` here and `poor_localization` on the Failures screen
 * (D-036). Across the reference database that is 26 images — every one with an
 * IoU below 0.50 and coverage above it, exactly as the distinction predicts.
 *
 * Both readings are true, but a badge saying only "Clean" beside a row the
 * Failures tab lists as a failure leaves the reader no way to reconcile them,
 * and the likeliest conclusion is that one of the two screens is broken.
 *
 * **Presentation only.** The stored verdict is untouched, so filtering,
 * counting and export continue to work on what the classifier decided — this
 * changes how that verdict is worded, not what it is.
 */
function present(row: ImageDiagnosis): {
  label: string;
  meaning: string;
  tone: string;
} {
  const label = LABEL[row.verdict] ?? row.verdict;
  const meaning = MEANING[row.verdict] ?? row.verdict;
  const loose = row.outcomes.poor_localization ?? 0;
  if (row.verdict !== "clean" || loose === 0) {
    return { label, meaning, tone: TONE[row.verdict] ?? "bg-[#A65C48]/10 text-[#A65C48]" };
  }
  return {
    label: "Mask clean · box loose",
    meaning:
      `every object covered and nothing extra, but ${loose} ` +
      `box${loose === 1 ? "" : "es"} scored too low an IoU to match — ` +
      "the Failures tab reports the same image as poor localization",
    tone: QUALIFIED,
  };
}

export function ImagesTable({
  runId,
  rows,
  filenames,
  dimensions,
  polygons,
}: {
  runId: string;
  rows: ImageDiagnosis[];
  filenames: Record<number, string>;
  /** Image dimensions by id, so the overlay can build its viewport. */
  dimensions: Record<number, [number, number]>;
  /** Predicted outlines by finding id, from `mask_findings`. */
  polygons: Record<number, number[][]>;
}) {
  const [verdict, setVerdict] = useState("All verdicts");
  // The image being inspected, with every finding on it. Nothing is
  // "selected" here — this view is the whole photograph, which is what the
  // Failures panel deliberately is not.
  const [open, setOpen] = useState<ImageDiagnosis | null>(null);
  const [opened, setOpened] = useState<{ imageId: number; rows: Finding[] } | null>(
    null,
  );
  useEffect(() => {
    if (!open) return;
    let live = true;
    api
      .imageFindings(runId, open.image_id)
      .then((found) => {
        if (live) setOpened({ imageId: open.image_id, rows: found });
      })
      .catch(() => {
        if (live) setOpened({ imageId: open.image_id, rows: [] });
      });
    return () => {
      live = false;
    };
  }, [runId, open]);
  const [size, setSize] = useState("All images");
  const [outcome, setOutcome] = useState("All outcomes");

  const verdicts = useMemo(
    () => ["All verdicts", ...Array.from(new Set(rows.map((r) => r.verdict)))],
    [rows],
  );

  const [layers, setLayers] = useState<Layers>(() =>
    defaultLayers(Object.keys(polygons).length > 0),
  );

  const viewport = useMemo(() => {
    if (!open) return null;
    const size = dimensions[open.image_id];
    if (!size) return null;
    const [width, height] = size;
    return { width, height, viewport: fitViewport(width, height) };
  }, [open, dimensions]);

  const wholeImage = useMemo(() => {
    const found = opened && open && opened.imageId === open.image_id ? opened.rows : [];
    const siblings = [];
    const predictions = [];
    for (const finding of found) {
      const truth = truthShapeOf(finding, layers);
      if (truth) siblings.push(truth);
      const prediction = predictionShapeOf(
        finding,
        polygons[finding.id] ?? null,
        layers,
      );
      if (prediction) predictions.push(prediction);
    }
    return { siblings, predictions };
  }, [opened, open, polygons, layers]);

  const visible = rows.filter((r) => {
    if (verdict !== "All verdicts" && r.verdict !== verdict) return false;
    if (size === "Single-object" && r.gt_count !== 1) return false;
    if (size === "Multi-object" && r.gt_count <= 1) return false;
    // "Contains a false negative" rather than "is a false negative": an image
    // holds several findings, so this narrows to images where that outcome
    // occurred at least once.
    if (outcome !== "All outcomes") {
      const key = OUTCOME_KEYS[outcome];
      if (!key || !(r.outcomes[key] > 0)) return false;
    }
    return true;
  });

  // Deliberately excludes `empty` from the denominator — see the docstring.
  const scored = rows.filter((r) => r.verdict !== "empty");
  const clean = scored.filter((r) => r.verdict === "clean").length;
  const qualified = scored.filter(
    (r) => r.verdict === "clean" && (r.outcomes.poor_localization ?? 0) > 0,
  ).length;
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
            value={outcome}
            options={["All outcomes", ...Object.keys(OUTCOME_KEYS)]}
            onChange={setOutcome}
            width="w-[165px]"
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
          // The number is the stored verdict rate and stays that way — this
          // says how many of those carry a loose box, rather than quietly
          // moving them out of the count.
          sub={
            qualified > 0
              ? `of ${num(scored.length)} with something to find · ${num(qualified)} with a loose box`
              : `of ${num(scored.length)} with something to find`
          }
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
                  <button
                    type="button"
                    onClick={() => setOpen(r)}
                    title="Open the whole image"
                    className="flex items-center gap-2.5 text-left group"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={api.imageUrl(r.image_id)}
                      alt=""
                      loading="lazy"
                      className="w-9 h-9 rounded object-cover bg-black/10 shrink-0 group-hover:ring-2 group-hover:ring-brass/50 transition-shadow"
                    />
                    <span>
                      <span className="font-mono text-[12px] text-ink group-hover:text-brass transition-colors">
                        #{r.image_id}
                      </span>
                      <span className="text-[11px] text-slate ml-2 truncate inline-block max-w-[200px] align-bottom">
                        {filenames[r.image_id] ?? ""}
                      </span>
                    </span>
                  </button>
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
                    className={`text-[10px] px-2 py-0.5 rounded whitespace-nowrap ${present(r).tone}`}
                    title={present(r).meaning}
                  >
                    {present(r).label}
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

      <Lightbox
        open={open !== null}
        onClose={() => setOpen(null)}
        title={open ? `Image #${open.image_id}` : ""}
        subtitle={
          open ? `${present(open).label} — ${present(open).meaning}` : undefined
        }
      >
        {open && viewport ? (
          <div className="flex flex-col gap-3 items-center">
            <div
              className="max-h-[72vh] w-full"
              style={{
                aspectRatio: `${viewport.width} / ${viewport.height}`,
                maxWidth: "min(100%, 62vh)",
              }}
            >
              {/* Nothing is singled out: every annotated object is a sibling
                  and every prediction an extra, which is the whole-image view
                  the Failures panel deliberately does not give. */}
              <Overlay
                imageId={open.image_id}
                width={viewport.width}
                height={viewport.height}
                viewport={viewport.viewport}
                truth={null}
                prediction={null}
                siblings={wholeImage.siblings}
                extras={wholeImage.predictions}
                showMasks
                showContext
              />
            </div>
            {/* The same four layers the failures panel offers, over the same
                overlay, so switching between the two views does not change
                what a control means. */}
            <div className="grid grid-cols-[46px_1fr_1fr] gap-1 items-center w-[240px]">
              <span />
              <span className="text-[9px] uppercase tracking-wide text-slate text-center">
                Outline
              </span>
              <span className="text-[9px] uppercase tracking-wide text-slate text-center">
                Box
              </span>

              <span className="text-[10px] text-slate">Truth</span>
              <LayerToggle
                label="Outline"
                on={layers.truthMask}
                onClick={() => setLayers((l) => ({ ...l, truthMask: !l.truthMask }))}
              />
              <LayerToggle
                label="Box"
                on={layers.truthBox}
                onClick={() => setLayers((l) => ({ ...l, truthBox: !l.truthBox }))}
              />

              <span className="text-[10px] text-slate">Pred</span>
              <LayerToggle
                label="Outline"
                on={layers.predictionMask}
                onClick={() =>
                  setLayers((l) => ({ ...l, predictionMask: !l.predictionMask }))
                }
              />
              <LayerToggle
                label="Box"
                on={layers.predictionBox}
                onClick={() =>
                  setLayers((l) => ({ ...l, predictionBox: !l.predictionBox }))
                }
              />
            </div>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={() => setLayers(ALL_LAYERS)}
                className="text-[10px] px-2 py-0.5 rounded border border-border/50 text-slate hover:border-brass/40 transition-colors"
              >
                Show all
              </button>
              <button
                type="button"
                onClick={() => setLayers(NO_LAYERS)}
                className="text-[10px] px-2 py-0.5 rounded border border-border/50 text-slate hover:border-brass/40 transition-colors"
              >
                Image only
              </button>
            </div>
            <OverlayLegend hasContext />
            <p className="text-[11px] text-slate max-w-[62vh] text-center leading-relaxed">
              {open.gt_count} annotated object{open.gt_count === 1 ? "" : "s"} ·{" "}
              {open.pred_count} prediction{open.pred_count === 1 ? "" : "s"} ·{" "}
              {Object.entries(open.outcomes)
                .filter(([, n]) => n > 0)
                .map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`)
                .join(" · ")}
            </p>
          </div>
        ) : (
          open && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={api.imageUrl(open.image_id)}
              alt={`Image ${open.image_id}`}
              className="max-w-full max-h-[75vh] object-contain rounded"
            />
          )
        )}
      </Lightbox>

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
