"use client";

import { useMemo, useState } from "react";
import type { Finding, MaskFinding } from "@/lib/api/rows";
import type { AlignedPair, Alignment } from "@/lib/compare/align";
import { api } from "@/lib/api/client";

/**
 * One ground-truth object, as each run saw it.
 *
 * **Only objects that exist in the ground truth appear here.** Each row is a
 * real annotated object and the two runs' verdicts on it, paired through
 * `(filename, truth box)`. False positives have no ground-truth object to pair
 * against and are reported separately, per image, as extra detections — see
 * the panel below. Putting them in this list would invent a correspondence the
 * data does not contain.
 */

const OUTCOME_TONE: Record<string, string> = {
  correct: "text-brass",
  false_negative: "text-[#B3452F]",
  false_positive: "text-[#B3452F]",
  poor_localization: "text-slate",
  wrong_class: "text-slate",
};

const label = (outcome: string) => outcome.replace(/_/g, " ");

function Side({
  finding,
  imageId,
  mask,
  title,
}: {
  finding: Finding;
  imageId: number;
  mask: MaskFinding | undefined;
  title: string;
}) {
  return (
    <div className="flex-1 min-w-0">
      <p className="text-[11px] text-slate mb-1.5 truncate">{title}</p>
      <div className="relative rounded-md overflow-hidden border border-border/40 bg-panel-dark/10 aspect-[4/3]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={api.imageUrl(imageId)}
          alt=""
          className="w-full h-full object-contain"
          loading="lazy"
        />
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
        <dt className="text-slate">Outcome</dt>
        <dd className={`text-right font-medium ${OUTCOME_TONE[finding.outcome] ?? "text-ink"}`}>
          {label(finding.outcome)}
        </dd>
        <dt className="text-slate">Confidence</dt>
        <dd className="text-right font-mono text-ink">
          {finding.confidence === null ? "n/a" : finding.confidence.toFixed(3)}
        </dd>
        <dt className="text-slate">Box IoU</dt>
        <dd className="text-right font-mono text-ink">
          {finding.iou === null ? "n/a" : finding.iou.toFixed(3)}
        </dd>
        <dt className="text-slate">Mask IoU</dt>
        <dd className="text-right font-mono text-ink">
          {mask?.mask_iou == null ? "n/a" : mask.mask_iou.toFixed(3)}
        </dd>
        <dt className="text-slate">Class</dt>
        <dd className="text-right text-ink truncate">
          {/* Never inferred for a wrong-class finding: the stored row is the
              only authority on what was predicted. */}
          {finding.class_name ?? "n/a"}
        </dd>
      </dl>
    </div>
  );
}

export function DiffViewer({
  labelA,
  labelB,
  alignment,
  masksA,
  masksB,
}: {
  labelA: string;
  labelB: string;
  alignment: Alignment;
  masksA: MaskFinding[];
  masksB: MaskFinding[];
}) {
  const [index, setIndex] = useState(0);
  const [differingOnly, setDifferingOnly] = useState(true);

  const visible = useMemo(
    () => (differingOnly ? alignment.pairs.filter((p) => p.differs) : alignment.pairs),
    [alignment.pairs, differingOnly],
  );

  const maskFor = (masks: MaskFinding[], findingId: number) =>
    masks.find((m) => m.finding_id === findingId);

  const safeIndex = visible.length ? Math.min(index, visible.length - 1) : 0;
  const pair: AlignedPair | undefined = visible[safeIndex];

  const extrasHere = pair
    ? alignment.extras.find((e) => e.filename === pair.filename)
    : undefined;

  return (
    <section className="bg-card border border-border/40 rounded-lg p-3">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-[13px] font-medium text-ink">Side by side</h3>
          <p className="text-[11px] text-slate mt-0.5">
            {alignment.pairs.length} ground-truth objects aligned ·{" "}
            {alignment.totalDiffering} where the runs disagree
          </p>
        </div>
        <label className="flex items-center gap-1.5 text-[11px] text-slate shrink-0 cursor-pointer">
          <input
            type="checkbox"
            checked={differingOnly}
            onChange={(e) => {
              setDifferingOnly(e.target.checked);
              setIndex(0);
            }}
            className="accent-[#c9a227]"
          />
          Disagreements only
        </label>
      </div>

      {visible.length === 0 ? (
        <p className="text-[12px] text-slate py-6 text-center">
          {alignment.pairs.length === 0
            ? "No aligned objects to show."
            : "The two runs agree on every aligned object."}
        </p>
      ) : (
        <>
          <div className="flex items-center justify-between gap-3 mb-3">
            <button
              type="button"
              onClick={() => setIndex((i) => Math.max(0, i - 1))}
              disabled={safeIndex === 0}
              className="px-3 py-1.5 rounded-md border border-border/60 text-[12px] text-ink disabled:opacity-40 hover:border-brass/50 transition-colors"
            >
              ← Previous
            </button>
            <span className="text-[11px] text-slate font-mono truncate">
              {safeIndex + 1} / {visible.length} · {pair?.filename.slice(0, 34)}
            </span>
            <button
              type="button"
              onClick={() => setIndex((i) => Math.min(visible.length - 1, i + 1))}
              disabled={safeIndex >= visible.length - 1}
              className="px-3 py-1.5 rounded-md border border-border/60 text-[12px] text-ink disabled:opacity-40 hover:border-brass/50 transition-colors"
            >
              {/* The label has to track the filter. Stepping through every
                  aligned object under a button that promises the next
                  disagreement would misdescribe what the button does — most of
                  those steps land on objects the two runs agreed about. */}
              {differingOnly ? "Next differing →" : "Next →"}
            </button>
          </div>

          {pair && (
            <div className="flex gap-4">
              <Side
                finding={pair.a}
                imageId={pair.imageIdA}
                mask={maskFor(masksA, pair.a.id)}
                title={labelA}
              />
              <Side
                finding={pair.b}
                imageId={pair.imageIdB}
                mask={maskFor(masksB, pair.b.id)}
                title={labelB}
              />
            </div>
          )}

          {/* Extras are image-level, never paired to the object above. */}
          {extrasHere && (extrasHere.a.length > 0 || extrasHere.b.length > 0) && (
            <div className="mt-3 pt-3 border-t border-border/30">
              <p className="text-[11px] text-slate mb-1">
                Extra detections on this image — matching no ground-truth object, so
                they are counted per image rather than paired
              </p>
              <div className="flex gap-4 text-[11px]">
                <span className="flex-1 text-ink">
                  {labelA}: <span className="font-mono">{extrasHere.a.length}</span>
                </span>
                <span className="flex-1 text-ink">
                  {labelB}: <span className="font-mono">{extrasHere.b.length}</span>
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
