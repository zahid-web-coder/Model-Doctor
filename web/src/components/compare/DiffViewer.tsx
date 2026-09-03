"use client";

import { useMemo, useState } from "react";
import type { Finding, ImageRow, MaskFinding } from "@/lib/api/rows";
import type { AlignedPair, Alignment } from "@/lib/compare/align";
import { fitViewport, isBox, zoomViewport } from "@/lib/compare/geometry";
import { api } from "@/lib/api/client";
import { Overlay, OverlayLegend, type Prediction, type Shape } from "./Overlay";
import { Lightbox } from "@/components/shared/Lightbox";
import type { Viewport } from "@/lib/compare/geometry";

/**
 * Everything one pane needs to redraw itself at full size.
 *
 * The pane assembles this rather than the modal reassembling it: the geometry
 * has already been resolved once, and deriving it twice from the same inputs
 * is how the two copies eventually disagree. The viewport travels with it, so
 * "Zoom to object" carries into the modal instead of silently resetting.
 */
interface ZoomView {
  title: string;
  imageId: number;
  width: number;
  height: number;
  viewport: Viewport;
  truth: Shape;
  prediction: Prediction | null;
  siblings: Shape[];
  extras: Prediction[];
}

/**
 * One ground-truth object, as each run saw it.
 *
 * **Only objects that exist in the ground truth appear here.** Each row is a
 * real annotated object and the two runs' verdicts on it, paired through
 * `(filename, truth box)`. False positives have no ground-truth object to pair
 * against and are reported separately, per image, as extra detections — see
 * the panel below. Putting them in this list would invent a correspondence the
 * data does not contain.
 *
 * Both panes are driven from the same object, so they show the same region of
 * the same file at the same zoom. That is what makes them comparable at a
 * glance: any difference on screen is a difference between the two runs, never
 * a difference in how the two panes were framed.
 */

const OUTCOME_TONE: Record<string, string> = {
  correct: "text-brass",
  false_negative: "text-[#B3452F]",
  false_positive: "text-[#B3452F]",
  poor_localization: "text-slate",
  wrong_class: "text-slate",
};

const label = (outcome: string) => outcome.replace(/_/g, " ");

/** Outcomes that carry a prediction worth drawing. */
const PREDICTED = new Set(["correct", "poor_localization", "wrong_class"]);

/**
 * What to call a drawn prediction.
 *
 * For a `wrong_class` finding the stored `class_name` is the **ground truth**,
 * not what the model said — the schema keeps one class per row and the
 * predicted class is not recoverable. Naming it here would be inventing it, so
 * the label says the class is unavailable and shows only the confidence, which
 * is stored.
 */
function predictionLabel(finding: Finding): string {
  const confidence =
    finding.confidence === null ? "n/a" : finding.confidence.toFixed(2);
  const name =
    finding.outcome === "wrong_class" ? "class n/a" : finding.class_name ?? "class n/a";
  return `${name} · ${confidence}`;
}

interface SideModel {
  title: string;
  finding: Finding;
  image: ImageRow | undefined;
  mask: MaskFinding | undefined;
  extras: Prediction[];
}

function Side({
  model,
  truth,
  siblings,
  zoomed,
  showMasks,
  showContext,
  onOpen,
}: {
  model: SideModel;
  truth: Shape;
  siblings: Shape[];
  zoomed: boolean;
  showMasks: boolean;
  showContext: boolean;
  onOpen: (view: ZoomView) => void;
}) {
  const { finding, image, mask } = model;

  const prediction: Prediction | null = PREDICTED.has(finding.outcome)
    ? {
        box: finding.pred_box,
        polygon: mask?.pred_polygon ?? null,
        tone: finding.outcome === "correct" ? "good" : "bad",
        label: predictionLabel(finding),
      }
    : null;

  // Dimensions are stored per image and are never guessed. Without them there
  // is no honest mapping from stored pixels to the screen, so the pane falls
  // back to the plain photograph and says why rather than drawing boxes at an
  // assumed scale.
  const drawable =
    image && typeof image.width === "number" && typeof image.height === "number";

  const viewport =
    drawable && zoomed && isBox(truth.box)
      ? zoomViewport(truth.box, image.width as number, image.height as number)
      : drawable
        ? fitViewport(image.width as number, image.height as number)
        : null;

  return (
    <div className="flex-1 min-w-0">
      <p className="text-[11px] text-slate mb-1.5 truncate">{model.title}</p>
      {/* The container takes the image's own shape, so a portrait frame is not
          letterboxed into a landscape box. `maxHeight` then caps a very tall
          image — and capping is safe precisely because the photograph lives
          inside the SVG: `preserveAspectRatio` letterboxes the bitmap and every
          shape drawn over it by the same amount, so they cannot come apart. */}
      <div
        className="relative rounded-md overflow-hidden border border-border/40 bg-panel-dark/10"
        style={
          drawable
            ? { aspectRatio: `${image.width} / ${image.height}`, maxHeight: "420px" }
            : { aspectRatio: "4 / 3" }
        }
      >
        {viewport && image ? (
          <button
            type="button"
            title="Open full size"
            onClick={() =>
              onOpen({
                title: model.title,
                imageId: image.id,
                width: image.width as number,
                height: image.height as number,
                viewport,
                truth,
                prediction,
                siblings,
                extras: model.extras,
              })
            }
            className="absolute inset-0 w-full h-full cursor-zoom-in"
          >
            <Overlay
              imageId={image.id}
              width={image.width as number}
              height={image.height as number}
              viewport={viewport}
              truth={truth}
              prediction={prediction}
              siblings={siblings}
              extras={model.extras}
              showMasks={showMasks}
              showContext={showContext}
            />
          </button>
        ) : (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={api.imageUrl(finding.image_id)}
              alt=""
              className="w-full h-full object-contain"
              loading="lazy"
            />
            <p className="absolute inset-x-0 bottom-0 bg-panel-dark/80 text-canvas text-[10px] px-2 py-1">
              Image dimensions were not recorded, so geometry cannot be placed.
            </p>
          </>
        )}
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
  imagesA,
  imagesB,
}: {
  labelA: string;
  labelB: string;
  alignment: Alignment;
  masksA: MaskFinding[];
  masksB: MaskFinding[];
  imagesA: ImageRow[];
  imagesB: ImageRow[];
}) {
  const [index, setIndex] = useState(0);
  const [differingOnly, setDifferingOnly] = useState(true);
  const [zoomed, setZoomed] = useState(true);
  const [showMasks, setShowMasks] = useState(true);
  const [showContext, setShowContext] = useState(true);
  const [zoom, setZoom] = useState<ZoomView | null>(null);

  const visible = useMemo(
    () => (differingOnly ? alignment.pairs.filter((p) => p.differs) : alignment.pairs),
    [alignment.pairs, differingOnly],
  );

  const maskIndexA = useMemo(
    () => new Map(masksA.map((m) => [m.finding_id, m])),
    [masksA],
  );
  const maskIndexB = useMemo(
    () => new Map(masksB.map((m) => [m.finding_id, m])),
    [masksB],
  );
  const imageIndexA = useMemo(() => new Map(imagesA.map((i) => [i.id, i])), [imagesA]);
  const imageIndexB = useMemo(() => new Map(imagesB.map((i) => [i.id, i])), [imagesB]);

  // Every aligned object on a given file, so the ones that are not under
  // discussion can be drawn as context. Ground truth is identical on both
  // sides by definition of the pairing, so one list serves both panes.
  const pairsByFile = useMemo(() => {
    const map = new Map<string, AlignedPair[]>();
    for (const pair of alignment.pairs) {
      const list = map.get(pair.filename);
      if (list) list.push(pair);
      else map.set(pair.filename, [pair]);
    }
    return map;
  }, [alignment.pairs]);

  const safeIndex = visible.length ? Math.min(index, visible.length - 1) : 0;
  const pair: AlignedPair | undefined = visible[safeIndex];

  const extrasHere = pair
    ? alignment.extras.find((e) => e.filename === pair.filename)
    : undefined;

  const siblings: Shape[] = useMemo(() => {
    if (!pair) return [];
    return (pairsByFile.get(pair.filename) ?? [])
      .filter((other) => other.key !== pair.key)
      .map((other) => ({ box: other.truthBox, polygon: other.truthPolygon }));
  }, [pair, pairsByFile]);

  const extraShapes = (
    findings: Finding[],
    masks: Map<number, MaskFinding>,
  ): Prediction[] =>
    findings.map((finding) => ({
      box: finding.pred_box,
      polygon: masks.get(finding.id)?.pred_polygon ?? null,
      // These are unmatched predictions by construction, so "failed or extra"
      // is what they are; the tone is carried explicitly now rather than
      // assumed by the renderer.
      tone: "bad" as const,
      label: finding.class_name ?? "prediction",
    }));

  return (
    <section className="bg-card border border-border/40 rounded-lg p-3">
      <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
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

          {/* View controls, not navigation: changing them holds the reader on
              the object they are looking at. */}
          <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
            <div className="flex rounded-md border border-border/60 overflow-hidden shrink-0">
              {([
                ["Fit", false],
                ["Zoom to object", true],
              ] as const).map(([text, value]) => (
                <button
                  key={text}
                  type="button"
                  onClick={() => setZoomed(value)}
                  aria-pressed={zoomed === value}
                  className={`px-3 py-1 text-[11px] transition-colors ${
                    zoomed === value
                      ? "bg-brass/15 text-ink font-medium"
                      : "text-slate hover:text-ink"
                  }`}
                >
                  {text}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-1.5 text-[11px] text-slate cursor-pointer">
                <input
                  type="checkbox"
                  checked={showMasks}
                  onChange={(e) => setShowMasks(e.target.checked)}
                  className="accent-[#c9a227]"
                />
                Masks
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-slate cursor-pointer">
                <input
                  type="checkbox"
                  checked={showContext}
                  onChange={(e) => setShowContext(e.target.checked)}
                  className="accent-[#c9a227]"
                />
                Context objects
              </label>
            </div>
          </div>

          {pair && (
            <>
              <div className="flex flex-col md:flex-row gap-4">
                <Side
                  model={{
                    title: labelA,
                    finding: pair.a,
                    image: imageIndexA.get(pair.imageIdA),
                    mask: maskIndexA.get(pair.a.id),
                    extras: extraShapes(extrasHere?.a ?? [], maskIndexA),
                  }}
                  truth={{ box: pair.truthBox, polygon: pair.truthPolygon }}
                  siblings={siblings}
                  onOpen={setZoom}
                  zoomed={zoomed}
                  showMasks={showMasks}
                  showContext={showContext}
                />
                <Side
                  model={{
                    title: labelB,
                    finding: pair.b,
                    image: imageIndexB.get(pair.imageIdB),
                    mask: maskIndexB.get(pair.b.id),
                    extras: extraShapes(extrasHere?.b ?? [], maskIndexB),
                  }}
                  truth={{ box: pair.truthBox, polygon: pair.truthPolygon }}
                  siblings={siblings}
                  onOpen={setZoom}
                  zoomed={zoomed}
                  showMasks={showMasks}
                  showContext={showContext}
                />
              </div>
              <div className="mt-3">
                <OverlayLegend hasContext={showContext && siblings.length > 0} />
              </div>
            </>
          )}

          {/* Extras are image-level, never paired to the object above. */}
          {extrasHere && (extrasHere.a.length > 0 || extrasHere.b.length > 0) && (
            <div className="mt-3 pt-3 border-t border-border/30">
              <p className="text-[11px] text-slate mb-1">
                Extra detections on this image — matching no ground-truth object, so
                they are counted per image rather than paired. Drawn thin and dashed
                on the pane that produced them.
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

      <Lightbox
        open={zoom !== null}
        onClose={() => setZoom(null)}
        title={zoom?.title ?? ""}
        subtitle="Ground truth and prediction drawn in the coordinates they were stored in."
        footer={<OverlayLegend hasContext={showContext && zoom !== null && zoom.siblings.length > 0} />}
      >
        {zoom && (
          <div
            className="w-full"
            style={{ aspectRatio: `${zoom.width} / ${zoom.height}`, maxHeight: "72vh" }}
          >
            <Overlay
              imageId={zoom.imageId}
              width={zoom.width}
              height={zoom.height}
              viewport={zoom.viewport}
              truth={zoom.truth}
              prediction={zoom.prediction}
              siblings={zoom.siblings}
              extras={zoom.extras}
              showMasks={showMasks}
              showContext={showContext}
            />
          </div>
        )}
      </Lightbox>
    </section>
  );
}
