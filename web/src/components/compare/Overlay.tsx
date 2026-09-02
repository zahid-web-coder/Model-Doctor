"use client";

import { api } from "@/lib/api/client";
import {
  isBox,
  isPolygon,
  pointsOf,
  rectOf,
  unitOf,
  viewBoxOf,
  type Viewport,
} from "@/lib/compare/geometry";

/**
 * One image with its stored geometry drawn over it.
 *
 * **The image is inside the SVG, not behind it.** An `<img>` with a separate
 * absolutely-positioned overlay has to keep two coordinate systems agreed —
 * one laid out by CSS, one by a `viewBox` — and they drift the moment the
 * container's aspect ratio stops matching the image's, or the moment zoom is
 * applied to one and not the other. Putting the bitmap in as an `<image>`
 * element makes drift impossible: the photograph and every box and polygon are
 * placed in the same coordinate space, in the units they were stored in, and
 * the `viewBox` moves all of them together. Zoom is then four numbers, not a
 * transform to be mirrored in two places.
 *
 * Nothing here knows which detector produced a run. It draws stored pixels.
 */

/**
 * Overlay colours are chosen for legibility on photographs, not to match the
 * app's palette — which is why they are the only saturated values in the UI.
 *
 * These were originally the design tokens: brass for truth, moss and rust for
 * predictions. All three are muted and two of them are warm browns, so on a
 * tan staircase an annotated object and a failed prediction rendered as nearly
 * the same colour. That is the one confusion this drawing must never allow.
 *
 * The set is now maximally separated by hue and bright enough to hold against
 * an arbitrary photograph: red for ground truth, green for a prediction that
 * matched, amber for one that failed. Context stays muted on purpose — it is
 * there to be seen past, not read.
 *
 * The stroke rule below still carries the distinction on its own, for a reader
 * who cannot rely on hue at all.
 */
const TONE = {
  truth: "#FF3B30", // bright red — the annotated object
  good: "#16E06A", // bright green — prediction that matched
  bad: "#FFB020", // bright amber — prediction that failed or was extra
  context: "#8B8272", // slate — deliberately quiet
} as const;

export interface Shape {
  box: number[] | null;
  polygon: number[][] | null;
}

export interface Prediction extends Shape {
  tone: "good" | "bad";
  label: string;
}

/**
 * Ground truth is solid, predictions are dashed — everywhere, without
 * exception.
 *
 * Colour alone would not carry this: it fails for a colour-blind reader, and it
 * fails again the moment a prediction is drawn in a colour that happens to sit
 * near the brass used for truth. Stroke style is the invariant, so an annotated
 * object can never be read as something a model claimed.
 */
const DASH = (unit: number) => `${unit * 6} ${unit * 4}`;

function Tag({
  text,
  x,
  y,
  unit,
  fill,
  bounds,
}: {
  text: string;
  x: number;
  y: number;
  unit: number;
  fill: string;
  /** The image, so a label can never be placed outside it. */
  bounds: { width: number; height: number };
}) {
  const fontSize = unit * 10;
  const padding = unit * 3;
  // Estimated rather than measured: SVG cannot report text extents before
  // paint, and a background that is a little wide costs nothing, while a
  // measurement pass would cost a render cycle per label.
  const width = text.length * fontSize * 0.58 + padding * 2;
  const height = fontSize + padding * 2;
  const margin = unit * 2;

  // Above the box by preference, below it when there is no room above.
  const preferred = y - height - margin;
  // **Then clamped into the image on both axes.** Only the top edge used to be
  // guarded, so a prediction label sat wherever the caller put it — below the
  // box — and an object filling the frame left nowhere for it to go. On this
  // project's staircases that is the common case, not the rare one: a box
  // covering 98.6% of the image leaves five pixels underneath, and the
  // confidence disappeared off the canvas. A label placed past the edge is
  // simply not drawn, and the number it carries is usually the one the reader
  // came for.
  const at = Math.max(
    margin,
    Math.min(preferred > 0 ? preferred : y + margin, bounds.height - height - margin),
  );
  const left = Math.max(margin, Math.min(x, bounds.width - width - margin));

  return (
    <g>
      <rect x={left} y={at} width={width} height={height} rx={unit * 2} fill={fill} />
      <text
        x={left + padding}
        y={at + padding + fontSize * 0.8}
        fontSize={fontSize}
        fill="#F4EFE3"
        fontFamily="ui-monospace, monospace"
      >
        {text}
      </text>
    </g>
  );
}

function Outline({
  shape,
  colour,
  unit,
  dashed,
  opacity = 1,
  strokeScale = 1,
  showMask,
}: {
  shape: Shape;
  colour: string;
  unit: number;
  dashed: boolean;
  opacity?: number;
  strokeScale?: number;
  showMask: boolean;
}) {
  return (
    <g opacity={opacity}>
      {showMask && isPolygon(shape.polygon) && (
        <polygon
          points={pointsOf(shape.polygon)}
          fill={colour}
          fillOpacity={0.18}
          stroke={colour}
          strokeWidth={unit * 1.2 * strokeScale}
          strokeDasharray={dashed ? DASH(unit) : undefined}
        />
      )}
      {isBox(shape.box) && (
        <rect
          {...rectOf(shape.box)}
          fill="none"
          stroke={colour}
          strokeWidth={unit * 2 * strokeScale}
          strokeDasharray={dashed ? DASH(unit) : undefined}
        />
      )}
    </g>
  );
}

export function Overlay({
  imageId,
  width,
  height,
  viewport,
  truth,
  prediction,
  siblings,
  extras,
  showMasks,
  showContext,
}: {
  imageId: number;
  width: number;
  height: number;
  viewport: Viewport;
  truth: Shape | null;
  prediction: Prediction | null;
  siblings: Shape[];
  extras: Shape[];
  showMasks: boolean;
  showContext: boolean;
}) {
  const unit = unitOf(viewport);

  return (
    <svg
      viewBox={viewBoxOf(viewport)}
      preserveAspectRatio="xMidYMid meet"
      className="w-full h-full block"
      role="img"
    >
      {/* preserveAspectRatio on the bitmap too: if the stored dimensions were
          ever wrong, this letterboxes visibly rather than stretching silently,
          so the mismatch is something a reader can see. */}
      <image
        href={api.imageUrl(imageId)}
        x={0}
        y={0}
        width={width}
        height={height}
        preserveAspectRatio="xMidYMid meet"
      />

      {/* Context first, so it can never paint over the object in question. */}
      {showContext &&
        siblings.map((sibling, index) => (
          <Outline
            key={`sibling-${index}`}
            shape={sibling}
            colour={TONE.context}
            unit={unit}
            dashed={false}
            opacity={0.45}
            strokeScale={0.6}
            showMask={false}
          />
        ))}

      {/* Extras belong to the image, not to the object being compared. They are
          drawn thin and unlabelled per-shape; the count and its meaning are
          stated in the panel below the viewer. */}
      {extras.map((extra, index) => (
        <Outline
          key={`extra-${index}`}
          shape={extra}
          colour={TONE.bad}
          unit={unit}
          dashed
          opacity={0.75}
          strokeScale={0.7}
          showMask={showMasks}
        />
      ))}

      {truth && (
        <Outline
          shape={truth}
          colour={TONE.truth}
          unit={unit}
          dashed={false}
          showMask={showMasks}
        />
      )}

      {prediction && (
        <Outline
          shape={prediction}
          colour={TONE[prediction.tone]}
          unit={unit}
          dashed
          showMask={showMasks}
        />
      )}

      {truth && isBox(truth.box) && (
        <Tag
          text="GT"
          x={rectOf(truth.box).x}
          y={rectOf(truth.box).y}
          unit={unit}
          fill={TONE.truth}
          bounds={{ width, height }}
        />
      )}

      {prediction && isBox(prediction.box) && (
        <Tag
          text={prediction.label}
          x={rectOf(prediction.box).x}
          y={rectOf(prediction.box).y + rectOf(prediction.box).height + unit * 14}
          unit={unit}
          fill={TONE[prediction.tone]}
          bounds={{ width, height }}
        />
      )}
    </svg>
  );
}

/** The key to the drawing, stated once beneath the two panes. */
export function OverlayLegend({ hasContext }: { hasContext: boolean }) {
  const items: { colour: string; dashed: boolean; label: string; faint?: boolean }[] = [
    { colour: TONE.truth, dashed: false, label: "Ground truth (solid)" },
    { colour: TONE.good, dashed: true, label: "Prediction, matched (dashed)" },
    { colour: TONE.bad, dashed: true, label: "Prediction, failed or extra (dashed)" },
  ];
  if (hasContext) {
    items.push({
      colour: TONE.context,
      dashed: false,
      label: "Other annotated objects — context only",
      faint: true,
    });
  }

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5 text-[10px] text-slate">
          <svg width="18" height="8" aria-hidden className={item.faint ? "opacity-50" : ""}>
            <line
              x1="0"
              y1="4"
              x2="18"
              y2="4"
              stroke={item.colour}
              strokeWidth="2"
              strokeDasharray={item.dashed ? "5 3" : undefined}
            />
          </svg>
          {item.label}
        </span>
      ))}
    </div>
  );
}
