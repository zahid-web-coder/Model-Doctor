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

const TONE = {
  truth: "#BB8F51", // brass
  good: "#66805A", // evidence-moss
  bad: "#A65C48", // evidence-rust
  context: "#8B8272", // slate
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
}: {
  text: string;
  x: number;
  y: number;
  unit: number;
  fill: string;
}) {
  const fontSize = unit * 10;
  const padding = unit * 3;
  // Estimated rather than measured: SVG cannot report text extents before
  // paint, and a background that is a little wide costs nothing, while a
  // measurement pass would cost a render cycle per label.
  const width = text.length * fontSize * 0.58 + padding * 2;
  const height = fontSize + padding * 2;
  // Above the box, unless that would leave the viewport.
  const top = y - height - unit * 2;
  const at = top > 0 ? top : y + unit * 2;

  return (
    <g>
      <rect x={x} y={at} width={width} height={height} rx={unit * 2} fill={fill} />
      <text
        x={x + padding}
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
        />
      )}

      {prediction && isBox(prediction.box) && (
        <Tag
          text={prediction.label}
          x={rectOf(prediction.box).x}
          y={rectOf(prediction.box).y + rectOf(prediction.box).height + unit * 14}
          unit={unit}
          fill={TONE[prediction.tone]}
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
