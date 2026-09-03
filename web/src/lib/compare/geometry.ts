/**
 * Turning stored geometry into SVG, without ever leaving the image's own
 * coordinate space.
 *
 * Every box and polygon Model Doctor stores is in **original-image pixels** —
 * `app/inference.py` and `app/rfdetr_adapter.py` both say so explicitly, and
 * both rescale detections back to that space before storing them. Whichever
 * detector produced a run, its geometry lands in the same frame as the ground
 * truth it is being compared against.
 *
 * So nothing here converts, normalises or rescales. The numbers are drawn as
 * they are stored, and a `viewBox` in the same units does the rest — which is
 * also how zoom works: change the window onto the image, not the coordinates.
 */

/** A window onto the image, in image pixels. */
export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The whole image. */
export const fitViewport = (width: number, height: number): Viewport => ({
  x: 0,
  y: 0,
  width,
  height,
});

export const viewBoxOf = (viewport: Viewport): string =>
  `${viewport.x} ${viewport.y} ${viewport.width} ${viewport.height}`;

const clamp = (value: number, low: number, high: number): number =>
  Math.min(Math.max(value, low), high);

/**
 * A window centred on one object, with room around it for context.
 *
 * Widened to the **image's** aspect ratio rather than the object's, so the
 * window fills its container instead of being letterboxed inside it. The
 * container keeps the image's shape at every zoom level, which is what stops
 * the layout jumping as the reader steps between objects of different shapes.
 *
 * Falls back to the whole image when the padded window would exceed it —
 * zooming "in" past the image bounds would otherwise scroll empty space into
 * view.
 */
export function zoomViewport(
  box: number[],
  imageWidth: number,
  imageHeight: number,
  padding = 0.6,
): Viewport {
  const [x1, y1, x2, y2] = box;
  const boxWidth = Math.max(x2 - x1, 1);
  const boxHeight = Math.max(y2 - y1, 1);

  let width = boxWidth * (1 + padding * 2);
  let height = boxHeight * (1 + padding * 2);

  const aspect = imageWidth / imageHeight;
  if (width / height < aspect) width = height * aspect;
  else height = width / aspect;

  if (width >= imageWidth || height >= imageHeight) {
    return fitViewport(imageWidth, imageHeight);
  }

  const x = clamp((x1 + x2) / 2 - width / 2, 0, imageWidth - width);
  const y = clamp((y1 + y2) / 2 - height / 2, 0, imageHeight - height);
  return { x, y, width, height };
}

/** A stored box is usable only when it has all four numbers. */
export function isBox(box: number[] | null | undefined): box is number[] {
  return Array.isArray(box) && box.length === 4 && box.every(Number.isFinite);
}

/** A stored polygon needs three points to bound an area, matching the backend. */
export function isPolygon(
  polygon: number[][] | null | undefined,
): polygon is number[][] {
  return Array.isArray(polygon) && polygon.length >= 3;
}

/** SVG `rect` attributes for a stored `[x1, y1, x2, y2]`. */
export function rectOf(box: number[]): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const [x1, y1, x2, y2] = box;
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  };
}

/** SVG `points` for a stored polygon. */
export const pointsOf = (polygon: number[][]): string =>
  polygon.map(([x, y]) => `${x},${y}`).join(" ");

/**
 * The rectangle a shape occupies — from its box when one is drawn, and from its
 * polygon's extent otherwise.
 *
 * **A label must not depend on the box layer being on.** Tags used to anchor on
 * `box` alone, so with boxes switched off — the default on a segmentation run,
 * where the outline is the whole point — the confidence score silently
 * disappeared. The polygon bounds the same object, so it can anchor the same
 * label. Returns null only when there is no geometry at all, which is a shape
 * that was never drawn.
 */
export function boundsOf(shape: {
  box: number[] | null;
  polygon: number[][] | null;
}): { x: number; y: number; width: number; height: number } | null {
  if (isBox(shape.box)) return rectOf(shape.box);
  if (!isPolygon(shape.polygon)) return null;
  const xs = shape.polygon.map(([x]) => x);
  const ys = shape.polygon.map(([, y]) => y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y };
}

/** A label's box, in image pixels. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * How big a label's background is, estimated from its text.
 *
 * Estimated rather than measured: SVG cannot report text extents before paint,
 * and a background that is a little wide costs nothing, while a measurement
 * pass would cost a render cycle per label.
 */
export function labelSize(
  text: string,
  unit: number,
): { width: number; height: number; fontSize: number; padding: number } {
  const fontSize = unit * 10;
  const padding = unit * 3;
  return {
    width: text.length * fontSize * 0.58 + padding * 2,
    height: fontSize + padding * 2,
    fontSize,
    padding,
  };
}

/**
 * Where a label sits: above its anchor by preference, below when there is no
 * room above, and clamped inside the image on both axes.
 *
 * A label placed past the edge is simply not drawn, and the number it carries
 * is usually the one the reader came for. On this project's staircases that is
 * the common case rather than the rare one — a prediction covering 98.6% of
 * its image leaves five pixels underneath.
 */
export function labelRect(
  text: string,
  x: number,
  y: number,
  unit: number,
  bounds: { width: number; height: number },
): Rect {
  const { width, height } = labelSize(text, unit);
  const margin = unit * 2;
  const preferred = y - height - margin;
  return {
    x: Math.max(margin, Math.min(x, bounds.width - width - margin)),
    y: Math.max(
      margin,
      Math.min(preferred > 0 ? preferred : y + margin, bounds.height - height - margin),
    ),
    width,
    height,
  };
}

/**
 * Push a label clear of the ones already placed.
 *
 * **Clamping alone loses labels.** Two predictions that both span the image
 * clamp to the same point against the bottom edge, and the second is drawn
 * exactly underneath the first — on image 737 that hid a 30.2% score behind a
 * 28.9% one, with nothing on screen to suggest a third prediction existed.
 * Overlapping shapes are the whole subject of this view, so their labels
 * collide by default rather than by accident.
 *
 * Steps down first, then up, and gives up rather than leaving the image: a
 * label drawn over another is still readable, and one outside the frame is not
 * drawn at all.
 */
export function withoutOverlap(
  rect: Rect,
  placed: Rect[],
  bounds: { width: number; height: number },
  gap: number,
): Rect {
  const hits = (candidate: Rect) =>
    placed.some(
      (other) =>
        candidate.x < other.x + other.width &&
        other.x < candidate.x + candidate.width &&
        candidate.y < other.y + other.height &&
        other.y < candidate.y + candidate.height,
    );
  if (!hits(rect)) return rect;

  const step = rect.height + gap;
  for (const direction of [1, -1]) {
    let candidate = rect;
    for (let attempt = 0; attempt <= placed.length; attempt += 1) {
      candidate = { ...candidate, y: candidate.y + direction * step };
      if (candidate.y < 0 || candidate.y + candidate.height > bounds.height) break;
      if (!hits(candidate)) return candidate;
    }
  }
  return rect;
}

/**
 * One on-screen unit, expressed in image pixels for the current viewport.
 *
 * Strokes and labels are sized from this rather than from a fixed number of
 * image pixels, so they stay the same size on screen whether the reader is
 * looking at the whole image or zoomed into one object. A fixed pixel stroke
 * would be invisible at fit and enormous at zoom.
 */
export const unitOf = (viewport: Viewport): number => viewport.width / 260;
