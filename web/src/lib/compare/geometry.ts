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
 * One on-screen unit, expressed in image pixels for the current viewport.
 *
 * Strokes and labels are sized from this rather than from a fixed number of
 * image pixels, so they stay the same size on screen whether the reader is
 * looking at the whole image or zoomed into one object. A fixed pixel stroke
 * would be invisible at fit and enormous at zoom.
 */
export const unitOf = (viewport: Viewport): number => viewport.width / 260;
