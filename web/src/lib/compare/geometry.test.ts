import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { boundsOf, labelRect, withoutOverlap, type Rect } from "./geometry.ts";

/**
 * `boundsOf` decides where a label can sit. Getting it wrong does not throw —
 * the tag is simply not drawn, and the confidence score a reader came for is
 * absent with nothing to say it ever existed.
 */

const TRIANGLE = [
  [30, 90],
  [10, 20],
  [70, 40],
];

describe("boundsOf", () => {
  test("prefers the box when one is drawn", () => {
    const at = boundsOf({ box: [10, 20, 90, 60], polygon: TRIANGLE });
    assert.deepEqual(at, { x: 10, y: 20, width: 80, height: 40 });
  });

  test("falls back to the polygon's extent when the box layer is off", () => {
    // The case that hid confidence scores: segmentation runs open with boxes
    // switched off, so `box` is null on every shape the reader is looking at.
    const at = boundsOf({ box: null, polygon: TRIANGLE });
    assert.deepEqual(at, { x: 10, y: 20, width: 60, height: 70 });
  });

  test("null only when nothing was drawn at all", () => {
    assert.equal(boundsOf({ box: null, polygon: null }), null);
  });

  test("a degenerate polygon anchors nothing", () => {
    // Two points bound no area — the same three-point rule the renderer uses
    // to decide whether to draw the polygon in the first place.
    assert.equal(boundsOf({ box: null, polygon: [[1, 1], [2, 2]] }), null);
  });

  test("a reversed box still bounds the object", () => {
    // Stored geometry is trusted, not repaired; `rectOf` already normalises
    // corner order, and a label must not land off-image because of it.
    const at = boundsOf({ box: [90, 60, 10, 20], polygon: null });
    assert.deepEqual(at, { x: 10, y: 20, width: 80, height: 40 });
  });
});

const IMAGE = { width: 1000, height: 1300 };
const overlaps = (a: Rect, b: Rect) =>
  a.x < b.x + b.width && b.x < a.x + a.width &&
  a.y < b.y + b.height && b.y < a.y + a.height;

describe("labelRect", () => {
  test("sits above its anchor when there is room", () => {
    const at = labelRect("staircase 74.3%", 200, 600, 4, IMAGE);
    assert.ok(at.y + at.height <= 600, "the label covered the object it names");
  });

  test("drops below the anchor rather than off the top edge", () => {
    const at = labelRect("GT", 200, 2, 4, IMAGE);
    assert.ok(at.y >= 0);
  });

  test("stays inside the image when the object fills the frame", () => {
    // The case that lost confidence scores outright: a prediction covering the
    // whole photograph leaves nowhere below it for a label to go.
    const at = labelRect("staircase 98.6%", 990, 1400, 4, IMAGE);
    assert.ok(at.x + at.width <= IMAGE.width, "clamped past the right edge");
    assert.ok(at.y + at.height <= IMAGE.height, "clamped past the bottom edge");
  });
});

describe("withoutOverlap", () => {
  const first: Rect = { x: 100, y: 500, width: 200, height: 40 };

  test("leaves a label alone when nothing is in its way", () => {
    assert.deepEqual(withoutOverlap(first, [], IMAGE, 4), first);
  });

  test("two predictions clamped to the same point do not stack invisibly", () => {
    // Image 737: three predictions, two of which clamped to identical
    // coordinates, so a 30.2% score was drawn exactly beneath a 28.9% one and
    // the reader had no way to know a third prediction existed.
    const second = withoutOverlap({ ...first }, [first], IMAGE, 4);
    assert.ok(!overlaps(second, first), "the second label was hidden by the first");
    const third = withoutOverlap({ ...first }, [first, second], IMAGE, 4);
    assert.ok(!overlaps(third, first) && !overlaps(third, second));
  });

  test("moves up when there is no room below", () => {
    const low: Rect = { x: 100, y: 1250, width: 200, height: 40 };
    const moved = withoutOverlap({ ...low }, [low], IMAGE, 4);
    assert.ok(!overlaps(moved, low));
    assert.ok(moved.y >= 0 && moved.y + moved.height <= IMAGE.height);
  });

  test("a label never leaves the image to avoid a collision", () => {
    // Overlapping is readable; off-canvas is not drawn at all, which is the
    // failure this whole path exists to prevent.
    const wall: Rect[] = Array.from({ length: 40 }, (_, i) => ({
      x: 0, y: i * 32, width: 1000, height: 32,
    }));
    const at = withoutOverlap({ x: 0, y: 640, width: 200, height: 32 }, wall, IMAGE, 4);
    assert.ok(at.y >= 0 && at.y + at.height <= IMAGE.height);
  });

  test("only moves vertically, so a label stays over its own object", () => {
    const moved = withoutOverlap({ ...first }, [first], IMAGE, 4);
    assert.equal(moved.x, first.x);
  });
});
