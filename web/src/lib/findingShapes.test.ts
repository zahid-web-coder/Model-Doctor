import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  ALL_LAYERS, NO_LAYERS, contextFor, defaultLayers, hasOutline, labelFor,
  predictionShapeOf, toneFor, truthShapeOf,
  type Layers,
} from "./findingShapes.ts";
import type { Finding, Outcome } from "./api/rows.ts";

/**
 * The overlay draws a claim about where the model looked and where the object
 * actually was. Getting a layer wrong here does not throw — it silently draws
 * a prediction where there was none, or hides ground truth that exists, and
 * the reader believes the picture.
 *
 * The structural cases matter most: a false positive has no ground truth and a
 * false negative has no prediction, and those absences are the finding rather
 * than missing data. Verified against run 5, where all 29 false positives
 * store no truth polygon and all 33 false negatives store no predicted one.
 */

function finding(overrides: Partial<Finding> = {}): Finding {
  return {
    id: 1, run_id: 5, image_id: 737,
    outcome: "correct" as Outcome,
    class_id: 0, class_name: "staircase",
    confidence: 0.962, iou: 0.9,
    pred_box: [10, 10, 90, 90],
    truth_box: [12, 12, 88, 88],
    truth_polygon: [[10, 10], [90, 10], [90, 90]],
    ...overrides,
  };
}

const PRED_POLY = [[11, 11], [89, 11], [89, 89]];
const NONE: Layers = {
  truthMask: false, truthBox: false, predictionMask: false, predictionBox: false,
};

describe("toneFor", () => {
  test("only a matched prediction reads as good", () => {
    assert.equal(toneFor("correct"), "good");
    for (const outcome of ["wrong_class", "poor_localization", "false_positive", "false_negative"] as const) {
      assert.equal(toneFor(outcome), "bad", `${outcome} was drawn as a success`);
    }
  });

  test("poor localization is not softened into a success", () => {
    // It landed on the object and still failed to match, which is exactly what
    // the overlay is being asked to show.
    assert.equal(toneFor("poor_localization"), "bad");
  });
});

describe("truthShapeOf", () => {
  test("a false positive has no ground truth to draw", () => {
    const shape = truthShapeOf(
      finding({ outcome: "false_positive", truth_box: null, truth_polygon: null }),
      ALL_LAYERS
    );
    assert.equal(shape, null);
  });

  test("the mask layer hides the polygon without hiding the box", () => {
    const shape = truthShapeOf(finding(), { ...ALL_LAYERS, truthMask: false });
    assert.equal(shape?.polygon, null);
    assert.deepEqual(shape?.box, [12, 12, 88, 88]);
  });

  test("the box layer hides the box without hiding the polygon", () => {
    const shape = truthShapeOf(finding(), { ...ALL_LAYERS, truthBox: false });
    assert.equal(shape?.box, null);
    assert.ok(shape?.polygon);
  });

  test("every layer off draws nothing at all", () => {
    assert.equal(truthShapeOf(finding(), NONE), null);
  });
});

describe("predictionShapeOf", () => {
  test("a false negative has no prediction to draw", () => {
    const shape = predictionShapeOf(
      finding({ outcome: "false_negative", pred_box: null, confidence: null }),
      null,
      ALL_LAYERS
    );
    assert.equal(shape, null);
  });

  test("the predicted polygon comes from the mask table, not the finding", () => {
    // They are separate tables; a finding never carries its own predicted
    // outline, so passing null must not fall back to the truth polygon.
    const shape = predictionShapeOf(finding(), PRED_POLY, ALL_LAYERS);
    assert.deepEqual(shape?.polygon, PRED_POLY);
    assert.notDeepEqual(shape?.polygon, finding().truth_polygon);
  });

  test("a finding the mask pass never covered still draws its box", () => {
    const shape = predictionShapeOf(finding(), null, ALL_LAYERS);
    assert.equal(shape?.polygon, null);
    assert.deepEqual(shape?.box, [10, 10, 90, 90]);
  });

  test("carries the tone its outcome earns", () => {
    const bad = predictionShapeOf(finding({ outcome: "false_positive" }), PRED_POLY, ALL_LAYERS);
    assert.equal(bad?.tone, "bad");
  });
});

describe("labelFor", () => {
  test("shows confidence when there is one", () => {
    assert.equal(labelFor(finding({ confidence: 0.962 })), "staircase 96.2%");
  });

  test("omits confidence rather than printing zero", () => {
    // A false negative has nothing to be confident about; rendering 0.0% would
    // be the same error as showing a null metric as a number.
    const label = labelFor(finding({ outcome: "false_negative", confidence: null }));
    assert.equal(label, "staircase");
    assert.ok(!label.includes("0"));
  });

  test("survives a finding with no class name", () => {
    assert.match(labelFor(finding({ class_name: null, confidence: null })), /prediction/);
  });
});

describe("hasOutline", () => {
  test("true when either side has a polygon", () => {
    assert.equal(hasOutline(finding(), null), true);
    assert.equal(hasOutline(finding({ truth_polygon: null }), PRED_POLY), true);
  });

  test("false when a box-only detector produced no outlines", () => {
    // The panel then shows the photograph rather than an empty SVG, which
    // would read as a broken viewer.
    assert.equal(hasOutline(finding({ truth_polygon: null }), null), false);
  });

  test("a degenerate polygon does not count as an outline", () => {
    // Two points bound no area. The backend uses the same three-point rule.
    assert.equal(hasOutline(finding({ truth_polygon: [[1, 1], [2, 2]] }), null), false);
  });
});

describe("defaultLayers", () => {
  test("a segmentation run opens on outlines, not boxes", () => {
    // The box is the polygon's axis-aligned hull, so showing both draws a
    // rectangle round every outline for no added information.
    const layers = defaultLayers(true);
    assert.equal(layers.truthBox, false);
    assert.equal(layers.predictionBox, false);
    assert.equal(layers.truthMask, true);
    assert.equal(layers.predictionMask, true);
  });

  test("a run with no outlines still shows its boxes", () => {
    // Otherwise a box-only detector would open on an empty overlay.
    assert.ok(defaultLayers(false).truthBox && defaultLayers(false).predictionBox);
  });

  test("masks are never off by default", () => {
    for (const outlines of [true, false]) {
      const layers = defaultLayers(outlines);
      assert.ok(layers.truthMask && layers.predictionMask);
    }
  });
});

describe("isolating one layer", () => {
  test("the predicted outline can be shown entirely on its own", () => {
    // The point of splitting the box switch per side: a reader comparing the
    // prediction against nothing else needs every other layer gone.
    const only: Layers = {
      truthMask: false, truthBox: false, predictionMask: true, predictionBox: false,
    };
    assert.equal(truthShapeOf(finding(), only), null);
    const pred = predictionShapeOf(finding(), PRED_POLY, only);
    assert.deepEqual(pred?.polygon, PRED_POLY);
    assert.equal(pred?.box, null);
  });

  test("truth outline against prediction box is reachable", () => {
    // Impossible while one switch governed both boxes.
    const mixed: Layers = {
      truthMask: true, truthBox: false, predictionMask: false, predictionBox: true,
    };
    assert.equal(truthShapeOf(finding(), mixed)?.box, null);
    assert.ok(truthShapeOf(finding(), mixed)?.polygon);
    assert.ok(predictionShapeOf(finding(), PRED_POLY, mixed)?.box);
    assert.equal(predictionShapeOf(finding(), PRED_POLY, mixed)?.polygon, null);
  });

  test("NO_LAYERS draws nothing at all", () => {
    assert.equal(truthShapeOf(finding(), NO_LAYERS), null);
    assert.equal(predictionShapeOf(finding(), PRED_POLY, NO_LAYERS), null);
  });
});

describe("contextFor", () => {
  /** The image-737 shape: the merged prediction belongs to a different finding. */
  const IMAGE_737 = [
    finding({ id: 1225, outcome: "poor_localization", pred_box: [0, 0, 90, 90] }),
    finding({ id: 1226, outcome: "false_positive", truth_box: null, truth_polygon: null }),
    finding({ id: 1228, outcome: "false_negative", pred_box: null, confidence: null }),
  ];

  test("a related prediction is not hidden because it belongs to another finding", () => {
    // The whole point. Inspecting #1228, the prediction the matcher gave to
    // #1225 — which also covers 80% of #1228 — must be drawn, or the merge
    // that caused this false negative is invisible.
    const ctx = contextFor(1228, IMAGE_737, { 1225: PRED_POLY }, ALL_LAYERS);
    const boxes = ctx.predictions.map((p) => p.box);
    assert.ok(
      boxes.some((b) => b && b[2] === 90),
      "the prediction assigned to #1225 was not offered as context",
    );
  });

  test("the finding being inspected is never duplicated into its own context", () => {
    const ctx = contextFor(1225, IMAGE_737, { 1225: PRED_POLY }, ALL_LAYERS);
    assert.equal(ctx.predictions.length, 1, "only #1226 predicts besides #1225");
    assert.equal(ctx.siblings.length, 1, "only #1228 is annotated besides #1225");
  });

  test("other annotated objects come through as siblings", () => {
    const ctx = contextFor(1226, IMAGE_737, {}, ALL_LAYERS);
    assert.equal(ctx.siblings.length, 2, "#1225 and #1228 are both annotated");
  });

  test("a false positive contributes a prediction but no sibling", () => {
    const ctx = contextFor(1225, IMAGE_737, {}, ALL_LAYERS);
    assert.ok(ctx.predictions.length >= 1);
    assert.ok(!ctx.siblings.some((s) => s.box === null && s.polygon === null));
  });

  test("context carries each prediction's own tone, not a blanket failure", () => {
    // Painting a neighbouring *correct* prediction as a failure would
    // misreport it, which is why extras carry a tone at all.
    const withCorrect = [
      finding({ id: 1, outcome: "correct" }),
      finding({ id: 2, outcome: "false_positive", truth_box: null, truth_polygon: null }),
    ];
    const ctx = contextFor(2, withCorrect, {}, ALL_LAYERS);
    assert.equal(ctx.predictions[0].tone, "good");
  });

  test("layers still govern what context draws", () => {
    const ctx = contextFor(1228, IMAGE_737, { 1225: PRED_POLY }, NO_LAYERS);
    assert.deepEqual(ctx, { siblings: [], predictions: [] });
  });

  test("an image with only this finding has no context", () => {
    const ctx = contextFor(1, [finding({ id: 1 })], {}, ALL_LAYERS);
    assert.deepEqual(ctx, { siblings: [], predictions: [] });
  });
});
