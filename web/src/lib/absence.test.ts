import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { absence, hasPrediction, hasTruth, type Field } from "./absence.ts";
import { ALL_OUTCOMES, type Outcome } from "./api/rows.ts";

/**
 * An empty cell must say why it is empty.
 *
 * The table used to render three different situations as the same "n/a": a
 * false positive with no ground truth, a false negative with no prediction,
 * and a wrong_class finding whose predicted class the schema cannot store.
 * Read as missing data, the first of those suggests the *image* was never
 * annotated — a different problem calling for a different response, and one
 * that would wrongly invite excluding the finding from the metrics.
 */

const FIELDS: Field[] = ["prediction", "truth", "confidence", "iou"];

describe("false positives", () => {
  test("say no ground-truth object matched, not that data is missing", () => {
    const entry = absence("false_positive", "truth");
    assert.ok(entry);
    assert.match(entry.long, /No matching ground-truth object/);
    assert.equal(entry.structural, true);
  });

  test("scope the statement to the prediction, never to the image", () => {
    // This is the distinction the whole module exists for. A false positive
    // says nothing about whether the image carries annotations, and the
    // wording must not let a reader conclude otherwise.
    const entry = absence("false_positive", "truth");
    assert.ok(entry);
    assert.match(entry.long, /Other objects in the same image may well be annotated/);
    assert.doesNotMatch(entry.long, /unlabelled|unannotated|no annotations/i);
  });

  test("still carry a prediction and a confidence", () => {
    // The model did predict something — that is what makes it a false
    // positive rather than nothing at all.
    assert.equal(absence("false_positive", "prediction"), null);
    assert.equal(absence("false_positive", "confidence"), null);
    assert.equal(hasPrediction("false_positive"), true);
    assert.equal(hasTruth("false_positive"), false);
  });

  test("have no IoU, because there is nothing to overlap with", () => {
    assert.ok(absence("false_positive", "iou"));
  });
});

describe("false negatives", () => {
  test("say the object was not detected", () => {
    const entry = absence("false_negative", "prediction");
    assert.ok(entry);
    assert.match(entry.long, /No matching prediction/);
    assert.match(entry.long, /model did not find it/);
    assert.equal(entry.structural, true);
  });

  test("make clear a ground-truth object does exist", () => {
    // The inverse of the false-positive case, and the one most likely to be
    // misread as "no data" when it means "the model missed it".
    const entry = absence("false_negative", "prediction");
    assert.ok(entry);
    assert.match(entry.long, /annotated object is present/i);
  });

  test("have no confidence, because nothing was predicted", () => {
    const entry = absence("false_negative", "confidence");
    assert.ok(entry);
    assert.match(entry.long, /no prediction to score/);
  });

  test("keep their ground-truth class", () => {
    assert.equal(absence("false_negative", "truth"), null);
    assert.equal(hasTruth("false_negative"), true);
    assert.equal(hasPrediction("false_negative"), false);
  });
});

describe("wrong_class", () => {
  test("explains the predicted class is not recorded rather than absent", () => {
    const entry = absence("wrong_class", "prediction");
    assert.ok(entry);
    assert.match(entry.long, /not recoverable/);
    assert.match(entry.long, /rather than inferred/);
  });

  test("counts as having made a prediction", () => {
    // Something was predicted; only *which* class is unavailable.
    assert.equal(hasPrediction("wrong_class"), true);
    assert.equal(hasTruth("wrong_class"), true);
  });
});

describe("matched outcomes", () => {
  test("correct and poor_localization have no absent fields", () => {
    for (const outcome of ["correct", "poor_localization"] as Outcome[]) {
      for (const field of FIELDS) {
        assert.equal(
          absence(outcome, field),
          null,
          `${outcome}.${field} should carry a value`
        );
      }
    }
  });
});

describe("the distinction that must not be lost", () => {
  test("no explanation conflates a false positive with an unlabelled image", () => {
    // An unlabelled image produces false positives; a false positive does not
    // imply an unlabelled image. Nothing here may state or imply the reverse.
    for (const outcome of ALL_OUTCOMES) {
      for (const field of FIELDS) {
        const entry = absence(outcome, field);
        if (!entry) continue;
        assert.doesNotMatch(
          entry.long,
          /image (has|had|was) (no|never)/i,
          `${outcome}.${field} makes a claim about the image`
        );
      }
    }
  });

  test("every absence is structural, so none reads as a data gap", () => {
    // If a non-structural absence is ever added it must be worded as a real
    // gap, and this test should be revisited rather than deleted.
    for (const outcome of ALL_OUTCOMES) {
      for (const field of FIELDS) {
        const entry = absence(outcome, field);
        if (entry) assert.equal(entry.structural, true, `${outcome}.${field}`);
      }
    }
  });
});

describe("presentation", () => {
  test("every explanation has both a short and a long form", () => {
    for (const outcome of ALL_OUTCOMES) {
      for (const field of FIELDS) {
        const entry = absence(outcome, field);
        if (!entry) continue;
        assert.ok(entry.short.length > 0, `${outcome}.${field} short`);
        assert.ok(entry.long.length > 20, `${outcome}.${field} long`);
      }
    }
  });

  test("short forms stay short enough for a dense table cell", () => {
    for (const outcome of ALL_OUTCOMES) {
      for (const field of FIELDS) {
        const entry = absence(outcome, field);
        if (entry) assert.ok(entry.short.length <= 14, `${outcome}.${field}`);
      }
    }
  });

  test("an unknown outcome degrades to no explanation, not a crash", () => {
    assert.equal(absence("something_new" as Outcome, "truth"), null);
  });
});
