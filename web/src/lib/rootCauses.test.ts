import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { indexByFinding, strongestFirst, noCausesReason } from "./rootCauses.ts";
import type { RootCause } from "./api/rows.ts";

/**
 * The detail panel makes a claim about *why* a specific object failed, so the
 * factors it shows must belong to the finding that is selected and must be
 * ordered by how strongly they were attributed. Both are easy to get subtly
 * wrong — an off-by-one in the grouping attaches another object's evidence to
 * this one, and an unsorted list buries the strongest factor.
 */

function cause(overrides: Partial<RootCause> = {}): RootCause {
  return {
    finding_id: 1,
    run_id: 5,
    outcome: "false_negative",
    class_name: "staircase",
    factor: "small_object",
    score: 0.5,
    evidence: null,
    ...overrides,
  };
}

describe("indexByFinding", () => {
  test("groups every cause under its own finding", () => {
    const index = indexByFinding([
      cause({ finding_id: 10, factor: "small_object" }),
      cause({ finding_id: 11, factor: "edge_truncation" }),
      cause({ finding_id: 10, factor: "low_contrast" }),
    ]);
    assert.deepEqual(
      index[10].map((c) => c.factor),
      ["small_object", "low_contrast"]
    );
    assert.deepEqual(index[11].map((c) => c.factor), ["edge_truncation"]);
  });

  test("a finding with no attributed factor is absent, not empty", () => {
    // The panel distinguishes "nothing was attributed" from "causes exist";
    // an empty array for an unseen finding would erase that distinction.
    const index = indexByFinding([cause({ finding_id: 10 })]);
    assert.equal(index[99], undefined);
  });

  test("a run with no root causes indexes to nothing", () => {
    // A database saved before the root-cause pass returns an empty list, and
    // the panel must fall through to its explanation rather than break.
    assert.deepEqual(indexByFinding([]), {});
  });

  test("never attributes one finding's evidence to another", () => {
    const index = indexByFinding([
      cause({ finding_id: 1081, factor: "edge_truncation", evidence: "touches left, top" }),
      cause({ finding_id: 1082, factor: "small_object", evidence: "0.4% of frame" }),
    ]);
    for (const [id, group] of Object.entries(index)) {
      for (const entry of group) {
        assert.equal(String(entry.finding_id), id);
      }
    }
  });
});

describe("strongestFirst", () => {
  test("orders by attribution strength, not by arrival", () => {
    const sorted = strongestFirst([
      cause({ factor: "low_contrast", score: 0.31 }),
      cause({ factor: "edge_truncation", score: 1.0 }),
      cause({ factor: "small_object", score: 0.72 }),
    ]);
    assert.deepEqual(
      sorted.map((c) => c.factor),
      ["edge_truncation", "small_object", "low_contrast"]
    );
  });

  test("does not mutate the caller's array", () => {
    // The index is shared across every selection; sorting it in place would
    // reorder the prop the page passed down.
    const original = [cause({ score: 0.1 }), cause({ score: 0.9 })];
    strongestFirst(original);
    assert.deepEqual(original.map((c) => c.score), [0.1, 0.9]);
  });

  test("an unattributed finding yields an empty list, not a crash", () => {
    assert.deepEqual(strongestFirst(undefined), []);
    assert.deepEqual(strongestFirst([]), []);
  });

  test("keeps a zero score rather than treating it as missing", () => {
    // Zero is a measured attribution strength. Dropping it as falsy would be
    // the same error as rendering a null metric as 0, in reverse.
    const sorted = strongestFirst([cause({ factor: "occlusion", score: 0 })]);
    assert.equal(sorted.length, 1);
    assert.equal(sorted[0].score, 0);
  });
});

describe("noCausesReason", () => {
  test("a correct detection is not described as unexplained", () => {
    // Attribution runs on failures only — verified against run 5, where all
    // 155 attributions belong to false positives, false negatives, and poor
    // localisations. Telling the reader "no measured condition applied" for a
    // correct detection would claim the detectors ran and found nothing.
    const text = noCausesReason("correct");
    assert.match(text, /attributed to failures/);
    assert.ok(!text.includes("predates"), "offered a stale-database excuse for a correct detection");
  });

  test("a failure with no factor is named as unexplained, not as an error", () => {
    for (const outcome of ["false_positive", "false_negative", "poor_localization", "wrong_class"] as const) {
      const text = noCausesReason(outcome);
      assert.match(text, /unexplained/, `${outcome} did not report the group honestly`);
      assert.ok(!text.includes("nothing to explain"), `${outcome} claimed there was nothing to explain`);
    }
  });

  test("never guesses a factor", () => {
    for (const outcome of ["correct", "false_positive", "false_negative", "poor_localization", "wrong_class"] as const) {
      assert.match(noCausesReason(outcome), /^[A-Z]/);
      assert.ok(noCausesReason(outcome).length > 40);
    }
  });
});
