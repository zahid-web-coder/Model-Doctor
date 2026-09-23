import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { SCAN_FINDINGS, findingForPlate } from "./scanFindings.ts";

/**
 * The scan cards are the one place on the landing page that labels a factor
 * "likely cause" or "chance". Those labels have to follow the engine's own
 * bar — `comparison.factor_qualifies`: lift above 1 *and* p below 0.05 — or
 * the card would be making exactly the unsupported claim the page argues
 * against. So the rule is checked against the figures the cards display,
 * rather than trusted to have been applied by hand.
 */
const SIGNIFICANCE = 0.05;

/** Displayed p values are strings; "<0.001" is below any threshold here. */
function pOf(p: string): number {
  return p.startsWith("<") ? Number(p.slice(1)) / 2 : Number(p);
}

describe("scan card verdicts", () => {
  test("each verdict is the engine's bar applied to the displayed figures", () => {
    for (const finding of SCAN_FINDINGS) {
      for (const f of finding.factors) {
        const qualifies = Number(f.lift) > 1 && pOf(f.p) < SIGNIFICANCE;
        assert.equal(
          f.verdict,
          qualifies ? "cause" : "chance",
          `finding ${finding.id}, ${f.factor}: ${f.lift}x, p ${f.p}`,
        );
      }
    }
  });

  test("a factor is never called a cause on the strength of lift alone", () => {
    // edge_truncation is on three in four failures in run 3 — and on nearly
    // as many successes. Present is not the same as responsible.
    const edge = SCAN_FINDINGS.flatMap((f) => f.factors)
      .filter((f) => f.factor === "edge_truncation");
    assert.ok(edge.length > 0);
    for (const f of edge) assert.equal(f.verdict, "chance");
  });

  test("a correct detection carries nothing to explain", () => {
    const correct = SCAN_FINDINGS.filter((f) => f.outcome === "Correct");
    assert.ok(correct.length > 0);
    for (const f of correct) assert.deepEqual(f.factors, []);
  });

  test("a missed object has no confidence to report", () => {
    for (const f of SCAN_FINDINGS.filter((x) => x.outcome === "Missed")) {
      assert.equal(f.confidence, null);
      assert.equal(f.iou, null);
    }
  });
});

describe("findingForPlate", () => {
  test("the same plate always reports the same finding", () => {
    assert.equal(findingForPlate(5).id, findingForPlate(5).id);
  });

  test("every finding is reachable from some plate", () => {
    const seen = new Set(
      Array.from({ length: SCAN_FINDINGS.length }, (_, i) => findingForPlate(i).id),
    );
    assert.equal(seen.size, SCAN_FINDINGS.length);
  });

  test("wraps indices past the end and below zero", () => {
    const n = SCAN_FINDINGS.length;
    assert.equal(findingForPlate(n).id, findingForPlate(0).id);
    assert.equal(findingForPlate(-1).id, findingForPlate(n - 1).id);
  });
});
