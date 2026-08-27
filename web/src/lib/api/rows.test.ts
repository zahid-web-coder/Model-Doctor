/**
 * Outcome normalisation — the repair that keeps `Outcomes` honest.
 *
 * `/runs/{id}/outcomes` groups by outcome, so a bucket with no findings is
 * absent from the payload rather than present as zero. Every screen that sums
 * or divides by those counts assumed all five keys existed, which produced
 * `NaN` — and because `NaN` is falsy, the guards that were meant to catch
 * missing data silently reported "no outcomes recorded" for runs that had real
 * failures. These tests pin the behaviour that fixed it.
 *
 * Run with Node's built-in runner, which strips TypeScript natively:
 *
 *     npm test
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { ALL_OUTCOMES, normaliseOutcomes, type Outcomes } from "./rows.ts";

// Case A — every outcome present. Nothing may be altered.
test("keeps every value when the payload is complete", () => {
  const complete: Outcomes = {
    correct: 191,
    false_positive: 48,
    false_negative: 27,
    poor_localization: 10,
    wrong_class: 3,
  };

  assert.deepEqual(normaliseOutcomes(complete), complete);
});

// Case B — the reported defect: a single-class dataset omits `wrong_class`.
test("fills an omitted zero-count outcome with zero", () => {
  const result = normaliseOutcomes({
    correct: 191,
    false_positive: 48,
    false_negative: 27,
    poor_localization: 10,
  });

  assert.equal(result.wrong_class, 0);
  // The keys that were present must survive untouched.
  assert.equal(result.correct, 191);
  assert.equal(result.false_positive, 48);
  assert.equal(result.false_negative, 27);
  assert.equal(result.poor_localization, 10);
});

test("the repaired payload sums without producing NaN", () => {
  const failures = normaliseOutcomes({
    correct: 180,
    false_negative: 41,
    false_positive: 37,
    poor_localization: 7,
  });
  const total = (["false_negative", "false_positive", "poor_localization", "wrong_class"] as const)
    .reduce((sum, key) => sum + failures[key], 0);

  assert.equal(Number.isNaN(total), false);
  assert.equal(total, 85);
});

// Case C — several outcomes missing at once.
test("fills every missing outcome, not just the first", () => {
  const result = normaliseOutcomes({ correct: 12 });

  assert.equal(result.correct, 12);
  for (const key of ALL_OUTCOMES) {
    assert.equal(typeof result[key], "number", `${key} must be a number`);
  }
  assert.equal(result.false_negative, 0);
  assert.equal(result.false_positive, 0);
  assert.equal(result.poor_localization, 0);
  assert.equal(result.wrong_class, 0);
});

test("always returns all five keys", () => {
  assert.deepEqual(Object.keys(normaliseOutcomes({})).sort(), [...ALL_OUTCOMES].sort());
});

// Case D — an empty payload is a real answer; an absent one is not.
test("an empty payload is a run with nothing recorded, and reads as zeros", () => {
  const result = normaliseOutcomes({});

  for (const key of ALL_OUTCOMES) assert.equal(result[key], 0);
});

test("a missing payload is never turned into a valid zero result", () => {
  // The distinction this whole fix rests on: "the run has zero of this
  // outcome" and "the outcomes could not be read" must not converge on the
  // same value. Normalisation repairs gaps *inside* a payload; it has no
  // payload to repair when the request failed, and must fail loudly rather
  // than manufacture a clean run out of an error.
  assert.throws(() => normaliseOutcomes(null as unknown as Partial<Outcomes>));
  assert.throws(() => normaliseOutcomes(undefined as unknown as Partial<Outcomes>));
});

test("is idempotent, so applying it twice is safe", () => {
  const once = normaliseOutcomes({ correct: 5, false_negative: 2 });

  assert.deepEqual(normaliseOutcomes(once), once);
});

test("does not mutate its input", () => {
  const input = { correct: 5 };
  normaliseOutcomes(input);

  assert.deepEqual(input, { correct: 5 });
});
