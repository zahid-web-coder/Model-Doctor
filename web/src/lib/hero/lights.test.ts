import { test } from "node:test";
import assert from "node:assert/strict";
import { stepLevel, strike, STRIKE, FADE_OUT } from "./lights.ts";

test("strike starts dark, stutters, and ends at full output", () => {
  assert.equal(strike(0), 0);
  assert.equal(strike(STRIKE), 1);
  assert.equal(strike(STRIKE + 5), 1);
  // It dips after the first flash — the stutter.
  assert.ok(strike(0.03) > strike(0.1));
  // The warm-up only rises.
  let last = strike(0.33);
  for (let t = 0.34; t < STRIKE; t += 0.05) {
    const now = strike(t);
    assert.ok(now >= last, `warm-up fell at t=${t}`);
    last = now;
  }
});

test("switching off fades to zero over FADE_OUT and stays there", () => {
  let level = 1;
  let t = 0;
  while (level > 0 && t < 1) {
    level = stepLevel(level, false, 0, 1 / 60);
    t += 1 / 60;
  }
  assert.equal(level, 0);
  assert.ok(Math.abs(t - FADE_OUT) < 0.03);
  assert.equal(stepLevel(0, false, 0, 1 / 60), 0);
});

test("switching on follows the strike from the moment of the switch", () => {
  assert.equal(stepLevel(0, true, 0, 1 / 60), 0);
  assert.equal(stepLevel(0.3, true, STRIKE + 0.1, 1 / 60), 1);
});
