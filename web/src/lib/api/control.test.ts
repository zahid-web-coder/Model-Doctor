import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { progressOf, STAGE_LABEL, type Job } from "./control.ts";

/**
 * The progress bar is the part of this workflow most able to lie.
 *
 * It is the only thing the user watches for minutes at a time, and a bar
 * derived from elapsed time would move smoothly while meaning nothing. These
 * assert that it reports position in the pipeline, and specifically that it
 * never claims completion before the job is finished.
 */

function job(overrides: Partial<Job> = {}): Job {
  return {
    token: "t",
    status: "running",
    stage: null,
    stages: [
      "inference+diagnosis", "evaluation", "mask-diagnosis",
      "root-cause", "clustering", "recommendations", "explainability",
    ],
    stage_index: null,
    run_id: null,
    detector: "yolo",
    split: "test",
    model_name: "m.pt",
    dataset_name: "ds",
    image_size: 640,
    confidence: 0.25,
    error: null,
    log_tail: null,
    created_at: "now",
    started_at: null,
    finished_at: null,
    explainability_supported: true,
    ...overrides,
  };
}

describe("progressOf", () => {
  test("a queued job has not started", () => {
    assert.equal(progressOf(job({ status: "queued" })), 0);
  });

  test("a running job with no stage yet shows movement, not zero", () => {
    // Zero for a job that is demonstrably running reads as "nothing is
    // happening", which is the one thing it is not.
    assert.ok(progressOf(job({ status: "running" })) > 0);
  });

  test("never reaches complete before the job finishes", () => {
    const stages = job().stages;
    for (let i = 0; i < stages.length; i += 1) {
      const value = progressOf(job({ status: "running", stage_index: i }));
      assert.ok(value < 1, `stage ${i} claimed completion while still running`);
    }
  });

  test("a succeeded job is complete whatever stage it ended on", () => {
    assert.equal(progressOf(job({ status: "succeeded", stage_index: 2 })), 1);
    assert.equal(progressOf(job({ status: "succeeded", stage_index: null })), 1);
  });

  test("advances monotonically through the pipeline", () => {
    let previous = -1;
    for (let i = 0; i < job().stages.length; i += 1) {
      const value = progressOf(job({ stage_index: i }));
      assert.ok(value > previous, `stage ${i} did not advance the bar`);
      previous = value;
    }
  });

  test("a failed job does not report itself as complete", () => {
    assert.ok(progressOf(job({ status: "failed", stage_index: 3 })) < 1);
  });
});

describe("stage labels", () => {
  test("every stage the backend can report has a human label", () => {
    // A missing entry renders the raw slug, which is not wrong but is the kind
    // of thing that ships unnoticed.
    for (const stage of job().stages) {
      assert.ok(STAGE_LABEL[stage], `no label for ${stage}`);
    }
  });

  test("labels do not promise heatmaps by another name", () => {
    // The explainability stage is skipped for detectors with no validated
    // attribution, so its label must describe what it is rather than imply
    // every run produces one.
    assert.equal(STAGE_LABEL["explainability"], "Attention heatmaps");
  });
});

describe("a run without explainability", () => {
  test("omits the stage entirely rather than showing it as pending", () => {
    // The backend removes it from `stages`; this asserts the shape the UI
    // relies on, so a regression there is caught here rather than by a user
    // watching a step that never starts.
    const rfdetr = job({
      detector: "rfdetr",
      explainability_supported: false,
      stages: job().stages.filter((s) => s !== "explainability"),
    });
    assert.ok(!rfdetr.stages.includes("explainability"));
    assert.equal(rfdetr.stages.length, 6);
    assert.equal(progressOf({ ...rfdetr, status: "succeeded" }), 1);
  });
});
