# Vision

## The problem

Object detection tooling is excellent at telling you *how much* a model fails
and silent about *why*.

A typical evaluation ends with a number: `mAP50 = 0.72`. That number is real,
comparable, and publishable. It is also nearly useless as an instruction. It
does not say which images broke, whether the model is missing objects or
inventing them, whether boxes are in roughly the right place but too loose,
whether two classes are being confused, or whether failures share a common
cause such as low light, small objects, or occlusion.

So engineers fall back on scrolling through prediction images by hand. That
does not scale, it is not reproducible, and it finds only the failures that
happen to be visually obvious.

## The thesis

**Failure analysis should be a first-class engineering workflow, not manual
inspection.**

The information needed to explain a failure already exists — predictions,
ground truth, confidence scores, and the image itself. What is missing is a
tool that systematically compares them, classifies what went wrong, groups
failures that share a cause, and reports the result in a form an engineer can
act on.

## What Model Doctor is

An **AI engineering developer tool**. The user is an ML engineer debugging a
model, not an end user consuming detections.

That framing drives concrete decisions:

- Prefer surfacing uncertain predictions over hiding them. A wrong detection at
  0.30 confidence is evidence, not noise.
- Structured, inspectable output beats pretty output.
- Reproducibility matters more than raw speed.
- The tool diagnoses; it never modifies the model.

## What Model Doctor is not

- **Not an object detection project.** It consumes detectors, it does not
  compete with them.
- **Not a training framework.** It never updates weights.
- **Not a labelling tool.** It reads ground truth; it does not create it.
- **Not a metrics dashboard.** Metrics are the starting point it moves beyond.

## Success criteria

Model Doctor succeeds when an engineer can point it at a model and a dataset
and receive an answer of the form:

> "Recall is low mainly because of one class. Its failures are concentrated in
> small, partially occluded instances. Most of the remaining errors are
> localisation, not detection — the objects are found but the boxes are loose."

That is a statement someone can act on. `mAP50 = 0.72` is not.

## Design principles

1. **Diagnose, never mutate.** Read-only with respect to models and datasets.
2. **Evidence over aggregates.** Always able to name the specific images.
3. **Detector-agnostic core.** Analysis depends on a normalised prediction
   format, never on a particular library's output types.
4. **Degrade, do not crash.** Missing resources produce guidance.
5. **Nothing about a dataset is hardcoded.** Class names, counts, and paths are
   discovered at runtime, always.
