/**
 * What each root-cause factor actually measures.
 *
 * **Transcribed from the detectors in `app/root_cause.py`, not invented.** Each
 * entry states the rule the corresponding detector applies, so a reader can
 * tell what caused a factor to fire without opening the backend. Where a
 * threshold is derived from the data rather than fixed, that is said, because
 * "small" meaning "small for this dataset" changes how the number should be
 * read.
 *
 * `SIGNATURE_SEPARATOR` mirrors the backend's own joiner: a cluster label is a
 * signature over the factors its members carry, so "small_object +
 * thin_structure" is the group of findings carrying both. Splitting on it lets
 * a combined label be explained from its parts rather than needing an entry of
 * its own.
 *
 * An unknown factor name returns `null` rather than a guess. New detectors can
 * be added to the backend without this file, and a screen that invents a
 * definition for one it has never heard of would be worse than one that shows
 * the raw name.
 */

export const SIGNATURE_SEPARATOR = " + ";

/** The label the backend gives failures no factor accounted for. */
export const UNEXPLAINED = "unexplained";

export interface FactorNote {
  /** Human-readable name for the factor. */
  title: string;
  /** The rule the detector applies, in one sentence. */
  rule: string;
  /** Why it plausibly relates to failure — the reason it is worth measuring. */
  why: string;
  /** Set when the threshold is derived from the run rather than configured. */
  calibrated?: boolean;
}

const NOTES: Record<string, FactorNote> = {
  small_object: {
    title: "Small object",
    rule: "The object's box covers a small share of the frame.",
    why: "Fewer pixels carry the evidence a detector needs, so small objects are missed or localised loosely more often than large ones.",
    calibrated: true,
  },
  thin_structure: {
    title: "Thin structure",
    rule: "The box is much longer than it is wide.",
    why: "A long, narrow object fills little of its own bounding box, so box-level overlap can look poor even when the object is found.",
    calibrated: true,
  },
  low_light: {
    title: "Low light",
    rule: "The region's mean luminance is below the configured threshold.",
    why: "Underexposed regions carry less contrast, so edges and texture the model relies on are weaker or absent.",
  },
  blur: {
    title: "Blur",
    rule: "The region's Laplacian variance is below the configured threshold — little high-frequency detail.",
    why: "Blur removes the sharp gradients that distinguish an object's boundary from its surroundings.",
  },
  edge_truncation: {
    title: "Edge truncation",
    rule: "The box touches, or comes within a margin of, the frame edge.",
    why: "A cut-off object is only partly visible, so the model sees less evidence than the annotation implies. On close-up datasets this is common rather than exceptional — worth knowing before concluding the model is weak.",
  },
  crowding: {
    title: "Crowding",
    rule: "A neighbouring annotation overlaps this object above the configured IoU threshold.",
    why: "Annotations do not record what is in front of what, so occlusion cannot be observed directly. Heavy overlap is the condition under which it occurs, and it is measurable.",
  },
};

/** The rule behind one factor, or `null` if this build does not know it. */
export function factorNote(name: string): FactorNote | null {
  return NOTES[name] ?? null;
}

/** Every factor in a cluster label, in the order the signature lists them. */
export function factorsInSignature(label: string): string[] {
  return label.split(SIGNATURE_SEPARATOR).map((part) => part.trim()).filter(Boolean);
}

/**
 * What a cluster label means, assembled from its parts.
 *
 * A combined signature is explained as the conjunction it is — the findings
 * carrying every listed factor — rather than as a category in its own right,
 * because that is exactly how the backend forms it.
 */
export function signatureSummary(label: string): string {
  if (label === UNEXPLAINED) {
    return "No factor fired for these failures. They are listed rather than hidden: these are the failures the current detectors cannot account for, which makes them the most interesting ones to look at.";
  }
  const parts = factorsInSignature(label);
  const known = parts.map((p) => factorNote(p)?.title.toLowerCase() ?? p);
  if (parts.length === 1) return `Failures where the ${known[0]} condition was present.`;
  return `Failures carrying every one of these conditions at once: ${known.join(", ")}.`;
}
