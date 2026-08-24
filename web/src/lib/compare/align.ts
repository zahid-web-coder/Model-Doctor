import type { Finding, ImageRow, Run } from "@/lib/api/rows";

/**
 * Pairing two runs' findings so they can be shown side by side.
 *
 * **`image_id` cannot be the key.** Every run inserts its own `images` rows, so
 * the same file is id 1 in one run and id 152 in another. Joining on it would
 * silently pair unrelated images.
 *
 * The key is `(filename, truth_box)`. Ground truth is a property of the
 * dataset, not of the model, so two runs over the same split describe the same
 * objects — and a finding that carries a truth box names one of them exactly.
 *
 * **False positives are not paired, ever.** A false positive has no truth box:
 * it is a detection the model invented, and there is no ground-truth object for
 * it to correspond to. Pairing them by proximity or by index would fabricate a
 * relationship the data does not contain, so they are carried as image-level
 * extras instead — "this run also produced N detections here that match
 * nothing" — which is what they actually are.
 */

/** Rounded so that float formatting differences cannot break a real match. */
const boxKey = (box: number[]): string =>
  box.map((v) => Math.round(v * 100) / 100).join(",");

export const truthKey = (filename: string, truthBox: number[]): string =>
  `${filename}|${boxKey(truthBox)}`;

export interface AlignedPair {
  key: string;
  filename: string;
  imageIdA: number;
  imageIdB: number;
  truthBox: number[];
  truthPolygon: number[][] | null;
  a: Finding;
  b: Finding;
  /** The two runs disagreed about this object's outcome. */
  differs: boolean;
}

export interface ImageExtras {
  filename: string;
  /** Detections matching no ground-truth object, per side. */
  a: Finding[];
  b: Finding[];
}

export interface Alignment {
  pairs: AlignedPair[];
  /** Ground-truth objects present in one run's findings but not the other's. */
  unmatchedA: Finding[];
  unmatchedB: Finding[];
  extras: ImageExtras[];
  totalDiffering: number;
}

const byId = (images: ImageRow[]): Map<number, ImageRow> =>
  new Map(images.map((image) => [image.id, image]));

/**
 * Pair up two runs' ground-truth-backed findings, and collect the rest.
 *
 * Findings with no truth box never enter the pairing. Findings whose image is
 * missing from the run's own image list are skipped rather than guessed at —
 * that would mean the two endpoints disagree, and inventing a filename to
 * proceed would corrupt every downstream key.
 */
export function alignRuns(
  findingsA: Finding[],
  imagesA: ImageRow[],
  findingsB: Finding[],
  imagesB: ImageRow[],
): Alignment {
  const imagesByIdA = byId(imagesA);
  const imagesByIdB = byId(imagesB);

  const keyed = (findings: Finding[], images: Map<number, ImageRow>) => {
    const withTruth = new Map<string, Finding>();
    const withoutTruth: { filename: string; finding: Finding }[] = [];
    for (const finding of findings) {
      const image = images.get(finding.image_id);
      if (!image) continue;
      if (finding.truth_box && finding.truth_box.length === 4) {
        withTruth.set(truthKey(image.filename, finding.truth_box), finding);
      } else {
        withoutTruth.push({ filename: image.filename, finding });
      }
    }
    return { withTruth, withoutTruth };
  };

  const left = keyed(findingsA, imagesByIdA);
  const right = keyed(findingsB, imagesByIdB);

  const filenameOf = (key: string) => key.slice(0, key.lastIndexOf("|"));

  const pairs: AlignedPair[] = [];
  for (const [key, a] of left.withTruth) {
    const b = right.withTruth.get(key);
    if (!b) continue;
    const filename = filenameOf(key);
    pairs.push({
      key,
      filename,
      imageIdA: a.image_id,
      imageIdB: b.image_id,
      truthBox: a.truth_box as number[],
      truthPolygon: a.truth_polygon ?? b.truth_polygon,
      a,
      b,
      differs: a.outcome !== b.outcome,
    });
  }

  const unmatchedA = [...left.withTruth]
    .filter(([key]) => !right.withTruth.has(key))
    .map(([, finding]) => finding);
  const unmatchedB = [...right.withTruth]
    .filter(([key]) => !left.withTruth.has(key))
    .map(([, finding]) => finding);

  // Extras are grouped per image, because that is the only level at which they
  // are comparable at all.
  const extrasByFile = new Map<string, ImageExtras>();
  const ensure = (filename: string): ImageExtras => {
    let entry = extrasByFile.get(filename);
    if (!entry) {
      entry = { filename, a: [], b: [] };
      extrasByFile.set(filename, entry);
    }
    return entry;
  };
  for (const { filename, finding } of left.withoutTruth) ensure(filename).a.push(finding);
  for (const { filename, finding } of right.withoutTruth) ensure(filename).b.push(finding);

  // Sorted by file, then by the pair key — never left to rely on the order the
  // findings happened to arrive in. An image usually holds several annotated
  // objects, so filename alone leaves ties, and a stable sort then settles them
  // by whichever run was passed first. That made object *n* a different object
  // depending on which run sat on the left, so swapping the two sides silently
  // moved the reader somewhere else. The key is built from the filename and the
  // truth box, so it is a property of the ground truth and identical whichever
  // way round the runs are given.
  pairs.sort(
    (x, y) => x.filename.localeCompare(y.filename) || x.key.localeCompare(y.key),
  );

  return {
    pairs,
    unmatchedA,
    unmatchedB,
    extras: [...extrasByFile.values()].sort((x, y) =>
      x.filename.localeCompare(y.filename),
    ),
    totalDiffering: pairs.filter((pair) => pair.differs).length,
  };
}

export type CompatibilityLevel = "ok" | "warning" | "blocked";

export interface Compatibility {
  level: CompatibilityLevel;
  headline: string;
  details: string[];
}

/**
 * Decide whether two runs can honestly be compared.
 *
 * Matching filenames are **not** sufficient, and assuming they were is the
 * mistake this guards against: two runs can cover the same files while
 * describing different objects in them, if the label export changed between
 * them. So the decisive test is whether the ground-truth key sets agree —
 * same files *and* same objects within them.
 *
 * A mismatch is reported as blocking rather than as a footnote. Every metric on
 * the page would otherwise be comparing two different questions.
 */
export function checkCompatibility(
  runA: Run,
  runB: Run,
  alignment: Alignment,
  findingsA: Finding[],
  findingsB: Finding[],
): Compatibility {
  const details: string[] = [];
  let level: CompatibilityLevel = "ok";

  if (runA.id === runB.id) {
    return {
      level: "blocked",
      headline: "Select two different runs.",
      details: [],
    };
  }

  if (runA.dataset_yaml !== runB.dataset_yaml) {
    level = "blocked";
    details.push(
      `Different datasets: "${runA.dataset_yaml}" against "${runB.dataset_yaml}".`,
    );
  }
  if (runA.split !== runB.split) {
    level = "blocked";
    details.push(`Different splits: "${runA.split}" against "${runB.split}".`);
  }

  const truthCount = (findings: Finding[]) =>
    findings.filter((f) => f.truth_box && f.truth_box.length === 4).length;
  const gtA = truthCount(findingsA);
  const gtB = truthCount(findingsB);

  if (alignment.unmatchedA.length || alignment.unmatchedB.length) {
    level = "blocked";
    details.push(
      `Ground truth differs: ${alignment.unmatchedA.length} object(s) appear only in the first run and ` +
        `${alignment.unmatchedB.length} only in the second. Matching filenames are not enough — ` +
        "the objects annotated within them must match too.",
    );
  } else if (gtA !== gtB) {
    level = "blocked";
    details.push(
      `Ground-truth object counts differ: ${gtA} against ${gtB}.`,
    );
  }

  if (runA.confidence_threshold !== runB.confidence_threshold) {
    if (level === "ok") level = "warning";
    details.push(
      `Different confidence thresholds: ${runA.confidence_threshold} against ` +
        `${runB.confidence_threshold}. Counts of detections are not directly comparable.`,
    );
  }
  if (runA.match_iou_threshold !== runB.match_iou_threshold) {
    if (level === "ok") level = "warning";
    details.push(
      `Different match IoU thresholds: ${runA.match_iou_threshold} against ` +
        `${runB.match_iou_threshold}. Outcomes were decided by different rules.`,
    );
  }
  if (runA.image_size !== runB.image_size) {
    if (level === "ok") level = "warning";
    details.push(
      `Different input resolutions: ${runA.image_size} against ${runB.image_size}. ` +
        "This is a fair comparison of two configurations, not of two architectures.",
    );
  }

  const headline =
    level === "blocked"
      ? "These runs cannot be compared."
      : level === "warning"
        ? "Comparable, with caveats."
        : `Comparable — ${alignment.pairs.length} ground-truth objects aligned.`;

  return { level, headline, details };
}
