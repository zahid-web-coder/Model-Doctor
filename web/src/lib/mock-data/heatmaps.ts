/**
 * Grad-CAM records, shaped like the `heatmaps` table.
 *
 * The `tone` field is a fixed Tailwind class, never an interpolated one. The
 * previous version built class names as `from-slate-${300 + i*100}`, which
 * Tailwind cannot see when it scans the source at build time — so none of
 * those classes were generated and every tile rendered with no background.
 * That is why this screen was blank.
 */
import { CLASSES } from "./classes";

export interface HeatmapImage {
  id: string;
  tone: string;
  failureType: string;
  className: string;
  confidence: number | null;
  factor: string;
}

const TONES = [
  "bg-gradient-to-tr from-slate-700 to-slate-500",
  "bg-gradient-to-tr from-slate-600 to-slate-400",
  "bg-gradient-to-tr from-stone-700 to-stone-500",
  "bg-gradient-to-tr from-zinc-700 to-zinc-500",
  "bg-gradient-to-tr from-neutral-700 to-neutral-500",
];
const TYPES = ["False Positive", "False Negative", "Wrong Class", "Poor Localization"];
const FACTORS = ["small_object", "thin_structure", "crowding", "low_contrast"];

export const mockHeatmaps: HeatmapImage[] = Array.from({ length: 24 }).map((_, i) => ({
  id: `hm-${2000 + i}`,
  tone: TONES[i % TONES.length],
  failureType: TYPES[i % TYPES.length],
  className: CLASSES[i % CLASSES.length],
  confidence: i % 7 === 0 ? null : 0.6 + ((i * 11) % 39) / 100,
  factor: FACTORS[i % FACTORS.length],
}));
