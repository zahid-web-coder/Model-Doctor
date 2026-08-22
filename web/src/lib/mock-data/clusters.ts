/**
 * Failure groups, shaped like `clusters` + `cluster_members`.
 *
 * Named after their factor signature rather than invented labels: the grouping
 * is deterministic on root-cause factors, not a learned clustering, so a group
 * is exactly "the failures sharing these factors".
 */
export interface Cluster {
  id: string;
  name: string;
  memberCount: number;
  sharePct: number;
  dominantFactor: string;
  impact: "High" | "Medium" | "Low";
  description: string;
}

export const mockClusters: Cluster[] = [
  { id: "c-1", name: "small_object", memberCount: 142, sharePct: 0.114, dominantFactor: "small_object", impact: "High",   description: "Objects below the 25th-percentile area threshold. Replicated across test and val at p<0.001." },
  { id: "c-2", name: "thin_structure", memberCount: 128, sharePct: 0.103, dominantFactor: "thin_structure", impact: "High",   description: "Thin annular structures, chiefly door_frame. Box IoU stays high while mask IoU collapses." },
  { id: "c-3", name: "small_object + thin_structure", memberCount: 112, sharePct: 0.090, dominantFactor: "small_object", impact: "High",   description: "Both discriminating factors present on the same finding." },
  { id: "c-4", name: "crowding", memberCount: 96,  sharePct: 0.077, dominantFactor: "crowding", impact: "Medium", description: "Significant on test (p=0.041) but not replicated on val (p=0.310)." },
  { id: "c-5", name: "low_contrast", memberCount: 86,  sharePct: 0.069, dominantFactor: "low_contrast", impact: "Medium", description: "Low local contrast between object and background." },
  { id: "c-6", name: "unexplained", memberCount: 74,  sharePct: 0.059, dominantFactor: "none", impact: "Medium", description: "No discriminating factor accounts for these. Listed rather than hidden." },
];

export const CLUSTER_STATS = {
  totalClusters: 24,
  highImpact: 7,
  totalFailures: 1247,
  coveragePct: 0.89,
};
