/**
 * Placeholder run history, shaped like the real `runs` table in SCHEMA.md.
 *
 * The model is a YOLO segmentation model and the splits are the real ones.
 * There is no accuracy or mAP column: that was excluded, and a number nobody
 * can trace back to the schema gets screenshotted and then believed.
 */
export interface Run {
  id: string;
  model: string;
  dataset: string;
  split: string;
  status: "Complete" | "Failed";
  date: string;
  totalImages: number;
  failureCount: number;
}

export const mockRuns: Run[] = [
  { id: "1287", model: "yolov8s-seg", dataset: "door_df_window", split: "test",  status: "Complete", date: "May 21, 2024", totalImages: 1045, failureCount: 836 },
  { id: "1286", model: "yolov8s-seg", dataset: "door_df_window", split: "val",   status: "Complete", date: "May 20, 2024", totalImages: 1500, failureCount: 921 },
  { id: "1285", model: "yolov8s-seg", dataset: "door_df_window", split: "train", status: "Complete", date: "May 19, 2024", totalImages: 5000, failureCount: 2500 },
  { id: "1284", model: "yolov8s-seg", dataset: "door_df_window", split: "test",  status: "Failed",   date: "May 18, 2024", totalImages: 450,  failureCount: 450 },
];

export function findRun(id: string): Run | undefined {
  return mockRuns.find((r) => r.id === id);
}
