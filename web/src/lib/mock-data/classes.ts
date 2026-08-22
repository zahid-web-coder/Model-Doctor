/**
 * The dataset's real classes. Kept in one place so no screen can invent one —
 * the earlier placeholders (Crack, Rust, Leak, Corrosion, Deformation) came
 * from nowhere and were never in this project.
 */
export const CLASSES = ["door", "door_frame", "window"] as const;
export type ClassName = (typeof CLASSES)[number];
