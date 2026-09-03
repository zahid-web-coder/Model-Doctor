import { api } from "@/lib/api/client";
import { ImagesTable } from "@/components/dashboard/ImagesTable";

/**
 * Image-level diagnosis from `/runs/{id}/image-diagnoses`.
 *
 * **A second lens over the same findings, not a replacement.** The Failures
 * screen answers "which object failed and why"; this answers "how many
 * photographs the model handled correctly, and when it did not, what shape the
 * mistake took". Both are true and neither derives the other.
 *
 * It exists because one prediction stretched across two annotated objects
 * produced a poor localisation *and* a false negative — two findings for one
 * model error, the second reading as "never saw it" when the model had covered
 * 80% of it. A finding knows only its own pairing; an image sees all of them.
 *
 * Empty for a run analysed before the pass existed, which the table reports as
 * not measured rather than as a clean run.
 */
export default async function ImagesPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const [diagnoses, images, masks] = await Promise.all([
    api.imageDiagnoses(runId),
    api.images(runId).catch(() => []),
    // Predicted outlines for the whole run, so opening an image can draw every
    // prediction on it without a second request per row.
    api.maskFindings(runId),
  ]);

  const filenames = Object.fromEntries(images.map((i) => [i.id, i.filename]));
  const dimensions = Object.fromEntries(
    images
      .filter((i) => typeof i.width === "number" && typeof i.height === "number")
      .map((i) => [i.id, [i.width as number, i.height as number] as [number, number]]),
  );
  const polygons = Object.fromEntries(
    masks
      .filter((m) => m.pred_polygon !== null)
      .map((m) => [m.finding_id, m.pred_polygon as number[][]]),
  );

  return (
    <ImagesTable
      runId={runId}
      rows={diagnoses}
      filenames={filenames}
      dimensions={dimensions}
      polygons={polygons}
    />
  );
}
