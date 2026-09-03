"use client";

import { useEffect, useMemo, useState } from "react";
import { Lightbox } from "@/components/shared/Lightbox";
import { LayerToggle } from "@/components/shared/LayerToggle";
import { Overlay, OverlayLegend } from "@/components/compare/Overlay";
import { api } from "@/lib/api/client";
import { fitViewport } from "@/lib/compare/geometry";
import {
  ALL_LAYERS,
  NO_LAYERS,
  contextFor,
  defaultLayers,
  predictionShapeOf,
  truthShapeOf,
  type Layers,
} from "@/lib/findingShapes";
import { OUTCOME_LABEL, type Finding, type GroupMember } from "@/lib/api/rows";

/**
 * The photographs behind one cluster's signature.
 *
 * **A signature is a claim about what the failures have in common, and until
 * now it could not be checked.** "small_object + thin_structure, 20 findings"
 * is a definition, not evidence: a reader could not see whether those twenty
 * really are small and thin, or whether the twenty-five `unexplained` ones
 * share something the detectors have no name for yet. That question is
 * answered by looking, so the cluster now opens into the same drawing the
 * Failures and Images screens use.
 *
 * The member being inspected is drawn at full weight against every other
 * object and prediction on its image, exactly as a finding is elsewhere — a
 * cluster member is a finding, and it would be strange for it to be drawn by
 * different rules because of how the reader arrived at it.
 *
 * Geometry is fetched per image on open rather than shipped with the page.
 * A cluster's members can span dozens of photographs, and the clusters screen
 * would otherwise have to load every finding in the run to draw one.
 */

const EMPTY: Finding[] = [];

export function ClusterImages({
  runId,
  label,
  members,
  dimensions,
  polygons,
}: {
  runId: string;
  /** The signature, used to title the viewer. */
  label: string;
  members: GroupMember[];
  /** Image dimensions by image id, so the overlay can build its viewport. */
  dimensions: Record<number, [number, number]>;
  /** Predicted outlines by finding id, from `mask_findings`. */
  polygons: Record<number, number[][]>;
}) {
  /** Index into `members`, or null when the viewer is closed. */
  const [at, setAt] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layers>(() =>
    defaultLayers(Object.keys(polygons).length > 0),
  );

  const member = at === null ? null : (members[at] ?? null);

  // Every finding on the member's image, so the one being inspected can be
  // drawn against its neighbours. Keyed by image id: without the key a slow
  // response for the previous member would render over the current one.
  const [fetched, setFetched] = useState<{ imageId: number; rows: Finding[] } | null>(
    null,
  );
  useEffect(() => {
    if (!member) return;
    let live = true;
    api
      .imageFindings(runId, member.image_id)
      .then((rows) => {
        if (live) setFetched({ imageId: member.image_id, rows });
      })
      .catch(() => {
        if (live) setFetched({ imageId: member.image_id, rows: [] });
      });
    return () => {
      live = false;
    };
  }, [runId, member]);

  const imageFindings =
    member && fetched && fetched.imageId === member.image_id ? fetched.rows : EMPTY;

  /** The member itself, with the geometry the list rows do not carry. */
  const selected = useMemo(
    () =>
      member ? (imageFindings.find((f) => f.id === member.finding_id) ?? null) : null,
    [member, imageFindings],
  );

  const context = useMemo(
    () =>
      member
        ? contextFor(member.finding_id, imageFindings, polygons, layers)
        : { siblings: [], predictions: [] },
    [member, imageFindings, polygons, layers],
  );

  const overlay = useMemo(() => {
    if (!member) return null;
    const size = dimensions[member.image_id];
    if (!size) return null;
    const [width, height] = size;
    return { width, height, viewport: fitViewport(width, height) };
  }, [member, dimensions]);

  if (members.length === 0) {
    return <p className="text-[11px] text-slate">No members returned.</p>;
  }

  const shown = members.slice(0, 6);
  const polygon = member ? (polygons[member.finding_id] ?? null) : null;

  return (
    <>
      <div className="flex flex-col gap-2">
        <div className="grid grid-cols-6 gap-1.5">
          {shown.map((m, index) => (
            <button
              key={m.finding_id}
              type="button"
              onClick={() => setAt(index)}
              title={`#${m.finding_id} · ${OUTCOME_LABEL[m.outcome] ?? m.outcome}`}
              className="group relative aspect-square rounded overflow-hidden bg-black/10 focus:outline-none focus:ring-2 focus:ring-brass/60"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={api.imageUrl(m.image_id)}
                alt=""
                loading="lazy"
                className="w-full h-full object-cover group-hover:opacity-80 transition-opacity"
              />
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setAt(0)}
          className="text-[11px] text-slate hover:text-brass transition-colors text-left"
        >
          {members.length > shown.length
            ? `View all ${members.length} images →`
            : `View ${members.length === 1 ? "the image" : "these images"} →`}
        </button>
      </div>

      <Lightbox
        open={member !== null}
        onClose={() => setAt(null)}
        title={member ? `${label} · finding #${member.finding_id}` : ""}
        subtitle={
          member
            ? `${member.class_name ?? "n/a"} · ${OUTCOME_LABEL[member.outcome] ?? member.outcome} · image #${member.image_id}`
            : undefined
        }
        position={at === null ? undefined : `${at + 1} of ${members.length}`}
        onPrev={at !== null && at > 0 ? () => setAt(at - 1) : undefined}
        onNext={
          at !== null && at < members.length - 1 ? () => setAt(at + 1) : undefined
        }
      >
        {member && (
          <div className="flex flex-col gap-3 items-center w-full">
            {overlay ? (
              <div
                className="max-h-[62vh] w-full"
                style={{
                  aspectRatio: `${overlay.width} / ${overlay.height}`,
                  maxWidth: "min(100%, 54vh)",
                }}
              >
                <Overlay
                  imageId={member.image_id}
                  width={overlay.width}
                  height={overlay.height}
                  viewport={overlay.viewport}
                  // Null until the geometry arrives, and null for good if this
                  // finding is not among the image's rows. The photograph and
                  // its context still draw; an overlay that waits for
                  // everything shows nothing while it waits.
                  truth={selected ? truthShapeOf(selected, layers) : null}
                  prediction={
                    selected ? predictionShapeOf(selected, polygon, layers) : null
                  }
                  siblings={context.siblings}
                  extras={context.predictions}
                  showMasks
                  showContext
                />
              </div>
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={api.imageUrl(member.image_id)}
                alt={`Image ${member.image_id}`}
                className="max-w-full max-h-[62vh] object-contain rounded"
              />
            )}

            <div className="flex flex-wrap items-center justify-center gap-1.5">
              <LayerToggle
                label="GT outline"
                on={layers.truthMask}
                onClick={() => setLayers({ ...layers, truthMask: !layers.truthMask })}
              />
              <LayerToggle
                label="GT box"
                on={layers.truthBox}
                onClick={() => setLayers({ ...layers, truthBox: !layers.truthBox })}
              />
              <LayerToggle
                label="Pred outline"
                on={layers.predictionMask}
                onClick={() =>
                  setLayers({ ...layers, predictionMask: !layers.predictionMask })
                }
              />
              <LayerToggle
                label="Pred box"
                on={layers.predictionBox}
                onClick={() =>
                  setLayers({ ...layers, predictionBox: !layers.predictionBox })
                }
              />
              <LayerToggle
                label="Show all"
                on={false}
                onClick={() => setLayers(ALL_LAYERS)}
              />
              <LayerToggle
                label="Image only"
                on={false}
                onClick={() => setLayers(NO_LAYERS)}
              />
            </div>

            <OverlayLegend
              hasContext={context.siblings.length + context.predictions.length > 0}
            />

            {/* Every member of the cluster, so the signature can be read across
                its findings rather than one photograph at a time — which is the
                only way to judge whether they have anything in common. */}
            <div className="w-full overflow-x-auto custom-scrollbar">
              <div className="flex gap-1.5 pb-1">
                {members.map((m, index) => (
                  <button
                    key={m.finding_id}
                    type="button"
                    onClick={() => setAt(index)}
                    title={`#${m.finding_id} · ${OUTCOME_LABEL[m.outcome] ?? m.outcome}`}
                    aria-current={index === at}
                    className={`shrink-0 w-12 h-12 rounded overflow-hidden bg-black/10 transition-all ${
                      index === at
                        ? "ring-2 ring-brass"
                        : "opacity-60 hover:opacity-100"
                    }`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={api.imageUrl(m.image_id)}
                      alt=""
                      loading="lazy"
                      className="w-full h-full object-cover"
                    />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </Lightbox>
    </>
  );
}
