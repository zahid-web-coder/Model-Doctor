"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { DRACO_PATH } from "@/lib/hero/asset";
import { LANDING_MODEL_URL } from "@/lib/hero/landing";
import { useHero } from "@/lib/hero/store";
import { useScan, useScanRequest } from "@/lib/hero/scan";
import { findingForPlate, type ScanFinding } from "@/lib/hero/scanFindings";
import { UNDER_BEAM } from "./InspectionSequence";

const PAYLOAD = /^Payload_\d+$/;

/** A pointer that moved further than this between down and up was orbiting. */
const CLICK_SLOP_PX = 5;

const mod1 = (v: number) => ((v % 1) + 1) % 1;

type Nodes = Record<string, THREE.Object3D | undefined>;

/**
 * Click a plate on the belt to scan it — or ask for an example by finding
 * from the 360° panel. 360° view only.
 *
 * **An invisible surface over the belt, not handlers on the plates.** The
 * plates are drawn by `Machine`, which the dashboard hero shares, and they are
 * one instanced mesh whose matrices change every frame. So this is a separate,
 * landing-only volume that tracks `MD_Root`'s frame — the same approach
 * `ArmDragger` takes for the arm — and a click on it is resolved to a plate by
 * position, using the formula `Machine` itself uses to lay the plates out.
 * `Machine` is not touched.
 *
 * **The belt only runs forward.** The carry is the forward distance from where
 * the belt is to where the chosen plate sits under the aperture, wrapped once.
 * Running it backwards to save a second would be the only moment the line ever
 * reversed, and it would read as a glitch.
 *
 * **An example request takes the plate that arrives soonest.** The panel
 * cannot see the belt, so it posts a finding and this picks the plate with the
 * shortest forward carry — the scan starts at once instead of after a lap.
 *
 * A pointer that travelled more than a few pixels was an orbit drag that
 * happened to end over the belt, and is ignored — otherwise every rotation
 * that finished on a plate would start a scan.
 */
export function PlateScanner({ enabled }: { enabled: boolean }) {
  const { nodes } = useGLTF(LANDING_MODEL_URL, DRACO_PATH) as unknown as { nodes: Nodes };

  const layout = useMemo(() => {
    const a = nodes.ConveyorStart?.position.x ?? -3;
    const b = nodes.ConveyorEnd?.position.x ?? 3;
    const from = Math.min(a, b);
    const to = Math.max(a, b);

    // Rest positions, which `Machine` reads and never writes — the instanced
    // copies carry the motion, and the source nodes stay where they were built.
    const rest = Object.values(nodes)
      .filter((n): n is THREE.Mesh => !!n && (n as THREE.Mesh).isMesh && PAYLOAD.test(n.name))
      .sort((x, y) => x.name.localeCompare(y.name))
      .map((m) => m.position.x);

    let top = 0.72;
    let depth = 0.5;
    let z = 0;
    const plate = nodes.InspectionObject as THREE.Mesh | undefined;
    if (plate) {
      plate.geometry.computeBoundingBox();
      const box = plate.geometry.boundingBox;
      if (box) {
        top = plate.position.y + box.max.y;
        depth = box.max.z - box.min.z;
      }
      z = plate.position.z;
    }
    return { from, to, span: to - from, rest, top, depth, z };
  }, [nodes]);

  /** Plate index → the belt progress at which it sits under the aperture. */
  const targetFor = useCallback((plate: number) => {
    const { from, span, rest } = layout;
    if (plate === rest.length) return UNDER_BEAM; // the inspection plate
    const lensX = from + UNDER_BEAM * span;
    return mod1((lensX - rest[plate]) / span);
  }, [layout]);

  /** Carry `plate` under the scanner and read it out as `finding`. */
  const scanPlate = useCallback((plate: number, finding: ScanFinding) => {
    const hero = useHero.getState();
    const progress = hero.objectProgress;
    useScan.setState({
      phase: "carry",
      finding,
      since: -1, // stamped by the render loop on its next frame
      from: progress,
      travel: mod1(targetFor(plate) - progress),
      armFrom: {
        armYaw: hero.armYaw,
        headTilt: hero.headTilt,
        headRotation: hero.headRotation,
      },
    });
  }, [targetFor]);

  // Example requests from the panel: the plate with the shortest carry.
  const request = useScanRequest((s) => s.request);
  useEffect(() => {
    if (!enabled || !request) return;
    const progress = useHero.getState().objectProgress;
    let best = 0;
    let shortest = Infinity;
    for (let plate = 0; plate <= layout.rest.length; plate += 1) {
      const travel = mod1(targetFor(plate) - progress);
      if (travel < shortest) { shortest = travel; best = plate; }
    }
    scanPlate(best, request.finding);
    // Consumed: a request left in the store would fire again the next time
    // the 360° view opens.
    useScanRequest.setState({ request: null });
  }, [enabled, request, layout, targetFor, scanPlate]);

  const root = nodes.MD_Root;

  // Follows MD_Root's world transform rather than being parented to it. A
  // mesh placed inside the loaded asset (via a portal) rendered correctly but
  // never received pointer events; one in the scene proper uses the same event
  // path as `ArmDragger`, and copying the matrix keeps it exactly aligned.
  const frame = useRef<THREE.Group>(null);
  useFrame(() => {
    if (!frame.current || !root) return;
    frame.current.matrix.copy(root.matrixWorld);
    frame.current.matrixWorldNeedsUpdate = true;
  });

  if (!enabled || !root) return null;

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > CLICK_SLOP_PX) return;
    e.stopPropagation();

    const local = root.worldToLocal(e.point.clone());
    const { from, to, span, rest } = layout;
    const progress = useHero.getState().objectProgress;

    // Which plate is under the pointer, by the layout `Machine` draws.
    let plate = -1;
    let gap = Infinity;
    rest.forEach((restX, i) => {
      let x = restX + progress * span;
      while (x > to) x -= span;
      const d = Math.abs(x - local.x);
      if (d < gap) { gap = d; plate = i; }
    });
    const inspection = rest.length;
    if (Math.abs(from + progress * span - local.x) < gap) plate = inspection;

    // Always the nearest plate, never nothing. An earlier version ignored
    // clicks in the gaps between plates — about a quarter of the belt — and a
    // click that silently does nothing reads as a broken control.
    if (plate < 0) return;
    scanPlate(plate, findingForPlate(plate));
  };

  const { from, to, span, top, depth, z } = layout;
  return (
    <group ref={frame} matrixAutoUpdate={false}>
      <mesh
        position={[(from + to) / 2, top + 0.05, z]}
        onClick={onClick}
        onPointerOver={() => { document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { document.body.style.cursor = ""; }}
      >
        <boxGeometry args={[span, 0.16, depth * 1.3]} />
        {/* Present for the raycaster, invisible to the eye. `visible={false}`
            would hide it from both. */}
        <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} />
      </mesh>
    </group>
  );
}
