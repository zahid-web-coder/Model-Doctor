"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { driveHinge, clamp01, HINGE } from "@/lib/hero/axes";
import { useHero } from "@/lib/hero/store";
import { createHeatmapTexture } from "@/lib/hero/heatmap";
import { createBeamMaterial } from "@/lib/hero/beamMaterial";
import { MODEL_URL, DRACO_PATH } from "@/lib/hero/asset";

/** Every node the runtime addresses by name. Checked on load, not assumed. */
export const REQUIRED = [
  "MD_Root", "ArmBase", "ArmShoulder", "ArmForearm", "ScannerHead",
  "ScannerLens", "Beam", "InspectionObject", "ConveyorStart", "ConveyorEnd",
] as const;

const ROLLER = /^Roller_\d+$/;
const PAYLOAD = /^Payload_\d+$/;
const IDENTITY = new THREE.Quaternion();

type Nodes = Record<string, THREE.Object3D | undefined>;

type Instanced = {
  mesh: THREE.InstancedMesh;
  members: THREE.Mesh[];
  rest: THREE.Vector3[];
} | null;

/**
 * Collapse a set of identical sibling meshes into one InstancedMesh.
 *
 * The rollers are 40 separate nodes in the asset because each turns about its
 * own axis, and joining them in Blender would have made them orbit a shared
 * origin. That is a modelling constraint, not a rendering one: here they are
 * one draw call whose per-instance matrices carry the spin.
 */
function useInstanced(nodes: Nodes, pattern: RegExp, parentName: string): Instanced {
  return useMemo(() => {
    const members = Object.values(nodes)
      .filter((n): n is THREE.Mesh => !!n && (n as THREE.Mesh).isMesh && pattern.test(n.name))
      .sort((a, b) => a.name.localeCompare(b.name));
    if (members.length === 0) return null;

    const mesh = new THREE.InstancedMesh(
      members[0].geometry, members[0].material, members.length,
    );
    mesh.name = `${parentName}_Instanced`;
    mesh.frustumCulled = false;
    mesh.castShadow = mesh.receiveShadow = true;
    return { mesh, members, rest: members.map((m) => m.position.clone()) };
  }, [nodes, pattern, parentName]);
}

export function Machine({ animate = true }: { animate?: boolean }) {
  const { scene, nodes } = useGLTF(MODEL_URL, DRACO_PATH) as unknown as {
    scene: THREE.Group;
    nodes: Nodes;
  };
  const [ready, setReady] = useState(false);

  const rollers = useInstanced(nodes, ROLLER, "Rollers");
  const payloads = useInstanced(nodes, PAYLOAD, "Payloads");

  // The rest pose the ASSET carries. Captured once, never written back, so the
  // angles in docs/HERO_3D_DESIGN.md live in exactly one place: the .glb.
  const rest = useMemo(() => ({
    ArmBase: nodes.ArmBase?.quaternion.clone(),
    ArmShoulder: nodes.ArmShoulder?.quaternion.clone(),
    ArmForearm: nodes.ArmForearm?.quaternion.clone(),
    ScannerHead: nodes.ScannerHead?.quaternion.clone(),
  }), [nodes]);

  // Travel extent read out of the asset rather than hardcoded. Lengthen the
  // conveyor in Blender and the runtime follows.
  const track = useMemo(() => {
    const a = nodes.ConveyorStart?.position.x ?? -3;
    const b = nodes.ConveyorEnd?.position.x ?? 3;
    return { from: Math.min(a, b), to: Math.max(a, b) };
  }, [nodes]);

  const heatmapRef = useRef<THREE.Mesh | null>(null);
  const plateTop = useRef(0);
  const plateSpan = useRef(0.32);
  const beamMaterial = useRef<THREE.ShaderMaterial | null>(null);
  const lensMaterial = useRef<THREE.MeshStandardMaterial | null>(null);

  useLayoutEffect(() => {
    scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const material = mesh.material as THREE.MeshStandardMaterial;
      if (!material || !material.isMeshStandardMaterial) return;
      // The probe is the whole reason metals read. Push its contribution up on
      // the metal families and keep the painted shell restrained, which is
      // what separates "painted metal" from "chrome toy".
      material.envMapIntensity = material.metalness > 0.5 ? 1.55 : 1.15;
      // Whiter and glossier than the surround, or the device disappears into
      // it. Value separation is what makes a white object read as white.
      if (material.name === "MD_Shell") material.roughness = 0.23;
      if (material.name === "MD_Metal") material.roughness = 0.2;
      if (material.name === "MD_Steel") material.roughness = 0.3;
    });

    const beam = nodes.Beam as THREE.Mesh | undefined;
    if (beam) {
      // The exported material is replaced outright rather than tuned. Blender's
      // emission strengths do not transfer, and more importantly a translucent
      // cone reads as smoked glass, not as light. See beamMaterial.ts.
      beam.material = createBeamMaterial();
      beam.renderOrder = 3;
      beamMaterial.current = beam.material as THREE.ShaderMaterial;
    }

    const label = nodes.ScannerLabel as THREE.Mesh | undefined;
    if (label) {
      // Polished dark metal. Matte ink reads as more legible on paper, but on
      // the shell it looks printed on; the specular version catches the room
      // and reads as machined badging.
      const material = (label.material as THREE.MeshStandardMaterial).clone();
      Object.assign(material, {
        color: new THREE.Color("#15181d"),
        metalness: 0.85,
        roughness: 0.2,
        envMapIntensity: 1.8,
      });
      material.needsUpdate = true;
      label.material = material;
    }

    const lens = nodes.ScannerLens as THREE.Mesh | undefined;
    if (lens) {
      const material = (lens.material as THREE.MeshStandardMaterial).clone();
      material.toneMapped = false;
      material.emissive = new THREE.Color("#bfe0ff");
      lens.material = material;
      lensMaterial.current = material;
    }

    setReady(true);
  }, [scene, nodes]);

  // Swap the individual roller and plate nodes for their instanced stand-ins.
  useEffect(() => {
    const groups = [rollers, payloads].filter((g): g is NonNullable<Instanced> => !!g);
    groups.forEach(({ mesh, members }) => {
      members.forEach((m) => { m.visible = false; });
      members[0].parent?.add(mesh);
    });
    return () => groups.forEach(({ mesh, members }) => {
      mesh.parent?.remove(mesh);
      members.forEach((m) => { m.visible = true; });
    });
  }, [rollers, payloads]);

  // The heatmap plane. Parented to MD_Root rather than to a plate, because it
  // is not owned by any one plate: it belongs to whichever plate is under the
  // aperture at this instant, and it moves between them as they pass.
  useEffect(() => {
    const root = nodes.MD_Root;
    const plate = nodes.InspectionObject as THREE.Mesh | undefined;
    if (!root || !plate) return;
    plate.geometry.computeBoundingBox();
    const box = plate.geometry.boundingBox;
    if (!box) return;
    const width = (box.max.x - box.min.x) * 0.82;
    const depth = (box.max.z - box.min.z) * 0.82;
    const texture = createHeatmapTexture();

    const overlay = new THREE.Mesh(
      new THREE.PlaneGeometry(width, depth),
      new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    overlay.name = "HeatmapOverlay";
    overlay.rotation.x = -Math.PI / 2;
    overlay.renderOrder = 2;
    root.add(overlay);
    heatmapRef.current = overlay;
    plateTop.current = plate.position.y + box.max.y + 0.0016;
    plateSpan.current = box.max.x - box.min.x;
    return () => {
      root.remove(overlay);
      overlay.geometry.dispose();
      texture?.dispose();
    };
  }, [nodes]);

  const matrix = useMemo(() => new THREE.Matrix4(), []);
  const slot = useMemo(() => new THREE.Vector3(), []);
  const lensWorld = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  const spinAxis = useMemo(() => new THREE.Vector3(0, 0, -1), []);
  const one = useMemo(() => new THREE.Vector3(1, 1, 1), []);
  const spin = useRef(0);

  useFrame((state, rawDelta) => {
    const s = useHero.getState();
    const span = track.to - track.from;
    // Frozen for reduced motion: the scene still renders, the loop still runs,
    // but nothing advances. Zeroing delta rather than skipping the callback
    // keeps every derived value (heatmap, beam, hinges) correct for the pose
    // it is frozen at, instead of leaving them wherever the last frame left.
    const delta = animate ? rawDelta : 0;

    // --- useFrame owns what loops (§3) ---------------------------------
    spin.current += s.conveyorSpeed * delta * 6.0;

    if (s.driveObject) {
      const next = s.objectProgress + s.conveyorSpeed * delta * 0.13;
      useHero.setState({ objectProgress: next > 1 ? next - 1 : next });
    }
    const progress = useHero.getState().objectProgress;

    if (rollers) {
      quat.setFromAxisAngle(spinAxis, spin.current);
      rollers.rest.forEach((position, index) => {
        matrix.compose(position, quat, one);
        rollers.mesh.setMatrixAt(index, matrix);
      });
      rollers.mesh.instanceMatrix.needsUpdate = true;
    }

    if (payloads) {
      const shift = progress * span;
      payloads.rest.forEach((position, index) => {
        let x = position.x + shift;
        while (x > track.to) x -= span;
        slot.set(x, position.y, position.z);
        matrix.compose(slot, IDENTITY, one);
        payloads.mesh.setMatrixAt(index, matrix);
      });
      payloads.mesh.instanceMatrix.needsUpdate = true;
    }

    if (nodes.InspectionObject) {
      nodes.InspectionObject.position.x = track.from + progress * span;
    }

    // Beam flicker is a small multiplier on the base intensity, never a write
    // to it: the timeline owns the base, useFrame owns the flicker (§3).
    const flicker = animate
      ? 1 + 0.05 * Math.sin(state.clock.elapsedTime * 11.0)
      : 1;
    const beam = s.beamIntensity * flicker;
    if (beamMaterial.current) {
      beamMaterial.current.uniforms.uIntensity.value = beam;
      beamMaterial.current.uniforms.uTime.value = state.clock.elapsedTime;
    }
    if (lensMaterial.current) {
      lensMaterial.current.emissiveIntensity = 5.5 * beam;
    }

    // --- Derived, never stored (§3) ------------------------------------
    // The heatmap belongs to whichever plate is currently under the aperture,
    // and only while it is there. Position comes from the aperture and the
    // plate positions, opacity from the distance between them and
    // beamIntensity. Nothing is stored, so nothing can fall out of step.
    if (heatmapRef.current && nodes.MD_Root && nodes.ScannerLens) {
      nodes.ScannerLens.getWorldPosition(lensWorld);
      nodes.MD_Root.worldToLocal(lensWorld);

      let bestX = 0;
      let bestZ = 0;
      let bestGap = Infinity;
      const consider = (x: number, z: number) => {
        const gap = Math.abs(x - lensWorld.x);
        if (gap < bestGap) { bestGap = gap; bestX = x; bestZ = z; }
      };
      if (nodes.InspectionObject) {
        consider(nodes.InspectionObject.position.x, nodes.InspectionObject.position.z);
      }
      if (payloads) {
        const shift = progress * span;
        payloads.rest.forEach((position) => {
          let x = position.x + shift;
          while (x > track.to) x -= span;
          consider(x, position.z);
        });
      }

      // A window about a plate wide: full strength directly under the beam,
      // nothing by the time the plate has cleared it.
      const half = plateSpan.current * 0.62;
      const w = clamp01(1 - bestGap / half);
      const eased = w * w * (3 - 2 * w);

      const overlay = heatmapRef.current;
      overlay.position.set(bestX, plateTop.current, bestZ);
      (overlay.material as THREE.MeshBasicMaterial).opacity = eased * s.beamIntensity;
      overlay.visible = eased > 0.002;
    }

    // --- Timeline-owned values, applied as deltas from the rest pose -----
    // Every one is a delta, so parameter zero is the pose as modelled and no
    // rest angle is duplicated here where it could drift from the .glb.
    if (rest.ArmBase && nodes.ArmBase) {
      // armYaw is normalised -1..1; 0.6 rad is the sweep the arm can make
      // without the aperture leaving the belt.
      driveHinge(nodes.ArmBase, rest.ArmBase, "Z", s.armYaw * 0.6);
    }
    if (rest.ArmShoulder && nodes.ArmShoulder) {
      driveHinge(nodes.ArmShoulder, rest.ArmShoulder, "Y", s.shoulderSwing);
    }
    if (rest.ArmForearm && nodes.ArmForearm) {
      driveHinge(nodes.ArmForearm, rest.ArmForearm, "Y", s.forearmSwing);
    }
    if (rest.ScannerHead && nodes.ScannerHead) {
      // Pan about the head's vertical, then tilt about its transverse axis.
      // Composed in that order so headTilt always means "tilt from level",
      // whatever the pan happens to be.
      driveHinge(nodes.ScannerHead, rest.ScannerHead, "Z", s.headRotation);
      nodes.ScannerHead.rotateOnAxis(HINGE.Y, s.headTilt);
    }
  });

  return <primitive object={scene} visible={ready} />;
}

useGLTF.preload(MODEL_URL, DRACO_PATH);
