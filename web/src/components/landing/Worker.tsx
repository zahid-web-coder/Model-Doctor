"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { DRACO_PATH } from "@/lib/hero/asset";
import { dressGuard, GUARD_MODELS_VERSION } from "@/lib/hero/guardMaterials";

export const GUARD_URL = `/models/guard.glb?v=${GUARD_MODELS_VERSION}`;

/**
 * A security guard by the service door, monitoring the line with a tablet.
 *
 * **A modelled figure, not primitives.** The first guard was capsules, spheres
 * and boxes assembled here. From the hero camera that was enough; in the 360°
 * view, where a visitor can orbit up to him, it read as a stack of parts —
 * limbs that did not join, a ball for a head, boxes for feet. He is now one
 * continuous body built in Blender (`md_guard.py` in the hero3d project): a
 * skinned skeleton with shoulders, elbows and knees, a shaped head with a
 * face, a vest and belt fitted to the body, a patrol cap and laced boots —
 * about 128 KB with Draco, fetched only where the 3D scene runs.
 *
 * **Geometry from Blender, materials here** — the same split as the machine.
 * The GLB's materials are placeholders named `Guard_*`, replaced by name from
 * `guardMaterials.ts`, which the patrolling guard shares.
 *
 * **Independently removable.** Delete the `<Worker />` line and nothing else
 * changes. He shares no material with the machine, and his idle loop writes
 * only his own transforms — so he cannot collide with the hero parameters or
 * with the scroll timeline.
 */

/** The tablet's display: a camera-wall layout, drawn once. */
function screenTexture(): THREE.CanvasTexture {
  const w = 320;
  const h = 224;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  g.fillStyle = "#0b1522";
  g.fillRect(0, 0, w, h);
  g.fillStyle = "#15283d";
  g.fillRect(0, 0, w, 22);
  g.fillStyle = "#9fdcae";
  g.fillRect(10, 8, 6, 6);
  g.fillStyle = "#c9d6e6";
  g.fillRect(24, 9, 70, 4);
  // A 3×2 wall of camera feeds, each a dim gradient with a label bar.
  for (let row = 0; row < 2; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      const x = 10 + col * 102;
      const y = 32 + row * 92;
      const grad = g.createLinearGradient(x, y, x + 94, y + 80);
      grad.addColorStop(0, "#2d4561");
      grad.addColorStop(1, "#16263a");
      g.fillStyle = grad;
      g.fillRect(x, y, 94, 80);
      g.fillStyle = "rgba(0,0,0,0.35)";
      g.fillRect(x, y + 68, 94, 12);
      g.fillStyle = row === 0 && col === 1 ? "#e8b86a" : "#8fb4d8";
      g.fillRect(x + 5, y + 72, 30, 4);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export function Worker({
  position = [-4.9, 0, -11.2] as [number, number, number],
  facing = 0.34,
}) {
  const { scene } = useGLTF(GUARD_URL, DRACO_PATH);
  const root = useRef<THREE.Group>(null);

  // The one material only he has: the tablet's display, lit. A person holding
  // a dark slab reads as a prop; a person looking at a lit screen reads as
  // working.
  const screen = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#000000", emissive: "#ffffff", emissiveIntensity: 0.9,
    emissiveMap: screenTexture(), roughness: 0.2, toneMapped: false,
  }), []);

  useEffect(() => dressGuard(scene, { Guard_Screen: screen }), [scene, screen]);

  useEffect(() => () => {
    screen.emissiveMap?.dispose();
    screen.dispose();
  }, [screen]);

  const head = useMemo(() => scene.getObjectByName("GuardHead") ?? null, [scene]);
  const headRest = useMemo(() => head?.quaternion.clone() ?? null, [head]);
  const turn = useMemo(() => new THREE.Euler(), []);
  const q = useMemo(() => new THREE.Quaternion(), []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    // Three motions on deliberately unrelated periods, so they never sync into
    // a visible pulse. Amplitudes are tiny — anything larger reads as a game
    // idle rather than a person standing still.
    if (head && headRest) {
      // Mostly down at the tablet, with an occasional look along the line.
      turn.set(0.12 + Math.sin(t * 0.23 + 1.1) * 0.03, -0.42 + Math.sin(t * 0.31) * 0.07, 0);
      head.quaternion.copy(headRest).multiply(q.setFromEuler(turn));
    }
    if (root.current) {
      root.current.rotation.z = Math.sin(t * 0.24) * 0.006;
      // Breathing: a rise of a few millimetres at the shoulders.
      root.current.scale.y = 1 + Math.sin(t * 0.9) * 0.0035;
    }
  });

  return (
    <group ref={root} position={position} rotation={[0, facing, 0]}>
      <primitive object={scene} />
    </group>
  );
}
