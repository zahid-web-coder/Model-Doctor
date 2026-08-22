"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

/**
 * A security guard by the service door, monitoring the line with a tablet.
 *
 * **Built from primitives rather than loaded as a character model.** He stands
 * about twelve metres from the camera and occupies maybe fifty pixels of frame
 * height. A rigged human GLB would cost hundreds of kilobytes and thousands of
 * triangles to deliver detail that is gone at that size — the silhouette and
 * the colour blocking carry the entire read, and capsules produce both.
 *
 * Detail is spent only where it survives: the vest, the cap, the tablet and
 * the reflective bands. Those four shapes are what say "security guard" at
 * fifty pixels. A face would not.
 *
 * **Independently removable.** Delete the `<Worker />` line and nothing else
 * changes. He is not in the machine's GLB, shares no material with it, and his
 * idle loop writes only his own transforms — so he cannot collide with the
 * hero parameters or with the scroll timeline.
 */

const SKIN = "#a37e5c";
const NAVY = "#2b3446";
const TROUSER = "#232a3a";
const VEST = "#3f5670";
const REFLECT = "#9aa4ad";
const BOOT = "#15181f";
const TABLET = "#2a2d33";
const BADGE = "#e8eef6";

export function Worker({
  position = [-4.9, 0, -11.2] as [number, number, number],
  facing = 0.34,
}) {
  const root = useRef<THREE.Group>(null);
  const chest = useRef<THREE.Group>(null);
  const head = useRef<THREE.Group>(null);

  const m = useMemo(() => {
    const make = (color: string, roughness: number, extra: object = {}) =>
      new THREE.MeshStandardMaterial({ color, roughness, envMapIntensity: 0.35, ...extra });
    return {
      skin: make(SKIN, 0.78),
      navy: make(NAVY, 0.82, { envMapIntensity: 0.3 }),
      trouser: make(TROUSER, 0.86, { envMapIntensity: 0.25 }),
      // A muted duty vest, not hi-vis. Neon green here is the brightest thing
      // in frame after the beam and pulls the eye straight off the scanner —
      // the guard is meant to be secondary.
      vest: make(VEST, 0.66, { envMapIntensity: 0.45 }),
      reflect: make(REFLECT, 0.34, { metalness: 0.12, envMapIntensity: 0.6 }),
      boot: make(BOOT, 0.65),
      tablet: make(TABLET, 0.42, { metalness: 0.2, envMapIntensity: 0.6 }),
      badge: make(BADGE, 0.4, { metalness: 0.1, envMapIntensity: 0.6 }),
      patch: make("#8c9ab4", 0.6),
      watch: make("#1a1d24", 0.35, { metalness: 0.5, envMapIntensity: 0.8 }),
    };
  }, []);

  // Legs, boots, belt, radio and pocket flaps, baked into two geometries.
  const lower = useMemo(() => {
    const box = (w: number, h: number, d: number, at: [number, number, number], rot?: number) => {
      const g = new THREE.BoxGeometry(w, h, d);
      if (rot) g.applyMatrix4(new THREE.Matrix4().makeRotationY(rot));
      g.translate(at[0], at[1], at[2]);
      return g;
    };
    const capsule = (r: number, len: number, at: [number, number, number]) => {
      const g = new THREE.CapsuleGeometry(r, len, 3, 8);
      g.translate(at[0], at[1], at[2]);
      return g;
    };
    const cyl = (r: number, h: number, at: [number, number, number]) => {
      const g = new THREE.CylinderGeometry(r, r, h, 14);
      g.translate(at[0], at[1], at[2]);
      return g;
    };
    return {
      trouser: mergeGeometries([
        capsule(0.078, 0.6, [-0.09, 0.44, 0]),
        capsule(0.078, 0.6, [0.1, 0.44, 0.04]),
        box(0.1, 0.13, 0.012, [-0.155, 0.5, 0.02], -0.5),
        box(0.1, 0.13, 0.012, [0.165, 0.5, 0.06], 0.5),
      ], false),
      dark: mergeGeometries([
        box(0.115, 0.096, 0.25, [-0.09, 0.048, 0.03]),
        box(0.115, 0.096, 0.25, [0.1, 0.048, 0.07]),
        cyl(0.152, 0.055, [0, 0.9, 0]),
        box(0.05, 0.13, 0.045, [0.155, 0.855, -0.02]),
      ], false),
    };
  }, []);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    // Three motions on deliberately unrelated periods, so they never sync into
    // a visible pulse. Amplitudes are tiny — anything larger reads as a game
    // idle rather than a person standing still.
    if (chest.current) chest.current.scale.set(1, 1 + Math.sin(t * 0.9) * 0.013, 1);
    if (head.current) {
      head.current.rotation.y = -0.42 + Math.sin(t * 0.31) * 0.07;
      head.current.rotation.x = Math.sin(t * 0.23 + 1.1) * 0.03;
    }
    if (root.current) {
      root.current.rotation.z = Math.sin(t * 0.24) * 0.008;
      root.current.position.y = position[1] + Math.sin(t * 0.48) * 0.005;
    }
  });

  return (
    <group ref={root} position={position} rotation={[0, facing, 0]}>
      {/* Everything below the waist is static, so it is two merged meshes —
          one per material — rather than eight nodes. */}
      <mesh geometry={lower.trouser} material={m.trouser} castShadow />
      <mesh geometry={lower.dark} material={m.boot} castShadow />

      <group ref={chest} position={[0, 0.72, 0]}>
        {/* Navy shirt, flattened front-to-back: a capsule has a circular
            section, which is why an unscaled one reads as a barrel. A torso is
            roughly twice as wide as it is deep, and that ratio is most of what
            makes a silhouette read as a person. */}
        <mesh position={[0, 0.3, 0]} scale={[1.18, 1, 0.6]} material={m.navy} castShadow>
          <capsuleGeometry args={[0.152, 0.34, 4, 12]} />
        </mesh>
        {/* Shoulder yoke — the widest point, and what gives the figure a neck
            rather than a bottle-top. */}
        <mesh position={[0, 0.46, 0]} scale={[1, 0.42, 0.58]} material={m.navy} castShadow>
          <capsuleGeometry args={[0.185, 0.1, 4, 12]} />
        </mesh>

        <mesh position={[0, 0.29, 0.008]} scale={[1.18, 1, 0.64]} material={m.vest} castShadow>
          <capsuleGeometry args={[0.158, 0.24, 4, 12]} />
        </mesh>
        <mesh position={[0, 0.26, 0.008]} scale={[1.18, 1, 0.64]} material={m.reflect}>
          <cylinderGeometry args={[0.162, 0.162, 0.036, 16]} />
        </mesh>
        <mesh position={[-0.078, 0.43, 0.055]} material={m.reflect}>
          <boxGeometry args={[0.034, 0.15, 0.018]} />
        </mesh>
        <mesh position={[0.078, 0.43, 0.055]} material={m.reflect}>
          <boxGeometry args={[0.034, 0.15, 0.018]} />
        </mesh>
        {/* Collar — a small step at the neckline, but it is what stops the
            torso reading as a single extruded capsule. */}
        <mesh position={[0, 0.53, 0.01]} scale={[1.1, 1, 0.7]} material={m.navy}>
          <cylinderGeometry args={[0.082, 0.094, 0.05, 12]} />
        </mesh>
        <mesh position={[-0.232, 0.31, 0.05]} rotation={[0, -0.35, 0]} material={m.patch}>
          <circleGeometry args={[0.032, 12]} />
        </mesh>
        <mesh position={[0, 0.29, 0.108]} material={m.navy}>
          <boxGeometry args={[0.012, 0.24, 0.012]} />
        </mesh>
        <mesh position={[0.058, 0.2, 0.104]} material={m.badge}>
          <boxGeometry args={[0.05, 0.068, 0.006]} />
        </mesh>
        <mesh position={[0.032, 0.36, 0.1]} rotation={[0, 0, 0.28]} material={m.boot}>
          <boxGeometry args={[0.006, 0.22, 0.004]} />
        </mesh>

        {/* Arms, bent to hold the tablet at waist height */}
        {[-1, 1].map((side) => (
          <group key={side}>
            <mesh
              position={[side * 0.222, 0.22, 0.012]}
              rotation={[0.26, 0, side * -0.13]}
              material={m.navy} castShadow
            >
              <capsuleGeometry args={[0.054, 0.25, 3, 8]} />
            </mesh>
            <mesh
              position={[side * 0.198, 0.03, 0.155]}
              rotation={[1.15, 0, side * -0.05]}
              material={m.navy} castShadow
            >
              <capsuleGeometry args={[0.047, 0.22, 3, 8]} />
            </mesh>
            <mesh position={[side * 0.152, -0.015, 0.248]} material={m.skin}>
              <sphereGeometry args={[0.046, 8, 6]} />
            </mesh>
            {/* Watch on one wrist only — an asymmetric detail is worth more at
                this size than two symmetric ones. */}
            {side === -1 && (
              <mesh position={[-0.176, 0.015, 0.218]} rotation={[1.15, 0, 0]} material={m.watch}>
                <cylinderGeometry args={[0.021, 0.021, 0.012, 10]} />
              </mesh>
            )}
          </group>
        ))}

        <mesh position={[0, -0.015, 0.255]} rotation={[-1.05, 0, 0]} material={m.tablet} castShadow>
          <boxGeometry args={[0.21, 0.008, 0.27]} />
        </mesh>
      </group>

      {/* Neck, head, and the cap */}
      <mesh position={[0, 1.3, 0]} material={m.skin}>
        <capsuleGeometry args={[0.045, 0.055, 2, 8]} />
      </mesh>
      <group ref={head} position={[0, 1.42, 0]}>
        <mesh material={m.skin} castShadow>
          <sphereGeometry args={[0.097, 12, 10]} />
        </mesh>
        <mesh position={[0, -0.03, 0.03]} material={m.trouser}>
          <sphereGeometry args={[0.088, 10, 8, 0, Math.PI * 2, 1.5, 0.9]} />
        </mesh>
        {/* Cap crown and peak. The peak is what makes the silhouette read as a
            uniform cap rather than hair. */}
        <mesh position={[0, 0.03, -0.006]} material={m.navy} castShadow>
          <sphereGeometry args={[0.103, 12, 10, 0, Math.PI * 2, 0, 1.25]} />
        </mesh>
        <mesh position={[0, 0.028, 0.098]} rotation={[0.2, 0, 0]} material={m.navy} castShadow>
          <boxGeometry args={[0.16, 0.014, 0.1]} />
        </mesh>
      </group>
    </group>
  );
}
