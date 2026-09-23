"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { grainedPolymer, paintedMetal } from "@/lib/hero/finishes";

/**
 * CCTV on all four walls: two bullet cameras per wall, each on a wall bracket,
 * panning slowly across the line and blinking a red "recording" lamp.
 *
 * **Why they belong here.** The room has two guards now; cameras on the walls
 * are the rest of the same story — this is a floor that is watched. They are
 * also the only detail on the far walls above head height, which were
 * otherwise empty concrete.
 *
 * **Aimed, not decorative.** Each head's rest direction is computed from its
 * mount towards the line (`TARGET`), tilted down by the real angle from its
 * height, so wherever the visitor orbits, every lens looks at the machine.
 * The pan is a slow sweep about that direction, each camera on its own period
 * and offset, so no two move together.
 *
 * Mount positions follow `Hall`'s walls (±13 m in x, ±12 m in z); move a wall
 * and these must move with it.
 */

const ROOM_X = 13;
const ROOM_Z = 12;
const MOUNT_H = 3.4;
const TARGET = new THREE.Vector3(0, 0.8, 0);

/** Mount point on the wall, and the wall's inward normal. */
const MOUNTS: { at: [number, number]; normal: [number, number] }[] = [
  { at: [-5.5, -ROOM_Z], normal: [0, 1] },
  { at: [5.5, -ROOM_Z], normal: [0, 1] },
  { at: [-5.5, ROOM_Z], normal: [0, -1] },
  { at: [5.5, ROOM_Z], normal: [0, -1] },
  { at: [-ROOM_X, -4.5], normal: [1, 0] },
  { at: [-ROOM_X, 4.5], normal: [1, 0] },
  { at: [ROOM_X, -4.5], normal: [-1, 0] },
  { at: [ROOM_X, 4.5], normal: [-1, 0] },
];

const ARM = 0.34; // wall to pan joint

function useCameraMaterials() {
  return useMemo(() => ({
    housing: paintedMetal({ color: "#e4e6e8", envMapIntensity: 0.9 }),
    bracket: paintedMetal({ color: "#d6d9dc", envMapIntensity: 0.8 }),
    dark: grainedPolymer({ color: "#16181c" }),
    glass: new THREE.MeshPhysicalMaterial({
      color: "#05070a", roughness: 0.05, metalness: 0.2,
      clearcoat: 1, clearcoatRoughness: 0.02, envMapIntensity: 1.4,
    }),
    // One lamp material per camera would let them blink independently; the
    // blink is driven by scaling each lamp instead, so one material serves all.
    lamp: new THREE.MeshStandardMaterial({
      color: "#220000", emissive: "#ff2a1a", emissiveIntensity: 2.2, toneMapped: false,
    }),
  }), []);
}

function Camera({
  at, normal, index, animate, m,
}: {
  at: [number, number];
  normal: [number, number];
  index: number;
  animate: boolean;
  m: ReturnType<typeof useCameraMaterials>;
}) {
  const pan = useRef<THREE.Group>(null);
  const lamp = useRef<THREE.Mesh>(null);

  // The wall faces `normal`; the mount's +Z points out of the wall.
  const wallYaw = Math.atan2(normal[0], normal[1]);
  const joint = new THREE.Vector3(at[0] + normal[0] * ARM, MOUNT_H - 0.12, at[1] + normal[1] * ARM);
  const toTarget = TARGET.clone().sub(joint);
  // Rest pan relative to the wall, and the downward tilt to the line.
  const restYaw = Math.atan2(toTarget.x, toTarget.z) - wallYaw;
  const tilt = Math.atan2(-toTarget.y, Math.hypot(toTarget.x, toTarget.z));
  const period = 13 + index * 1.7;

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (pan.current) {
      pan.current.rotation.y = restYaw + (animate ? Math.sin(t * ((2 * Math.PI) / period) + index) * 0.32 : 0);
    }
    if (lamp.current) {
      // A short blink every two seconds, staggered per camera.
      const on = !animate || ((t + index * 0.37) % 2) < 0.18;
      lamp.current.scale.setScalar(on ? 1 : 0.001);
    }
  });

  return (
    <group position={[at[0], MOUNT_H, at[1]]} rotation={[0, wallYaw, 0]}>
      {/* Wall plate and the arm out to the pan joint */}
      <RoundedBox args={[0.16, 0.2, 0.03]} radius={0.008} smoothness={3}
        position={[0, 0, 0.015]} material={m.bracket} castShadow />
      <mesh position={[0, 0, ARM / 2]} rotation={[Math.PI / 2, 0, 0]} material={m.bracket} castShadow>
        <cylinderGeometry args={[0.022, 0.026, ARM, 16]} />
      </mesh>
      <mesh position={[0, -0.06, ARM]} material={m.bracket} castShadow>
        <cylinderGeometry args={[0.026, 0.026, 0.12, 16]} />
      </mesh>

      {/* Pan, then tilt, then the camera body along +Z */}
      <group ref={pan} position={[0, -0.12, ARM]}>
        <mesh material={m.dark}>
          <cylinderGeometry args={[0.034, 0.034, 0.03, 20]} />
        </mesh>
        <group rotation={[tilt, 0, 0]} position={[0, -0.04, 0]}>
          {/* Housing: a bullet body with a sun hood over it */}
          <mesh position={[0, 0, 0.1]} rotation={[Math.PI / 2, 0, 0]} material={m.housing} castShadow>
            <cylinderGeometry args={[0.048, 0.048, 0.26, 28]} />
          </mesh>
          <mesh position={[0, 0, -0.03]} material={m.housing} castShadow>
            <sphereGeometry args={[0.048, 24, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
          </mesh>
          <RoundedBox args={[0.13, 0.012, 0.32]} radius={0.004} smoothness={2}
            position={[0, 0.056, 0.12]} material={m.housing} castShadow />
          {/* Lens: a dark bezel and glass, recessed in the front */}
          <mesh position={[0, 0, 0.232]} rotation={[Math.PI / 2, 0, 0]} material={m.dark}>
            <cylinderGeometry args={[0.044, 0.044, 0.008, 28]} />
          </mesh>
          <mesh position={[0, 0, 0.2365]} material={m.glass}>
            <circleGeometry args={[0.034, 28]} />
          </mesh>
          <mesh ref={lamp} position={[0.03, -0.026, 0.237]} material={m.lamp}>
            <sphereGeometry args={[0.0055, 10, 8]} />
          </mesh>
          {/* Cable out of the back */}
          <mesh position={[0, -0.03, -0.06]} rotation={[0.6, 0, 0]} material={m.dark}>
            <cylinderGeometry args={[0.006, 0.006, 0.08, 8]} />
          </mesh>
        </group>
      </group>
    </group>
  );
}

export function SecurityCameras({ animate = true }: { animate?: boolean }) {
  const m = useCameraMaterials();
  return (
    <>
      {MOUNTS.map((mount, i) => (
        <Camera key={i} index={i} at={mount.at} normal={mount.normal} animate={animate} m={m} />
      ))}
    </>
  );
}
