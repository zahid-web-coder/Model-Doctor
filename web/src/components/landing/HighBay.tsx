"use client";

import { useMemo } from "react";
import * as THREE from "three";
import { SpotLight, Sparkles } from "@react-three/drei";

/**
 * Overhead high-bay lighting: where the key light comes from, made visible.
 *
 * **The key light already existed; it had no source.** A pool of light on the
 * line with nothing above it reads as a render, not a room. So the first
 * fixture hangs exactly where `LandingScene`'s key spot sits, and its shaft is
 * aimed where that spot is aimed — world origin, three's default target. Move
 * the key and this must move with it, or the shaft and the pool it lands in
 * stop agreeing.
 *
 * **The shaft carries no light of its own.** drei's `SpotLight` is a real
 * light plus a volumetric cone; here it is used for the cone alone, at zero
 * intensity, because the scene's key light is the carefully tuned one and a
 * second light on the same axis would double the pool on the belt.
 *
 * **Shafts end in the air, never on a surface.** The volume is an open
 * cylinder `distance` long, and without a depth buffer it is cut off hard
 * wherever it passes through geometry — the first version reached the floor
 * and drew a sharp ellipse around the arm. A per-frame depth pass would soften
 * that, at the cost of rendering the scene twice. Instead the cone stops above
 * the machine and `attenuation` fades it out before its end, which is also how
 * a shaft of light looks in a real hall: it dissolves, it does not land.
 *
 * **One background pool, for depth.** The far half of the room was a flat
 * brown wall. A second fixture over the back of the hall, with its own cheap
 * (shadowless) spot, puts a lit patch of floor behind the line — which is what
 * tells the eye the room goes back that far.
 *
 * **Dust is what makes the shaft read as air.** A handful of slow sparkles
 * inside the key cone. Frozen under reduced motion: they are pure decoration,
 * and decoration that moves is motion.
 */

/** Must match the key `spotLight` in `LandingScene`. */
const KEY = { position: [1.6, 5.2, 2.0] as const, target: [0, 0, 0] as const };

/** Behind and left of the line, over open floor. */
const BACK = { position: [-5.2, 6.2, -5.4] as const, target: [-5.2, 0, -5.4] as const };

/** The ceiling the fixtures hang from — the walls in `Hall` are 10 m tall. */
const CEILING_Y = 10;

function Fixture({ position }: { position: readonly [number, number, number] }) {
  const [x, y, z] = position;
  const drop = CEILING_Y - y;
  return (
    <group position={[x, y, z]}>
      {/* Suspension rod up to the ceiling. */}
      <mesh position={[0, drop / 2 + 0.1, 0]}>
        <cylinderGeometry args={[0.012, 0.012, drop, 6]} />
        <meshStandardMaterial color="#3a3833" roughness={0.6} metalness={0.5} />
      </mesh>
      {/* Housing: a shallow dark bell. */}
      <mesh position={[0, 0.12, 0]}>
        <cylinderGeometry args={[0.2, 0.34, 0.2, 24, 1, true]} />
        <meshStandardMaterial
          color="#2c2a26" roughness={0.45} metalness={0.7} side={2}
          envMapIntensity={0.8}
        />
      </mesh>
      <mesh position={[0, 0.22, 0]}>
        <cylinderGeometry args={[0.2, 0.2, 0.02, 24]} />
        <meshStandardMaterial color="#2c2a26" roughness={0.45} metalness={0.7} />
      </mesh>
      {/* The lamp face. Driven past the bloom threshold on purpose: this is
          the one thing in the room that should glow. */}
      <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.31, 32]} />
        <meshStandardMaterial
          color="#fff2df" emissive="#fff2df" emissiveIntensity={6}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

export function HighBay({ animate = true }: { animate?: boolean }) {
  // The background lamp's aim point, placed *in the scene*. A spot light reads
  // its target's world matrix, and a target that is not in the graph never has
  // one computed — it silently aims at the origin instead.
  const backTarget = useMemo(() => new THREE.Object3D(), []);

  return (
    <group>
      <Fixture position={KEY.position} />
      <SpotLight
        position={[...KEY.position]}
        target-position={[...KEY.target]}
        intensity={0}
        castShadow={false}
        volumetric
        color="#fff0d8"
        distance={2.6}
        angle={0.42}
        attenuation={2.4}
        anglePower={4}
        radiusTop={0.3}
        radiusBottom={0.95}
        opacity={0.34}
      />
      <Sparkles
        position={[1.05, 3.3, 1.3]}
        scale={[1.5, 2.6, 1.5]}
        count={45}
        size={1.3}
        speed={animate ? 0.18 : 0}
        opacity={0.4}
        noise={0.6}
        color="#ffe7c2"
      />

      <Fixture position={BACK.position} />
      {/* The background pool. No shadows: it lights open floor, and a second
          shadow map is the most expensive thing that could be added here.

          A plain spot, separate from the shaft below, because drei's
          `distance` sets both the cone's length *and* the light's range: one
          short enough to end in the air would stop the light 1.8 m short of
          the floor it exists to light. */}
      <primitive object={backTarget} position={[...BACK.target]} />
      <spotLight
        position={[...BACK.position]}
        target={backTarget}
        intensity={22}
        decay={1.3}
        distance={14}
        angle={0.5}
        penumbra={0.9}
        color="#ffeede"
      />
      <SpotLight
        position={[...BACK.position]}
        target-position={[...BACK.target]}
        intensity={0}
        castShadow={false}
        volumetric
        color="#ffeede"
        distance={4.4}
        angle={0.5}
        attenuation={3.8}
        anglePower={4}
        radiusTop={0.3}
        radiusBottom={1.35}
        opacity={0.22}
      />
    </group>
  );
}
