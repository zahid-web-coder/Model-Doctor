"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { scroll, clamp01 } from "@/lib/hero/landing";

/**
 * The two ends of the line: the machine that feeds plates onto it, and the
 * rack that stores them once they have been verified.
 *
 * Both are procedural for the same reason the verification station is — they
 * are cabinets and racks, which is boxes, and modelling them would mean a
 * second GLB to download and keep in step with an asset whose kinematic
 * assertions currently pass.
 *
 * They sit at the conveyor's own endpoints, read from the GLB: `ConveyorStart`
 * is x -3 and `ConveyorEnd` is x +3, both at y 0.7.
 */

const BELT_Y = 0.655;
const START_X = -3.0;
const END_X = 3.0;

function useMetals() {
  return useMemo(() => ({
    shell: new THREE.MeshStandardMaterial({
      color: "#d9d4c8", roughness: 0.44, metalness: 0.3, envMapIntensity: 1.05,
    }),
    dark: new THREE.MeshStandardMaterial({
      color: "#2a2a2c", roughness: 0.4, metalness: 0.7, envMapIntensity: 1.15,
    }),
    plate: new THREE.MeshStandardMaterial({
      color: "#1e1f22", roughness: 0.55, metalness: 0.45, envMapIntensity: 0.9,
    }),
  }), []);
}

/**
 * The infeed: a hopper cabinet at the head of the line with a stack of blanks
 * in it and a pusher that indexes one forward.
 *
 * The pusher is keyed to the conveyor rather than to a clock, so it strokes
 * when the line is moving and rests when it is not — a feeder cycling against
 * a stopped belt is the kind of detail that reads as wrong without anyone
 * being able to say why.
 */
export function InfeedDevice() {
  const m = useMetals();
  const pusher = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (!pusher.current) return;
    // One stroke per plate pitch. Sawtooth rather than a sine: a pusher snaps
    // out and returns, it does not oscillate.
    const cycle = (state.clock.elapsedTime * 0.55) % 1;
    const stroke = cycle < 0.35 ? cycle / 0.35 : 1 - clamp01((cycle - 0.35) / 0.45);
    pusher.current.position.x = -0.34 + stroke * 0.26;
  });

  return (
    <group position={[START_X - 0.42, 0, 0]}>
      {/* Cabinet */}
      <mesh position={[0, 0.46, 0]} material={m.shell} castShadow receiveShadow>
        <boxGeometry args={[0.72, 0.92, 0.86]} />
      </mesh>
      {/* Dark service panel and vent on the operator side */}
      <mesh position={[0, 0.5, 0.436]} material={m.dark}>
        <boxGeometry args={[0.5, 0.42, 0.02]} />
      </mesh>
      {[0.16, 0.1, 0.04].map((y) => (
        <mesh key={y} position={[0, y + 0.12, 0.437]} material={m.dark}>
          <boxGeometry args={[0.42, 0.022, 0.024]} />
        </mesh>
      ))}

      {/* Magazine above, holding the stack of blanks waiting to go on. */}
      <mesh position={[0.02, 1.03, 0]} material={m.dark} castShadow>
        <boxGeometry args={[0.46, 0.24, 0.6]} />
      </mesh>
      {[0, 1, 2, 3].map((i) => (
        <mesh key={i} position={[0.02, 0.94 + i * 0.036, 0]} material={m.plate} castShadow>
          <boxGeometry args={[0.4, 0.026, 0.5]} />
        </mesh>
      ))}

      {/* Outfeed lip, level with the rollers so a plate can slide on. */}
      <mesh position={[0.44, BELT_Y + 0.02, 0]} material={m.dark} castShadow>
        <boxGeometry args={[0.24, 0.035, 0.62]} />
      </mesh>
      {/* The pusher that indexes a plate out onto the belt. */}
      <mesh ref={pusher} position={[-0.34, BELT_Y + 0.06, 0]} material={m.shell} castShadow>
        <boxGeometry args={[0.1, 0.07, 0.44]} />
      </mesh>

      {/* Feet */}
      {[-0.26, 0.26].map((z) => (
        <mesh key={z} position={[0, 0.014, z]} material={m.dark}>
          <boxGeometry args={[0.6, 0.028, 0.14]} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * The storage rack at the tail of the line: verified plates come off the belt
 * and stack into it.
 *
 * How full it is follows `scroll.t` rather than a clock, so the stack grows as
 * the story advances and the line reads as having produced something by the
 * end. A rack that filled on a timer would be full before the first plate had
 * been inspected.
 */
export function StorageDevice() {
  const m = useMetals();
  const SLOTS = 8;

  return (
    <group position={[END_X + 0.46, 0, 0]}>
      {/* Infeed lip, level with the rollers, so plates have something to run
          onto rather than ending in mid-air. */}
      <mesh position={[-0.42, BELT_Y + 0.02, 0]} material={m.dark} castShadow>
        <boxGeometry args={[0.26, 0.035, 0.62]} />
      </mesh>

      {/* The cabinet. Same footprint, height and material family as the
          infeed at the other end, because they are a matched pair on one line
          — the first version was an open rack of bare struts that shared
          nothing with the rest of the machine and read as scaffolding. */}
      <mesh position={[0, 0.46, 0]} material={m.shell} castShadow receiveShadow>
        <boxGeometry args={[0.72, 0.92, 0.86]} />
      </mesh>

      {/* Collection well, sunk into the top: the plates stack down inside it
          rather than balancing on a shelf. */}
      <mesh position={[0, 0.95, 0]} material={m.dark} castShadow>
        <boxGeometry args={[0.62, 0.06, 0.76]} />
      </mesh>
      {[-0.35, 0.35].map((z) => (
        <mesh key={z} position={[0, 1.06, z]} material={m.shell} castShadow>
          <boxGeometry args={[0.6, 0.28, 0.05]} />
        </mesh>
      ))}
      {[-0.29, 0.29].map((x) => (
        <mesh key={x} position={[x, 1.06, 0]} material={m.shell} castShadow>
          <boxGeometry args={[0.05, 0.28, 0.72]} />
        </mesh>
      ))}

      {/* Glazed inspection window on the operator side, so the stack inside is
          visible and the cabinet is not a blank box. */}
      <mesh position={[0, 0.52, 0.436]} material={m.dark}>
        <boxGeometry args={[0.52, 0.44, 0.02]} />
      </mesh>
      <mesh position={[0, 0.52, 0.448]}>
        <planeGeometry args={[0.44, 0.36]} />
        <meshStandardMaterial
          color="#20303f" roughness={0.12} metalness={0.1}
          transparent opacity={0.55} envMapIntensity={1.4}
        />
      </mesh>

      {/* Counter readout — a small lit strip, so the cabinet looks like it is
          keeping count of what it has taken. */}
      <mesh position={[0.0, 0.82, 0.44]}>
        <planeGeometry args={[0.3, 0.05]} />
        <meshStandardMaterial
          color="#0d1b2a" emissive="#5fa8e8" emissiveIntensity={0.9}
          roughness={0.3} toneMapped={false}
        />
      </mesh>

      {/* The stack itself, sitting down in the well. */}
      <Stack slots={SLOTS} material={m.plate} />

      {/* Feet, matching the infeed. */}
      {[-0.26, 0.26].map((z) => (
        <mesh key={z} position={[0, 0.014, z]} material={m.dark} receiveShadow>
          <boxGeometry args={[0.6, 0.028, 0.14]} />
        </mesh>
      ))}
    </group>
  );
}

function Stack({ slots, material }: { slots: number; material: THREE.Material }) {
  const group = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!group.current) return;
    // Verification runs from t 0.6; the cabinet fills across that beat and the
    // hand-off, so by the last frame it holds a completed batch.
    const filled = Math.round(clamp01((scroll.t - 0.55) / 0.4) * slots);
    group.current.children.forEach((child, i) => { child.visible = i < filled; });
  });

  return (
    <group ref={group}>
      {Array.from({ length: slots }, (_, i) => (
        <mesh
          key={i}
          position={[0, 0.99 + i * 0.03, 0]}
          material={material}
          castShadow
          visible={false}
        >
          <boxGeometry args={[0.48, 0.026, 0.6]} />
        </mesh>
      ))}
    </group>
  );
}
