"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { scroll, clamp01 } from "@/lib/hero/landing";
import {
  EDGE_RADIUS, brushedMetal, grainedPolymer, paintedMetal,
} from "@/lib/hero/finishes";

/**
 * The two ends of the line: the machine that feeds plates onto it, and the
 * rack that stores them once they have been verified.
 *
 * Both are procedural for the same reason the verification station is — they
 * are cabinets and racks, which is boxes, and modelling them would mean a
 * second GLB to download and keep in step with an asset whose kinematic
 * assertions currently pass.
 *
 * **What makes a box read as equipment.** The first versions were a painted
 * block with a dark panel on it, and read as a prop. Real line cabinets carry
 * the same handful of cues, and those are what is built here, once, and
 * shared by both so the pair matches:
 *
 * - raised on levelling feet over a recessed plinth, not sitting on the floor
 * - double doors: a centre seam, hinges, handles
 * - a control station: red emergency stop on its yellow plate, start and stop
 *   buttons, a status lamp
 * - a warning label and ventilation louvres
 *
 * They sit at the conveyor's own endpoints, read from the GLB: `ConveyorStart`
 * is x -3 and `ConveyorEnd` is x +3, both at y 0.7. The operator side — the
 * one the camera sees — is +Z.
 */

const BELT_Y = 0.655;
const START_X = -3.0;
const END_X = 3.0;

/** Cabinet envelope, shared by both ends. */
const W = 0.72;          // along the line (x)
const D = 0.86;          // across it (z)
const TOP = 0.92;
const BASE = 0.07;       // body starts here; feet and plinth below
const BODY_H = TOP - BASE;
const FRONT = D / 2;     // the operator face, +Z

/**
 * The same colours as before, with the machine's finishes, plus the handful of
 * colours control hardware always has: e-stop red on yellow, start green.
 */
function useMetals() {
  return useMemo(() => ({
    shell: paintedMetal({ color: "#d9d4c8" }),
    dark: brushedMetal({ color: "#2a2a2c" }),
    plate: grainedPolymer({ color: "#1e1f22" }),
    rubber: grainedPolymer({ color: "#111214", envMapIntensity: 0.5 }),
    steel: brushedMetal({ color: "#9a9ea4" }),
    estop: new THREE.MeshPhysicalMaterial({
      color: "#b3141c", roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.1,
    }),
    yellow: new THREE.MeshPhysicalMaterial({
      color: "#e2b21a", roughness: 0.45, clearcoat: 0.6, clearcoatRoughness: 0.2,
    }),
    green: new THREE.MeshPhysicalMaterial({
      color: "#1f8a3c", roughness: 0.35, clearcoat: 1, clearcoatRoughness: 0.1,
    }),
    lamp: new THREE.MeshStandardMaterial({
      color: "#1c4d2a", emissive: "#3dff7a", emissiveIntensity: 2.8, toneMapped: false,
    }),
    glass: new THREE.MeshPhysicalMaterial({
      color: "#1a2530", roughness: 0.05, metalness: 0, clearcoat: 1,
      clearcoatRoughness: 0.03, transparent: true, opacity: 0.45, envMapIntensity: 1.5,
    }),
  }), []);
}

type Metals = ReturnType<typeof useMetals>;

/** Cylinders are Y-up; this turns one to face out of the operator side. */
const FACE_OUT: [number, number, number] = [Math.PI / 2, 0, 0];

/** Body, recessed plinth, and four levelling feet. */
function Cabinet({ m }: { m: Metals }) {
  const inset = 0.035;
  return (
    <>
      <RoundedBox
        args={[W, BODY_H, D]} radius={EDGE_RADIUS * 1.5} smoothness={3}
        position={[0, BASE + BODY_H / 2, 0]} material={m.shell} castShadow receiveShadow
      />
      {/* Kick plinth, set back so the body reads as standing on something. */}
      <mesh position={[0, BASE / 2, 0]} material={m.dark} receiveShadow>
        <boxGeometry args={[W - inset * 2, BASE, D - inset * 2]} />
      </mesh>
      {/* Levelling feet at the corners: threaded post and a rubber pad. */}
      {[-1, 1].flatMap((sx) => [-1, 1].map((sz) => (
        <group key={`${sx}${sz}`} position={[sx * (W / 2 - 0.05), 0, sz * (D / 2 - 0.05)]}>
          <mesh position={[0, 0.034, 0]} material={m.steel} castShadow>
            <cylinderGeometry args={[0.011, 0.011, 0.05, 12]} />
          </mesh>
          <mesh position={[0, 0.006, 0]} material={m.rubber} receiveShadow>
            <cylinderGeometry args={[0.034, 0.038, 0.012, 20]} />
          </mesh>
        </group>
      )))}
    </>
  );
}

/**
 * Double doors on the operator face: a centre seam, a hinge pair on each
 * outer edge, and a handle either side of the seam. Hardware only — the
 * doors are the body's own face, so they share its finish and its highlight.
 */
function Doors({ m }: { m: Metals }) {
  const y0 = BASE + 0.05;
  const y1 = TOP - 0.05;
  const mid = (y0 + y1) / 2;
  return (
    <>
      <mesh position={[0, mid, FRONT + 0.0015]} material={m.dark}>
        <boxGeometry args={[0.004, y1 - y0, 0.003]} />
      </mesh>
      {[-1, 1].map((side) => (
        <group key={side}>
          {[y0 + 0.12, y1 - 0.12].map((y) => (
            <mesh key={y} position={[side * (W / 2 - 0.022), y, FRONT + 0.006]} material={m.dark}>
              <cylinderGeometry args={[0.009, 0.009, 0.055, 12]} />
            </mesh>
          ))}
          {/* Handle: a bar on two standoffs. */}
          <group position={[side * 0.045, mid + 0.04, FRONT]}>
            <RoundedBox
              args={[0.014, 0.16, 0.014]} radius={0.005} smoothness={3}
              position={[0, 0, 0.03]} material={m.steel} castShadow
            />
            {[-0.065, 0.065].map((dy) => (
              <mesh key={dy} position={[0, dy, 0.015]} rotation={FACE_OUT} material={m.steel}>
                <cylinderGeometry args={[0.005, 0.005, 0.03, 8]} />
              </mesh>
            ))}
          </group>
        </group>
      ))}
    </>
  );
}

/**
 * The control station: a small dark housing with an emergency stop — red
 * mushroom on a yellow plate, the one piece of industrial hardware everyone
 * recognises — start and stop buttons, and a lit status lamp.
 */
function ControlPanel({ m, x }: { m: Metals; x: number }) {
  return (
    <group position={[x, TOP - 0.13, FRONT]}>
      <RoundedBox
        args={[0.19, 0.13, 0.05]} radius={0.008} smoothness={3}
        position={[0, 0, 0.025]} material={m.dark} castShadow
      />
      {/* E-stop */}
      <mesh position={[-0.05, 0, 0.052]} material={m.yellow}>
        <boxGeometry args={[0.07, 0.07, 0.004]} />
      </mesh>
      <mesh position={[-0.05, 0, 0.062]} rotation={FACE_OUT} material={m.estop} castShadow>
        <cylinderGeometry args={[0.024, 0.02, 0.018, 24]} />
      </mesh>
      {/* Start, stop */}
      <mesh position={[0.03, 0.028, 0.056]} rotation={FACE_OUT} material={m.green}>
        <cylinderGeometry args={[0.012, 0.012, 0.01, 18]} />
      </mesh>
      <mesh position={[0.03, -0.022, 0.056]} rotation={FACE_OUT} material={m.rubber}>
        <cylinderGeometry args={[0.012, 0.012, 0.01, 18]} />
      </mesh>
      {/* Status lamp */}
      <mesh position={[0.07, 0.028, 0.054]} material={m.lamp}>
        <sphereGeometry args={[0.007, 12, 8]} />
      </mesh>
    </group>
  );
}

/** A yellow warning triangle with its exclamation mark. */
function WarningLabel({ at }: { at: [number, number, number] }) {
  const yellow = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#e8b71c", roughness: 0.5, polygonOffset: true, polygonOffsetFactor: -1,
  }), []);
  const black = useMemo(() => new THREE.MeshStandardMaterial({
    color: "#111111", roughness: 0.6, polygonOffset: true, polygonOffsetFactor: -2,
  }), []);
  return (
    <group position={at}>
      <mesh material={yellow}>
        <circleGeometry args={[0.045, 3, Math.PI / 2]} />
      </mesh>
      <mesh position={[0, 0.004, 0.0005]} material={black}>
        <planeGeometry args={[0.007, 0.03]} />
      </mesh>
      <mesh position={[0, -0.02, 0.0005]} material={black}>
        <planeGeometry args={[0.007, 0.007]} />
      </mesh>
    </group>
  );
}

/** Ventilation louvres: angled slats in a dark frame. */
function Louvres({ m, at }: { m: Metals; at: [number, number, number] }) {
  return (
    <group position={at}>
      <mesh material={m.dark}>
        <boxGeometry args={[0.2, 0.12, 0.004]} />
      </mesh>
      {[-0.04, -0.013, 0.013, 0.04].map((y) => (
        <mesh key={y} position={[0, y, 0.006]} rotation={[0.5, 0, 0]} material={m.shell}>
          <boxGeometry args={[0.18, 0.018, 0.003]} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * The infeed: a cabinet at the head of the line with an open magazine of
 * blanks on top and a pusher that indexes one forward.
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
      <Cabinet m={m} />
      <Doors m={m} />
      <ControlPanel m={m} x={0.2} />
      <Louvres m={m} at={[-0.18, BASE + 0.16, FRONT + 0.002]} />
      <WarningLabel at={[-0.18, TOP - 0.16, FRONT + 0.002]} />

      {/* The magazine: an open cage of corner posts and guide rails, with the
          stack of blanks visible inside it rather than hidden in a box. */}
      <mesh position={[0.02, TOP + 0.012, 0]} material={m.dark} receiveShadow>
        <boxGeometry args={[0.5, 0.024, 0.64]} />
      </mesh>
      {[-1, 1].flatMap((sx) => [-1, 1].map((sz) => (
        <mesh key={`${sx}${sz}`} position={[0.02 + sx * 0.23, TOP + 0.16, sz * 0.3]} material={m.steel} castShadow>
          <boxGeometry args={[0.022, 0.3, 0.022]} />
        </mesh>
      )))}
      {[-1, 1].map((sz) => (
        <mesh key={sz} position={[0.02, TOP + 0.3, sz * 0.3]} material={m.steel} castShadow>
          <boxGeometry args={[0.48, 0.018, 0.018]} />
        </mesh>
      ))}
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <mesh key={i} position={[0.02, TOP + 0.04 + i * 0.03, 0]} material={m.plate} castShadow>
          <boxGeometry args={[0.4, 0.024, 0.5]} />
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

      {/* The cabinet. Same envelope and hardware as the infeed at the other
          end, because they are a matched pair on one line. */}
      <Cabinet m={m} />
      <Doors m={m} />
      <ControlPanel m={m} x={-0.2} />
      <WarningLabel at={[0.2, TOP - 0.16, FRONT + 0.002]} />

      {/* Glazed window in the right-hand door, framed, so the cabinet is not a
          blank box and reads as holding something. */}
      <mesh position={[0.19, 0.48, FRONT + 0.004]} material={m.dark}>
        <boxGeometry args={[0.24, 0.3, 0.006]} />
      </mesh>
      <mesh position={[0.19, 0.48, FRONT + 0.008]} material={m.glass}>
        <planeGeometry args={[0.21, 0.27]} />
      </mesh>

      {/* Counter readout — a small lit strip, so the cabinet looks like it is
          keeping count of what it has taken. */}
      <mesh position={[0.17, 0.7, FRONT + 0.004]}>
        <planeGeometry args={[0.2, 0.04]} />
        <meshStandardMaterial
          color="#0d1b2a" emissive="#5fa8e8" emissiveIntensity={0.9}
          roughness={0.3} toneMapped={false}
        />
      </mesh>

      {/* Collection well, sunk into the top: the plates stack down inside it
          rather than balancing on a shelf. */}
      <mesh position={[0, TOP + 0.03, 0]} material={m.dark} castShadow>
        <boxGeometry args={[0.62, 0.06, 0.76]} />
      </mesh>
      {[-0.35, 0.35].map((z) => (
        <RoundedBox
          key={z} args={[0.6, 0.28, 0.05]} radius={EDGE_RADIUS * 0.7} smoothness={3}
          position={[0, TOP + 0.14, z]} material={m.shell} castShadow
        />
      ))}
      {[-0.29, 0.29].map((x) => (
        <RoundedBox
          key={x} args={[0.05, 0.28, 0.72]} radius={EDGE_RADIUS * 0.7} smoothness={3}
          position={[x, TOP + 0.14, 0]} material={m.shell} castShadow
        />
      ))}

      {/* The stack itself, sitting down in the well. */}
      <Stack slots={SLOTS} material={m.plate} />
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
          position={[0, TOP + 0.07 + i * 0.03, 0]}
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
