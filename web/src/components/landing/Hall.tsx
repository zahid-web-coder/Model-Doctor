"use client";

import { useMemo } from "react";
import { MeshReflectorMaterial } from "@react-three/drei";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { createConcreteTexture, createHazardTexture } from "@/lib/hero/hallTextures";

/**
 * The inspection hall: walls, floor, service door and hazard bollards.
 *
 * **Deliberately not in the GLB.** A backdrop inside the asset would inflate
 * the file every page downloads, cast shadows the machine does not need, and
 * tie the environment to a Blender re-export every time it changes.
 *
 * **Merged where nothing animates.** The four walls are one mesh and the
 * door's metal fittings are another. None of them move, so a separate node per
 * part bought nothing and cost a draw call each. Tiling is baked into the UVs
 * before merging, which is what lets walls of different widths share one
 * material without stretching the panel bays.
 *
 * Landing page only. The dashboard hero is one panel of a dashboard and has no
 * use for a room.
 */

const WALL_H = 10;
const ROOM_X = 13;
const ROOM_Z = 12;
const BAY = 3.6; // metres per concrete bay, used to derive the UV tiling

const WALLS = [
  { pos: [-ROOM_X, WALL_H / 2, 0], rot: [0, Math.PI / 2, 0], w: ROOM_Z * 2 },
  { pos: [0, WALL_H / 2, -ROOM_Z], rot: [0, 0, 0], w: ROOM_X * 2 },
  { pos: [ROOM_X, WALL_H / 2, 0], rot: [0, -Math.PI / 2, 0], w: ROOM_Z * 2 },
  { pos: [0, WALL_H / 2, ROOM_Z], rot: [0, Math.PI, 0], w: ROOM_X * 2 },
] as const;

function buildWalls() {
  const parts = WALLS.map(({ pos, rot, w }) => {
    const geometry = new THREE.PlaneGeometry(w, WALL_H);
    const uv = geometry.attributes.uv;
    const tiles = Math.round(w / BAY);
    for (let i = 0; i < uv.count; i += 1) uv.setX(i, uv.getX(i) * tiles);
    uv.needsUpdate = true;
    geometry.applyMatrix4(new THREE.Matrix4().compose(
      new THREE.Vector3(pos[0], pos[1], pos[2]),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(rot[0], rot[1], rot[2])),
      new THREE.Vector3(1, 1, 1),
    ));
    return geometry;
  });
  return mergeGeometries(parts, false);
}

function buildDoorMetal() {
  const parts = [
    { geo: new THREE.BoxGeometry(0.42, 0.075, 0.05), at: [0.12, 2.36, 0.075] },
    { geo: new THREE.BoxGeometry(1.16, 0.3, 0.012), at: [0, 0.2, 0.068] },
    { geo: new THREE.BoxGeometry(0.13, 0.028, 0.03), at: [0.46, 1.12, 0.085] },
  ].map(({ geo, at }) => {
    geo.translate(at[0], at[1], at[2]);
    return geo;
  });
  return mergeGeometries(parts, false);
}

export function Hall() {
  const concrete = useMemo(() => createConcreteTexture(512), []);
  const hazard = useMemo(() => {
    const map = createHazardTexture(128);
    map?.repeat.set(1, 3);
    return map;
  }, []);
  const walls = useMemo(() => buildWalls(), []);
  const doorMetal = useMemo(() => buildDoorMetal(), []);

  return (
    <group>
      {/* Four walls, one mesh. Tone comes from the lighting rather than from
          per-wall material colours. */}
      <mesh geometry={walls} receiveShadow>
        <meshStandardMaterial
          map={concrete} color="#a89e8e" roughness={0.88} metalness={0}
          envMapIntensity={0.42}
        />
      </mesh>

      <group position={[-5.6, 0, -11.94]}>
        <mesh position={[0, 1.3, 0.03]}>
          <planeGeometry args={[1.42, 2.7]} />
          <meshStandardMaterial color="#726c62" roughness={0.78} />
        </mesh>
        <mesh position={[0, 1.24, 0.05]} receiveShadow>
          <planeGeometry args={[1.18, 2.46]} />
          <meshStandardMaterial
            color="#3c3a35" roughness={0.52} metalness={0.3}
            envMapIntensity={0.6}
          />
        </mesh>
        <mesh position={[-0.02, 1.62, 0.065]}>
          <planeGeometry args={[0.3, 1.06]} />
          <meshStandardMaterial
            color="#5c5b57" roughness={0.3} metalness={0.35}
            envMapIntensity={0.6}
          />
        </mesh>
        {/* Closer arm, kick plate and lever — one mesh, none of them move. */}
        <mesh geometry={doorMetal}>
          <meshStandardMaterial
            color="#8f8d87" roughness={0.32} metalness={0.68}
            envMapIntensity={0.75}
          />
        </mesh>
      </group>

      {/* Hazard bollards. The yellow is the only saturated colour in the room,
          which is why the reference has it — the eye finds it at once. */}
      {[-6.6, -4.6].map((x) => (
        <mesh key={x} position={[x, 0.42, -11.6]} castShadow receiveShadow>
          <cylinderGeometry args={[0.085, 0.095, 0.84, 14]} />
          <meshStandardMaterial
            map={hazard} roughness={0.62} metalness={0.1}
            envMapIntensity={0.35}
          />
        </mesh>
      ))}

      <mesh position={[-5.6, 0.004, -10.55]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[9.5, 0.075]} />
        <meshStandardMaterial color="#a98a1c" roughness={0.6} envMapIntensity={0.25} />
      </mesh>

      {/* Polished concrete, pulled well back from a mirror. A sealed floor
          returns a soft, broken suggestion of what is above it; a crisp
          reflection reads as wet glass.

          512, up from 256. At 256 the reflection under the arm broke into
          visible stair-stepped blocks — the one artifact on the page that
          read as cheap. 1024 would be four times 512's cost for a reflection
          this heavily blurred, where the difference does not survive. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[70, 70]} />
        <MeshReflectorMaterial
          resolution={512}
          mixBlur={2.6}
          mixStrength={0.9}
          blur={[900, 320]}
          mirror={0.18}
          depthScale={1.3}
          minDepthThreshold={0.2}
          maxDepthThreshold={1.6}
          color="#7d7566"
          roughness={0.78}
          metalness={0.08}
        />
      </mesh>
    </group>
  );
}
