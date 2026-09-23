"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { SpotLight, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { DRACO_PATH } from "@/lib/hero/asset";
import { hallLevel } from "@/lib/hero/lights";
import { addFur } from "@/lib/hero/fur";
import { dressGuard, guardMaterials, GUARD_MODELS_VERSION } from "@/lib/hero/guardMaterials";

export const WALKER_URL = `/models/guard-walker.glb?v=${GUARD_MODELS_VERSION}`;
export const DOG_URL = `/models/guard-dog.glb?v=${GUARD_MODELS_VERSION}`;

/**
 * A second guard, patrolling behind the line with his dog.
 *
 * He walks the length of the line, stops at the robot and again at the
 * verification station to look them over, then walks on, turns at the end and
 * comes back; the loop is about 45 seconds. His German Shepherd walks at his
 * right on a lead, a little ahead, as a handler's dog does.
 *
 * **The dog follows; she is not scripted.** Her place is a point beside him,
 * and she moves towards it at her own pace — so she lags into his turns,
 * swings round him when he about-faces, and catches up when he sets off.
 * She faces the way she is moving, and her trot, like his walk, is advanced
 * by distance covered, so her paws cannot skate. When he stops she stands,
 * looks about, sniffs the floor and wags. The lead is a real tube from his
 * fist to her collar that sags more the closer they are.
 *
 * **In the dark he takes a torch.** Throw the hall lights off and he raises
 * a torch in his free hand — lighting the floor ahead as he walks, and the
 * belt when he stops to check it. Lights on, it is gone.
 *
 * **Jointed, not baked.** His GLB (`md_guard.py --walker`) is the door
 * guard's body cut at the hips, knees, shoulders and elbows, each part's
 * origin on its hinge. The walk is procedural: one gait phase drives every
 * joint, advanced by *distance travelled*, not time — so the feet cannot
 * slide, and as he slows into a stop the stride shrinks with his speed and
 * the legs come to rest together rather than freezing mid-step.
 *
 * **Behind the line, on purpose.** The route runs 1.7 m behind the belt,
 * clear of the cabinets and the station, so he never crosses between the
 * camera and the machine; the belt hides his legs for part of each pass,
 * which is what puts him *in* the room rather than on top of it.
 *
 * **Independently removable**, like the door guard: delete the
 * `<GuardPatrol />` line and nothing else changes.
 */

const PATH_Z = -1.7;
const SPEED = 0.95;        // m/s — an unhurried patrol
const STRIDE = 1.25;       // metres per gait cycle (two steps)
const RAMP = 0.45;         // seconds to reach speed, and to stop
const TURN_RATE = Math.PI / 1.5; // rad/s when turning on the spot
const RAISE = 0.6;         // seconds to raise or lower the torch

type Phase =
  | { kind: "walk"; from: number; to: number; heading: number; dur: number }
  | { kind: "turn"; from: number; to: number; dur: number }
  | { kind: "inspect"; dur: number };

const FACE_LINE = 0;               // +Z, towards the belt
const EAST = Math.PI / 2;          // +X
const WEST = -Math.PI / 2;

/** The patrol as a list of timed phases, looped. */
function buildRoute(): { phases: (Phase & { start: number })[]; total: number } {
  const spec: (["walk", number] | ["face", number] | ["inspect", number])[] = [
    ["walk", -0.35], ["face", FACE_LINE], ["inspect", 4.5], ["face", EAST],
    ["walk", 2.0], ["face", FACE_LINE], ["inspect", 3.6], ["face", EAST],
    ["walk", 4.3], ["face", WEST],
    ["walk", -4.3], ["face", EAST],
  ];
  const phases: (Phase & { start: number })[] = [];
  let x = -4.3;
  let heading = EAST;
  let t = 0;
  for (const [kind, value] of spec) {
    let phase: Phase;
    if (kind === "walk") {
      phase = { kind, from: x, to: value, heading, dur: Math.abs(value - x) / SPEED + RAMP };
      x = value;
    } else if (kind === "face") {
      phase = { kind: "turn", from: heading, to: value,
        dur: Math.abs(value - heading) / TURN_RATE + 0.25 };
      heading = value;
    } else {
      phase = { kind, dur: value };
    }
    phases.push({ ...phase, start: t });
    t += phase.dur;
  }
  return { phases, total: t };
}

const smooth = (t: number) => {
  const c = Math.min(1, Math.max(0, t));
  return c * c * (3 - 2 * c);
};

/**
 * Distance covered `t` seconds into a walk of `dur` seconds, accelerating
 * over the first RAMP seconds and braking over the last — so the speed, and
 * with it the stride, is continuous from standing to walking and back.
 */
function walked(t: number, dur: number): { dist: number; speed: number } {
  const r = RAMP;
  if (t < r) return { dist: (SPEED * t * t) / (2 * r), speed: (SPEED * t) / r };
  const cruise = dur - r;
  if (t < cruise) return { dist: SPEED * (t - r / 2), speed: SPEED };
  const left = dur - t;
  const total = SPEED * (dur - r);
  return { dist: total - (SPEED * left * left) / (2 * r), speed: (SPEED * left) / r };
}

type Joints = Record<
  "thighL" | "thighR" | "shinL" | "shinR" | "armL" | "armR" | "foreL" | "foreR" | "head",
  THREE.Object3D | undefined
>;

/** Everything the walk writes each frame, found once in the loaded asset. */
type Rig = {
  scene: THREE.Object3D;
  j: Joints;
  hand?: THREE.Object3D;
  torch?: THREE.Object3D;
  tip?: THREE.Object3D;
  aim?: THREE.Object3D;
  route: ReturnType<typeof buildRoute>;
  dog: DogRig;
  // Gait phase, accumulated across walks so a new walk starts where the last
  // step ended instead of snapping the legs back to phase zero.
  gait: { phase: number; lastDist: number; lastIndex: number };
  a: THREE.Vector3;
  b: THREE.Vector3;
};

function makeRig(scene: THREE.Object3D, dogScene: THREE.Object3D): Rig {
  const find = (name: string) => scene.getObjectByName(name);
  return {
    scene,
    j: {
      thighL: find("WalkerThighL"), thighR: find("WalkerThighR"),
      shinL: find("WalkerShinL"), shinR: find("WalkerShinR"),
      armL: find("WalkerArmL"), armR: find("WalkerArmR"),
      foreL: find("WalkerForearmL"), foreR: find("WalkerForearmR"),
      head: find("WalkerHead"),
    },
    hand: find("WalkerLeashHand"),
    torch: find("WalkerTorch"),
    tip: find("WalkerTorchTip"),
    aim: find("WalkerTorchAim"),
    route: buildRoute(),
    dog: makeDogRig(dogScene),
    gait: { phase: 0, lastDist: 0, lastIndex: -1 },
    a: new THREE.Vector3(),
    b: new THREE.Vector3(),
  };
}

/** One frame of the patrol: place him, pose every joint, aim the torch. */
function drive(
  rig: Rig, g: THREE.Object3D, dogRoot: THREE.Object3D, leash: THREE.BufferGeometry,
  lights: (THREE.SpotLight | null)[], torchLight: THREE.SpotLight | null,
  clock: number, dt: number, animate: boolean,
) {
  const { j, route } = rig;
  // Reduced motion: he stands at the robot, torch on the belt, and stays.
  const t = animate ? clock % route.total : route.phases[2].start + 1.5;

  let index = route.phases.findIndex((p) => t < p.start + p.dur);
  if (index < 0) index = route.phases.length - 1;
  const p = route.phases[index];
  const local = t - p.start;

  let x = -4.3;
  let heading = EAST;
  let stride = 0;        // 0..1: how much of a full walking stride
  let raise = 0;         // 0..1: torch raised onto the belt
  let dist = 0;

  // Standing still: where the last walk ended.
  for (let i = index - 1; i >= 0; i -= 1) {
    const q = route.phases[i];
    if (q.kind === "walk") { x = q.to; break; }
  }

  if (p.kind === "walk") {
    const w = walked(local, p.dur);
    dist = w.dist;
    x = p.from + Math.sign(p.to - p.from) * w.dist;
    heading = p.heading;
    stride = w.speed / SPEED;
  } else if (p.kind === "turn") {
    const u = local / p.dur;
    heading = p.from + (p.to - p.from) * smooth(u);
    // Small steps while turning on the spot.
    stride = 0.3 * Math.sin(Math.PI * Math.min(1, u));
    dist = local * 0.55;
  } else {
    heading = FACE_LINE;
    raise = smooth(local / RAISE) * smooth((p.dur - local) / RAISE);
  }

  // Advance the gait by distance, so the feet keep pace with the ground.
  const G = rig.gait;
  if (index !== G.lastIndex) { G.lastIndex = index; G.lastDist = dist; }
  G.phase += ((dist - G.lastDist) / STRIDE) * Math.PI * 2;
  G.lastDist = dist;
  const phi = G.phase;
  const s = Math.sin(phi);

  g.position.set(x, stride * 0.012 * Math.cos(2 * phi), PATH_Z);
  g.rotation.set(0, heading, stride * 0.022 * s);

  // Legs: thighs swing opposite; each knee flexes early in its swing.
  const knee = (ph: number) => 0.05 + 0.6 * Math.max(0, Math.cos(ph + 0.5)) ** 2;
  if (j.thighL) j.thighL.rotation.x = -0.38 * s * stride;
  if (j.thighR) j.thighR.rotation.x = 0.38 * s * stride;
  if (j.shinL) j.shinL.rotation.x = knee(phi) * stride;
  if (j.shinR) j.shinR.rotation.x = knee(phi + Math.PI) * stride;

  // Arms swing against the legs. The right hand holds the lead: forward and
  // a little out towards the dog, with only a small swing.
  const sweep = animate ? Math.sin(clock * 0.7) : 0;
  // In the dark the left hand carries the torch: raised ahead of him, and
  // higher still onto the belt when he stops to check it.
  const dark = 1 - hallLevel.value;
  if (j.armL) {
    const free = 0.3 * s * stride;
    // Rest aim is already angled ahead; walking adds a little, so the pool
    // lands about a metre in front of him, and checking adds enough to put
    // it on the belt 1.7 m away.
    const held = -0.1 - 0.06 * s * stride - 0.35 * raise;
    j.armL.rotation.x = free + (held - free) * dark;
    j.armL.rotation.z = 0.12 * sweep * raise * dark;
  }
  if (j.foreL) {
    const free = -0.12 - 0.14 * Math.max(0, -s) * stride;
    j.foreL.rotation.x = free + (-0.15 - 0.2 * raise - free) * dark;
  }
  if (j.armR) {
    j.armR.rotation.x = -0.28 - 0.06 * s * stride;
    j.armR.rotation.z = -0.16;
  }
  if (j.foreR) j.foreR.rotation.x = -0.55;

  // Head: a glance at the line now and then while walking; down at the
  // belt, scanning, while inspecting.
  if (j.head) {
    const toLine = Math.max(-0.6, Math.min(0.6, FACE_LINE - heading));
    const glance = animate ? smooth((Math.sin(clock * 0.35) - 0.3) / 0.5) : 0;
    j.head.rotation.set(
      0.08 + 0.3 * raise,
      toLine * glance * (1 - raise) + 0.22 * sweep * raise,
      0,
    );
  }

  g.updateMatrixWorld();

  // The torch and its light, only in the dark.
  if (rig.torch) rig.torch.visible = dark > 0.02;
  if (rig.tip && rig.aim) {
    rig.tip.getWorldPosition(rig.a);
    rig.aim.getWorldPosition(rig.b);
    for (const light of lights) {
      if (!light) continue;
      light.position.copy(rig.a);
      light.target.position.copy(rig.b);
      light.target.updateMatrixWorld();
    }
    // Dimmed to nothing rather than hidden: a light that comes and goes
    // changes the scene's light count, which recompiles every material —
    // a visible hitch at the moment the lever is thrown.
    if (torchLight) torchLight.intensity = (6 + 5 * raise) * dark;
    setBeam(lights[1] ?? null, 0.22 * dark);
  }

  driveDog(rig.dog, dogRoot, g.position, heading, stride > 0.05 || p.kind === "turn",
    clock, dt, animate);

  // The lead, from his fist to her collar.
  if (rig.hand && rig.dog.collar) {
    rig.hand.getWorldPosition(rig.a);
    rig.dog.collar.getWorldPosition(rig.b);
    updateLeash(leash, rig.a, rig.b);
  }
}

/** The beam's volume opacity lives in a uniform drei only sets on render. */
function setBeam(light: THREE.SpotLight | null, opacity: number) {
  const volume = light?.children.find((c) => (c as THREE.Mesh).isMesh) as THREE.Mesh | undefined;
  const material = volume?.material as THREE.ShaderMaterial | undefined;
  if (material?.uniforms?.opacity) material.uniforms.opacity.value = opacity;
}

// ---------------------------------------------------------------------------
// The dog
// ---------------------------------------------------------------------------

/** Her place, in his frame: at his right (-X, he faces +Z), a little ahead. */
const HEEL = new THREE.Vector3(-0.62, 0, 0.35);
const DOG_STRIDE = 0.72;   // metres per trot cycle
const DOG_MAX = 1.8;       // m/s she can close a gap at
const DOG_GAIN = 3.2;      // how briskly she makes up distance

type DogRig = {
  root: THREE.Object3D;
  collar?: THREE.Object3D;
  head?: THREE.Object3D;
  tail?: THREE.Object3D;
  upper: Record<"FL" | "FR" | "BL" | "BR", THREE.Object3D | undefined>;
  lower: Record<"FL" | "FR" | "BL" | "BR", THREE.Object3D | undefined>;
  pos: THREE.Vector3;
  heading: number;
  phase: number;
  placed: boolean;
  target: THREE.Vector3;
};

function makeDogRig(scene: THREE.Object3D): DogRig {
  const find = (name: string) => scene.getObjectByName(name);
  const legs = ["FL", "FR", "BL", "BR"] as const;
  return {
    root: scene,
    collar: find("DogCollar"),
    head: find("DogHead"),
    tail: find("DogTail"),
    upper: Object.fromEntries(legs.map((l) => [l, find(`DogUpper${l}`)])) as DogRig["upper"],
    lower: Object.fromEntries(legs.map((l) => [l, find(`DogLower${l}`)])) as DogRig["lower"],
    pos: new THREE.Vector3(),
    heading: EAST,
    phase: 0,
    placed: false,
    target: new THREE.Vector3(),
  };
}

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

function driveDog(
  d: DogRig, root: THREE.Object3D, guard: THREE.Vector3, guardHeading: number,
  guardMoving: boolean, clock: number, dt: number, animate: boolean,
) {
  // Where she should be: his heel point, turned with him.
  const c = Math.cos(guardHeading);
  const sn = Math.sin(guardHeading);
  d.target.set(
    guard.x + HEEL.x * c + HEEL.z * sn,
    0,
    guard.z - HEEL.x * sn + HEEL.z * c,
  );
  if (!d.placed || !animate) {
    d.pos.copy(d.target);
    d.heading = guardHeading;
    d.placed = true;
  }

  // Close the gap at her own pace.
  const step = Math.min(dt, 0.1);
  const gap = d.target.clone().sub(d.pos);
  const dist = gap.length();
  const speed = Math.min(DOG_MAX, dist * DOG_GAIN);
  const move = dist > 1e-4 ? gap.multiplyScalar((speed * step) / dist) : gap.set(0, 0, 0);
  d.pos.add(move);
  const moved = move.length();

  // Face where she is going; when she has arrived, the way he faces.
  const want = speed > 0.12 ? Math.atan2(move.x, move.z) : guardHeading;
  d.heading += wrap(want - d.heading) * Math.min(1, step * 5);

  // Trot: diagonal pairs together, advanced by ground covered.
  d.phase += (moved / DOG_STRIDE) * Math.PI * 2;
  const a = Math.min(1, speed / 0.9);
  const flex = (ph: number) => Math.max(0, Math.cos(ph + 0.5)) ** 2;
  const pairs = { FL: 0, BR: 0, FR: Math.PI, BL: Math.PI } as const;
  for (const leg of ["FL", "FR", "BL", "BR"] as const) {
    const ph = d.phase + pairs[leg];
    const front = leg[0] === "F";
    const up = d.upper[leg];
    const low = d.lower[leg];
    if (up) up.rotation.x = -0.42 * Math.sin(ph) * a;
    if (low) low.rotation.x = (front ? 0.7 : -0.55) * flex(ph) * a;
  }

  root.position.set(d.pos.x, 0.012 * Math.cos(2 * d.phase) * a, d.pos.z);
  root.rotation.set(0, d.heading, 0);

  // Tail: a relaxed wag, livelier when she is stopped beside him.
  if (d.tail) {
    const wag = animate ? Math.sin(clock * (guardMoving ? 5 : 7.5)) : 0;
    d.tail.rotation.set(-0.25, wag * (guardMoving ? 0.18 : 0.32), 0);
  }
  // Head: level on the move with the odd dip to sniff; when stopped, a look
  // along the line and back.
  if (d.head) {
    const sniff = animate ? smooth((Math.sin(clock * 0.45 + 2) - 0.75) / 0.2) : 0;
    const look = animate && !guardMoving ? Math.sin(clock * 0.5) * 0.5 : 0;
    d.head.rotation.set(0.05 + 0.55 * sniff, look, 0);
  }
}

// ---------------------------------------------------------------------------
// The lead
// ---------------------------------------------------------------------------

const LEASH_LENGTH = 1.3;
const LEASH_SEGMENTS = 24;
const LEASH_SIDES = 8;
const LEASH_RADIUS = 0.01;

function makeLeashGeometry(): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const verts = (LEASH_SEGMENTS + 1) * LEASH_SIDES;
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts * 3), 3));
  g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(verts * 3), 3));
  const index: number[] = [];
  for (let i = 0; i < LEASH_SEGMENTS; i += 1) {
    for (let k = 0; k < LEASH_SIDES; k += 1) {
      const a = i * LEASH_SIDES + k;
      const b = i * LEASH_SIDES + ((k + 1) % LEASH_SIDES);
      const c2 = a + LEASH_SIDES;
      const d = b + LEASH_SIDES;
      index.push(a, c2, b, b, c2, d);
    }
  }
  g.setIndex(index);
  return g;
}

const lp = { p: new THREE.Vector3(), t: new THREE.Vector3(), n: new THREE.Vector3(),
  b: new THREE.Vector3(), mid: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0),
  ctrl: new THREE.Vector3() };

/**
 * A quadratic curve from hand to collar, its middle pulled down by however
 * much slack the lead has at this distance — taut when she is far out,
 * hanging in a loop when she is close.
 */
function updateLeash(g: THREE.BufferGeometry, a: THREE.Vector3, b: THREE.Vector3) {
  const span = a.distanceTo(b);
  const sag = Math.sqrt(Math.max(0, LEASH_LENGTH * LEASH_LENGTH - span * span)) * 0.5;
  lp.ctrl.copy(a).add(b).multiplyScalar(0.5);
  lp.ctrl.y -= sag * 2;
  const pos = g.getAttribute("position") as THREE.BufferAttribute;
  const nor = g.getAttribute("normal") as THREE.BufferAttribute;
  for (let i = 0; i <= LEASH_SEGMENTS; i += 1) {
    const t = i / LEASH_SEGMENTS;
    const u = 1 - t;
    // Point and tangent of the quadratic Bézier.
    lp.p.set(0, 0, 0).addScaledVector(a, u * u).addScaledVector(lp.ctrl, 2 * u * t)
      .addScaledVector(b, t * t);
    // Never below the floor, whatever the slack.
    lp.p.y = Math.max(lp.p.y, 0.01);
    lp.t.copy(lp.ctrl).sub(a).multiplyScalar(2 * u).addScaledVector(b.clone().sub(lp.ctrl), 2 * t)
      .normalize();
    lp.n.crossVectors(lp.t, lp.up);
    if (lp.n.lengthSq() < 1e-6) lp.n.set(1, 0, 0);
    lp.n.normalize();
    lp.b.crossVectors(lp.n, lp.t).normalize();
    for (let k = 0; k < LEASH_SIDES; k += 1) {
      const ang = (k / LEASH_SIDES) * Math.PI * 2;
      const cx = Math.cos(ang);
      const cy = Math.sin(ang);
      const nx = lp.n.x * cx + lp.b.x * cy;
      const ny = lp.n.y * cx + lp.b.y * cy;
      const nz = lp.n.z * cx + lp.b.z * cy;
      const v = i * LEASH_SIDES + k;
      nor.setXYZ(v, nx, ny, nz);
      pos.setXYZ(v, lp.p.x + nx * LEASH_RADIUS, lp.p.y + ny * LEASH_RADIUS, lp.p.z + nz * LEASH_RADIUS);
    }
  }
  pos.needsUpdate = true;
  nor.needsUpdate = true;
  g.computeBoundingSphere();
}

export function GuardPatrol({ animate = true }: { animate?: boolean }) {
  const { scene } = useGLTF(WALKER_URL, DRACO_PATH);
  const { scene: dogScene } = useGLTF(DOG_URL, DRACO_PATH);
  const root = useRef<THREE.Group>(null);
  const dogRoot = useRef<THREE.Group>(null);
  const leash = useMemo(() => makeLeashGeometry(), []);
  const rig = useRef<Rig | null>(null);
  const torch = useRef<THREE.SpotLight>(null);
  const beam = useRef<THREE.SpotLight>(null);

  useEffect(() => {
    dressGuard(scene);
    dressGuard(dogScene);
    rig.current = makeRig(scene, dogScene);
  }, [scene, dogScene]);

  // Her coat: shells over the tan and black fur (see fur.ts).
  useEffect(() => addFur(dogScene, ["Dog_Tan", "Dog_Black"]), [dogScene]);

  useEffect(() => () => leash.dispose(), [leash]);

  useFrame((state, dt) => {
    if (!rig.current || !root.current || !dogRoot.current) return;
    drive(rig.current, root.current, dogRoot.current, leash, [torch.current, beam.current],
      torch.current, state.clock.elapsedTime, dt, animate);
  });

  return (
    <>
      <group ref={root} position={[-4.3, 0, PATH_Z]} rotation={[0, EAST, 0]}>
        <primitive object={scene} />
      </group>
      <group ref={dogRoot}>
        <primitive object={dogScene} />
      </group>
      <mesh geometry={leash} material={guardMaterials().Leash} castShadow frustumCulled={false} />
      <spotLight
        ref={torch} color="#fff1d0" angle={0.34} penumbra={0.7}
        distance={8} decay={1.5} intensity={0} castShadow={false}
      />
      {/* The visible beam: volume only, and short, so it dissolves in the
          air instead of being cut off hard where it meets the floor — see
          HighBay for why. */}
      <SpotLight
        ref={beam} intensity={0} castShadow={false} volumetric color="#fff1d0"
        distance={0.9} angle={0.3} attenuation={1.1} anglePower={5}
        radiusTop={0.02} radiusBottom={0.26} opacity={0}
      />
    </>
  );
}
