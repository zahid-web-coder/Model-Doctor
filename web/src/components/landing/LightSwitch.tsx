"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { paintedMetal } from "@/lib/hero/finishes";
import { useHallLights } from "@/lib/hero/lights";

/**
 * The hall's lighting isolator — a big lever on the back wall beside the
 * service door — and the exit sign over the door.
 *
 * **Operate it like the real thing.** In the 360° view, grab the handle and
 * pull it up for on or down for off; it follows the pointer while held and
 * snaps to whichever end it is nearer when let go. A plain click throws it
 * the other way. The 360° panel has the same control for keyboards.
 *
 * The lever is drawn from the store, not the other way round: flip the lights
 * from the panel and the handle swings to match.
 *
 * The exit sign is always lit, as a real one must be: with the hall lights
 * off it is one of the few things still glowing, which is what keeps the dark
 * room reading as a room.
 */

const WALL_Z = -12;
const PANEL = new THREE.Vector3(-4.05, 1.35, WALL_Z);
const DOOR_X = -5.6;
const BOX = { w: 0.26, h: 0.4, d: 0.12 };
/** Handle angle at each end of its travel. Up is on. */
const UP = 0.95;
const DOWN = -0.95;
/** Pixels of vertical drag for the handle's full travel. */
const THROW_PX = 160;
const CLICK_SLOP_PX = 5;

function faceTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 400;
  const g = c.getContext("2d")!;
  g.fillStyle = "#7d858d";
  g.fillRect(0, 0, 256, 400);
  // Title plate
  g.fillStyle = "#1d2024";
  g.fillRect(24, 26, 208, 58);
  g.fillStyle = "#f1f3f5";
  g.font = "bold 30px sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText("HALL LIGHTS", 128, 56);
  // Hazard strip
  for (let i = -2; i < 12; i += 1) {
    g.fillStyle = i % 2 ? "#e0b300" : "#1a1a1a";
    g.beginPath();
    g.moveTo(i * 24, 360); g.lineTo(i * 24 + 24, 360);
    g.lineTo(i * 24 + 44, 384); g.lineTo(i * 24 + 20, 384);
    g.fill();
  }
  // ON above, OFF below — beside the lever's travel on the right.
  g.font = "bold 40px sans-serif";
  g.fillStyle = "#2fbf62";
  g.fillText("ON", 128, 150);
  g.fillStyle = "#e0452f";
  g.fillText("OFF", 128, 290);
  g.strokeStyle = "#1d2024";
  g.lineWidth = 4;
  g.beginPath(); g.moveTo(128, 185); g.lineTo(128, 255); g.stroke();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function exitTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 96;
  const g = c.getContext("2d")!;
  g.fillStyle = "#0d6b3a";
  g.fillRect(0, 0, 256, 96);
  g.fillStyle = "#e9fff2";
  g.font = "bold 52px sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText("EXIT", 150, 50);
  g.beginPath();
  g.moveTo(22, 48); g.lineTo(52, 26); g.lineTo(52, 38); g.lineTo(74, 38);
  g.lineTo(74, 58); g.lineTo(52, 58); g.lineTo(52, 70); g.closePath();
  g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

type Drag = { y: number; from: number; moved: boolean };

export function LightSwitch({ enabled }: { enabled: boolean }) {
  const lever = useRef<THREE.Group>(null);
  const lampOn = useRef<THREE.MeshStandardMaterial>(null);
  const lampOff = useRef<THREE.MeshStandardMaterial>(null);
  const drag = useRef<Drag | null>(null);
  const angle = useRef(UP);
  const getState = useThree((s) => s.get);

  const m = useMemo(() => ({
    box: paintedMetal({ color: "#7d858d", envMapIntensity: 0.8 }),
    face: new THREE.MeshStandardMaterial({ map: faceTexture(), roughness: 0.5, envMapIntensity: 0.6 }),
    steel: new THREE.MeshStandardMaterial({ color: "#b9bec4", roughness: 0.3, metalness: 0.85, envMapIntensity: 1 }),
    grip: new THREE.MeshStandardMaterial({ color: "#b3261e", roughness: 0.5, envMapIntensity: 0.5 }),
    exit: new THREE.MeshStandardMaterial({
      color: "#000000", emissive: "#ffffff", emissiveIntensity: 1.1,
      emissiveMap: exitTexture(), toneMapped: false,
    }),
    housing: new THREE.MeshStandardMaterial({ color: "#e5e3de", roughness: 0.5, envMapIntensity: 0.5 }),
  }), []);

  // Drag is tracked on the window, like the arm's: a pointer that slides off
  // the handle mid-pull must not drop the gesture.
  useEffect(() => {
    if (!enabled) return;
    const setOrbit = (on: boolean) => {
      const controls = getState().controls as { enabled: boolean } | null;
      if (controls) controls.enabled = on;
    };
    const onMove = (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const dy = d.y - e.clientY;             // up the screen is up the travel
      if (Math.abs(dy) > CLICK_SLOP_PX) d.moved = true;
      angle.current = Math.min(UP, Math.max(DOWN, d.from + (dy / THROW_PX) * (UP - DOWN)));
    };
    const onUp = () => {
      const d = drag.current;
      if (!d) return;
      drag.current = null;
      document.body.style.cursor = "";
      setOrbit(true);
      const on = useHallLights.getState().on;
      // A click throws it; a pull lands on whichever end it is nearer.
      const next = d.moved ? angle.current > 0 : !on;
      if (next !== on) useHallLights.setState({ on: next });
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      drag.current = null;
      setOrbit(true);
    };
  }, [enabled, getState]);

  useFrame((_, dt) => {
    const on = useHallLights.getState().on;
    // Not held: the handle swings to where the lights say it is.
    if (!drag.current) {
      const target = on ? UP : DOWN;
      angle.current += (target - angle.current) * Math.min(1, dt * 14);
    }
    if (lever.current) lever.current.rotation.x = -angle.current;
    if (lampOn.current) lampOn.current.emissiveIntensity = on ? 2.2 : 0.05;
    if (lampOff.current) lampOff.current.emissiveIntensity = on ? 0.05 : 2.2;
  });

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (!enabled) return;
    e.stopPropagation();
    drag.current = { y: e.clientY, from: angle.current, moved: false };
    document.body.style.cursor = "grabbing";
    const controls = getState().controls as { enabled: boolean } | null;
    if (controls) controls.enabled = false;
  };
  const hover = {
    onPointerOver: () => { if (enabled && !drag.current) document.body.style.cursor = "grab"; },
    onPointerOut: () => { if (!drag.current) document.body.style.cursor = ""; },
  };

  const pivot: [number, number, number] = [BOX.w / 2 + 0.02, 0, BOX.d / 2];

  return (
    <group>
      <group position={PANEL}>
        {/* Enclosure, with its printed face */}
        <RoundedBox args={[BOX.w, BOX.h, BOX.d]} radius={0.012} smoothness={3}
          position={[0, 0, BOX.d / 2]} material={m.box} castShadow receiveShadow />
        <mesh position={[0, 0, BOX.d + 0.0008]}>
          <planeGeometry args={[BOX.w - 0.03, BOX.h - 0.03]} />
          <primitive object={m.face} attach="material" />
        </mesh>
        {/* Status lamps: green on, red off */}
        {[{ y: 0.075, ref: lampOn, c: "#35e070" }, { y: -0.085, ref: lampOff, c: "#ff4a32" }].map((l) => (
          <mesh key={l.y} position={[-0.075, l.y, BOX.d + 0.006]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.013, 0.013, 0.012, 20]} />
            <meshStandardMaterial ref={l.ref} color="#111" emissive={l.c}
              emissiveIntensity={0.05} toneMapped={false} />
          </mesh>
        ))}
        {/* Conduit down the wall from the box */}
        <mesh position={[-0.06, -BOX.h / 2 - 0.5, 0.03]} castShadow>
          <cylinderGeometry args={[0.016, 0.016, 1.0, 12]} />
          <primitive object={m.steel} attach="material" />
        </mesh>

        {/* The lever, on the box's right flank: boss, arm and grip */}
        <group position={pivot}>
          <mesh rotation={[0, 0, Math.PI / 2]} material={m.steel} castShadow>
            <cylinderGeometry args={[0.032, 0.032, 0.04, 24]} />
          </mesh>
          <group ref={lever}>
            <mesh position={[0.012, 0, 0.09]} rotation={[Math.PI / 2, 0, 0]} material={m.steel} castShadow>
              <cylinderGeometry args={[0.012, 0.014, 0.18, 14]} />
            </mesh>
            <mesh position={[0.012, 0, 0.215]} rotation={[Math.PI / 2, 0, 0]} material={m.grip} castShadow>
              <capsuleGeometry args={[0.022, 0.08, 6, 16]} />
            </mesh>
          </group>
        </group>

        {/* The grab volume: generous, because from orbit distance the handle
            is a few pixels across. Covers the box and the lever's sweep. */}
        {enabled && (
          <mesh position={[0.06, 0, 0.14]} onPointerDown={onPointerDown} {...hover}>
            <boxGeometry args={[0.46, 0.62, 0.28]} />
            <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} />
          </mesh>
        )}
      </group>

      {/* Exit sign over the service door */}
      <group position={[DOOR_X, 2.92, WALL_Z]}>
        <RoundedBox args={[0.42, 0.17, 0.06]} radius={0.01} smoothness={3}
          position={[0, 0, 0.03]} material={m.housing} castShadow />
        <mesh position={[0, 0, 0.0605]}>
          <planeGeometry args={[0.38, 0.13]} />
          <primitive object={m.exit} attach="material" />
        </mesh>
      </group>
    </group>
  );
}
