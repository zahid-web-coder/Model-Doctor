"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import * as THREE from "three";
import { EDGE_RADIUS, brushedMetal, grainedPolymer, paintedMetal } from "@/lib/hero/finishes";
import { useHero } from "@/lib/hero/store";
import { useMonitorHover } from "@/lib/hero/scan";
import { scroll, beatProgress, STATION_X, clamp01, ease, DIAGNOSIS } from "@/lib/hero/landing";

/**
 * The verification station: a portal the belt runs through, a measuring bar
 * that sweeps a plate as it passes, and a monitor reporting the result.
 *
 * **Built here, not in Blender.** It is a frame — uprights, a crossbeam, a bar
 * and a screen — which is boxes and one cylinder. Modelling it would have
 * meant a second GLB to download, a second export to keep in step, and a
 * change to an asset whose 30 kinematic assertions currently pass.
 *
 * The status lamp is green when a batch verifies, but it is deliberately not
 * the message: a lamp says we grade parts, and the product diagnoses models.
 * What the screen reports is that the finding held on a second, independent
 * split — the honest meaning of verification here, and the stronger claim. The
 * lamp is a status cue next to that, and it is paired with a legible word so
 * it does not depend on colour alone.
 */

const HEIGHT = 1.62;
const WIDTH = 0.52;   // half-span across the belt

/**
 * The monitor's picture, drawn on a canvas.
 *
 * Real content rather than a blue rectangle: at the distance the camera ends
 * up, a flat emissive panel reads as a light box, and what makes it read as a
 * screen is a header bar, rows, and something that looks like a number.
 */
function createScreenTexture(size = 512) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = Math.round(size * 0.62);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const W = canvas.width;
  const H = canvas.height;

  ctx.fillStyle = "#0e1622";
  ctx.fillRect(0, 0, W, H);

  // Header
  ctx.fillStyle = "#162232";
  ctx.fillRect(0, 0, W, H * 0.16);
  ctx.fillStyle = "#7fd6a0";
  ctx.font = `600 ${Math.round(H * 0.085)}px ui-monospace, monospace`;
  ctx.fillText("VERIFIED", W * 0.05, H * 0.115);
  ctx.fillStyle = "#6f8aa8";
  ctx.font = `${Math.round(H * 0.07)}px ui-monospace, monospace`;
  ctx.fillText("REPLICATION CHECK", W * 0.36, H * 0.112);

  // Rows, from the same measured figures the diagnosis beat shows.
  const rows = DIAGNOSIS.replication.runs;
  ctx.font = `${Math.round(H * 0.078)}px ui-monospace, monospace`;
  rows.forEach((r, i) => {
    const y = H * (0.36 + i * 0.19);
    ctx.fillStyle = "#8fa6bd";
    ctx.fillText(r.label, W * 0.05, y);
    ctx.fillStyle = "#f0d78a";
    ctx.textAlign = "right";
    ctx.fillText(`${r.lift}x`, W * 0.72, y);
    ctx.fillStyle = "#7fd6a0";
    ctx.fillText(r.p, W * 0.95, y);
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.07)";
    ctx.fillRect(W * 0.05, y + H * 0.035, W * 0.9, 1);
  });

  ctx.fillStyle = "#5d7590";
  ctx.font = `${Math.round(H * 0.062)}px ui-monospace, monospace`;
  ctx.fillText(DIAGNOSIS.replication.factor, W * 0.05, H * 0.9);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export function VerificationStation() {
  const bar = useRef<THREE.Mesh>(null);
  const barMaterial = useRef<THREE.MeshStandardMaterial>(null);
  const screenMaterial = useRef<THREE.MeshStandardMaterial>(null);
  const lampMaterial = useRef<THREE.MeshStandardMaterial>(null);
  const amberMaterial = useRef<THREE.MeshStandardMaterial>(null);
  const lampLight = useRef<THREE.PointLight>(null);

  // Twice the old 512: the camera ends this story pushed into this screen,
  // and at 512 its figures softened into blur on the way in.
  const screen = useMemo(() => createScreenTexture(1024), []);
  // In the 360° view the monitor is readable on hover: at any orbit distance
  // its figures are a few pixels tall, so the page shows them full size.
  const freeOrbit = useHero((s) => s.freeOrbit);

  // One material for every structural part. They are the same painted steel,
  // and a shared instance is one fewer program for the renderer to swap.
  // Clear-coated paint and brushed dark steel — the same finishes as the
  // machine and the cabinets (`lib/hero/finishes.ts`), same colours as before.
  const steel = useMemo(() => paintedMetal({ color: "#d9d4c8", envMapIntensity: 1.1 }), []);
  const darkSteel = useMemo(() => brushedMetal({ color: "#2a2a2c", envMapIntensity: 1.2 }), []);
  // Monitor parts. A bezel is matte black plastic, not metal; the glass is a
  // near-invisible layer whose only job is to reflect the room across the
  // picture — which is what makes a display read as a screen, not a sticker.
  const monitor = useMemo(() => ({
    bezel: grainedPolymer({ color: "#141518", envMapIntensity: 0.6 }),
    // Tinted dark, not white: a white film at any opacity lays a milky veil
    // over the picture — the first version washed the navy screen to grey.
    // Real cover glass deepens the black slightly and adds the reflections.
    glass: new THREE.MeshPhysicalMaterial({
      color: "#070b10", roughness: 0.02, clearcoat: 1, clearcoatRoughness: 0.02,
      transparent: true, opacity: 0.16, envMapIntensity: 1.0, depthWrite: false,
    }),
    led: new THREE.MeshStandardMaterial({
      color: "#113a1d", emissive: "#3dff7a", emissiveIntensity: 2.6, toneMapped: false,
    }),
  }), []);

  useFrame(() => {
    const k = beatProgress(scroll.t, "verification");
    const sweep = ease(clamp01(k));

    if (bar.current) {
      bar.current.position.y = HEIGHT - 0.22 - sweep * 0.52;
      bar.current.visible = k > 0.02;
    }
    if (barMaterial.current) {
      // Brightest mid-sweep, so the measurement has a moment rather than a
      // steady state.
      barMaterial.current.emissiveIntensity = 2.2 * Math.sin(sweep * Math.PI);
    }
    if (screenMaterial.current) {
      // The screen comes up as the sweep completes and stays lit into the
      // hand-off, because the camera is about to fly into it.
      // Peaks near 1, not 2.4. The landing's post pass tone-maps the whole
      // frame, `toneMapped: false` notwithstanding, and ACES pushes bright
      // colours towards white: at 2.4 the green and gold figures came out
      // grey. Below ~1 they keep their colour.
      screenMaterial.current.emissiveIntensity = 0.45 + 0.6 * ease(clamp01((k - 0.5) / 0.5));
    }
    // Green only once the check has actually passed — a lamp that is green
    // before the measurement means nothing. Amber runs while it is working
    // and drops away as the green comes up, so the tower reads as a state
    // change rather than as two lights that happen to be on.
    const pass = ease(clamp01((k - 0.62) / 0.3));
    const working = ease(clamp01(k / 0.25)) * (1 - pass);
    if (lampMaterial.current) lampMaterial.current.emissiveIntensity = 3.2 * pass;
    if (amberMaterial.current) amberMaterial.current.emissiveIntensity = 0.25 + 2.4 * working;
    if (lampLight.current) lampLight.current.intensity = 2.6 * pass;
  });

  return (
    <group position={[STATION_X, 0, 0]}>
      {/* Uprights, either side of the belt. */}
      {[-WIDTH, WIDTH].map((z) => (
        <RoundedBox
          key={z} args={[0.11, HEIGHT, 0.13]} radius={EDGE_RADIUS} smoothness={3}
          position={[0, HEIGHT / 2, z]} material={steel} castShadow receiveShadow
        />
      ))}
      {[-WIDTH, WIDTH].map((z) => (
        <mesh key={`foot${z}`} position={[0, 0.018, z]} material={darkSteel} receiveShadow>
          <boxGeometry args={[0.3, 0.036, 0.3]} />
        </mesh>
      ))}

      <RoundedBox
        args={[0.14, 0.13, WIDTH * 2 + 0.13]} radius={EDGE_RADIUS} smoothness={3}
        position={[0, HEIGHT, 0]} material={steel} castShadow
      />
      <RoundedBox
        args={[0.2, 0.14, WIDTH * 1.5]} radius={EDGE_RADIUS} smoothness={3}
        position={[0, HEIGHT - 0.12, 0]} material={darkSteel} castShadow
      />

      {/* The measuring bar. Emissive rather than lit, so it reads as an
          instrument rather than a painted rod. */}
      <mesh ref={bar} position={[0, HEIGHT - 0.22, 0]}>
        <boxGeometry args={[0.035, 0.02, WIDTH * 1.7]} />
        <meshStandardMaterial
          ref={barMaterial}
          color="#8fc4ff" emissive="#3d9bff" emissiveIntensity={0}
          roughness={0.3} metalness={0.1} toneMapped={false}
        />
      </mesh>

      {/* Andon stack light on the crossbeam — amber while the bar is sweeping,
          green once the check has passed.

          A two-lamp tower rather than a single bead: a stack light is the one
          piece of factory signage everyone reads instantly, and having a lit
          amber above an unlit green means the green landing is a change of
          state rather than a light that was simply always on. Both lenses are
          solid enough to be visible unlit, so the fixture reads as a fixture
          from the first frame instead of appearing when it fires.

          Colour is never the only channel: the screen below states the result
          in words, and this agrees with it. */}
      <group position={[0, HEIGHT + 0.07, WIDTH * 0.55]}>
        <mesh position={[0, 0.01, 0]} material={darkSteel} castShadow>
          <cylinderGeometry args={[0.035, 0.042, 0.05, 12]} />
        </mesh>
        {/* Amber — the line is working. */}
        <mesh position={[0, 0.075, 0]}>
          <cylinderGeometry args={[0.036, 0.036, 0.08, 14]} />
          <meshStandardMaterial
            ref={amberMaterial}
            color="#c08a2a" emissive="#ffa726" emissiveIntensity={0.25}
            roughness={0.3} metalness={0} toneMapped={false}
          />
        </mesh>
        {/* Green — the finding replicated. */}
        <mesh position={[0, 0.158, 0]}>
          <cylinderGeometry args={[0.036, 0.036, 0.08, 14]} />
          <meshStandardMaterial
            ref={lampMaterial}
            color="#2f6b46" emissive="#38d97a" emissiveIntensity={0}
            roughness={0.3} metalness={0} toneMapped={false}
          />
        </mesh>
        <mesh position={[0, 0.205, 0]} material={darkSteel}>
          <cylinderGeometry args={[0.03, 0.038, 0.022, 12]} />
        </mesh>
        {/* A real point light, so the green throws onto the crossbeam and the
            plate under it. Without it the lamp glows but changes nothing, and
            an indicator that does not affect its surroundings reads as a
            decal. */}
        <pointLight
          ref={lampLight}
          position={[0, 0.158, 0.05]}
          color="#4fe28c"
          intensity={0}
          distance={1.6}
          decay={2}
        />
      </group>

      {/* The monitor. On a bracket off the near upright, angled to the
          approach so it is square to the lens by the time it fills the frame.
          It was a bare plane floating beside the frame before, which read as a
          second screen hanging in mid-air. */}
      <group
        position={[0, 1.16, WIDTH + 0.12]}
        rotation={[0, 0.5, 0]}
        onPointerOver={freeOrbit ? (e) => {
          e.stopPropagation();
          useMonitorHover.setState({ hover: true });
          document.body.style.cursor = "zoom-in";
        } : undefined}
        onPointerOut={freeOrbit ? () => {
          useMonitorHover.setState({ hover: false });
          document.body.style.cursor = "";
        } : undefined}
      >
        {/* Mount, back to the upright: a VESA plate on the monitor, a swivel
            knuckle, the arm, and a clamp collar where it meets the upright. A
            bare bar reads as a stick holding a picture up. */}
        <mesh position={[0, 0, -0.022]} material={darkSteel} castShadow>
          <boxGeometry args={[0.11, 0.11, 0.012]} />
        </mesh>
        <mesh position={[0, -0.02, -0.045]} material={darkSteel} castShadow>
          <cylinderGeometry args={[0.022, 0.022, 0.06, 20]} />
        </mesh>
        <RoundedBox
          args={[0.032, 0.032, 0.21]} radius={0.008} smoothness={3}
          position={[0, -0.02, -0.16]} material={darkSteel} castShadow
        />
        <RoundedBox
          args={[0.07, 0.085, 0.06]} radius={0.01} smoothness={3}
          position={[0, -0.02, -0.28]} material={darkSteel} castShadow
        />
        {/* Bezel */}
        <RoundedBox
          args={[0.52, 0.34, 0.03]} radius={EDGE_RADIUS * 0.8} smoothness={3}
          material={monitor.bezel} castShadow
        />
        {/* Power lamp, bottom right of the bezel. */}
        <mesh position={[0.235, -0.158, 0.0155]} material={monitor.led}>
          <sphereGeometry args={[0.0035, 10, 8]} />
        </mesh>
        {/* Picture, inset so the bezel actually surrounds it */}
        <mesh position={[0, 0.008, 0.017]}>
          <planeGeometry args={[0.47, 0.28]} />
          <meshStandardMaterial
            ref={screenMaterial}
            map={screen ?? undefined}
            emissiveMap={screen ?? undefined}
            color="#ffffff"
            emissive="#ffffff"
            emissiveIntensity={0.5}
            roughness={0.28}
            metalness={0}
            toneMapped={false}
          />
        </mesh>
        {/* Cover glass, a hair in front of the picture. */}
        <mesh position={[0, 0.008, 0.0185]} material={monitor.glass}>
          <planeGeometry args={[0.47, 0.28]} />
        </mesh>
      </group>
    </group>
  );
}
