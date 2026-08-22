"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import gsap from "gsap";
import { useHero, type CameraPreset } from "@/lib/hero/store";

/**
 * Camera presets, tweened by GSAP. Values are the Blender HeroCamera converted
 * to Y-up: (x, y, z)_blender -> (x, z, -y).
 *
 * Close and long-lensed. The reference is a product shot: the pod dominates
 * the frame and the conveyor recedes behind it, which needs the camera near
 * the machine and the lens compressing the run rather than a wide view that
 * turns the conveyor into the subject.
 */
export const PRESETS: Record<CameraPreset, { position: [number, number, number]; target: [number, number, number] }> = {
  idle: { position: [2.3, 1.62, 1.85], target: [0.02, 1.24, 0.02] },
  inspect: { position: [1.42, 1.44, 1.18], target: [0.06, 1.2, 0.0] },
  wide: { position: [3.6, 2.05, 2.95], target: [-0.15, 1.05, 0.05] },
};

/**
 * GSAP owns the camera; OrbitControls owns it only when explicitly handed
 * over. §3's rule is that exactly one system writes each value — an orbiter
 * fighting a tween is precisely the jitter that rule prevents, so the handover
 * is a switch rather than something that just happens.
 *
 * In the dashboard `freeOrbit` stays false: an orbit control inside a
 * scrolling page swallows the wheel, and the hero is a piece of the page
 * rather than a viewer.
 */
export function CameraRig({ animate = true }: { animate?: boolean }) {
  const { camera } = useThree();
  const preset = useHero((s) => s.cameraPreset);
  const freeOrbit = useHero((s) => s.freeOrbit);
  const controls = useRef<React.ComponentRef<typeof OrbitControls>>(null);

  const rig = useMemo(() => ({
    x: PRESETS.idle.position[0], y: PRESETS.idle.position[1],
    z: PRESETS.idle.position[2], tx: PRESETS.idle.target[0],
    ty: PRESETS.idle.target[1], tz: PRESETS.idle.target[2],
  }), []);

  useEffect(() => {
    if (freeOrbit) return;
    const next = PRESETS[preset] ?? PRESETS.idle;
    // Reduced motion still gets the right framing, it just arrives there
    // rather than travelling: duration 0 sets the pose without a tween.
    const tween = gsap.to(rig, {
      x: next.position[0], y: next.position[1], z: next.position[2],
      tx: next.target[0], ty: next.target[1], tz: next.target[2],
      duration: animate ? 1.35 : 0, ease: "power3.inOut", overwrite: true,
    });
    return () => { tween.kill(); };
  }, [preset, freeOrbit, rig, animate]);

  useFrame(() => {
    if (freeOrbit) return;
    camera.position.set(rig.x, rig.y, rig.z);
    camera.lookAt(rig.tx, rig.ty, rig.tz);
  });

  return freeOrbit ? (
    <OrbitControls
      ref={controls}
      makeDefault
      target={[0.02, 1.2, 0.02]}
      enablePan
      enableZoom
      enableDamping
      dampingFactor={0.06}
      rotateSpeed={0.75}
      zoomSpeed={0.9}
      panSpeed={0.7}
      minDistance={0.55}
      maxDistance={10.5}
      minPolarAngle={0.12}
      maxPolarAngle={Math.PI * 0.495}
    />
  ) : null;
}
