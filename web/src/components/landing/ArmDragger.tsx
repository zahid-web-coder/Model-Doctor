"use client";

import { useEffect, useRef, useState } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import { useHero } from "@/lib/hero/store";

/**
 * Grab the machine and move it.
 *
 * Nudge buttons were the first version and they were wrong: swinging the arm
 * through its range took a dozen clicks, and a control you have to hammer is
 * worse than no control. Here you put the pointer on the scanner and drag —
 * horizontally to swing the arm, vertically to tilt the head — which is how
 * you would move the real thing.
 *
 * **An invisible grab volume rather than the arm's own meshes.** The arm is a
 * kinematic chain of thirty-odd nodes that sweeps through its range, so
 * hit-testing the geometry means the target moves as you drag it and slips out
 * from under the pointer at the edges. A single volume covering the working
 * envelope stays put, and at this size nobody can tell the difference except
 * that it works.
 *
 * OrbitControls is disabled for the duration, or the same drag would spin the
 * camera and the arm at once.
 */

const YAW_LIMIT = 1;
const TILT_LIMIT = 0.4;
// Pixels to radians. Tuned so a drag across roughly a third of a laptop screen
// covers the arm's full sweep — far enough to feel deliberate, short enough
// that the full range is reachable without repositioning.
const YAW_PER_PIXEL = 0.0045;
const TILT_PER_PIXEL = 0.0022;

const clamp = (v: number, limit: number) => Math.max(-limit, Math.min(limit, v));

type Drag = { x: number; y: number; yaw: number; tilt: number };

export function ArmDragger({ enabled }: { enabled: boolean }) {
  const controls = useThree((s) => s.controls) as { enabled: boolean } | null;
  const drag = useRef<Drag | null>(null);
  const [hovered, setHovered] = useState(false);

  // The cursor is the only affordance the volume has — it is invisible, so
  // without this there is nothing to say the machine can be grabbed.
  useEffect(() => {
    if (!enabled) return;
    document.body.style.cursor = drag.current ? "grabbing" : hovered ? "grab" : "";
    return () => { document.body.style.cursor = ""; };
  }, [hovered, enabled]);

  // Move and release are tracked on the window, not on the mesh. A pointer
  // that leaves the volume mid-drag — which happens constantly near the limits
  // — would otherwise drop the gesture and strand the arm.
  useEffect(() => {
    if (!enabled) return;

    const onMove = (e: PointerEvent) => {
      const start = drag.current;
      if (!start) return;
      useHero.setState({
        armYaw: clamp(start.yaw + (e.clientX - start.x) * YAW_PER_PIXEL, YAW_LIMIT),
        headTilt: clamp(start.tilt + (e.clientY - start.y) * TILT_PER_PIXEL, TILT_LIMIT),
      });
    };
    const onUp = () => {
      if (!drag.current) return;
      drag.current = null;
      document.body.style.cursor = "";
      if (controls) controls.enabled = true;
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      // Leaving orbit mode mid-drag must not leave the camera disabled.
      if (controls) controls.enabled = true;
    };
  }, [enabled, controls]);

  if (!enabled) return null;

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const s = useHero.getState();
    drag.current = { x: e.clientX, y: e.clientY, yaw: s.armYaw, tilt: s.headTilt };
    document.body.style.cursor = "grabbing";
    if (controls) controls.enabled = false;
  };

  return (
    <mesh
      // Centred on the arm's working envelope: the pedestal stands at
      // x -0.86 / z 0.54 and the head reaches to about y 1.9.
      position={[-0.62, 1.15, 0.42]}
      onPointerDown={onPointerDown}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <boxGeometry args={[1.5, 1.9, 1.3]} />
      {/* Transparent, not `visible={false}`. An invisible object can be
          skipped by the raycaster entirely, which would leave this silently
          inert — the handler never fires and there is nothing to see. A fully
          transparent material is still hit-tested. `depthWrite` off so it
          cannot occlude the machine it wraps. */}
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}
