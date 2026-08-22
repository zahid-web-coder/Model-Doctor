"use client";

import { useEffect, useRef, useState } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import { useHero } from "@/lib/hero/store";
import { poseFromDrag } from "@/lib/hero/armControl";

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
      useHero.setState(poseFromDrag(start, e.clientX - start.x, e.clientY - start.y));
    };
    const onUp = () => {
      drag.current = null;
      document.body.style.cursor = "";
      // Unconditional, deliberately. Any path that leaves the controls
      // disabled — a pointerdown whose pointerup was swallowed, a release
      // outside the window — makes the camera silently stop responding, and
      // the only symptom is "sometimes dragging does nothing". Re-enabling on
      // every release costs nothing and closes all of those at once.
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
      // Wrapped tight around the scanner pod, not the whole arm.
      //
      // The first version was a 1.5 x 1.9 x 1.3 box over the arm's entire
      // working envelope, which from most angles covered a good part of the
      // viewport — so a drag meant for the camera landed on the arm instead
      // and the orbit appeared to be broken at random. A small target on the
      // one part that obviously represents the machine keeps the two gestures
      // out of each other's way: grab the pod to aim it, drag anywhere else to
      // orbit.
      position={[-0.35, 1.55, 0.5]}
      onPointerDown={onPointerDown}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <boxGeometry args={[0.78, 0.62, 0.78]} />
      {/* Transparent, not `visible={false}`. An invisible object can be
          skipped by the raycaster entirely, which would leave this silently
          inert — the handler never fires and there is nothing to see. A fully
          transparent material is still hit-tested. `depthWrite` off so it
          cannot occlude the machine it wraps. */}
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}
