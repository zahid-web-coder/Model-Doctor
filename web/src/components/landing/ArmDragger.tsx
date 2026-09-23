"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useHero } from "@/lib/hero/store";
import { poseFromDrag, screen } from "@/lib/hero/armControl";
import { cancelScan } from "@/lib/hero/scan";

/**
 * Keeps the screen-to-arm mapping in step with wherever the camera has been
 * orbited to.
 *
 * Both axes move the scanner along a circle — yaw about the world vertical
 * through the arm's base, tilt about the head's own transverse axis. Whether
 * either reads as left/right or up/down *on screen* depends on where you are
 * standing, and this view lets you stand anywhere. Tilt is worse still: its
 * axis is carried by the head, so it also moves as the arm swings.
 *
 * So both are measured rather than assumed. The tangential direction for a
 * positive rotation is `ω × r`; dotting it with the camera's right or up
 * vector gives the sign directly.
 *
 * Without this the controls are correct from one angle and inverted from the
 * opposite one, which is indistinguishable from a plain sign bug until you
 * notice it depends on where you orbited — which is why picking a constant
 * failed twice before this.
 */
function useScreenSigns() {
  const { camera, scene } = useThree();
  const v = useMemo(() => ({
    right: new THREE.Vector3(),
    up: new THREE.Vector3(),
    basePos: new THREE.Vector3(),
    headPos: new THREE.Vector3(),
    lensPos: new THREE.Vector3(),
    axis: new THREE.Vector3(),
    radius: new THREE.Vector3(),
    tangent: new THREE.Vector3(),
    quat: new THREE.Quaternion(),
    forward: new THREE.Vector3(),
  }), []);

  useFrame(() => {
    const base = scene.getObjectByName("ArmBase");
    const head = scene.getObjectByName("ScannerHead");
    const lens = scene.getObjectByName("ScannerLens") ?? head;
    if (!base || !head || !lens) return;

    base.getWorldPosition(v.basePos);
    head.getWorldPosition(v.headPos);
    lens.getWorldPosition(v.lensPos);

    // Screen axes in world. Right is flattened to the ground plane — there is
    // no roll here and the vertical component only adds noise to a left/right
    // question. Up is the camera's own, so it stays correct when looking down.
    camera.getWorldDirection(v.forward);
    v.right.set(-v.forward.z, 0, v.forward.x).normalize();
    v.up.set(0, 1, 0).applyQuaternion(camera.quaternion).normalize();

    // Yaw: the scanner swings about the world +Y through the arm's base.
    // Tangent = omega x r.
    v.radius.subVectors(v.lensPos, v.basePos);
    v.tangent.set(0, 1, 0).cross(v.radius);
    const yawDot = v.tangent.dot(v.right);
    // Ignore the degenerate band where the tangent is nearly edge-on: the sign
    // is meaningless there and would flap frame to frame, reversing the
    // controls under the visitor's hand mid-drag.
    if (Math.abs(yawDot) > 0.05) screen.yawSign = yawDot > 0 ? 1 : -1;

    // Tilt: the head pitches about its own local (0,0,-1) — the asset's
    // Blender +Y — carried into world space by the head's current rotation.
    // Its axis therefore moves as the arm swings, which is the second reason
    // a constant could never be right.
    v.axis.set(0, 0, -1).applyQuaternion(head.getWorldQuaternion(v.quat)).normalize();
    v.radius.subVectors(v.lensPos, v.headPos);
    v.tangent.crossVectors(v.axis, v.radius);
    const tiltDot = v.tangent.dot(v.up);
    // `tiltDot > 0` means a positive headTilt lifts the beam on screen. The
    // stored sign multiplies a screen direction where +1 is *down*, so it is
    // the negation.
    if (Math.abs(tiltDot) > 0.01) screen.tiltSign = tiltDot > 0 ? -1 : 1;
  });
}

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

  // Hooks run unconditionally; the early return below is after them.
  useScreenSigns();

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
      // Taking the arm by hand ends a plate scan, which would otherwise keep
      // easing it back to rest underneath the drag.
      cancelScan();
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
