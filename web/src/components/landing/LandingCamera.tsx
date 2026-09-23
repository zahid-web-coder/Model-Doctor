"use client";

import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { CAMERA_PATH, scroll, ease, lerp, clamp01 } from "@/lib/hero/landing";
import { useHero } from "@/lib/hero/store";

/**
 * The camera, as a pure function of `scroll.t`.
 *
 * No tween of its own and no OrbitControls: GSAP owns `t`, this reads it, and
 * that is the entire ownership story. Two systems writing the camera is the
 * jitter §3 exists to prevent, and it is worst here because it only appears
 * when a scroll lands mid-tween.
 *
 * A small pointer parallax is layered on top. It is additive and bounded, so
 * it never fights the path — it moves the eye, not the shot.
 */
export function LandingCamera({ parallax = true }: { parallax?: boolean }) {
  const { camera, pointer } = useThree();
  const freeOrbit = useHero((s) => s.freeOrbit);
  const position = useRef(new THREE.Vector3(...CAMERA_PATH[0].position));
  const target = useRef(new THREE.Vector3(...CAMERA_PATH[0].target));
  // Scratch vector for the narrow-screen dolly; allocated once, not per frame.
  const offset = useRef(new THREE.Vector3());
  const drift = useRef(new THREE.Vector2());

  useFrame((_, delta) => {
    // Orbit mode hands the camera over entirely. Writing the path here as
    // well would drag the camera back to the scroll position on every frame,
    // which reads as the controls being broken rather than as a conflict.
    if (freeOrbit) return;

    const t = scroll.t;

    // Find the segment `t` falls in. Five keys, so a scan is cheaper than any
    // structure that would avoid it.
    let i = 0;
    while (i < CAMERA_PATH.length - 2 && t > CAMERA_PATH[i + 1].at) i += 1;
    const a = CAMERA_PATH[i];
    const b = CAMERA_PATH[i + 1];
    const span = b.at - a.at;
    const k = ease(span === 0 ? 0 : Math.min(1, Math.max(0, (t - a.at) / span)));

    position.current.set(
      lerp(a.position[0], b.position[0], k),
      lerp(a.position[1], b.position[1], k),
      lerp(a.position[2], b.position[2], k),
    );
    target.current.set(
      lerp(a.target[0], b.target[0], k),
      lerp(a.target[1], b.target[1], k),
      lerp(a.target[2], b.target[2], k),
    );

    // Narrow screens, opening shot only. The hero keys are framed for 16:10:
    // the line sits in the right ~55% beside the copy. On a narrower screen
    // the same shot puts the copy on top of the infeed again, so the camera
    // trucks further left and eases back, in proportion to how much narrower
    // the screen is, and hands back to the authored path by the inspection
    // beat — the close-ups are framed on the machine, not beside the copy.
    const aspect = (camera as THREE.PerspectiveCamera).aspect || 1.6;
    const narrow = clamp01((1.6 - aspect) / (1.6 - 1.25));
    const hero = 1 - ease(clamp01((t - 0.2) / 0.2));
    const shift = narrow * hero;
    if (shift > 0) {
      // Screen-left for the hero's viewing direction, on the ground plane.
      position.current.x -= 1.0 * shift;
      position.current.z += 0.9 * shift;
      target.current.x -= 1.0 * shift;
      target.current.z += 0.9 * shift;
      // And back along the view, so the far end of the line — the storage
      // cabinet — still fits inside the right edge.
      offset.current.subVectors(position.current, target.current);
      position.current.addScaledVector(offset.current, 0.28 * shift);
    }

    if (parallax) {
      // Chase the pointer rather than tracking it, so a flick of the mouse
      // does not snap the camera.
      const rate = Math.min(1, delta * 2.4);
      drift.current.x = lerp(drift.current.x, pointer.x, rate);
      drift.current.y = lerp(drift.current.y, pointer.y, rate);
      // Bounded hard. Past a few centimetres this stops reading as a living
      // camera and starts reading as a wobble.
      position.current.x += drift.current.x * 0.16;
      position.current.y += drift.current.y * 0.09;
    }

    camera.position.copy(position.current);
    camera.lookAt(target.current);
  });

  // Orbit takes over only when explicitly handed the camera, and starts from
  // wherever the path left it, so the handover has no jump.
  return freeOrbit ? (
    <OrbitControls
      makeDefault
      target={[0.0, 1.0, 0.0]}
      enablePan
      enableZoom
      enableDamping
      dampingFactor={0.06}
      rotateSpeed={0.7}
      zoomSpeed={0.8}
      minDistance={0.6}
      // Inside the room: the walls stand at 12–13m, and orbiting past them
      // puts the camera outside the building looking at the back of a plane.
      maxDistance={12}
      // Unconstrained on both axes. The earlier version stopped just above the
      // horizon so you could never end up under the floor, but that also made
      // the control feel fenced in — and "under the floor" is a view you have
      // to work to reach and can leave again immediately. Free rotation is
      // what a 360 view is for.
      minPolarAngle={0}
      maxPolarAngle={Math.PI}
    />
  ) : null;
}
