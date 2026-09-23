"use client";

import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import { hallLevel, stepLevel, useHallLights } from "@/lib/hero/lights";

/**
 * The room's own lights, and the one place the hall's light level is advanced.
 *
 * These were inline in `LandingScene`; they moved here when the lights got a
 * switch, so everything that dims with it sits together. Each keeps a floor
 * when off — a real hall at night is dark, not black, and a pure black frame
 * would read as the page failing to load.
 */
export function HallLighting() {
  const key = useRef<THREE.SpotLight>(null);
  const fillA = useRef<THREE.DirectionalLight>(null);
  const fillB = useRef<THREE.DirectionalLight>(null);
  const ambient = useRef<THREE.AmbientLight>(null);
  const switched = useRef({ on: true, at: -10 });

  useFrame((state, dt) => {
    const on = useHallLights.getState().on;
    const now = state.clock.elapsedTime;
    const s = switched.current;
    if (on !== s.on) { s.on = on; s.at = now; }
    const L = stepLevel(hallLevel.value, on, now - s.at, Math.min(dt, 0.1));
    hallLevel.value = L;

    if (key.current) key.current.intensity = 58 * L;
    if (fillA.current) fillA.current.intensity = 0.62 * (0.16 + 0.84 * L);
    if (fillB.current) fillB.current.intensity = 0.62 * (0.22 + 0.78 * L);
    if (ambient.current) {
      ambient.current.intensity = 0.075 + 0.1 * (1 - L);
      // Night light is cool; lamplight is warm.
      ambient.current.color.setRGB(0.96 - 0.3 * (1 - L), 0.93 - 0.2 * (1 - L), 0.87);
    }
    state.scene.environmentIntensity = 0.2 + 0.8 * L;
  });

  return (
    <>
      {/* The key. Decay and a finite distance are what create falloff — a
          directional light has neither, so on its own it lights the far wall
          exactly as brightly as the pod and the room reads flat. */}
      <spotLight
        ref={key}
        position={[1.6, 5.2, 2.0]} angle={0.62} penumbra={0.85}
        intensity={58} distance={22} decay={1.25} color="#fff2df"
        castShadow shadow-mapSize={[2048, 2048]} shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
      <directionalLight ref={fillA} position={[3.6, 4.4, 3.2]} intensity={0.62} color="#fff4e6" />
      {/* Rim from behind. On a rounded white pod this is what describes the
          silhouette — more than any amount of extra geometry would. */}
      <directionalLight ref={fillB} position={[-2.6, 3.1, -3.4]} intensity={0.62} color="#fff8f0" />
      <ambientLight ref={ambient} intensity={0.075} color="#f6ecdd" />
    </>
  );
}

/** Runs `pass` with the frame's renderer; mounted either side of a pass. */
function RendererPass({ pass }: { pass: (gl: THREE.WebGLRenderer) => void }) {
  useFrame((state) => pass(state.gl));
  return null;
}

let savedAutoClear = false;
const clearOn = (gl: THREE.WebGLRenderer) => { savedAutoClear = gl.autoClear; gl.autoClear = true; };
const clearRestore = (gl: THREE.WebGLRenderer) => { gl.autoClear = savedAutoClear; };

/**
 * `ContactShadows`, with its render target actually cleared.
 *
 * The post-processing composer turns the renderer's `autoClear` off for the
 * whole frame, and drei's contact shadows render into their own target
 * assuming it is on — so each frame's shadow was drawn *over* the last. With
 * everything anchored in place nobody saw it; the patrolling guard left a
 * trail of footprints the length of the hall. `autoClear` is switched on for
 * exactly that pass and restored after it: the two hooks sit either side of
 * the shadows, and R3F runs same-priority frame callbacks in mount order.
 */
export function ClearedContactShadows(props: React.ComponentProps<typeof ContactShadows>) {
  return (
    <>
      <RendererPass pass={clearOn} />
      <ContactShadows {...props} />
      <RendererPass pass={clearRestore} />
    </>
  );
}
