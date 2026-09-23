"use client";

import { useState } from "react";
import { PerformanceMonitor } from "@react-three/drei";
import {
  Bloom, EffectComposer, N8AO, Noise, ToneMapping, Vignette,
} from "@react-three/postprocessing";
import { BlendFunction, ToneMappingMode } from "postprocessing";

/**
 * The landing scene's finishing pass: contact shadow, glow, grade.
 *
 * **Ambient occlusion is the one that matters.** Without it every object sits
 * a hair above the floor — the pod's base, the conveyor legs, the bollards —
 * because nothing darkens the crease where two surfaces meet. `ContactShadows`
 * handles the floor directly under the machine; AO handles every other corner,
 * including the ones the key light never reaches.
 *
 * **Bloom only on what is actually emitting.** The threshold sits at 2.4, so
 * only strong HDR values glow: the lens (emissive 5.5 at full beam), the beam
 * shader and the fixture lamps (6). It started at 1.05 and then 1.6, and at
 * both the white shell directly under the key light crossed it — the top of
 * the pod wore a halo, reading as a lamp rather than a machine. 2.4 clears the
 * shell's highlight and still catches everything that is actually a light.
 * Too low and every highlight bleeds, and the scene reads as a soft-focus
 * photograph of a model rather than a machine.
 *
 * **Tone mapping stays ACES.** `EffectComposer` switches the renderer's own
 * tone mapping off and expects it back as an effect. ACES is what the scene
 * rendered under before this pass existed, and every light intensity and
 * material value in it was tuned against that curve — a different operator
 * here would silently re-grade all of them.
 *
 * **Degrades rather than stutters.** `PerformanceMonitor` watches the frame
 * rate; on a sustained decline AO drops to half resolution and a cheaper
 * quality tier, and multisampling goes. Bloom and the grade stay — they cost
 * little, and they are most of what the pass is for.
 *
 * Grain is left out under reduced motion: it changes every frame, which is
 * motion however fine it is.
 */
export function LandingEffects({ animate = true }: { animate?: boolean }) {
  const [full, setFull] = useState(true);

  return (
    <>
      <PerformanceMonitor
        onDecline={() => setFull(false)}
        onFallback={() => setFull(false)}
        flipflops={3}
      />
      <EffectComposer multisampling={full ? 4 : 0} enableNormalPass={false}>
        <N8AO
          aoRadius={0.55}
          distanceFalloff={0.8}
          intensity={2.4}
          color="#1a150f"
          quality={full ? "medium" : "performance"}
          halfRes={!full}
        />
        <Bloom
          mipmapBlur
          luminanceThreshold={2.4}
          luminanceSmoothing={0.3}
          intensity={0.9}
          radius={0.62}
        />
        <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
        <Vignette offset={0.32} darkness={0.5} />
        <Noise
          opacity={animate ? 0.028 : 0}
          premultiply
          blendFunction={BlendFunction.SOFT_LIGHT}
        />
      </EffectComposer>
    </>
  );
}
