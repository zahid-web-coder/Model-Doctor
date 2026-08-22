"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { Machine } from "./Machine";
import { CameraRig } from "./CameraRig";

/**
 * A warm industrial environment, generated in-engine.
 *
 * Drei's <Environment> renders its children into a cubemap rather than
 * downloading one, so this is a real IBL probe with no network fetch and no
 * CSP problem. That matters twice over: a preset's CDN request suspends the
 * whole Canvas with no error when it cannot complete, and metals are black
 * without a probe — a mirror with nothing to reflect returns nothing.
 *
 * The lightformers are the reflections. A large warm softbox overhead is what
 * the roller tops catch; the narrow strips are the long specular streaks that
 * run down the rails and read as a room the machine is standing in.
 */
function IndustrialEnvironment() {
  return (
    <Environment resolution={256} frames={1} background={false}>
      <Lightformer
        form="rect" intensity={0.85} color="#fff3e0"
        position={[0, 6, 1]} rotation={[-Math.PI / 2, 0, 0]} scale={[12, 8, 1]}
      />
      <Lightformer
        form="rect" intensity={0.5} color="#ffe9cc"
        position={[7, 3, 3]} rotation={[0, -Math.PI / 2.6, 0]} scale={[9, 6, 1]}
      />
      <Lightformer
        form="rect" intensity={0.3} color="#dce8ff"
        position={[-8, 2.5, 1]} rotation={[0, Math.PI / 2.4, 0]} scale={[9, 5, 1]}
      />
      <Lightformer
        form="rect" intensity={1.1} color="#ffffff"
        position={[-2, 5, -6]} rotation={[Math.PI / 3, 0, 0]} scale={[10, 0.6, 1]}
      />
      <Lightformer
        form="rect" intensity={0.6} color="#fff6ea"
        position={[3, 4.5, 6]} rotation={[-Math.PI / 3, 0, 0]} scale={[10, 0.5, 1]}
      />
      <Lightformer
        form="rect" intensity={0.2} color="#d8c8ac"
        position={[0, -3, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[14, 14, 1]}
      />
    </Environment>
  );
}

/**
 * The dashboard hero.
 *
 * This is the machine and its lighting, and deliberately not the full landing
 * scene: the hall shell, the worker and the post-processing stack belong to
 * the landing page, where the canvas is the page. Here it is one panel of a
 * dashboard that stays on screen the whole session, so it carries the subject
 * and nothing that only pays off at full-bleed.
 *
 * `paused` drives `frameloop`. Without it the render loop runs for the whole
 * session on a panel the user scrolled past minutes ago, which is the
 * performance failure that gets missed because nothing looks wrong.
 */
export default function HeroModel({
  paused = false,
  animate = true,
}: {
  paused?: boolean;
  animate?: boolean;
}) {
  return (
    <Canvas
      frameloop={paused ? "never" : "always"}
      camera={{ position: [2.3, 1.62, 1.85], fov: 32, near: 0.1, far: 60 }}
      dpr={[1, 2]}
      shadows
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#8a8071"]} />
      <fogExp2 attach="fog" args={["#6f6757", 0.012]} />

      <IndustrialEnvironment />

      {/* The key. Decay and a finite distance are what create falloff — a
          directional light has neither, so on its own it lights the far
          background exactly as brightly as the pod and the scene reads flat. */}
      <spotLight
        position={[1.6, 5.2, 2.0]} angle={0.62} penumbra={0.85}
        intensity={58} distance={22} decay={1.25} color="#fff2df"
        castShadow shadow-mapSize={[1024, 1024]} shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
      <directionalLight position={[3.6, 4.4, 3.2]} intensity={0.62} color="#fff4e6" />
      {/* Rim from behind. On a rounded white pod this is what describes the
          silhouette — more than any amount of extra geometry would. */}
      <directionalLight position={[-2.6, 3.1, -3.4]} intensity={0.62} color="#fff8f0" />
      <ambientLight intensity={0.075} color="#f6ecdd" />

      <Suspense fallback={null}>
        <Machine animate={animate} />
      </Suspense>

      <ContactShadows
        position={[0, 0.004, 0]} scale={11} far={2.2} resolution={512}
        opacity={0.78} blur={1.3} color="#332c22"
      />
      <CameraRig animate={animate} />
    </Canvas>
  );
}
