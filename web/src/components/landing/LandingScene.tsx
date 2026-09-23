"use client";

import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Environment, Lightformer } from "@react-three/drei";
import { Machine } from "./Machine";
import { Hall } from "./Hall";
import { LandingCamera } from "./LandingCamera";
import { InspectionSequence } from "./InspectionSequence";
import { VerificationStation } from "./VerificationStation";
import { InfeedDevice, StorageDevice } from "./LineDevices";
import { Worker, GUARD_URL } from "./Worker";
import { GuardPatrol, WALKER_URL, DOG_URL } from "./GuardPatrol";
import { ArmDragger } from "./ArmDragger";
import { LandingEffects } from "./LandingEffects";
import { HighBay } from "./HighBay";
import { SecurityCameras } from "./SecurityCameras";
import { HallLighting, ClearedContactShadows } from "./HallLighting";
import { LightSwitch } from "./LightSwitch";
import { PlateScanner } from "./PlateScanner";
import { SurfaceDetail } from "./SurfaceDetail";
import { LANDING_MODEL_URL } from "@/lib/hero/landing";
import { DRACO_PATH } from "@/lib/hero/asset";

/**
 * The same in-engine probe the dashboard hero uses. Rendered to a cubemap
 * rather than downloaded, so there is no CDN fetch to suspend the canvas and
 * no CSP problem — and metals need a probe at all, since a mirror with nothing
 * to reflect returns nothing.
 */
function IndustrialEnvironment() {
  return (
    <Environment resolution={256} frames={1} background={false}>
      <Lightformer form="rect" intensity={0.85} color="#fff3e0"
        position={[0, 6, 1]} rotation={[-Math.PI / 2, 0, 0]} scale={[12, 8, 1]} />
      <Lightformer form="rect" intensity={0.5} color="#ffe9cc"
        position={[7, 3, 3]} rotation={[0, -Math.PI / 2.6, 0]} scale={[9, 6, 1]} />
      <Lightformer form="rect" intensity={0.3} color="#dce8ff"
        position={[-8, 2.5, 1]} rotation={[0, Math.PI / 2.4, 0]} scale={[9, 5, 1]} />
      <Lightformer form="rect" intensity={1.1} color="#ffffff"
        position={[-2, 5, -6]} rotation={[Math.PI / 3, 0, 0]} scale={[10, 0.6, 1]} />
      <Lightformer form="rect" intensity={0.6} color="#fff6ea"
        position={[3, 4.5, 6]} rotation={[-Math.PI / 3, 0, 0]} scale={[10, 0.5, 1]} />
      <Lightformer form="rect" intensity={0.2} color="#d8c8ac"
        position={[0, -3, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[14, 14, 1]} />
    </Environment>
  );
}

/**
 * The landing scene.
 *
 * A separate composition from the dashboard hero, not a second copy of it. The
 * machine, its animation logic, the beam shader and the heatmap are the same
 * modules; what differs is everything around them — a room instead of a void,
 * a scroll-driven camera instead of a fixed preset, and a verification station
 * that the dashboard has no use for.
 *
 * The GLB is the same file at the same URL, so drei's loader cache serves both
 * routes from one download.
 */
export default function LandingScene({
  paused = false,
  animate = true,
  interactive = false,
}: {
  paused?: boolean;
  animate?: boolean;
  /** Orbit mode: the pointer owns the camera and the arm, not the timeline. */
  interactive?: boolean;
}) {
  return (
    <Canvas
      frameloop={paused ? "never" : "always"}
      camera={{ position: [6.4, 3.5, 7.2], fov: 34, near: 0.1, far: 90 }}
      dpr={[1, 1.75]}
      shadows
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#8a8071"]} />
      <fogExp2 attach="fog" args={["#6f6757", 0.012]} />

      <IndustrialEnvironment />

      {/* Key, fill, rim and ambient — and the light switch's level, which
          everything that dims reads. */}
      <HallLighting />

      <Hall />
      <HighBay animate={animate} />
      <SecurityCameras animate={animate} />
      <LightSwitch enabled={interactive} />

      <Suspense fallback={null}>
        <Machine animate={animate} url={LANDING_MODEL_URL} />
        {/* After Machine, so its material tunings are in place to multiply. */}
        <SurfaceDetail url={LANDING_MODEL_URL} />
        {/* Click a plate to scan it — 360° view only, like the arm grab. */}
        <PlateScanner enabled={interactive} />
      </Suspense>

      <InfeedDevice />
      <VerificationStation />
      <StorageDevice />

      {/* The guard by the service door: life and scale in the room. Remove
          this line and the environment is unchanged. Its own boundary, so
          the machine never waits on his download. */}
      <Suspense fallback={null}>
        <Worker />
      </Suspense>
      {/* A second guard on patrol behind the line, with his dog. Removable
          the same way. */}
      <Suspense fallback={null}>
        <GuardPatrol animate={animate} />
      </Suspense>

      <ClearedContactShadows
        position={[0, 0.004, 0]} scale={11} far={2.2} resolution={1024}
        opacity={0.78} blur={1.3} color="#332c22"
      />

      {/* Only mounted in orbit mode: an invisible grab volume that swallowed
          pointer events during the story would block nothing visible and be
          impossible to diagnose. */}
      <ArmDragger enabled={interactive} />

      <LandingCamera parallax={animate} />
      <InspectionSequence animate={animate} />

      {/* Last, so it composites everything above. */}
      <LandingEffects animate={animate} />
    </Canvas>
  );
}

useGLTF.preload(LANDING_MODEL_URL, DRACO_PATH);
useGLTF.preload(GUARD_URL, DRACO_PATH);
useGLTF.preload(WALKER_URL, DRACO_PATH);
useGLTF.preload(DOG_URL, DRACO_PATH);
