"use client";

import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { useGLTF, OrbitControls, Center } from "@react-three/drei";

/**
 * The Draco decoder is served from `/draco`, not from Google's CDN.
 *
 * drei defaults to a gstatic URL, which means the hero silently fails on a
 * blocked network and, worse, pins us to a decoder version we do not control.
 * The files in `public/draco` are copied from the exact `three` build in
 * `package.json`, so decoder and loader can never drift apart.
 */
const DRACO_PATH = "/draco/";
const MODEL_URL = "/models/hero.glb";

useGLTF.preload(MODEL_URL, DRACO_PATH);

function Machine() {
  const { scene } = useGLTF(MODEL_URL, DRACO_PATH);

  // The loader caches by URL, so every mount shares one scene graph. Cloning
  // keeps a second canvas (landing + dashboard) from stealing the first one's
  // nodes.
  const model = useMemo(() => scene.clone(true), [scene]);

  return (
    <Center>
      <primitive object={model} />
    </Center>
  );
}

/**
 * Phase 0 probe. Deliberately minimal: this exists to prove the asset pipeline
 * survives Next's bundler and SSR, not to look like anything. The real scene
 * arrives in phase 2.
 *
 * Lighting is three plain lights on purpose. `<Environment preset>` fetches an
 * HDRI from a CDN and suspends the whole canvas until it arrives, which is a
 * network dependency masquerading as a lighting choice. The real scene builds
 * its probe from `Lightformer`s locally.
 */
export default function HeroModel() {
  return (
    <Canvas
      camera={{ position: [2.4, 1.4, 2.4], fov: 35 }}
      dpr={[1, 2]}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#d8d2c6"]} />
      <hemisphereLight args={["#fff6e8", "#7a7368", 1.6]} />
      <directionalLight position={[3, 5, 2]} intensity={2.4} />
      <directionalLight position={[-4, 2, -3]} intensity={0.8} />
      <Suspense fallback={null}>
        <Machine />
      </Suspense>
      <OrbitControls makeDefault />
    </Canvas>
  );
}
