"use client";

import { useEffect } from "react";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { DRACO_PATH } from "@/lib/hero/asset";
import { surfaceSet, type SurfaceKind } from "@/lib/hero/surfaceTextures";

/**
 * Gives the landing machine's materials a real finish.
 *
 * **Why the first version read as polystyrene.** A bright, evenly diffuse
 * white with a fine bump on it is exactly what thermocol looks like — and
 * that is what the first pass produced: the blockout's flat white, plus an
 * orange-peel normal map. What makes a white machine read as *painted metal*
 * is the opposite: a smooth, slightly off-white base under a glossy clear
 * coat, so the lights and the room reflect in it crisply. That reflection is
 * the cue; texture is not.
 *
 * So the finishes, by material:
 *
 * - **Shell** — clear-coated paint: no bump at all, the base pulled a little
 *   off pure white so the diffuse does not blow out under the key light, and a
 *   near-mirror clear coat on top that carries the highlights.
 * - **Metal, steel** — anisotropic reflection. Brushed and machined metal
 *   stretches highlights along the grain; that is a property of how it
 *   reflects, not a pattern painted on it, and a physical material draws it
 *   directly.
 * - **Trim, plates** — a faint grain, which is what rubber and matte polymer
 *   actually have.
 *
 * **Landing only, by construction.** It dresses the landing's own GLB, whose
 * materials are separate objects from the dashboard's even where names match.
 *
 * **After `Machine`, and building on it.** Mounted after it, so `Machine`'s
 * layout effect has already tuned roughness and environment response; those
 * values are carried into the replacement materials rather than overwritten.
 * Marked in `userData`, so a remount or hot reload never stacks a second pass.
 */

type Finish =
  | { type: "clearcoat"; tint: number; clearcoatRoughness: number; smudge: SurfaceKind }
  | { type: "anisotropic"; anisotropy: number; smudge: SurfaceKind }
  | { type: "grain"; normal: number };

const FINISHES: Record<string, Finish> = {
  MD_Shell: { type: "clearcoat", tint: 0.9, clearcoatRoughness: 0.06, smudge: "painted" },
  MD_Metal: { type: "anisotropic", anisotropy: 0.6, smudge: "brushed" },
  MD_Steel: { type: "anisotropic", anisotropy: 0.45, smudge: "brushed" },
  MD_DarkTrim: { type: "grain", normal: 0.06 },
  MD_Plate: { type: "grain", normal: 0.07 },
};

/** A physical copy of a standard material, carrying every value `Machine` set. */
function toPhysical(source: THREE.MeshStandardMaterial): THREE.MeshPhysicalMaterial {
  const physical = new THREE.MeshPhysicalMaterial();
  THREE.MeshStandardMaterial.prototype.copy.call(physical, source);
  // `MeshStandardMaterial.copy` resets `defines` to STANDARD only, which
  // compiles the shader without the physical branch — clear coat and
  // anisotropy would be set and silently never drawn.
  physical.defines = { STANDARD: "", PHYSICAL: "" };
  physical.name = source.name;
  return physical;
}

function dress(source: THREE.MeshStandardMaterial): THREE.MeshStandardMaterial {
  const finish = FINISHES[source.name];
  if (!finish) return source;

  if (finish.type === "grain") {
    const set = surfaceSet("grain");
    source.normalMap = set.normal;
    source.normalScale.set(finish.normal, finish.normal);
    source.roughnessMap = set.roughness;
    source.userData.surfaceDetail = true;
    source.needsUpdate = true;
    return source;
  }

  const physical = toPhysical(source);
  physical.roughnessMap = surfaceSet(finish.smudge).roughness;
  if (finish.type === "clearcoat") {
    physical.color.multiplyScalar(finish.tint);
    physical.clearcoat = 1;
    physical.clearcoatRoughness = finish.clearcoatRoughness;
  } else {
    physical.anisotropy = finish.anisotropy;
  }
  physical.userData.surfaceDetail = true;
  return physical;
}

export function SurfaceDetail({ url }: { url: string }) {
  const { scene } = useGLTF(url, DRACO_PATH) as unknown as { scene: THREE.Group };

  useEffect(() => {
    // One replacement per source material, so parts that shared a material
    // before still share one after — including the instanced rollers and
    // plates, which hold their own reference and are visited like any mesh.
    const replaced = new Map<THREE.Material, THREE.Material>();

    scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const swap = (material: THREE.Material) => {
        const standard = material as THREE.MeshStandardMaterial;
        if (!standard?.isMeshStandardMaterial || standard.userData.surfaceDetail) {
          return material;
        }
        let next = replaced.get(material);
        if (!next) {
          next = dress(standard);
          replaced.set(material, next);
        }
        return next;
      };
      mesh.material = Array.isArray(mesh.material)
        ? mesh.material.map(swap)
        : swap(mesh.material);
    });
  }, [scene]);

  return null;
}
