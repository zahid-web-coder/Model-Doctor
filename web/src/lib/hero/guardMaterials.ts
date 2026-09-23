import * as THREE from "three";
import { surfaceSet } from "./surfaceTextures";

/**
 * The guards' materials, shared by the door guard, the patrolling one and
 * his dog.
 *
 * Their GLBs (`md_guard.py` in the hero3d project) carry placeholder
 * materials named `Guard_*`; `dressGuard` swaps each for the one here, so
 * the look is tuned in code next to the rest of the scene — the same split as
 * the machine. Built once and kept for the page's life, like `surfaceSet`:
 * two figures share them, and neither may dispose what the other still draws.
 */

let cache: Record<string, THREE.Material> | null = null;

/**
 * Bumped whenever a guard or dog GLB is rebuilt. The files keep their names,
 * so without it a browser that already holds the old model — an open tab
 * under hot reload, or a cached copy on the live site — keeps drawing it
 * against code written for the new one (a torch it should not have, a lead
 * with nothing to hang from).
 */
export const GUARD_MODELS_VERSION = "2026-09-23d";

export function guardMaterials(): Record<string, THREE.Material> {
  if (cache) return cache;
  const weave = surfaceSet("grain").normal;
  // Cloth has sheen: a soft brightening towards grazing angles as the light
  // catches the fibres. A plain matte material has none, and uniform matte
  // colour is most of why a clothed figure reads as moulded plastic. The
  // grain map adds the weave, so a sleeve is not one smooth gradient.
  const cloth = (color: string, roughness: number, envMapIntensity: number) =>
    new THREE.MeshPhysicalMaterial({
      color, roughness, envMapIntensity,
      // Kept low: under the key light a strong, pale sheen turned the navy
      // uniform grey, and a deep weave read as fleece.
      sheen: 0.4, sheenRoughness: 0.8,
      sheenColor: new THREE.Color(color).lerp(new THREE.Color("#ffffff"), 0.18),
      normalMap: weave, normalScale: new THREE.Vector2(0.14, 0.14),
    });
  const make = (color: string, roughness: number, extra: object = {}) =>
    new THREE.MeshStandardMaterial({ color, roughness, envMapIntensity: 0.35, ...extra });

  cache = {
    // Skin carries a faint warm sheen for the same reason: flat skin reads as
    // a mannequin's.
    Guard_Skin: new THREE.MeshPhysicalMaterial({
      color: "#a37e5c", roughness: 0.58, envMapIntensity: 0.4,
      sheen: 0.3, sheenRoughness: 0.5, sheenColor: new THREE.Color("#ffd9c4"),
    }),
    Guard_Lip: make("#8d604a", 0.45, { envMapIntensity: 0.4 }),
    Guard_Shirt: cloth("#2b3446", 0.82, 0.3),
    Guard_Trouser: cloth("#232a3a", 0.86, 0.25),
    Guard_Cap: cloth("#1f2636", 0.7, 0.35),
    // A muted duty vest, not hi-vis. Neon green here would be the brightest
    // thing in frame after the beam and pull the eye straight off the
    // scanner — the guards are meant to be secondary.
    Guard_Vest: cloth("#3f5670", 0.66, 0.45),
    Guard_Reflect: make("#9aa4ad", 0.34, { metalness: 0.12, envMapIntensity: 0.6 }),
    // Polished duty leather: the highlight on a toe cap is what says "boot".
    Guard_Leather: make("#15181f", 0.36, { envMapIntensity: 0.7 }),
    Guard_Rubber: make("#0e1014", 0.8),
    Guard_Tablet: make("#2a2d33", 0.42, { metalness: 0.2, envMapIntensity: 0.6 }),
    Guard_Badge: make("#c9ced6", 0.3, { metalness: 0.85, envMapIntensity: 0.9 }),
    Guard_Hair: make("#1b1612", 0.75),
    Guard_Eye: make("#1a1410", 0.15, { envMapIntensity: 0.8 }),
    Guard_Sclera: make("#cfc6ba", 0.3),

    // The dog. Fur is cloth-like — rough, with sheen along the silhouette —
    // and the weave map breaks it up the same way.
    Dog_Tan: cloth("#a8743f", 0.85, 0.35),
    Dog_Black: cloth("#1b1714", 0.8, 0.35),
    Dog_Nose: make("#0c0b0a", 0.25, { envMapIntensity: 0.9 }),
    Dog_Eye: make("#2a1a0c", 0.1, { envMapIntensity: 1 }),
    Dog_Collar: make("#8e1f1a", 0.55),
    Dog_Tag: make("#d4a53c", 0.25, { metalness: 0.9, envMapIntensity: 1.1 }),
    Dog_Tongue: make("#b5524f", 0.35, { envMapIntensity: 0.5 }),
    // The torch's lens, lit (the patrolling guard carries it in the dark).
    Guard_Lens: new THREE.MeshStandardMaterial({
      color: "#000000", emissive: "#fff1d0", emissiveIntensity: 1.6, toneMapped: false,
    }),
    // The lead: red nylon to match her collar. Black was correct for a duty
    // lead and invisible against his trousers and the floor.
    Leash: make("#a3241c", 0.55, { envMapIntensity: 0.5 }),
  };
  // Keep the GLB's names on the replacements, so later passes (the dog's fur)
  // can still find a material by what it is.
  for (const [name, material] of Object.entries(cache)) material.name = name;
  return cache;
}

/** Swap a guard GLB's placeholder materials for the real ones, by name. */
export function dressGuard(scene: THREE.Object3D, extra: Record<string, THREE.Material> = {}) {
  const materials = { ...guardMaterials(), ...extra };
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const name = (mesh.material as THREE.Material).name;
    const replacement = materials[name];
    if (replacement) mesh.material = replacement;
    mesh.castShadow = name !== "Guard_Screen" && name !== "Guard_Lens";
  });
}
