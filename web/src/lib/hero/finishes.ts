import * as THREE from "three";
import { surfaceSet } from "./surfaceTextures";

/**
 * Real-world finishes for the procedural parts of the landing line: the
 * infeed, the storage cabinet and the verification station.
 *
 * These are the same three finishes `SurfaceDetail` gives the machine, so the
 * whole line reads as one factory rather than a real robot standing among
 * cardboard boxes — which is how it looked once the robot alone had them.
 *
 * - `paintedMetal` — powder-coated sheet steel: a dielectric paint (so low
 *   metalness), with a glossy clear coat that carries the room's reflections.
 *   A bright, evenly diffuse surface with no coat is what reads as foam.
 * - `brushedMetal` — bare or dark-finished steel: metallic, with anisotropic
 *   highlights stretched along the grain.
 * - `grainedPolymer` — rubber, polymer and the plates: matte, with a faint
 *   grain, and nothing that shines.
 *
 * Each call returns a new material; callers memoise them, as before.
 */

type Base = { color: THREE.ColorRepresentation; envMapIntensity?: number };

export function paintedMetal({ color, envMapIntensity = 1.05 }: Base) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.38,
    metalness: 0.05,
    clearcoat: 1,
    clearcoatRoughness: 0.08,
    envMapIntensity,
    roughnessMap: surfaceSet("painted").roughness,
  });
}

export function brushedMetal({ color, envMapIntensity = 1.15 }: Base) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.36,
    metalness: 0.75,
    anisotropy: 0.45,
    envMapIntensity,
    roughnessMap: surfaceSet("brushed").roughness,
  });
}

export function grainedPolymer({ color, envMapIntensity = 0.9 }: Base) {
  const set = surfaceSet("grain");
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.1,
    envMapIntensity,
    normalMap: set.normal,
    normalScale: new THREE.Vector2(0.07, 0.07),
    roughnessMap: set.roughness,
  });
}

/**
 * Edge radius for sheet-metal parts, in metres. Nothing manufactured has a
 * perfectly sharp 90° edge; a few millimetres of radius catches a highlight
 * along every edge, and its absence is the plainest sign of an untouched CG
 * box. The machine's own parts are bevelled in Blender for the same reason.
 */
export const EDGE_RADIUS = 0.012;
