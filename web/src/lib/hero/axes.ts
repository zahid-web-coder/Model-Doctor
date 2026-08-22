import * as THREE from "three";

/**
 * Blender is Z-up; glTF and three.js are Y-up. The exporter maps
 * (x, y, z)_blender -> (x, z, -y)_gltf, which is a rotation, so it carries
 * axes across without changing any angle:
 *
 *   Blender +X  ->  three +X
 *   Blender +Y  ->  three -Z
 *   Blender +Z  ->  three +Y
 *
 * Every hinge in the asset was authored against a Blender axis, so this table
 * is the one place that conversion is written down. Deriving it per node is
 * how a sign error ends up in one joint and not the others.
 */
export const HINGE = {
  X: new THREE.Vector3(1, 0, 0),
  Y: new THREE.Vector3(0, 0, -1),
  Z: new THREE.Vector3(0, 1, 0),
} as const;

export type HingeAxis = keyof typeof HINGE;

const scratch = new THREE.Quaternion();

/**
 * Rotate a node by `angle` about a Blender-space axis, starting from the rest
 * pose the asset itself carries.
 *
 * The asset stores each joint's rest rotation (ArmBase 33.3 deg, ArmShoulder
 * 17, ArmForearm 46, ScannerHead -63). Driving a joint therefore applies a
 * DELTA rather than an absolute angle — parameter zero means "as modelled",
 * and no rest value has to be duplicated here where it could drift from the
 * asset.
 *
 * Post-multiplying applies the delta in the node's own local frame, which is
 * what a hinge is.
 */
export function driveHinge(
  node: THREE.Object3D,
  restQuaternion: THREE.Quaternion,
  blenderAxis: HingeAxis,
  angle: number,
) {
  scratch.setFromAxisAngle(HINGE[blenderAxis], angle);
  node.quaternion.copy(restQuaternion).multiply(scratch);
}

export const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
