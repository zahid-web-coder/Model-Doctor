import * as THREE from "three";

/**
 * Shell fur for the dog.
 *
 * **How it works.** Each furred part is drawn again, `LAYERS` times, as one
 * `InstancedMesh`: every instance is the same surface pushed out along its
 * normals a little further (`gl_InstanceID` says how far). A procedural hash
 * gives each cell of the surface one hair of random height; a shell keeps
 * only the cells whose hair reaches it, and within a cell only a disc that
 * shrinks with height — so hairs taper to points. Seen together the shells
 * read as a coat, darker at the roots where light cannot reach, and drooping
 * slightly under their own weight.
 *
 * **Why this and not a texture.** A flat fur texture on a smooth body still
 * has a smooth, plastic silhouette; it is the edge — the outline breaking into
 * hair — that says "fur". Shells give that outline at the cost of one extra
 * draw per part and no download: the hair pattern is computed in the shader.
 *
 * Uses the UVs `md_dog.py` projects (one UV unit is 0.12 m), so `DENSITY`
 * hairs per unit is a real spacing.
 */

const LAYERS = 20;
/**
 * Hairs per UV unit (0.12 m): about 4 mm apart. The first version packed them
 * at 1.5 mm, far below a pixel at any distance the page is seen from, and the
 * coat averaged into a soft haze — the dog looked blurred, not furred. Hair
 * reads through strands you can resolve, a direction, and contrast between
 * strands, so each hair here is coarse enough to see.
 */
const DENSITY = 30;
/** Hairs per clump side: strands lean together into tufts as they rise. */
const CLUMP = 3;

/** Coat length, metres, per part: the muzzle and legs are short-coated. */
const LENGTH: Record<string, number> = {
  DogBody: 0.024,
  DogTail: 0.032,
  DogHead: 0.009,
};
const DEFAULT_LENGTH = 0.011;

function furMaterial(base: THREE.Material, length: number): THREE.Material {
  const m = base.clone() as THREE.MeshStandardMaterial;
  // The strands are the detail; the base's weave map would only add noise.
  m.normalMap = null;
  m.onBeforeCompile = (shader) => {
    shader.uniforms.furLength = { value: length };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        `#include <common>
        uniform float furLength;
        varying float vFurLayer;
        varying vec2 vFurUv;`,
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
        vFurLayer = float(gl_InstanceID + 1) / ${LAYERS.toFixed(1)};
        vFurUv = uv * ${DENSITY.toFixed(1)};
        transformed += normalize(objectNormal) * vFurLayer * furLength;
        // Combed: the coat lies back towards the tail and down, more so
        // towards the tips — a Shepherd's coat is flat, not a pile carpet.
        // (She faces +Z in her own frame.)
        transformed += normalize(vec3(0.0, -0.55, -1.0))
          * vFurLayer * vFurLayer * furLength * 1.1;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
        varying float vFurLayer;
        varying vec2 vFurUv;
        float furHash(vec2 p) {
          p = fract(p * vec2(123.34, 456.21));
          p += dot(p, p + 45.32);
          return fract(p.x * p.y);
        }`,
      )
      .replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
        vec2 cell = floor(vFurUv);
        vec2 local = fract(vFurUv) - 0.5;
        // Each hair starts somewhere in its cell, not on a grid...
        local -= (vec2(furHash(cell + 7.1), furHash(cell + 3.3)) - 0.5) * 0.5;
        // ...and leans into its clump as it rises, making tufts.
        vec2 clump = (floor(cell / ${CLUMP.toFixed(1)}) + 0.5) * ${CLUMP.toFixed(1)};
        local -= (clump - (cell + 0.5)) * 0.32 * vFurLayer;
        float height = 0.4 + 0.6 * furHash(cell);
        float t = vFurLayer / height;
        float radius = 0.5 * pow(max(0.0, 1.0 - t), 0.6);
        if (t > 1.0 || length(local) > radius) discard;
        float shade = 0.72 + 0.5 * furHash(cell + 11.7);`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
        // Dark at the roots where light cannot reach; each strand its own
        // shade; tips lighter and a touch warmer, as sun-bleached guard hairs
        // are. The contrast between neighbouring strands is what makes them
        // read as separate hairs.
        diffuseColor.rgb *= mix(0.3, 1.0, vFurLayer) * shade;
        diffuseColor.rgb = mix(diffuseColor.rgb,
          diffuseColor.rgb * 1.35 + vec3(0.035, 0.022, 0.006),
          smoothstep(0.55, 1.0, t));`,
      );
  };
  m.customProgramCacheKey = () => `fur2-${length}`;
  return m;
}

/**
 * Add fur shells to every mesh in `root` whose material is named in
 * `materials`. Returns a disposer that removes them.
 */
export function addFur(root: THREE.Object3D, materials: string[]): () => void {
  const targets: { mesh: THREE.Mesh; part: string }[] = [];
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh || (mesh as THREE.InstancedMesh).isInstancedMesh) return;
    const name = (mesh.material as THREE.Material).name;
    if (!materials.includes(name)) return;
    // The part is the named node the mesh belongs to (multi-material parts
    // load as a group of meshes).
    let part = mesh.name;
    let p: THREE.Object3D | null = mesh;
    while (p && !/^Dog[A-Z]/.test(p.name)) p = p.parent;
    if (p) part = p.name.replace(/_\d+$/, "").replace(/(Upper|Lower)(FL|FR|BL|BR)$/, "Leg");
    targets.push({ mesh, part });
  });

  const made: { shells: THREE.InstancedMesh; material: THREE.Material }[] = [];
  const restore: (() => void)[] = [];
  for (const { mesh, part } of targets) {
    const material = furMaterial(mesh.material as THREE.Material,
      LENGTH[part] ?? DEFAULT_LENGTH);
    const shells = new THREE.InstancedMesh(mesh.geometry, material, LAYERS);
    // The surface under the coat is the roots: dark, so the gaps between
    // strands read as depth rather than as more of the same colour.
    const skin = (mesh.material as THREE.MeshStandardMaterial).clone();
    skin.color.multiplyScalar(0.45);
    const original = mesh.material as THREE.Material;
    mesh.material = skin;
    restore.push(() => { mesh.material = original; skin.dispose(); });
    shells.castShadow = false;
    shells.receiveShadow = true;
    // Identity instances: the layer comes from gl_InstanceID, not a matrix.
    shells.frustumCulled = false;
    mesh.add(shells);
    made.push({ shells, material });
  }
  return () => {
    for (const undo of restore) undo();
    for (const { shells, material } of made) {
      shells.removeFromParent();
      shells.dispose();
      material.dispose();
    }
  };
}
