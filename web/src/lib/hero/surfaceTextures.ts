import * as THREE from "three";

/**
 * Surface detail for the machine, generated in the browser rather than
 * shipped in the GLB.
 *
 * **Why here and not in the asset.** The same reason as `hallTextures.ts`: a
 * texture map outweighs the whole compressed machine. Baking these into the
 * landing GLB took it from 175 KB to 612 KB for exactly this look; generated
 * here, all six maps take about 40 ms once (measured cold, first call) and
 * nothing on the wire. The asset carries only the UVs they need
 * (`md_realism.py`), one UV unit per texture repeat at a real-world size.
 *
 * **What they add.** Every material in the blockout is one flat value. Real
 * paint, brushed steel and rubber vary across the surface, and that variation
 * is what breaks a reflection up into something that reads as a material
 * rather than a mirror of one number:
 *
 * - painted — a faint orange peel, and gentle smudging in the roughness
 * - brushed — long streaks along one axis, in both normal and roughness
 * - grain   — fine, even texture for polymer, rubber and the plates
 *
 * **Tiling by construction.** The noise is built on a periodic lattice, so
 * every map wraps with no seam and needs no blending pass.
 */

export type SurfaceKind = "painted" | "brushed" | "grain";

export type SurfaceSet = {
  normal: THREE.DataTexture;
  roughness: THREE.DataTexture;
};

const SIZE = 256;

/** Deterministic PRNG, so the page looks the same on every load. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const smooth = (t: number) => t * t * (3 - 2 * t);

/**
 * Tileable value noise: a random lattice of `cx` × `cy` cells, wrapped, and
 * smoothly interpolated. Unequal cell counts make it anisotropic, which is
 * how the brushed streaks are drawn.
 */
function lattice(cx: number, cy: number, rand: () => number): Float32Array {
  const grid = new Float32Array(cx * cy);
  for (let i = 0; i < grid.length; i += 1) grid[i] = rand();
  const out = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y += 1) {
    const gy = (y / SIZE) * cy;
    const y0 = Math.floor(gy);
    const ty = smooth(gy - y0);
    const r0 = (y0 % cy) * cx;
    const r1 = ((y0 + 1) % cy) * cx;
    for (let x = 0; x < SIZE; x += 1) {
      const gx = (x / SIZE) * cx;
      const x0 = Math.floor(gx);
      const tx = smooth(gx - x0);
      const c0 = x0 % cx;
      const c1 = (x0 + 1) % cx;
      const top = grid[r0 + c0] + (grid[r0 + c1] - grid[r0 + c0]) * tx;
      const bottom = grid[r1 + c0] + (grid[r1 + c1] - grid[r1 + c0]) * tx;
      out[y * SIZE + x] = top + (bottom - top) * ty;
    }
  }
  return out;
}

/** Octaves of lattice noise, summed and normalised to 0..1. */
function fbm(octaves: readonly [number, number, number][], seed: number): Float32Array {
  const rand = mulberry32(seed);
  const sum = new Float32Array(SIZE * SIZE);
  let weightTotal = 0;
  for (const [cx, cy, weight] of octaves) {
    const layer = lattice(cx, cy, rand);
    for (let i = 0; i < sum.length; i += 1) sum[i] += layer[i] * weight;
    weightTotal += weight;
  }
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < sum.length; i += 1) {
    sum[i] /= weightTotal;
    if (sum[i] < min) min = sum[i];
    if (sum[i] > max) max = sum[i];
  }
  const span = max - min || 1;
  for (let i = 0; i < sum.length; i += 1) sum[i] = (sum[i] - min) / span;
  return sum;
}

/** Cell counts per octave: [cells across U, cells across V, weight]. */
const HEIGHT: Record<SurfaceKind, readonly [number, number, number][]> = {
  // Orange peel: a fine undulation — millimetres, not centimetres. At 8 cells
  // per 35 cm repeat the first version put ~4 cm lumps on the shell.
  painted: [[24, 24, 1], [48, 48, 0.5], [96, 96, 0.25]],
  // Streaks: few cells along U, many across V.
  brushed: [[2, 64, 1], [4, 128, 0.6], [8, 256, 0.35]],
  // Grain: even and fine, but short of the top octave — at viewing distance
  // that one reads as speckle, not texture.
  grain: [[16, 16, 0.6], [32, 32, 1], [64, 64, 0.7]],
};

const SMUDGE: Record<SurfaceKind, readonly [number, number, number][]> = {
  painted: [[2, 2, 1], [4, 4, 0.5], [8, 8, 0.25]],
  brushed: [[2, 32, 1], [4, 64, 0.5]],
  grain: [[3, 3, 1], [6, 6, 0.5], [12, 12, 0.25]],
};

/**
 * How far roughness varies. The map is a multiplier on the material's own
 * roughness and cannot exceed 1, so it spans `1 − spread .. 1`: the surface
 * averages a few percent glossier than the flat value, never rougher.
 */
const SPREAD: Record<SurfaceKind, number> = {
  painted: 0.08,
  brushed: 0.16,
  grain: 0.14,
};

function toTexture(rgba: Uint8Array): THREE.DataTexture {
  const texture = new THREE.DataTexture(rgba, SIZE, SIZE, THREE.RGBAFormat);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 4;
  // Data, not colour: no sRGB decode on either map.
  texture.colorSpace = THREE.NoColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function normalMap(height: Float32Array, strength: number): THREE.DataTexture {
  const rgba = new Uint8Array(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y += 1) {
    const up = ((y - 1 + SIZE) % SIZE) * SIZE;
    const down = ((y + 1) % SIZE) * SIZE;
    const row = y * SIZE;
    for (let x = 0; x < SIZE; x += 1) {
      const left = (x - 1 + SIZE) % SIZE;
      const right = (x + 1) % SIZE;
      const dx = (height[row + right] - height[row + left]) * strength;
      const dy = (height[down + x] - height[up + x]) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (row + x) * 4;
      rgba[i] = ((-dx / len) * 0.5 + 0.5) * 255;
      rgba[i + 1] = ((-dy / len) * 0.5 + 0.5) * 255;
      rgba[i + 2] = ((1 / len) * 0.5 + 0.5) * 255;
      rgba[i + 3] = 255;
    }
  }
  return toTexture(rgba);
}

/** Roughness multiplier in G (three's roughnessMap channel); R, B, A at 1. */
function roughnessMap(smudge: Float32Array, spread: number): THREE.DataTexture {
  const rgba = new Uint8Array(SIZE * SIZE * 4);
  for (let i = 0; i < smudge.length; i += 1) {
    const v = 1 - spread * smudge[i];
    rgba[i * 4] = 255;
    rgba[i * 4 + 1] = v * 255;
    rgba[i * 4 + 2] = 255;
    rgba[i * 4 + 3] = 255;
  }
  return toTexture(rgba);
}

const cache = new Map<SurfaceKind, SurfaceSet>();

/**
 * One map pair per kind, built on first use and kept for the page's life:
 * three kinds × two maps at 256² is under 1.6 MB of GPU memory, and building
 * them again on every mount would be the only cost this module has.
 */
export function surfaceSet(kind: SurfaceKind): SurfaceSet {
  const hit = cache.get(kind);
  if (hit) return hit;
  const seed = { painted: 11, brushed: 23, grain: 37 }[kind];
  const set = {
    normal: normalMap(fbm(HEIGHT[kind], seed), 6),
    roughness: roughnessMap(fbm(SMUDGE[kind], seed + 101), SPREAD[kind]),
  };
  cache.set(kind, set);
  return set;
}
