import * as THREE from "three";

/**
 * Textures for the hall, generated on a canvas rather than shipped as files.
 *
 * Two reasons. The scene has to stay self-contained — no fetches — and these
 * patterns are cheap to draw and expensive to download: a 1K concrete map is
 * larger than the entire compressed machine. Everything here is procedural,
 * tiles, and costs a few hundred KB of GPU memory.
 */

/**
 * Precast concrete panelling: vertical joints at a regular pitch, a horizontal
 * course line, and enough low-frequency mottling that the surface is not flat.
 *
 * The mottling matters more than it sounds. A perfectly even wall reads as a
 * grey card at any distance, and it is the thing that makes a CG background
 * look like a backdrop rather than a room.
 */
export function createConcreteTexture(size = 512) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#b9b3a8";
  ctx.fillRect(0, 0, size, size);

  // Low-frequency blotching, from soft radial splats. Cheaper than real noise
  // and indistinguishable once it is out of focus.
  for (let i = 0; i < 90; i += 1) {
    const x = (i * 97.3) % size;
    const y = (i * 231.7) % size;
    const r = 26 + ((i * 53) % 70);
    const shade = 176 + ((i * 37) % 26);
    const blob = ctx.createRadialGradient(x, y, 0, x, y, r);
    blob.addColorStop(0, `rgba(${shade},${shade - 4},${shade - 12},0.16)`);
    blob.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = blob;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  // Horizontal courses at fixed heights. The texture spans the wall's full
  // height with no vertical tiling, so these land at real positions.
  for (const at of [0.18, 0.52, 0.83]) {
    const y = Math.round(size * at);
    ctx.fillStyle = "rgba(96,90,82,0.40)";
    ctx.fillRect(0, y, size, 2);
    ctx.fillStyle = "rgba(226,221,212,0.22)";
    ctx.fillRect(0, y + 2, size, 1);
  }

  // Vertical panel joint at each edge of the bay: a dark recess with a lit
  // edge beside it, which is what makes a seam read as a groove rather than a
  // drawn line.
  for (const x of [0, size]) {
    ctx.fillStyle = "rgba(96,90,82,0.55)";
    ctx.fillRect(x - 2, 0, 4, size);
    ctx.fillStyle = "rgba(226,221,212,0.32)";
    ctx.fillRect(x + 2, 0, 1, size);
  }

  // Skirting: the wall darkens into the floor. Real rooms have a shadow
  // gradient there, and its absence is why a flat wall meeting a flat floor
  // reads as two planes intersecting rather than as a corner.
  const skirt = ctx.createLinearGradient(0, size, 0, size * 0.86);
  skirt.addColorStop(0, "rgba(38,33,27,0.72)");
  skirt.addColorStop(0.35, "rgba(50,44,36,0.34)");
  skirt.addColorStop(1, "rgba(60,53,44,0)");
  ctx.fillStyle = skirt;
  ctx.fillRect(0, size * 0.86, size, size * 0.14);

  // A darker band up near the ceiling too, so the room falls away above.
  const top = ctx.createLinearGradient(0, 0, 0, size * 0.16);
  top.addColorStop(0, "rgba(44,39,32,0.42)");
  top.addColorStop(1, "rgba(44,39,32,0)");
  ctx.fillStyle = top;
  ctx.fillRect(0, 0, size, size * 0.16);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

/** Yellow-and-black hazard banding, for the bollards by the door. */
export function createHazardTexture(size = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#d8b022";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#1c1b19";
  ctx.save();
  ctx.translate(size / 2, size / 2);
  ctx.rotate(-Math.PI / 4);
  for (let i = -size; i < size; i += size / 4) {
    ctx.fillRect(i, -size, size / 8, size * 2);
  }
  ctx.restore();

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
