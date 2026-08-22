import * as THREE from "three";

/**
 * The heatmap the beam leaves on a plate: a radial ramp through the usual
 * blue-to-white sequence, generated rather than shipped.
 *
 * Deliberately not part of the GLB. §2 keeps HeatmapOverlay as a runtime
 * component and §3 derives its opacity from the beam, so baking it into the
 * asset would put the appearance of a measurement somewhere nothing can keep
 * it in step with what the beam is doing.
 */
export function createHeatmapTexture(size = 256) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0, size / 2, size / 2, size / 2,
  );
  gradient.addColorStop(0.0, "#ffffff");
  gradient.addColorStop(0.1, "#ff2b16");
  gradient.addColorStop(0.28, "#ff9b00");
  gradient.addColorStop(0.46, "#f2e900");
  gradient.addColorStop(0.62, "#4bd44b");
  gradient.addColorStop(0.8, "#20b6e8");
  gradient.addColorStop(1.0, "#1b3fd8");

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
