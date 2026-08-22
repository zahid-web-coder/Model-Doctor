/**
 * Copies the Draco decoder out of the installed `three` into `public/draco`.
 *
 * Runs on postinstall so the decoder is always the one that matches the loader
 * in `node_modules`. Vendoring the files instead would work today and rot the
 * first time someone bumps three — the loader would move and the decoder would
 * not, and Draco failures surface as an empty canvas rather than an error.
 *
 * `public/draco` is gitignored for the same reason: it is generated, not
 * authored.
 */
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, "..", "node_modules", "three", "examples", "jsm", "libs", "draco", "gltf");
const to = join(here, "..", "public", "draco");

try {
  const files = await readdir(from);
  await mkdir(to, { recursive: true });
  await Promise.all(files.map((f) => copyFile(join(from, f), join(to, f))));
  console.log(`draco: copied ${files.length} files from three into public/draco`);
} catch (err) {
  // Do not fail the install. A missing decoder breaks the 3D hero, which is
  // one route; failing here would break every build including the dashboard's.
  console.warn(`draco: could not copy decoder (${err.message}). The 3D hero will not load.`);
}
