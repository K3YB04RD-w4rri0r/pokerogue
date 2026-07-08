/**
 * Custom Node.js ESM loader for the standalone headless runner.
 *
 * Handles file types that Vite normally transforms but Node.js doesn't understand:
 * - .frag/.vert (GLSL shaders) — returns empty string (headless mode doesn't render)
 * - import.meta.env — provides defaults
 */

import { URL as NodeURL } from "node:url";

const RAW_EXTENSIONS = new Set([".frag", ".vert", ".glsl"]);

export async function resolve(specifier, context, nextResolve) {
  // Strip ?raw query params from specifiers
  if (specifier.includes("?raw")) {
    return nextResolve(specifier.split("?")[0], context);
  }
  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  const parsed = new NodeURL(url);
  const ext = parsed.pathname.match(/\.[^.]+$/)?.[0] ?? "";

  // Handle .frag, .vert, .glsl files — return empty string module
  if (RAW_EXTENSIONS.has(ext)) {
    return {
      format: "module",
      source: "export default '';",
      shortCircuit: true,
    };
  }

  return nextLoad(url, context);
}
