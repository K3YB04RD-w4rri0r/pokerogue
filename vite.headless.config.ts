/**
 * Vite build configuration for the headless RL Node.js bundle.
 *
 * Bundles the RL runner (src/rl/cli.ts) into a single Node.js-executable file
 * at dist/rl/cli.js. All game source code from src/ and test mock infrastructure
 * from test/ are bundled together, while heavy runtime dependencies (phaser, jsdom,
 * i18next, etc.) are left as external requires resolved from node_modules.
 *
 * Usage:
 *   pnpm rl:build            # Build the headless bundle
 *   pnpm rl:run               # Run the built CLI
 *   node dist/rl/cli.js       # Run directly
 *
 * Key design decisions:
 * - Uses Vite's SSR build mode (ssr: true) which targets Node.js by default
 * - Reuses vite-tsconfig-paths plugin for #app/*, #enums/*, #rl/*, #test/* aliases
 * - Externalizes phaser, jsdom, i18next, and other large npm packages
 * - Handles .frag/.vert/.glsl shader imports (returns empty string for headless)
 * - Replaces import.meta.env references with sensible defaults for Node.js
 * - Produces source maps for debugging
 */

import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import path from "node:path";

/**
 * Custom Vite plugin to handle GLSL shader imports (.frag, .vert, .glsl).
 *
 * In the browser build, these are imported with ?raw and Vite inlines the text.
 * In headless mode, shaders are never used (no WebGL rendering), so we return
 * an empty string export for any shader file.
 */
function glslShaderPlugin() {
  return {
    name: "headless-glsl-shader",
    transform(code: string, id: string) {
      const shaderExtensions = [".frag", ".vert", ".glsl"];
      const cleanId = id.split("?")[0]; // Strip ?raw query
      if (shaderExtensions.some(ext => cleanId.endsWith(ext))) {
        return {
          code: "export default '';",
          map: null,
        };
      }
      return null;
    },
  };
}

// Dependencies that should NOT be bundled -- they live in node_modules and
// will be require()'d / import()'d at runtime by the Node.js process.
//
// Criteria for externalization:
// - Large packages that would bloat the bundle (phaser, jsdom)
// - Packages with native/WASM components (jsdom -> parse5, etc.)
// - Packages that work correctly as Node.js imports without bundling
const EXTERNAL_DEPS = [
  // Core game dependency -- Phaser has its own CJS/ESM builds
  "phaser",
  // NOTE: phaser3-rex-plugins is NOT externalized because its ESM exports
  // lack .js extensions, which breaks Node.js module resolution. Bundle it instead.

  // DOM emulation for headless mode
  "jsdom",

  // i18n stack
  "i18next",
  "i18next-browser-languagedetector",
  "i18next-http-backend",
  // NOTE: i18next-korean-postposition-processor is intentionally NOT externalized.
  // It is a pure CJS module with no ESM exports field, so when loaded at runtime
  // via Node.js ESM import, the default export becomes the entire module.exports
  // wrapper object (which lacks .type), rather than the actual plugin instance
  // (module.exports.default). Bundling it lets Vite handle CJS->ESM interop correctly.

  // Other runtime deps
  // NOTE: crypto-js is intentionally NOT externalized. It is a CJS module
  // that doesn't support named ESM exports (e.g., `import { AES, enc }`).
  // Bundling it lets Vite handle CJS->ESM interop correctly.
  "json-stable-stringify",
  "jszip",
  "compare-versions",
  "core-js",
  "@material/material-color-utilities",

  // Node.js built-in modules
  "node:fs",
  "node:path",
  "node:url",
  "node:fs/promises",
  "node:crypto",
  "node:process",
  "node:child_process",
  "node:os",
  "node:util",
  "node:events",
  "node:stream",
  "node:buffer",
  "node:http",
  "node:https",
  "node:net",
  "node:tls",
  "node:zlib",
  "fs",
  "path",
  "url",
  "crypto",
  "os",
  "util",
  "events",
  "stream",
  "buffer",
  "http",
  "https",
  "net",
  "tls",
  "zlib",
];

// biome-ignore lint/style/noDefaultExport: required for Vite config
export default defineConfig({
  plugins: [
    // Resolve #app/*, #enums/*, #rl/*, #test/*, etc. from tsconfig.json paths
    tsconfigPaths(),

    // Handle .frag/.vert/.glsl shader files (return empty string)
    glslShaderPlugin(),

    // Populate i18next namespace list at build time.
    // Without this, `const nsEn = []` in i18n.ts stays empty, so i18next.init()
    // doesn't pre-load any namespaces, causing all i18next.t() calls to return
    // raw keys (e.g. "rockSmash.name" instead of "Rock Smash").
    require("./src/plugins/vite/namespaces-i18n-plugin").LocaleNamespace(),
  ],

  // ── Define: replace import.meta.env references ────────────────────
  // Vite's define replaces these at build time. For the headless bundle,
  // we provide safe defaults that avoid browser-specific behavior.
  define: {
    // Mode: "production" avoids dev-only code paths (debug logging, etc.)
    "import.meta.env.MODE": JSON.stringify("production"),
    // Disable i18n debug output
    "import.meta.env.VITE_I18N_DEBUG": JSON.stringify("0"),
    // No server URL -- MockFetch intercepts all API calls
    "import.meta.env.VITE_SERVER_URL": JSON.stringify("http://localhost:8001"),
    // Bypass login in headless mode
    "import.meta.env.VITE_BYPASS_LOGIN": JSON.stringify("1"),
    // Bypass tutorial in headless mode
    "import.meta.env.VITE_BYPASS_TUTORIAL": JSON.stringify("1"),
    // Not running in test mode (we have our own mock infrastructure)
    "import.meta.env.NODE_ENV": JSON.stringify("production"),
    // Port (unused in headless but referenced in some modules)
    "import.meta.env.VITE_PORT": JSON.stringify("8000"),
    // OAuth IDs (unused in headless)
    "import.meta.env.VITE_DISCORD_CLIENT_ID": JSON.stringify(""),
    "import.meta.env.VITE_GOOGLE_CLIENT_ID": JSON.stringify(""),
    // SSR flag
    "import.meta.env.SSR": JSON.stringify(true),
    // DEV flag
    "import.meta.env.DEV": JSON.stringify(false),
    "import.meta.env.PROD": JSON.stringify(true),
  },

  // ── Build Configuration ───────────────────────────────────────────
  build: {
    // Target Node.js 18+ (matches engines.node in package.json which requires >=24.9.0)
    target: "node18",

    // Output directory
    outDir: "dist/rl",

    // Enable source maps for debugging headless runs
    sourcemap: true,

    // Don't minify -- we want readable stack traces for debugging RL issues
    minify: false,

    // Don't clear the output dir on each build (other dist/ content may exist)
    emptyOutDir: true,

    // SSR build mode: targets Node.js, uses CJS-compatible output,
    // and handles node: protocol imports correctly
    ssr: true,

    // Use the CLI entry point as the SSR entry
    // (Vite SSR mode uses rollupOptions.input for the entry)
    rollupOptions: {
      input: {
        cli: path.resolve(__dirname, "src/rl/cli.ts"),
      },

      output: {
        // ESM format (package.json has "type": "module")
        format: "esm",

        // Single-file output naming
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",

        // Preserve module structure for readable output
        // (helps with debugging and understanding the bundle)
        inlineDynamicImports: false,
      },

      // External dependencies: do not bundle these
      external: (id: string) => {
        // Exact matches (bare specifiers like "phaser", "jsdom", etc.)
        if (EXTERNAL_DEPS.includes(id)) return true;

        // Prefix matches (e.g., "phaser/src/..." or "jsdom/lib/...")
        for (const dep of EXTERNAL_DEPS) {
          if (id.startsWith(dep + "/")) return true;
        }

        // Node.js built-in modules (handles bare "fs" and "node:fs" patterns)
        if (id.startsWith("node:")) return true;

        // Resolved absolute paths into node_modules should also be external.
        // Vite may resolve some imports to absolute paths before calling this function.
        if (id.includes("/node_modules/")) {
          // Check if the resolved path is for one of our external deps
          for (const dep of EXTERNAL_DEPS) {
            if (id.includes(`/node_modules/${dep}/`)) return true;
          }
        }

        // Everything else gets bundled (src/*, test/test-utils/*, etc.)
        return false;
      },

      // Suppress warnings about mixed module formats and circular deps
      // (the game codebase has many circular imports that work fine at runtime)
      onwarn(warning, defaultHandler) {
        // Suppress circular dependency warnings (game codebase has many)
        if (warning.code === "CIRCULAR_DEPENDENCY") return;
        // Suppress "Module level directives cause errors when bundled"
        if (warning.code === "MODULE_LEVEL_DIRECTIVE") return;
        defaultHandler(warning);
      },
    },

    // Increase chunk size warning limit (game code is large)
    chunkSizeWarningLimit: 10000,
  },

  // ── SSR Configuration ───────────────────────────────────────────
  // In SSR mode, Vite automatically externalizes all node_modules by default.
  // We need to explicitly opt-in packages that must be bundled (e.g. because
  // they have CJS/ESM interop issues when loaded at runtime by Node.js).
  ssr: {
    noExternal: [
      // CJS module with broken ESM interop: default import gives wrapper object
      // instead of plugin instance, causing i18next.use() to fail
      "i18next-korean-postposition-processor",
      // Must be bundled because its ESM exports lack .js extensions
      "phaser3-rex-plugins",
      // CJS module that doesn't support named ESM exports (import { AES, enc })
      "crypto-js",
    ],
  },

  // ── Resolve Configuration ─────────────────────────────────────────
  // NOTE: We intentionally do NOT alias Phaser to its source build here
  // (unlike vite.interactive.config.ts) because Phaser is externalized.
  // Adding an alias would resolve "phaser" to an absolute path before
  // the external check runs, preventing externalization. Since Phaser
  // is loaded at runtime from node_modules, no alias is needed.

  // ── esbuild Configuration ─────────────────────────────────────────
  esbuild: {
    // Keep original function/class names for debugging and phase name detection
    // (the game uses constructor.name / phaseName for phase identification)
    keepNames: true,
  },
});
