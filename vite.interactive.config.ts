/**
 * Vite config for the rendered RL browser mode.
 *
 * Extends the default game config (which already handles Phaser, i18n, etc.)
 * and adds the RL WebSocket bridge plugin for Python control.
 *
 * Usage:
 *   npx vite --config vite.interactive.config.ts
 *   Then: python3 tools/play.py --rendered --port <PORT>
 */

import { sharedConfig } from "./vite.config";
import { defineConfig, loadEnv, type UserConfig } from "vite";
import { rlBridgePlugin } from "./src/rl/vite-ws-plugin";

// biome-ignore lint/style/noDefaultExport: required for Vite
export default defineConfig(async config => {
  const { mode, command } = config;
  const envPort = Number(loadEnv(mode, process.cwd()).VITE_PORT);
  const shared = await sharedConfig(config);

  return {
    ...shared,
    base: "",
    publicDir: command === "serve" ? "assets" : false,
    server: {
      port: Number.isNaN(envPort) ? 8000 : envPort,
    },
    plugins: [...(shared.plugins || []), rlBridgePlugin()],
  } satisfies UserConfig;
});
