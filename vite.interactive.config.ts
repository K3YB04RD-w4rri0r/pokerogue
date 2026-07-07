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

import { defineConfig, loadEnv, type UserConfig } from "vite";
import { rlBridgePlugin } from "./src/rl/vite-ws-plugin";
import { sharedConfig } from "./vite.config";

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
      watch: {
        // Don't file-watch the static asset tree: assets/ holds tens of
        // thousands of sprites and watching them exhausts the default Linux
        // inotify budget — the dev server dies mid-session with
        // "Error: ENOSPC ... watch '<assets/...>'". They're served, not
        // transformed, so watching them buys nothing.
        ignored: ["**/assets/**", "**/locales/**", "**/dist/**", "**/.rl-verify/**"],
      },
    },
    plugins: [...(shared.plugins || []), rlBridgePlugin()],
  } satisfies UserConfig;
});
