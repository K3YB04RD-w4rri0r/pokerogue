import type { Plugin, ViteDevServer } from "vite";

export function rlBridgePlugin(): Plugin {
  return {
    name: "rl-bridge",

    configureServer(server: ViteDevServer) {
      // We need the ws package for the WebSocket server.
      // Vite depends on it internally, so it's available in node_modules.
      import("ws").then(({ WebSocketServer, WebSocket: WS }) => {
        const wss = new WebSocketServer({ noServer: true });
        let pythonWs: InstanceType<typeof WS> | null = null;
        let browserWs: InstanceType<typeof WS> | null = null;
        // The browser's `ready` can arrive before Python connects (the relay
        // never buffers). Keep the LAST ready and replay it to a late Python
        // client so the handshake can't be lost to ordering.
        let pendingReady: string | null = null;
        // Last un-answered browser->python state/game_over frame: replayed to
        // a NEWLY connected python so a mid-step handover (crash, Ctrl+C,
        // second client) doesn't leave browser-waiting-for-action vs
        // python-waiting-for-state in a mutual infinite wait.
        let pendingState: string | null = null;

        server.httpServer?.on("upgrade", (req, socket, head) => {
          // Only handle /ws/rl path - leave Vite's HMR WebSocket alone
          if (!req.url?.startsWith("/ws/rl")) {
            return;
          }

          wss.handleUpgrade(req, socket, head, ws => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const role = url.searchParams.get("role") || "python";

            if (role === "browser") {
              console.log("[rl-bridge] Browser connected");
              // A new browser session (page reload, second tab) replaces the
              // old one — close the stale socket so two games can't interleave
              // messages onto the same Python client.
              if (browserWs && browserWs !== ws && browserWs.readyState === WS.OPEN) {
                console.log("[rl-bridge] Replacing previous browser session");
                browserWs.close();
              }
              browserWs = ws;
              ws.on("message", data => {
                const text = data.toString();
                // Buffer the newest ready for a Python client that hasn't
                // connected yet (ordering race — see pendingReady above)
                try {
                  const frameType = JSON.parse(text)?.type;
                  if (frameType === "ready") {
                    pendingReady = text;
                    pendingState = null; // fresh session; older state is void
                  } else if (frameType === "state" || frameType === "game_over") {
                    pendingState = text;
                  }
                } catch {
                  /* non-JSON frames relay as-is */
                }
                // Relay browser -> Python (only from the CURRENT browser session)
                if (ws === browserWs && pythonWs?.readyState === WS.OPEN) {
                  pythonWs.send(text);
                }
              });
              ws.on("close", () => {
                if (ws === browserWs) {
                  console.log("[rl-bridge] Browser disconnected");
                  browserWs = null;
                  pendingReady = null;
                  pendingState = null;
                  // Tell the surviving Python client instead of letting its
                  // recv() block forever (audit H3)
                  if (pythonWs?.readyState === WS.OPEN) {
                    pythonWs.send(JSON.stringify({ type: "error", message: "browser disconnected" }));
                  }
                }
              });
            } else {
              console.log("[rl-bridge] Python connected");
              if (pythonWs && pythonWs !== ws && pythonWs.readyState === WS.OPEN) {
                console.log("[rl-bridge] Replacing previous python session");
                pythonWs.close();
              }
              pythonWs = ws;
              // Replay the buffered ready (browser booted before Python
              // connected) and the last un-answered state (mid-step handover).
              if (pendingReady && browserWs?.readyState === WS.OPEN) {
                console.log("[rl-bridge] Replaying buffered browser ready to python");
                ws.send(pendingReady);
              }
              if (pendingState && browserWs?.readyState === WS.OPEN) {
                console.log("[rl-bridge] Replaying pending state to python (mid-episode handover)");
                ws.send(pendingState);
              }
              ws.on("message", data => {
                const text = data.toString();
                try {
                  const parsed = JSON.parse(text);
                  if (parsed?.type === "start") {
                    // Session is live — the buffered ready is now stale (a
                    // handover after this point must NOT look like a fresh boot)
                    pendingReady = null;
                  }
                  if (typeof parsed?.action === "number") {
                    pendingState = null; // state answered
                  }
                } catch {
                  /* non-JSON frames relay as-is */
                }
                // Relay Python -> browser (only from the CURRENT python client)
                if (ws === pythonWs && browserWs?.readyState === WS.OPEN) {
                  browserWs.send(text);
                }
              });
              ws.on("close", () => {
                if (ws === pythonWs) {
                  console.log("[rl-bridge] Python disconnected");
                  pythonWs = null;
                }
              });
            }
          });
        });
      });
    },

    transformIndexHtml: {
      order: "pre",
      handler(html, ctx) {
        // Only inject the bridge when ?rl=true is in the URL
        if (!ctx.originalUrl?.includes("rl=true")) {
          return html;
        }

        // Inject TWO scripts BEFORE the closing </head> tag:
        //
        // 1. A regular (non-module) inline script that clears all storage
        //    synchronously. Regular scripts execute immediately during HTML
        //    parsing, BEFORE any deferred/module scripts. This guarantees
        //    storage is wiped before the game's main.ts module reads it.
        //
        // 2. A module script that imports the browser bridge (async).
        const bridgeScript = `
    <script>
      localStorage.clear();
      sessionStorage.clear();
    </script>
    <script type="module">
      import("/src/rl/browser-bridge.ts");
    </script>`;

        return html.replace("</head>", `${bridgeScript}\n  </head>`);
      },
    },
  };
}
