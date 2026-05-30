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

        server.httpServer?.on("upgrade", (req, socket, head) => {
          // Only handle /ws/rl path - leave Vite's HMR WebSocket alone
          if (!req.url?.startsWith("/ws/rl")) return;

          wss.handleUpgrade(req, socket, head, (ws) => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const role = url.searchParams.get("role") || "python";

            if (role === "browser") {
              console.log("[rl-bridge] Browser connected");
              browserWs = ws;
              ws.on("message", (data) => {
                // Relay browser -> Python
                if (pythonWs?.readyState === WS.OPEN) {
                  pythonWs.send(data.toString());
                }
              });
              ws.on("close", () => {
                console.log("[rl-bridge] Browser disconnected");
                browserWs = null;
              });
            } else {
              console.log("[rl-bridge] Python connected");
              pythonWs = ws;
              ws.on("message", (data) => {
                // Relay Python -> browser
                if (browserWs?.readyState === WS.OPEN) {
                  browserWs.send(data.toString());
                }
              });
              ws.on("close", () => {
                console.log("[rl-bridge] Python disconnected");
                pythonWs = null;
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
