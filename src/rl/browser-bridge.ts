/**
 * Browser-side RL bridge.
 *
 * Loaded dynamically when ?rl=true is in the URL via a Vite plugin.
 * Connects the running browser game to a Python controller via WebSocket.
 *
 * Flow:
 * 1. Display a status indicator on-screen.
 * 2. Wait for globalScene to exist, disable tutorials, set gender.
 * 3. Create PhaseRouter (hooks UI.setMode / Phase.end) BEFORE TitlePhase fires.
 * 4. Connect to WebSocket relay at ws://localhost:PORT/ws/rl?role=browser.
 * 5. Main loop: auto-handle setup phases (title, gender, starters, evolution),
 *    advance to decision -> serialize state/actions -> send to Python
 *    -> receive action -> execute.
 * 6. On game_over, notify Python and stop.
 *
 * URL parameters:
 *   ?rl=true             — required, enables the bridge
 *   ?seed=abc123         — optional, battle RNG seed for reproducibility
 *   ?delay=500           — optional, ms between actions (watchability)
 *   ?starters=A,B,C      — optional, custom starting party (SpeciesId names)
 *   ?override=KEY=VALUE  — optional, repeatable game override (same surface
 *                          as the headless CLI's --override flag)
 *   ?rewardConfig={...}  — optional, URL-encoded partial RewardConfig JSON
 *                          (same as the headless CLI's --reward-config)
 *   ?waves=N             — optional, end the session after N*50 decisions
 *                          (mirrors the headless CLI's --waves step cap)
 *   ?timeout=N           — optional, no-progress decision timeout in seconds
 *                          (default 30; escape hatch for waits the progress
 *                          probe can't see, e.g. very slow asset loads)
 */

import { EVOLVE_MOVE } from "#app/constants";
import { globalScene } from "#app/global-scene";
import { Button } from "#enums/buttons";
import { LearnMoveSituation } from "#enums/learn-move-situation";
import { PlayerGender } from "#enums/player-gender";
import { UiMode } from "#enums/ui-mode";
import { EvolutionPhase } from "#phases/evolution-phase";
import { buildActionLabels } from "#rl/action-labels";
import { applyOverrideValues } from "#rl/apply-overrides";
import { buildTerminalGameState, EpisodeRewardTracker, resolveExecutedAction, SETUP_PHASES } from "#rl/episode-runtime";
import type { PhaseRouter, PhaseState } from "#rl/phase-router";
import { createPhaseRouter, DecisionPhase, parseStarterCsv } from "#rl/phase-router";
import { encodeObservation } from "#rl/spaces";
import { buildGameState as buildFullGameState } from "#rl/state-builder";
import Phaser from "phaser";

// ── Visual Indicator ──────────────────────────────────────────────────

const INDICATOR_STYLES =
  "position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.8);color:#0f0;"
  + "padding:8px 16px;border-radius:4px;z-index:99999;font-family:monospace;font-size:14px;"
  + "pointer-events:none;";

function createIndicator(): HTMLDivElement {
  const el = document.createElement("div");
  el.id = "rl-bridge-indicator";
  el.style.cssText = INDICATOR_STYLES;
  el.textContent = "[RL] Waiting for game to start...";
  document.body.appendChild(el);
  return el;
}

function updateIndicator(el: HTMLDivElement, text: string, bg?: string): void {
  el.textContent = `[RL] ${text}`;
  if (bg) {
    el.style.background = bg;
  }
}

// ── URL Parameters ────────────────────────────────────────────────────

interface UrlParams {
  seed?: string;
  /** Delay (ms) after executing an action, giving the browser time to animate.
   *  Default: 500. Set to 0 to disable. Use ?delay=1000 for slower animations. */
  renderDelay: number;
  /** Custom starting party from &starters=MEWTWO,LUGIA,... (default: daily-run starters) */
  starters?: ReturnType<typeof parseStarterCsv>;
  /** Game overrides from repeated &override=KEY=VALUE params (same surface as
   *  the headless CLI's --override; VALUE is JSON if parseable, else a raw string). */
  overrides: Record<string, unknown> | null;
  /** Partial RewardConfig from &rewardConfig=<url-encoded JSON> (same as --reward-config). */
  rewardConfig: Record<string, number> | null;
  /** Step budget from &waves=N: the session ends (type "done") after N*50
   *  decisions, mirroring the headless CLI's --waves cap. Omit = unbounded. */
  waves: number | null;
  /** Decision-timeout override in SECONDS from &timeout=N (default: the
   *  router's 30s). For slow machines / software-rendered browsers. */
  timeoutMs: number | null;
  /** v9: mask enemy private info (fog of war) */
  fogOfWar: boolean;
}

function parseUrlParams(): UrlParams {
  const params = new URLSearchParams(window.location.search);
  const starters = params.get("starters");

  // &override=KEY=VALUE, repeatable (mirrors the CLI's --override flag)
  let overrides: Record<string, unknown> | null = null;
  for (const body of params.getAll("override")) {
    const eq = body.indexOf("=");
    if (eq <= 0) {
      console.warn(`[RL Bridge] Invalid &override (expected KEY=VALUE): ${body}`);
      continue;
    }
    const key = body.slice(0, eq);
    const rawValue = body.slice(eq + 1);
    let value: unknown;
    try {
      value = JSON.parse(rawValue);
    } catch {
      value = rawValue; // plain string (e.g. double)
    }
    overrides = { ...(overrides ?? {}), [key]: value };
  }

  let rewardConfig: Record<string, number> | null = null;
  const rawRewardConfig = params.get("rewardConfig");
  if (rawRewardConfig) {
    try {
      const parsed = JSON.parse(rawRewardConfig);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("must be a JSON object");
      }
      rewardConfig = parsed as Record<string, number>;
    } catch (err) {
      console.warn(`[RL Bridge] Invalid &rewardConfig (ignored): ${err}`);
    }
  }

  const rawWaves = Number(params.get("waves"));
  const waves = Number.isFinite(rawWaves) && rawWaves > 0 ? Math.floor(rawWaves) : null;

  const rawTimeout = Number(params.get("timeout"));
  const timeoutMs = Number.isFinite(rawTimeout) && rawTimeout > 0 ? Math.floor(rawTimeout * 1000) : null;

  return {
    seed: params.get("seed") || undefined,
    fogOfWar: params.get("fog") === "1" || params.get("fog") === "true",
    renderDelay: Number(params.get("delay") ?? 500),
    starters: starters ? parseStarterCsv(starters) : undefined,
    overrides,
    rewardConfig,
    waves,
    timeoutMs,
  };
}

// ── Wait for Game Ready ──────────────────────────────────────────────

/**
 * Wait for globalScene to be available, then apply RL-mode overrides:
 * - Disable tutorials (prevents blocking message dialogs)
 * - Set player gender to FEMALE
 *
 * Does NOT wait for battle — the PhaseRouter + main loop auto-handle
 * TitlePhase, SelectGenderPhase, and SelectStarterPhase.
 */
async function waitForGameReady(indicator: HTMLDivElement): Promise<void> {
  console.log("[RL Bridge] Waiting for game to initialize...");

  // Phase 1: Wait for globalScene and disable tutorials ASAP
  while (true) {
    try {
      if (globalScene) {
        globalScene.enableTutorials = false;
        break;
      }
    } catch {
      // globalScene may not be ready yet
    }
    await sleep(100);
  }
  console.log("[RL Bridge] Tutorials disabled");

  // Phase 2: Wait for gameData to exist so we can set gender
  while (true) {
    try {
      if (globalScene.gameData) {
        globalScene.gameData.gender = PlayerGender.FEMALE;
        break;
      }
    } catch {
      // gameData created during LoginPhase, may not exist yet
    }
    await sleep(100);
  }
  console.log("[RL Bridge] Gender set to FEMALE");

  // Phase 3: Monkey-patch getSession to prevent loading saved sessions from the
  // API server. When bypassLogin=false, TitlePhase.start() calls getSession()
  // which fetches session data from the server (ignoring localStorage.clear()).
  // By returning null, TitlePhase will show "New Game" instead of "Continue",
  // and our executeTitleAction() will start fresh at wave 1.
  globalScene.gameData.getSession = async (_slotId: number) => {
    console.log("[RL Bridge] Intercepted getSession() — returning null (fresh start)");
    return null;
  };
  console.log("[RL Bridge] getSession patched (no saved session will load)");

  updateIndicator(indicator, "Game ready. Auto-starting...", "rgba(0,80,0,0.8)");
}

// ── WebSocket Helpers ─────────────────────────────────────────────────

function connectWS(): Promise<WebSocket> {
  const port = window.location.port || "8080";
  const wsUrl = `ws://localhost:${port}/ws/rl?role=browser`;
  console.log("[RL Bridge] Connecting to", wsUrl);

  return new Promise<WebSocket>((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      console.log("[RL Bridge] WebSocket connected");
      resolve(ws);
    };
    ws.onerror = ev => {
      console.error("[RL Bridge] WebSocket error:", ev);
      reject(new Error("WebSocket connection failed"));
    };
  });
}

/**
 * Wait for an `{"action": <number>}` message from the Python controller.
 * Resolves with the action index, or -1 if the socket closes.
 */
function waitForAction(ws: WebSocket): Promise<number> {
  return new Promise<number>(resolve => {
    if (ws.readyState !== WebSocket.OPEN) {
      resolve(-1);
      return;
    }

    const onMessage = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data);
        if (typeof msg.action === "number") {
          cleanup();
          resolve(msg.action);
        }
      } catch {
        // Ignore non-JSON or malformed messages
      }
    };

    const onClose = () => {
      cleanup();
      resolve(-1);
    };

    function cleanup() {
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
    }

    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);
  });
}

function sendWS(ws: WebSocket, obj: Record<string, unknown>): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

/**
 * Wait for a specific message type from the Python controller.
 * Used for the handshake: browser sends "ready", waits for "start".
 * Returns false if the socket closes before receiving the expected message.
 */
function waitForMessageType(ws: WebSocket, expectedType: string, timeoutMs = 60000): Promise<boolean> {
  return new Promise<boolean>(resolve => {
    if (ws.readyState !== WebSocket.OPEN) {
      resolve(false);
      return;
    }

    const timer = setTimeout(() => {
      cleanup();
      resolve(false);
    }, timeoutMs);

    const onMessage = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === expectedType) {
          cleanup();
          resolve(true);
        }
      } catch {
        // Ignore non-JSON
      }
    };

    const onClose = () => {
      cleanup();
      resolve(false);
    };

    function cleanup() {
      clearTimeout(timer);
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
    }

    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);
  });
}

// ── Game State ────────────────────────────────────────────────────────
// GameState is now built by the shared state-builder module (buildFullGameState)

// ── Action Labels ─────────────────────────────────────────────────────

// Action labels come from the shared #rl/action-labels module (same labels
// as the headless CLI).

// ── Utility ───────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

/** Encode a Float32Array bit-exactly to base64 (browser btoa; decoded as "<f4"). */
function obsToBase64(obs: Float32Array): string {
  const bytes = new Uint8Array(obs.buffer, obs.byteOffset, obs.byteLength);
  let bin = "";
  const CHUNK = 0x8000; // String.fromCharCode arg-count limit safety
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

// ── Setup Phases ──────────────────────────────────────────────────────

// SETUP_PHASES comes from the shared #rl/episode-runtime module.

// ── Message Auto-Dismiss ──────────────────────────────────────────────

/**
 * Dismiss any blocking MESSAGE dialogs (tutorials, battle narration, etc.)
 * that are currently showing. In headless mode, MockText.showText immediately
 * fires callbacks so messages never block. In the browser, messages require
 * the player to press ACTION. This replicates the headless behavior.
 */
async function dismissBlockingMessages(): Promise<void> {
  let dismissed = 0;
  const MAX_DISMISS = 50;
  while (dismissed < MAX_DISMISS) {
    await sleep(100);
    try {
      const uiMode = globalScene.ui?.getMode();
      if (uiMode === UiMode.MESSAGE) {
        const handler = globalScene.ui.getHandler();
        if (handler?.active) {
          (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
          dismissed++;
          continue;
        }
      }
      // Not in MESSAGE mode — stop dismissing
      break;
    } catch {
      break;
    }
  }
  if (dismissed > 0) {
    console.log(`[RL Bridge] Dismissed ${dismissed} blocking messages`);
  }
}

/**
 * Advance to the next decision point while auto-dismissing any MESSAGE
 * dialogs that would otherwise block the game from reaching a decision phase.
 *
 * In the browser, battle narration ("Go! Bulbasaur!", ability triggers, etc.)
 * shows as MESSAGE and waits for player input. This auto-presses ACTION every
 * 300ms so the game flows automatically while the user watches in the browser.
 */
async function advanceWithAutoDismiss(router: PhaseRouter, timeoutMs?: number | null): Promise<PhaseState> {
  let running = true;

  // Background loop (fire-and-forget): press ACTION whenever stuck in MESSAGE mode
  void (async () => {
    while (running) {
      await sleep(300);
      if (!running) {
        break;
      }
      try {
        const uiMode = globalScene.ui?.getMode();
        if (uiMode === UiMode.MESSAGE) {
          const handler = globalScene.ui.getHandler();
          if (handler?.active) {
            (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
          }
        }
      } catch {
        // ignore
      }
    }
  })();

  try {
    return await router.advanceToNextDecision(timeoutMs ?? undefined);
  } finally {
    running = false;
    // Let the dismiss loop finish its current iteration
    await sleep(0);
  }
}

// ── Instant Evolution ─────────────────────────────────────────────────

/**
 * Replace the evolution CINEMATIC with its LOGIC ("replicate the logic,
 * cut the animations"). The cinematic path proved unreliable under
 * automation even time-warped — observed both as a freeze and as a
 * silently-skipped evolution — and its temp display-pokemon asset load can
 * hang forever on a broken asset. The actual evolution logic is three
 * steps, extracted verbatim from EvolutionPhase.handleSuccessEvolution /
 * postEvolve:
 *   1. pokemon.evolve(evolution, species)  — species/form/stats/name are
 *      set SYNCHRONOUSLY; only the promise tail loads sprites, so it is
 *      raced against a timeout (broken/slow assets cost sprites, not runs).
 *   2. queue LearnMovePhase for each EVOLVE_MOVE level move.
 *   3. queue EndEvolutionPhase (restores MESSAGE mode).
 * FormChangePhase (a subclass) keeps its own start() — different logic —
 * and remains covered by the cinematic fast-forward instead.
 */
function installInstantEvolution(): void {
  const proto = EvolutionPhase.prototype as unknown as {
    start(): Promise<void> | void;
    end(): void;
  };
  if ((proto.start as { __rlInstant?: boolean }).__rlInstant) {
    return;
  }
  const originalStart = proto.start;
  proto.start = async function (this: {
    phaseName: string;
    validate?(): boolean;
    end(): void;
    pokemon: any;
    evolution: unknown;
    lastLevel: number;
    fusionSpeciesEvolved?: boolean;
  }) {
    if (this.phaseName !== "EvolutionPhase") {
      return originalStart.call(this as never);
    }
    try {
      if (this.validate && !this.validate()) {
        return this.end();
      }
      const pokemon = this.pokemon;
      const before = pokemon?.name;
      await Promise.race([
        pokemon.evolve(this.evolution, pokemon.species),
        sleep(15000).then(() =>
          console.warn("[RL Bridge] evolve() asset load did not settle in 15s — continuing (sprites may lag)"),
        ),
      ]);
      const learnSituation = this.fusionSpeciesEvolved
        ? LearnMoveSituation.EVOLUTION_FUSED
        : pokemon.fusionSpecies
          ? LearnMoveSituation.EVOLUTION_FUSED_BASE
          : LearnMoveSituation.EVOLUTION;
      const levelMoves = (pokemon.getLevelMoves(this.lastLevel + 1, true, false, false, learnSituation) ?? []).filter(
        (lm: [number, number]) => lm[0] === EVOLVE_MOVE,
      );
      for (const lm of levelMoves) {
        globalScene.phaseManager.unshiftNew("LearnMovePhase", globalScene.getPlayerParty().indexOf(pokemon), lm[1]);
      }
      globalScene.phaseManager.unshiftNew("EndEvolutionPhase");
      console.log(`[RL Bridge] Instant evolution: ${before} -> ${pokemon?.name}`);
    } catch (err) {
      console.error("[RL Bridge] Instant evolution failed; ending phase:", err);
    }
    this.end();
  };
  (proto.start as { __rlInstant?: boolean }).__rlInstant = true;
}

// ── Cinematic Fast-Forward ────────────────────────────────────────────

/**
 * Cinematic phases whose animations are pure presentation for RL purposes:
 * evolutions, form changes and egg hatches run 15-20s of clock-driven beats
 * (delayedCalls, tween counters, fanfares) that on slow/software-rendered
 * browsers stretch into minutes and read as a hang. The underlying LOGIC
 * (pokemon.evolve(), learn-move queueing, dex updates) is clock-independent,
 * so instead of replicating it we "cut the animations, keep the logic" by
 * time-warping the scene clocks while one of these phases is current — the
 * exact same game code runs, just compressed. Rendered-only; headless mock
 * tweens already complete synchronously.
 */
const CINEMATIC_PHASES = new Set([
  "EvolutionPhase",
  "EndEvolutionPhase",
  "FormChangePhase",
  "EggHatchPhase",
  "EggSummaryPhase",
]);
const CINEMATIC_TIMESCALE = 50;

/** Start the cinematic watcher; returns a stop function. */
function startCinematicFastForward(): () => void {
  let warped = false;
  const setScale = (scale: number) => {
    try {
      const scene = globalScene as unknown as { time?: { timeScale: number }; tweens?: { timeScale: number } };
      if (scene.time) {
        scene.time.timeScale = scale;
      }
      if (scene.tweens) {
        scene.tweens.timeScale = scale;
      }
    } catch {
      /* scene mid-teardown */
    }
  };
  const tick = setInterval(() => {
    try {
      const phaseName = globalScene.phaseManager?.getCurrentPhase()?.phaseName ?? "";
      const wantWarp = CINEMATIC_PHASES.has(phaseName);
      if (wantWarp && !warped) {
        warped = true;
        console.log(`[RL Bridge] Fast-forwarding cinematic: ${phaseName} (x${CINEMATIC_TIMESCALE})`);
        setScale(CINEMATIC_TIMESCALE);
      } else if (!wantWarp && warped) {
        warped = false;
        setScale(1);
      }
    } catch {
      /* ignore — next tick retries */
    }
  }, 200);
  return () => {
    clearInterval(tick);
    if (warped) {
      setScale(1);
    }
  };
}

// ── Phase Change Wait ─────────────────────────────────────────────────

/**
 * Wait for the game to transition away from a specific phase.
 * In the browser, initBattle() is async — TitlePhase stays current while
 * assets load. This waits (with timeout) until the phase changes.
 */
async function waitForPhaseChange(currentPhaseName: string, timeoutMs = 30000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    await sleep(100);
    try {
      const phase = globalScene.phaseManager?.getCurrentPhase();
      const phaseName = phase?.phaseName;
      // If phase changed or no current phase, we're done
      if (!phaseName || phaseName !== phaseNameForDecision(currentPhaseName)) {
        return;
      }
    } catch {
      return; // If we can't check, assume it changed
    }
  }
  console.warn(`[RL Bridge] Timed out waiting for phase change from ${currentPhaseName}`);
}

/**
 * Map DecisionPhase enum values back to phase class names for comparison.
 */
function phaseNameForDecision(decision: string): string | null {
  switch (decision) {
    case DecisionPhase.TITLE:
      return "TitlePhase";
    case DecisionPhase.SELECT_GENDER:
      return "SelectGenderPhase";
    case DecisionPhase.SELECT_STARTER:
      return "SelectStarterPhase";
    case DecisionPhase.EVOLUTION:
      return "EvolutionPhase";
    case DecisionPhase.FORM_CHANGE:
      return "FormChangePhase";
    default:
      return null;
  }
}

// ── Seed ──────────────────────────────────────────────────────────────

/**
 * Apply seed for deterministic battles.
 * Sets both Phaser's global RNG and the game's own seed (BattleScene.seed),
 * then re-derives wave seeds. Without setSeed+resetSeed, the game seed
 * (generated by randomString(24) using Math.random()) would be random
 * regardless of the URL seed parameter.
 */
function applySeed(seed: string): void {
  try {
    Phaser.Math.RND.sow([seed]);
    globalScene.setSeed(seed);
    globalScene.resetSeed();
    console.log(`[RL Bridge] Seed applied: ${seed}`);
  } catch (err) {
    console.warn("[RL Bridge] Failed to apply seed:", err);
  }
}

// ── Main Bridge Loop ──────────────────────────────────────────────────

async function startBridge(): Promise<void> {
  const indicator = createIndicator();
  const urlParams = parseUrlParams();

  if (urlParams.seed) {
    console.log(`[RL Bridge] Seed from URL: ${urlParams.seed}`);
  }

  // Step 1: Wait for globalScene + set overrides (tutorials, gender)
  await waitForGameReady(indicator);

  // Step 1b: Apply game overrides from &override=KEY=VALUE (same config
  // surface as the headless CLI's --override). Must run before the battle is
  // created so battle-creation reads (STARTING_WAVE_OVERRIDE, movesets, ...)
  // see the overridden values.
  if (urlParams.overrides) {
    console.log("[RL Bridge] Applying overrides:", urlParams.overrides);
    await applyOverrideValues(urlParams.overrides);
  }

  // Step 2: Create PhaseRouter ASAP — must be before TitlePhase fires
  // so the setMode hook catches it. TitlePhase waits indefinitely for input,
  // so even if it fires before the hook, detectCurrentDecision() will find it.
  const router: PhaseRouter = createPhaseRouter({ verbose: true, starterSpecies: urlParams.starters });

  // Debug handle for bug hunting (browser devtools / automated probes):
  // window.__rlDebug.state() -> current phase, UI mode, handler flags.
  (window as unknown as { __rlDebug?: unknown }).__rlDebug = {
    scene: globalScene,
    router,
    state: () => {
      try {
        const phase = globalScene.phaseManager?.getCurrentPhase();
        const mode = globalScene.ui?.getMode();
        const handler = globalScene.ui?.getHandler() as unknown as {
          active?: boolean;
          awaitingActionInput?: boolean;
        };
        const msgHandler = globalScene.ui?.getMessageHandler() as unknown as {
          awaitingActionInput?: boolean;
          pendingPrompt?: boolean;
        };
        return {
          phase: phase?.phaseName ?? null,
          uiMode: mode != null ? UiMode[mode] : null,
          handlerActive: handler?.active ?? null,
          handlerAwaiting: handler?.awaitingActionInput ?? null,
          messageAwaiting: msgHandler?.awaitingActionInput ?? null,
          atDecision: router.isAtDecisionPoint(),
        };
      } catch (err) {
        return { error: String(err) };
      }
    },
  };

  // Dismiss any blocking messages that appeared during boot
  await dismissBlockingMessages();

  // Step 3: Connect to the WebSocket relay
  let ws: WebSocket;
  try {
    ws = await connectWS();
  } catch (err) {
    updateIndicator(indicator, "WebSocket connection failed!", "rgba(150,0,0,0.8)");
    console.error("[RL Bridge] Failed to connect WebSocket:", err);
    return;
  }

  // Handle unexpected close
  let wsOpen = true;
  ws.onclose = () => {
    wsOpen = false;
    console.warn("[RL Bridge] WebSocket closed");
    updateIndicator(indicator, "WebSocket disconnected", "rgba(150,0,0,0.8)");
  };

  updateIndicator(indicator, "Connected! Waiting for Python...", "rgba(0,100,0,0.8)");

  // Notify Python that the bridge is ready and wait for "start" handshake.
  // This prevents the game from auto-starting without Python running.
  sendWS(ws, { type: "ready" });
  console.log("[RL Bridge] Sent 'ready', waiting for Python 'start' signal...");

  const started = await waitForMessageType(ws, "start", 120000);
  if (!started) {
    updateIndicator(indicator, "Python not connected — start play.py --rendered", "rgba(150,0,0,0.8)");
    console.error("[RL Bridge] Python did not respond with 'start'. Run: python3 tools/play.py --rendered");
    ws.close();
    return;
  }
  console.log("[RL Bridge] Python confirmed — starting game!");
  updateIndicator(indicator, "Python connected! Auto-starting game...", "rgba(0,100,0,0.8)");

  // Step 4: Main loop
  let step = 0;
  // &waves=N ends the session after N*50 decisions (same step budget the
  // headless CLI derives from --waves). null = unbounded (watching mode).
  const maxSteps = urlParams.waves !== null ? urlParams.waves * 50 : null;
  // Reward bookkeeping shared with the headless CLI (episode-runtime.ts) so a
  // rendered episode reports the same rewards headless training would.
  const tracker = new EpisodeRewardTracker(urlParams.rewardConfig ?? undefined);
  // Set when the episode ends by a cap — the final done message then carries
  // the true final observation/reward (mirrors the headless CLI).
  let capPayload: Record<string, unknown> | null = null;
  // Evolutions: run the logic, skip the cinematic entirely
  installInstantEvolution();
  // Compress remaining cinematics (form change, egg hatch) — logic intact
  const stopCinematicFastForward = startCinematicFastForward();
  // Track which setup phases have been handled to avoid re-processing them.
  // In the browser, async asset loading means phases can stay "current" longer
  // than in headless mode, causing re-detection.
  const handledSetupPhases = new Set<string>();

  try {
    while (wsOpen) {
      // Advance to the next decision point, auto-dismissing any blocking
      // MESSAGE dialogs (battle narration, tutorials, etc.) along the way
      let state: PhaseState;
      try {
        state = await advanceWithAutoDismiss(router, urlParams.timeoutMs);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("Timeout")) {
          console.warn("[RL Bridge] Timeout waiting for decision at step", step);
          sendWS(ws, { type: "error", message: `Timeout at step ${step}` });
          break;
        }
        if (msg.includes("destroyed")) {
          break;
        }
        throw err;
      }

      // Auto-handle setup phases (title, gender, starters, evolution, form_change)
      // Skip if we already handled this exact phase (prevents re-processing when
      // the phase stays current due to async asset loading in the browser).
      if (SETUP_PHASES.has(state.phase) && !handledSetupPhases.has(state.phase)) {
        handledSetupPhases.add(state.phase);
        console.log("[RL Bridge] Auto-handling setup phase:", state.phase);

        // Apply seed BEFORE executeTitleAction so it takes effect before
        // the first battle is created (newBattle → resetSeed uses this.seed).
        if (state.phase === DecisionPhase.TITLE && urlParams.seed) {
          applySeed(urlParams.seed);
        }

        await router.executeAction(0);

        // Wait for the phase to actually change before continuing.
        // In browser, initBattle() is async — TitlePhase stays current while
        // assets load. Without this wait, the loop immediately re-detects TITLE.
        await waitForPhaseChange(state.phase);
        continue;
      }
      if (SETUP_PHASES.has(state.phase) && handledSetupPhases.has(state.phase)) {
        // Already handled — the phase is still playing out. In the browser this
        // can require player input to advance: an EvolutionPhase sits in
        // EVOLUTION_SCENE mode and waits for ACTION at each "…is evolving / evolved
        // into…" beat (EvolutionSceneUiHandler.processInput only fires when
        // awaitingActionInput), and trailing MESSAGE dialogs need ACTION too.
        // Headless never hits this (mock tweens fire the callbacks synchronously),
        // so just sleeping here spins forever. Press ACTION to drive it. ACTION is
        // safe when nothing is awaiting input (the handler ignores it) and never
        // cancels the evolution (only CANCEL does).
        try {
          const mode = globalScene.ui?.getMode();
          if (mode === UiMode.MESSAGE || mode === UiMode.EVOLUTION_SCENE) {
            const handler = globalScene.ui.getHandler();
            if (handler?.active) {
              (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
            }
          }
        } catch {
          /* best-effort; fall through to the retry sleep */
        }
        await sleep(200);
        continue;
      }

      // Hide the pokemon info container if it's still visible (e.g. after
      // a successful catch — AttemptCapturePhase.catch() shows IVs/stats
      // overlay but the hide tween may not finish before the next decision).
      try {
        const pic = globalScene.pokemonInfoContainer;
        if (pic && (pic as any).shown) {
          pic.setVisible(false);
          (pic as any).shown = false;
        }
      } catch (_) {
        /* ignore */
      }

      // Check for game over
      if (state.phase === DecisionPhase.GAME_OVER || router.isGameOver()) {
        // The live scene is already post-reset at game over — reuse the last
        // decision state patched to terminal truth, and report the terminal
        // reward (both shared with the headless CLI via episode-runtime).
        const reward = tracker.rewardOnArrival(step, true, router.isVictory());
        const base = tracker.getLastGameState() ?? buildFullGameState(state, step);
        const gameState = buildTerminalGameState(base, router.isVictory());
        sendWS(ws, {
          type: "game_over",
          step,
          victory: router.isVictory(),
          gameState,
          reward,
          wave: (gameState as { battle?: { wave_index?: number } }).battle?.wave_index ?? 0,
        });
        updateIndicator(
          indicator,
          `Game over (${router.isVictory() ? "victory" : "defeat"}) at step ${step}`,
          "rgba(100,100,0,0.8)",
        );
        break;
      }

      // Reward earned by the previous action (0 on the very first state) —
      // same bookkeeping the headless CLI uses.
      const reward = tracker.rewardOnArrival(step, false, false);

      // Build state payload
      const actionLabels = buildActionLabels(state);
      const gameState = buildFullGameState(state, step);
      tracker.noteDecisionState(gameState);
      const wave = (gameState as { battle?: { wave_index?: number } }).battle?.wave_index ?? 0;
      // TS-encoded observation + mask, additive alongside the full gameState
      // (the Python tools may keep encoding locally — the two encoders are
      // bitwise parity-verified, so either source is valid).
      const obs = encodeObservation(gameState, { fogOfWar: urlParams.fogOfWar });

      // REAL wave cap (+ N*50 step backstop), matching the headless CLI: end
      // as truncated at a genuine decision state so the final observation is
      // real, not zeros. null waves = unbounded (watching mode).
      if (urlParams.waves !== null && (wave > urlParams.waves || (maxSteps !== null && step >= maxSteps))) {
        capPayload = {
          reason: wave > urlParams.waves ? "wave_cap" : "step_cap",
          reward,
          obsB64: obsToBase64(obs),
          mask: state.actionMask,
          wave,
          gameState,
        };
        break;
      }

      // Send state to Python
      sendWS(ws, {
        type: "state",
        step,
        phase: state.phase,
        actions: actionLabels,
        gameState,
        metadata: state.metadata,
        reward,
        obsB64: obsToBase64(obs),
        mask: state.actionMask,
        wave,
      });

      // Update on-screen indicator
      updateIndicator(
        indicator,
        `Step ${step} | Wave ${wave || "?"} | ${state.phase} | ${actionLabels.length} actions`,
      );

      // Wait for action from Python
      const action = await waitForAction(ws);

      // WebSocket closed while waiting
      if (action === -1) {
        console.warn("[RL Bridge] Lost connection while waiting for action");
        break;
      }

      // Validate action against mask and execute (invalid actions fall back
      // to the first valid one — same rule as the headless CLI)
      const { executed, wasValid } = resolveExecutedAction(state, action);
      if (!wasValid) {
        console.warn(
          `[RL Bridge] Invalid action ${action} for phase ${state.phase}. `
            + `Valid: [${state.validActions.join(", ")}]. Using action ${executed}.`,
        );
      }

      // Pre-action bookkeeping for the next step's reward (must run before
      // executeAction: the modifier phase is gone once the action resolves)
      tracker.notePreAction(state, executed);
      await router.executeAction(executed);

      // Give the browser time to animate the action before detecting the
      // next decision point. Without this delay, the TUI shows the next
      // state (e.g., "switch") while the browser is still rendering the
      // current phase (e.g., "fight"). Configurable via ?delay=N (ms).
      if (urlParams.renderDelay > 0) {
        await sleep(urlParams.renderDelay);
      }

      step++;
    }
  } catch (err) {
    console.error("[RL Bridge] Error in main loop:", err);
    sendWS(ws, { type: "error", message: String(err) });
  }

  // Send completion message
  sendWS(ws, { type: "done", steps: step, ...(capPayload ?? {}) });

  updateIndicator(indicator, `Done (${step} steps)`, "rgba(100,100,0,0.8)");

  // Cleanup
  stopCinematicFastForward();
  router.destroy();
  console.log(`[RL Bridge] Session ended after ${step} steps`);
}

// ── Entry Point ───────────────────────────────────────────────────────

startBridge().catch(err => {
  console.error("[RL Bridge] Fatal error:", err);
});
