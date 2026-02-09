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
 *   ?rl=true         — required, enables the bridge
 *   ?seed=abc123     — optional, sets battle RNG seed for reproducibility
 */

import { globalScene } from "#app/global-scene";
import { createPhaseRouter, DecisionPhase } from "#rl/phase-router";
import type { PhaseState, PhaseRouter } from "#rl/phase-router";
import { getAvailableModifiers } from "#rl/modifier-api";
import { buildGameState as buildFullGameState } from "#rl/state-builder";
import { UiMode } from "#enums/ui-mode";
import { Button } from "#enums/buttons";
import { PlayerGender } from "#enums/player-gender";
import Phaser from "phaser";
import {
  ACTION_FIGHT_ENEMY_START,
  ACTION_FIGHT_ENEMY2_START,
  ACTION_FIGHT_ALLY_START,
  ACTION_SWITCH_START,
  ACTION_BALL_START,
  ACTION_RUN,
  ACTION_TERA_ENEMY_START,
  ACTION_TERA_ENEMY2_START,
  ACTION_TERA_ALLY_START,
  ACTION_SELECT_REWARD_START,
  ACTION_REROLL,
  ACTION_SKIP,
  ACTION_BUY_SHOP_START,
  ACTION_PARTY_TARGET_START,
  MAX_MOVES,
} from "#rl/spaces";

// ── Visual Indicator ──────────────────────────────────────────────────

const INDICATOR_STYLES =
  "position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.8);color:#0f0;" +
  "padding:8px 16px;border-radius:4px;z-index:99999;font-family:monospace;font-size:14px;" +
  "pointer-events:none;";

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
}

function parseUrlParams(): UrlParams {
  const params = new URLSearchParams(window.location.search);
  return {
    seed: params.get("seed") || undefined,
    renderDelay: Number(params.get("delay") ?? 500),
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
    ws.onerror = (ev) => {
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
  return new Promise<number>((resolve) => {
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
  return new Promise<boolean>((resolve) => {
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

interface ActionInfo {
  index: number;
  label: string;
}

const BALL_NAMES = ["Poke Ball", "Great Ball", "Ultra Ball", "Rogue Ball", "Master Ball"];
const TIER_NAMES = ["COMMON", "GREAT", "ULTRA", "ROGUE", "MASTER", "LUXURY"];

/**
 * Get the name of an enemy Pokemon by its field slot for better action labels in doubles.
 * Returns "Enemy" / "Enemy 2" as fallback if the enemy field can't be read.
 */
function getEnemyName(slot: 0 | 1): string {
  try {
    const enemyField = globalScene.getEnemyField()?.filter(p => p?.isActive()) ?? [];
    if (slot < enemyField.length) {
      return enemyField[slot].species?.name ?? (slot === 0 ? "Enemy" : "Enemy 2");
    }
  } catch {
    // ignore
  }
  return slot === 0 ? "Enemy" : "Enemy 2";
}

/**
 * Get the name of the player's ally Pokemon (second active slot) for ally targeting labels.
 */
function getAllyName(): string {
  try {
    const playerField = globalScene.getPlayerField()?.filter(p => p?.isActive()) ?? [];
    if (playerField.length > 1) {
      return playerField[1].species?.name ?? "Ally";
    }
  } catch {
    // ignore
  }
  return "Ally";
}

function getMoveName(moveIndex: number): string {
  try {
    const phase = globalScene.phaseManager.getCurrentPhase();
    if (phase?.is("CommandPhase")) {
      const pokemon = (phase as any).getPokemon();
      const moveset = pokemon.getMoveset(false);
      if (moveIndex < moveset.length) {
        const move = moveset[moveIndex].getMove();
        const ppLeft = moveset[moveIndex].getMovePp() - moveset[moveIndex].ppUsed;
        const ppMax = moveset[moveIndex].getMovePp();
        return `${move.name} (${ppLeft}/${ppMax} PP, pow:${move.power || "-"})`;
      }
    }
  } catch {
    // ignore
  }
  return `Move ${moveIndex}`;
}

function getPartyName(slot: number): string {
  try {
    const party = globalScene.getPlayerParty();
    if (slot < party.length) {
      const p = party[slot];
      const hpPct = Math.round((p.hp / p.getMaxHp()) * 100);
      return `${p.species?.name ?? "?"} Lv${p.level} (${hpPct}% HP)`;
    }
  } catch {
    // ignore
  }
  return `Slot ${slot}`;
}

/**
 * Build human-readable labels for every valid action in the current PhaseState.
 */
function buildActionLabels(state: PhaseState): ActionInfo[] {
  const actions: ActionInfo[] = [];

  for (const idx of state.validActions) {
    // Try phase-specific label first
    const phaseLabel = getPhaseSpecificLabel(idx, state.phase);
    if (phaseLabel !== null) {
      actions.push({ index: idx, label: phaseLabel });
      continue;
    }

    let label = `Action ${idx}`;

    // Fight -> Enemy (0-3)
    if (idx >= ACTION_FIGHT_ENEMY_START && idx < ACTION_FIGHT_ENEMY_START + MAX_MOVES) {
      label = `Fight: ${getMoveName(idx - ACTION_FIGHT_ENEMY_START)} -> ${getEnemyName(0)}`;
    }
    // Fight -> Enemy 2 (4-7)
    else if (idx >= ACTION_FIGHT_ENEMY2_START && idx < ACTION_FIGHT_ENEMY2_START + MAX_MOVES) {
      label = `Fight: ${getMoveName(idx - ACTION_FIGHT_ENEMY2_START)} -> ${getEnemyName(1)}`;
    }
    // Fight -> Ally (8-11)
    else if (idx >= ACTION_FIGHT_ALLY_START && idx < ACTION_FIGHT_ALLY_START + MAX_MOVES) {
      label = `Fight: ${getMoveName(idx - ACTION_FIGHT_ALLY_START)} -> ${getAllyName()}`;
    }
    // Switch (12-16)
    else if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
      label = `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
    }
    // Ball (17-21)
    else if (idx >= ACTION_BALL_START && idx < ACTION_BALL_START + 5) {
      label = `Throw: ${BALL_NAMES[idx - ACTION_BALL_START] ?? "Ball"}`;
    }
    // Run (22)
    else if (idx === ACTION_RUN) {
      label = "Run away";
    }
    // Tera -> Enemy (23-26)
    else if (idx >= ACTION_TERA_ENEMY_START && idx < ACTION_TERA_ENEMY_START + MAX_MOVES) {
      label = `Tera + ${getMoveName(idx - ACTION_TERA_ENEMY_START)} -> ${getEnemyName(0)}`;
    }
    // Tera -> Enemy 2 (27-30)
    else if (idx >= ACTION_TERA_ENEMY2_START && idx < ACTION_TERA_ENEMY2_START + MAX_MOVES) {
      label = `Tera + ${getMoveName(idx - ACTION_TERA_ENEMY2_START)} -> ${getEnemyName(1)}`;
    }
    // Tera -> Ally (31-34)
    else if (idx >= ACTION_TERA_ALLY_START && idx < ACTION_TERA_ALLY_START + MAX_MOVES) {
      label = `Tera + ${getMoveName(idx - ACTION_TERA_ALLY_START)} -> ${getAllyName()}`;
    }
    // Select reward (35-37)
    else if (idx >= ACTION_SELECT_REWARD_START && idx < ACTION_SELECT_REWARD_START + 3) {
      const ri = idx - ACTION_SELECT_REWARD_START;
      label = getRewardLabel(ri);
    }
    // Reroll (38)
    else if (idx === ACTION_REROLL) {
      label = getRerollLabel();
    }
    // Skip (39)
    else if (idx === ACTION_SKIP) {
      label = "Skip / Decline";
    }
    // Buy shop (40-51)
    else if (idx >= ACTION_BUY_SHOP_START && idx < ACTION_BUY_SHOP_START + 12) {
      const si = idx - ACTION_BUY_SHOP_START;
      label = getShopLabel(si);
    }
    // Party target (52-57)
    else if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
      label = `Apply to: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
    }

    actions.push({ index: idx, label });
  }

  return actions;
}

/**
 * For non-command phases where action indices carry phase-specific meaning,
 * return a label or null to fall through to the generic labeler.
 */
function getPhaseSpecificLabel(idx: number, phase: string): string | null {
  switch (phase) {
    case DecisionPhase.CHECK_SWITCH:
      if (idx === 0) return "Accept switch";
      if (idx === ACTION_SKIP) return "Decline switch";
      return null;

    case DecisionPhase.SWITCH:
      if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
        return `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
      }
      return null;

    case DecisionPhase.LEARN_MOVE:
      if (idx === ACTION_SKIP) return "Don't learn move";
      if (idx >= 0 && idx < MAX_MOVES) return `Replace move slot ${idx}`;
      return null;

    case DecisionPhase.GAME_OVER:
      return idx === 0 ? "Continue (retry)" : "Quit";

    case DecisionPhase.REVIVAL_BLESSING:
      if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
        return `Revive: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
      }
      return null;

    case DecisionPhase.SELECT_BIOME:
      return `Pick biome option ${idx}`;

    case DecisionPhase.MODIFIER_TARGET:
      if (idx === ACTION_SKIP) return "Cancel (back to items)";
      if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
        return `Apply to: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
      }
      return null;

    case DecisionPhase.MYSTERY_ENCOUNTER:
      return `Encounter option ${idx}`;

    case DecisionPhase.EVOLUTION:
    case DecisionPhase.FORM_CHANGE:
      return "Continue";

    default:
      return null;
  }
}

function getRewardLabel(rewardIndex: number): string {
  try {
    const mods = getAvailableModifiers();
    if (mods && rewardIndex < mods.rewards.length) {
      const r = mods.rewards[rewardIndex];
      return `Select reward ${rewardIndex}: ${r.name} [${TIER_NAMES[r.tier] ?? "?"}]`;
    }
  } catch {
    // ignore
  }
  return `Select reward ${rewardIndex}`;
}

function getRerollLabel(): string {
  try {
    const mods = getAvailableModifiers();
    if (mods) {
      return `Reroll modifiers (cost: $${mods.rerollCost})`;
    }
  } catch {
    // ignore
  }
  return "Reroll modifiers";
}

function getShopLabel(shopIndex: number): string {
  try {
    const mods = getAvailableModifiers();
    if (mods && shopIndex < mods.shop.length) {
      const s = mods.shop[shopIndex];
      return `Buy: ${s.name} ($${s.cost})`;
    }
  } catch {
    // ignore
  }
  return `Buy shop item ${shopIndex}`;
}

// ── Utility ───────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Setup Phases ──────────────────────────────────────────────────────

/** Phases that the bridge auto-handles (user won't see them since we wait for battle). */
const SETUP_PHASES = new Set<string>([
  DecisionPhase.TITLE,
  DecisionPhase.SELECT_GENDER,
  DecisionPhase.SELECT_STARTER,
  DecisionPhase.EVOLUTION,
  DecisionPhase.FORM_CHANGE,
]);

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
async function advanceWithAutoDismiss(router: PhaseRouter): Promise<PhaseState> {
  let running = true;

  // Background loop: press ACTION whenever stuck in MESSAGE mode
  const dismissLoop = (async () => {
    while (running) {
      await sleep(300);
      if (!running) break;
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
    return await router.advanceToNextDecision();
  } finally {
    running = false;
    // Let the dismiss loop finish its current iteration
    await sleep(0);
  }
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
    case DecisionPhase.TITLE: return "TitlePhase";
    case DecisionPhase.SELECT_GENDER: return "SelectGenderPhase";
    case DecisionPhase.SELECT_STARTER: return "SelectStarterPhase";
    case DecisionPhase.EVOLUTION: return "EvolutionPhase";
    case DecisionPhase.FORM_CHANGE: return "FormChangePhase";
    default: return null;
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

  // Step 2: Create PhaseRouter ASAP — must be before TitlePhase fires
  // so the setMode hook catches it. TitlePhase waits indefinitely for input,
  // so even if it fires before the hook, detectCurrentDecision() will find it.
  const router: PhaseRouter = createPhaseRouter({ verbose: true });

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
        state = await advanceWithAutoDismiss(router);
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
      } else if (SETUP_PHASES.has(state.phase) && handledSetupPhases.has(state.phase)) {
        // Already handled — wait a bit and retry
        console.log("[RL Bridge] Setup phase already handled, waiting for transition:", state.phase);
        await sleep(200);
        continue;
      }

      // Check for game over
      if (state.phase === DecisionPhase.GAME_OVER || router.isGameOver()) {
        const gameState = buildFullGameState(state, step);
        sendWS(ws, {
          type: "game_over",
          step,
          victory: router.isVictory(),
          gameState,
        });
        updateIndicator(
          indicator,
          `Game over (${router.isVictory() ? "victory" : "defeat"}) at step ${step}`,
          "rgba(100,100,0,0.8)",
        );
        break;
      }

      // Build state payload
      const actionLabels = buildActionLabels(state);
      const gameState = buildFullGameState(state, step);

      // Send state to Python
      sendWS(ws, {
        type: "state",
        step,
        phase: state.phase,
        actions: actionLabels,
        gameState,
        metadata: state.metadata,
      });

      // Update on-screen indicator
      updateIndicator(
        indicator,
        `Step ${step} | Wave ${(gameState as any).battle?.wave_index ?? "?"} | ${state.phase} | ${actionLabels.length} actions`,
      );

      // Wait for action from Python
      const action = await waitForAction(ws);

      // WebSocket closed while waiting
      if (action === -1) {
        console.warn("[RL Bridge] Lost connection while waiting for action");
        break;
      }

      // Validate action against mask and execute
      if (!state.actionMask[action]) {
        console.warn(
          `[RL Bridge] Invalid action ${action} for phase ${state.phase}. ` +
            `Valid: [${state.validActions.join(", ")}]. Using first valid action.`,
        );
        await router.executeAction(state.validActions[0] ?? 0);
      } else {
        await router.executeAction(action);
      }

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
  sendWS(ws, { type: "done", steps: step });

  updateIndicator(
    indicator,
    `Done (${step} steps)`,
    "rgba(100,100,0,0.8)",
  );

  // Cleanup
  router.destroy();
  console.log(`[RL Bridge] Session ended after ${step} steps`);
}

// ── Entry Point ───────────────────────────────────────────────────────

startBridge().catch((err) => {
  console.error("[RL Bridge] Fatal error:", err);
});
