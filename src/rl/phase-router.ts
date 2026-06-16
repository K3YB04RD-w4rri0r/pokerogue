/**
 * Phase Decision Router for the RL Framework.
 *
 * Detects which game phase is active, determines valid actions, and routes
 * RL agent decisions to the appropriate game handler. Replaces PhaseInterceptor's
 * Vitest-coupled polling with an event-driven, Promise-based approach.
 *
 * Key design:
 * - Hooks UI.prototype.setMode to detect when decision phases pause for input
 * - Hooks Phase.prototype.end to detect when non-setMode phases complete
 * - Provides a waitForDecision() API that resolves when a decision point is reached
 * - Each decision phase has a dedicated handler for action validation and execution
 *
 * Does NOT import from 'vitest'. Uses existing src/rl/modifier-api.ts for modifier handling.
 */

import { MAX_TERAS_PER_ARENA } from "#app/constants";
import { getGameMode } from "#app/game-mode";
import { globalScene } from "#app/global-scene";
import { Phase } from "#app/phase";
import { allMoves } from "#data/data-lists";
import { BattleType } from "#enums/battle-type";
import { BattlerIndex } from "#enums/battler-index";
import { BiomeId } from "#enums/biome-id";
import { Button } from "#enums/buttons";
import { Command } from "#enums/command";
import { GameModes } from "#enums/game-modes";
import { MoveTarget } from "#enums/move-target";
import { MoveUseMode } from "#enums/move-use-mode";
import { SwitchType } from "#enums/switch-type";
import { UiMode } from "#enums/ui-mode";
import { getMoveTargets } from "#moves/move-utils";
import type { CommandPhase } from "#phases/command-phase";
import { EncounterPhase } from "#phases/encounter-phase";
import { SelectStarterPhase } from "#phases/select-starter-phase";
import type { SelectTargetPhase } from "#phases/select-target-phase";
import {
  getAvailableModifiers,
  getEligiblePokemon,
  rerollModifiers,
  selectRewardModifier,
  selectShopModifier,
  skipModifiers,
} from "#rl/modifier-api";
import {
  ACTION_BALL_START,
  ACTION_BUY_SHOP_START,
  ACTION_FIGHT_ALLY_START,
  ACTION_FIGHT_ENEMY_START,
  ACTION_FIGHT_ENEMY2_START,
  ACTION_PARTY_TARGET_START,
  ACTION_REROLL,
  ACTION_RUN,
  ACTION_SELECT_REWARD_START,
  ACTION_SKIP,
  ACTION_SPACE_SIZE,
  ACTION_SWITCH_START,
  ACTION_TERA_ALLY_START,
  ACTION_TERA_ENEMY_START,
  ACTION_TERA_ENEMY2_START,
  MAX_MOVES,
  MAX_PARTY_SIZE,
  MAX_REWARD_OPTIONS,
  MAX_SHOP_OPTIONS,
  NUM_POKEBALL_TYPES,
} from "#rl/spaces";
import { generateStarters } from "#test/test-utils/game-manager-utils";
import { UI } from "#ui/ui";
import { canTerastallize } from "#utils/pokemon-utils";

// ─── Decision Phase Enum ──────────────────────────────────────────────

/** All decision phases the RL agent may encounter. */
export enum DecisionPhase {
  COMMAND = "command",
  SELECT_TARGET = "target",
  SELECT_MODIFIER = "modifier",
  MODIFIER_TARGET = "modifier_target",
  SWITCH = "switch",
  CHECK_SWITCH = "check_switch",
  LEARN_MOVE = "learn_move",
  EVOLUTION = "evolution",
  SELECT_STARTER = "starter",
  MYSTERY_ENCOUNTER = "mystery",
  GAME_OVER = "game_over",
  SELECT_BIOME = "select_biome",
  REVIVAL_BLESSING = "revival_blessing",
  FORM_CHANGE = "form_change",
  TITLE = "title",
  SELECT_GENDER = "select_gender",
  UNKNOWN = "unknown",
}

// ─── Phase State ──────────────────────────────────────────────────────

/** Describes the current decision state: which phase, what actions are valid. */
export interface PhaseState {
  /** Which decision phase is active */
  phase: DecisionPhase;
  /** Indices of valid actions in the action space */
  validActions: number[];
  /** Boolean mask over the full ACTION_SPACE_SIZE action space */
  actionMask: boolean[];
  /** Phase-specific metadata for the RL agent */
  metadata: Record<string, unknown>;
}

// ─── Phase Router Interface ───────────────────────────────────────────

export interface PhaseRouter {
  /** Check if the game is currently at a decision point */
  isAtDecisionPoint(): boolean;

  /** Get the current phase state with valid actions, or null if not at a decision point */
  getCurrentPhaseState(): PhaseState | null;

  /** Execute an action for the current decision phase */
  executeAction(action: number): Promise<void>;

  /** Advance the game to the next decision point (skip non-decision phases) */
  advanceToNextDecision(): Promise<PhaseState>;

  /** Register a callback invoked each time a decision point is reached */
  onDecision(callback: (state: PhaseState) => void): void;

  /** Return true if the game has ended (game over or victory) */
  isGameOver(): boolean;

  /** Return true if the game ended in a victory */
  isVictory(): boolean;

  /** Drain accumulated info messages (e.g. from auto-skipped phases like IV Scanner) */
  drainInfoMessages(): string[];

  /** Clean up hooks and restore prototypes */
  destroy(): void;
}

// ─── Internal Types ───────────────────────────────────────────────────

/** Pending resolve/reject for the decision wait promise */
interface PendingDecision {
  resolve: (state: PhaseState) => void;
  reject: (error: Error) => void;
}

// ─── Phase Name <-> DecisionPhase Mapping ─────────────────────────────

/**
 * Map from phase phaseName strings to DecisionPhase enum values.
 * Not all phases here are "endBySetMode" - some end via callbacks.
 */
const PHASE_NAME_TO_DECISION: Record<string, DecisionPhase> = {
  CommandPhase: DecisionPhase.COMMAND,
  SelectTargetPhase: DecisionPhase.SELECT_TARGET,
  SelectModifierPhase: DecisionPhase.SELECT_MODIFIER,
  SwitchPhase: DecisionPhase.SWITCH,
  CheckSwitchPhase: DecisionPhase.CHECK_SWITCH,
  LearnMovePhase: DecisionPhase.LEARN_MOVE,
  EvolutionPhase: DecisionPhase.EVOLUTION,
  SelectStarterPhase: DecisionPhase.SELECT_STARTER,
  MysteryEncounterPhase: DecisionPhase.MYSTERY_ENCOUNTER,
  PostMysteryEncounterPhase: DecisionPhase.MYSTERY_ENCOUNTER,
  GameOverPhase: DecisionPhase.GAME_OVER,
  SelectBiomePhase: DecisionPhase.SELECT_BIOME,
  RevivalBlessingPhase: DecisionPhase.REVIVAL_BLESSING,
  FormChangePhase: DecisionPhase.FORM_CHANGE,
  TitlePhase: DecisionPhase.TITLE,
  SelectGenderPhase: DecisionPhase.SELECT_GENDER,
};

/**
 * Phases where setMode() indicates the phase is paused and waiting for input.
 * For these phases, when setMode fires, the phase has NOT called this.end() yet.
 */
const END_BY_SET_MODE_PHASES = new Set<string>([
  "TitlePhase",
  "SelectGenderPhase",
  "CommandPhase",
  "SelectStarterPhase",
  "SelectModifierPhase",
  "MysteryEncounterPhase",
  "PostMysteryEncounterPhase",
]);

/**
 * UiModes that indicate a decision point (the UI is waiting for player input).
 * Used to filter noise from non-decision setMode calls.
 */
const DECISION_UI_MODES = new Set<UiMode>([
  UiMode.COMMAND,
  UiMode.FIGHT,
  UiMode.TARGET_SELECT,
  UiMode.MODIFIER_SELECT,
  UiMode.PARTY,
  UiMode.CONFIRM,
  UiMode.SUMMARY,
  UiMode.MYSTERY_ENCOUNTER,
  UiMode.OPTION_SELECT,
  UiMode.STARTER_SELECT,
  UiMode.TITLE,
  UiMode.BALL,
  UiMode.EVOLUTION_SCENE,
]);

/**
 * Phases that are "callback-based" - they pause for input via a callback argument
 * passed to setMode, and end via this.end() called from within that callback.
 * We detect these by their setMode calls (not by endBySetMode).
 */
const CALLBACK_DECISION_PHASES = new Set<string>([
  "SelectTargetPhase",
  "SwitchPhase",
  "CheckSwitchPhase",
  "LearnMovePhase",
  "RevivalBlessingPhase",
  "SelectBiomePhase",
  "GameOverPhase",
]);

/**
 * Phases that are NOT decision points but may block on UI input (showText prompt,
 * CONFIRM dialog, or special UI mode). The polling fallback auto-skips them by
 * pressing ACTION or selecting "yes/skip" on CONFIRM dialogs.
 *
 * These bypass our setMode hook because they use setModeWithoutClear,
 * setModeForceTransition, setOverlayMode, or showText with prompt=true.
 */
const AUTO_SKIP_PHASES = new Set<string>([
  "EggLapsePhase", // PH2: CONFIRM dialog for skip preference
  "EggHatchPhase", // PH3: EGG_HATCH_SCENE requires ACTION
  "EggSummaryPhase", // PH4: EGG_HATCH_SUMMARY requires CANCEL
  "EndCardPhase", // PH5: Classic victory card, waits for ACTION
  "UnlockPhase", // PH6: "Unlocked X" message, waits for ACTION
  "EvolutionPhase", // PH7: EVOLUTION_SCENE + possible pause CONFIRM
  "FormChangePhase", // Extends EvolutionPhase, same patterns
  "ModifierRewardPhase", // PH8: "Obtained X" showText with prompt
  "RibbonModifierRewardPhase", // PH9: Same pattern as PH8
  "GameOverModifierRewardPhase", // PH10: Same pattern as PH8
  "MoneyRewardPhase", // PH11: "Won X money" showText with prompt
  "TrainerVictoryPhase", // PH12: Trainer defeat dialogue
  "LevelCapPhase", // PH13: "Level cap raised" message
  // Full-party capture: the "Your party is full. Release a Pokemon?" showText
  // prompt blocks in MESSAGE mode BEFORE the CONFIRM dialog the dedicated
  // setMode handlers cover. The poll advances the prompt; the CONFIRM branch
  // below routes to the release option (cursor 2).
  "AttemptCapturePhase",
]);

// ─── Implementation ───────────────────────────────────────────────────

export function createPhaseRouter(options?: { verbose?: boolean }): PhaseRouter {
  const verbose = options?.verbose ?? false;

  // ── State ──────────────────────────────────────────────────────────
  let pendingDecision: PendingDecision | null = null;
  let currentPhaseState: PhaseState | null = null;
  let decisionCallbacks: Array<(state: PhaseState) => void> = [];
  let gameOverFlag = false;
  let victoryFlag = false;
  let destroyed = false;

  // Track the last setMode call to detect decision phases for callback-based phases
  let lastSetModePhase: string | null = null;
  let lastSetModeUiMode: UiMode | null = null;

  // Two-step modifier targeting: stores the selected modifier while agent picks a Pokemon
  let pendingModifierAction: {
    source: "reward" | "shop";
    index: number;
    cost: number;
  } | null = null;

  // Timeout handle for the advance loop
  let advanceTimeoutId: ReturnType<typeof setTimeout> | null = null;

  // Re-entry guard: prevent executeTitleAction from being called multiple times.
  // In the browser, initBattle() is async (loads real assets). While assets load,
  // TitlePhase stays current and the bridge's setup loop re-detects TITLE,
  // calling executeTitleAction() again. Each call adds 3 starters to the party.
  let titleActionExecuted = false;

  // ── Info Messages ─────────────────────────────────────────────────
  // Accumulated info strings from auto-skipped phases (e.g. IV Scanner).
  // Drained by cli.ts between decisions and forwarded to the JSON protocol.
  let infoMessages: string[] = [];

  // ── Modifier Handler Monkey-Patch ──────────────────────────────────
  // Installed lazily on first modifier phase access. Patches show() to
  // kill lingering tweens/timers from a previous show() call, preventing
  // the alternating-wave shop rendering bug (see cleanupModifierUI docs).
  let modifierHandlerPatched = false;

  // ── Prototype Hooks ────────────────────────────────────────────────
  const originalSetMode = UI.prototype.setMode;
  const originalPhaseEnd = Phase.prototype.end;

  /**
   * Hooked setMode: when a decision phase calls setMode with a decision UiMode,
   * we detect that a decision point has been reached and resolve the pending promise.
   */
  function hookedSetMode(this: UI, mode: UiMode, ...args: unknown[]): Promise<void> {
    const ret = originalSetMode.apply(this, [mode, ...args]);

    if (destroyed) {
      return ret;
    }

    const currentPhase = globalScene.phaseManager?.getCurrentPhase();
    if (!currentPhase) {
      return ret;
    }

    const phaseName = currentPhase.phaseName;

    // Track every setMode call for context
    lastSetModePhase = phaseName;
    lastSetModeUiMode = mode;

    // Check if this is a decision point
    const isEndBySetMode = END_BY_SET_MODE_PHASES.has(phaseName);
    const isCallbackPhase = CALLBACK_DECISION_PHASES.has(phaseName);
    const isDecisionMode = DECISION_UI_MODES.has(mode);

    if (verbose) {
      console.log(
        `[PhaseRouter] setMode: phase=${phaseName} mode=${UiMode[mode]} isDecision=${isDecisionMode} endBySetMode=${isEndBySetMode} callbackPhase=${isCallbackPhase}`,
      );
    }

    // Auto-handle AttemptCapturePhase (full party) — the game shows a multi-step
    // UI when catching a Pokemon with a full party of 6:
    //   CONFIRM ("fullParty") → PARTY (RELEASE mode) → release selected Pokemon
    // Without handling, the game hangs permanently (P0 bug).
    if (phaseName === "AttemptCapturePhase") {
      // CONFIRM ("fullParty") and PARTY (RELEASE mode) are driven by the
      // polling state machine in tryAutoSkipPhase — the flow needs multiple
      // inputs at UI states that open asynchronously (options submenu, goodbye
      // prompt), so per-setMode setTimeout chains race the UI. The poll reads
      // the actual UI state each 50ms tick and acts on it.
      if (mode === UiMode.CONFIRM || mode === UiMode.PARTY) {
        return ret;
      }

      if (mode === UiMode.SUMMARY || mode === UiMode.POKEDEX_PAGE) {
        // Summary/Pokedex view during capture — auto-dismiss by pressing CANCEL
        setTimeout(() => {
          const phase = globalScene.phaseManager.getCurrentPhase();
          if (phase?.phaseName !== "AttemptCapturePhase") {
            return;
          }
          const handler = globalScene.ui.getHandler();
          if (!handler) {
            return;
          }
          (handler as { processInput(button: Button): boolean }).processInput(Button.CANCEL);
          if (verbose) {
            console.log(`[PhaseRouter] AttemptCapturePhase: auto-dismissed ${UiMode[mode]}`);
          }
        }, 0);
        return ret;
      }
    }

    // Auto-skip ScanIvsPhase (IV Scanner) — not a decision for the RL agent.
    // The phase shows a CONFIRM dialog asking to view IVs. We auto-decline
    // and log all enemy IVs that exceed the player's dex baseline.
    if (phaseName === "ScanIvsPhase" && mode === UiMode.CONFIRM) {
      const statNames = ["HP", "ATK", "DEF", "SP.ATK", "SP.DEF", "SPD"];
      for (const enemy of globalScene.getEnemyField()) {
        if (!enemy) {
          continue;
        }
        const enemyIvs = enemy.ivs;
        const dexIvs = globalScene.gameData.dexData[enemy.species.getRootSpeciesId()]?.ivs;
        const ivDetails: string[] = [];
        for (let s = 0; s < 6; s++) {
          const label = statNames[s];
          const eiv = enemyIvs[s];
          const div = dexIvs ? dexIvs[s] : 0;
          if (eiv > div) {
            ivDetails.push(`${label}=${eiv}${eiv === 31 ? "(perfect)" : ""}`);
          }
        }
        const summary = ivDetails.length > 0 ? ivDetails.join(", ") : "none above baseline";
        infoMessages.push(`IVScanner ${enemy.name}: ${summary}`);
      }
      if (verbose) {
        for (const msg of infoMessages) {
          console.log(`[PhaseRouter] ${msg}`);
        }
      }
      // Auto-decline after the current call stack completes
      setTimeout(() => {
        const phase = globalScene.phaseManager.getCurrentPhase();
        if (phase?.phaseName === "ScanIvsPhase") {
          globalScene.ui.setMode(UiMode.MESSAGE);
          phase.end();
        }
      }, 0);
      return ret;
    }

    if ((isEndBySetMode || isCallbackPhase) && isDecisionMode) {
      const decisionPhase = PHASE_NAME_TO_DECISION[phaseName];
      if (decisionPhase !== undefined) {
        // Small delay to let the UI handler fully initialize
        // (handlers set active=true and awaitingActionInput=true asynchronously)
        setTimeout(() => {
          if (!destroyed) {
            resolveDecisionPoint(decisionPhase, phaseName, mode);
          }
        }, 0);
      }
    }

    return ret;
  }

  /**
   * Hooked Phase.end: detect game over conditions.
   */
  function hookedPhaseEnd(this: Phase): void {
    const phaseName = this.phaseName;

    // Detect game over
    if (phaseName === "GameOverPhase") {
      gameOverFlag = true;
      // Check if it was a victory
      const phase = this as unknown as { isVictory?: boolean };
      if ("isVictory" in phase) {
        victoryFlag = !!phase.isVictory;
      }
    }

    originalPhaseEnd.apply(this);
  }

  // Install hooks
  UI.prototype.setMode = function (mode: UiMode, ...args: unknown[]): Promise<void> {
    return hookedSetMode.call(this, mode, ...args);
  };
  Phase.prototype.end = function (): void {
    hookedPhaseEnd.call(this);
  };

  // Trampoline the phase pump to prevent a headless stack overflow. The game's
  // pump is recursive: Phase.end() -> shiftPhase() -> startCurrentPhase() ->
  // phase.start() -> ... -> phase.end() -> shiftPhase() -> ... In the browser,
  // tween/animation gaps make phases async so the stack unwinds between them.
  // In headless the mock tweens fire onComplete SYNCHRONOUSLY, so an entire
  // queued phase chain runs as one unbroken recursion — and a deep wave's long
  // chain (berries + weather + per-Pokemon turn phases) overflows V8's stack.
  // Flatten it: a re-entrant shiftPhase (a phase ending mid-pump) just requests
  // another iteration and unwinds; the outermost call drives the chain in a
  // loop at a flat stack depth. Phase ORDER is unchanged, so RNG/determinism
  // are unaffected. Installed once on the reused phaseManager (idempotent);
  // semantically transparent, so no teardown is required.
  // RL_NO_TRAMPOLINE=1 disables this (recursive pump restored) — used to prove
  // the trampoline is behaviour-neutral by diffing phase logs + observations
  // against the unpatched pump at shallow waves (where the unpatched pump does
  // not yet overflow).
  // The trampoline is only needed in the headless Node env, where mock tweens
  // fire onComplete synchronously so the recursive phase pump can overflow the
  // stack. In the browser (rendered RL mode) tweens are async — the pump unwinds
  // between phases, so there is no overflow — and `process` is undefined there,
  // so we must not touch it (doing so threw "process is not defined" and stalled
  // the bridge on the gender-select screen).
  const inNodeEnv = typeof process !== "undefined" && process.env != null;
  const pm = globalScene.phaseManager as unknown as { shiftPhase: () => void; __rlTrampolined?: boolean } | undefined;
  if (pm && !pm.__rlTrampolined && inNodeEnv && !process.env.RL_NO_TRAMPOLINE) {
    const originalShift = pm.shiftPhase.bind(pm);
    let pumping = false;
    let pending = false;
    pm.shiftPhase = function rlTrampolinedShiftPhase(): void {
      if (pumping) {
        pending = true; // re-entrant (phase ended during the pump) — defer, unwind
        return;
      }
      pumping = true;
      try {
        originalShift();
        while (pending) {
          pending = false;
          originalShift();
        }
      } finally {
        pumping = false;
      }
    };
    pm.__rlTrampolined = true;
  }

  // ── Decision Resolution ────────────────────────────────────────────

  function resolveDecisionPoint(decision: DecisionPhase, phaseName: string, uiMode: UiMode): void {
    // If a modifier target is pending and the hook fires for SelectModifierPhase,
    // present MODIFIER_TARGET instead of SELECT_MODIFIER
    if (pendingModifierAction && decision === DecisionPhase.SELECT_MODIFIER) {
      decision = DecisionPhase.MODIFIER_TARGET;
    }

    const state = buildPhaseState(decision, phaseName, uiMode);
    if (!state) {
      return;
    }

    // Only resolve if there's something actionable (at least one valid action)
    // Exception: GAME_OVER and setup phases always resolve
    const isSetupOrTerminal = [
      DecisionPhase.GAME_OVER,
      DecisionPhase.TITLE,
      DecisionPhase.SELECT_GENDER,
      DecisionPhase.SELECT_STARTER,
      DecisionPhase.EVOLUTION,
    ].includes(decision);

    if (state.validActions.length === 0 && !isSetupOrTerminal) {
      return;
    }

    if (verbose) {
      console.log(
        `[PhaseRouter] Decision resolved: ${decision} validActions=${state.validActions.length} pending=${!!pendingDecision}`,
      );
    }

    currentPhaseState = state;

    // Notify listeners
    for (const cb of decisionCallbacks) {
      try {
        cb(state);
      } catch (e) {
        console.error("[PhaseRouter] Decision callback error:", e);
      }
    }

    // Resolve pending promise
    if (pendingDecision) {
      const pd = pendingDecision;
      pendingDecision = null;
      pd.resolve(state);
    }
  }

  // ── Phase State Builders ───────────────────────────────────────────

  function buildPhaseState(decision: DecisionPhase, phaseName: string, uiMode: UiMode): PhaseState | null {
    // TitlePhase stays current while initBattle() runs asynchronously, so it
    // can be re-detected by the poll a timing-dependent number of times.
    // Present TITLE exactly once per episode — otherwise the number of
    // decision steps (and any seeded action RNG consuming them) varies
    // between identical runs, breaking determinism.
    if (decision === DecisionPhase.TITLE && titleActionExecuted) {
      return null;
    }

    const metadata: Record<string, unknown> = {
      phaseName,
      uiMode,
      uiModeName: UiMode[uiMode],
    };

    let actionMask: boolean[];

    switch (decision) {
      case DecisionPhase.COMMAND:
        actionMask = buildCommandActionMask(metadata);
        break;
      case DecisionPhase.SELECT_TARGET:
        actionMask = buildSelectTargetActionMask(metadata);
        break;
      case DecisionPhase.SELECT_MODIFIER:
        actionMask = buildModifierActionMask(metadata);
        break;
      case DecisionPhase.MODIFIER_TARGET:
        actionMask = buildModifierTargetActionMask(metadata);
        break;
      case DecisionPhase.SWITCH:
        actionMask = buildSwitchActionMask(metadata, false);
        break;
      case DecisionPhase.CHECK_SWITCH:
        actionMask = buildCheckSwitchActionMask(metadata);
        break;
      case DecisionPhase.LEARN_MOVE:
        actionMask = buildLearnMoveActionMask(metadata);
        break;
      case DecisionPhase.REVIVAL_BLESSING:
        actionMask = buildRevivalBlessingActionMask(metadata);
        break;
      case DecisionPhase.SELECT_BIOME:
        actionMask = buildSelectBiomeActionMask(metadata);
        break;
      case DecisionPhase.GAME_OVER:
        actionMask = buildGameOverActionMask(metadata);
        break;
      case DecisionPhase.MYSTERY_ENCOUNTER:
        actionMask = buildMysteryEncounterActionMask(metadata);
        break;
      // Auto-handled phases: accept first/only option
      case DecisionPhase.EVOLUTION:
      case DecisionPhase.FORM_CHANGE:
      case DecisionPhase.TITLE:
      case DecisionPhase.SELECT_GENDER:
      case DecisionPhase.SELECT_STARTER:
        actionMask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
        // Mark action 0 as valid as a placeholder for "proceed"
        actionMask[0] = true;
        metadata.autoHandle = true;
        break;
      default:
        actionMask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
        actionMask[0] = true;
        metadata.autoHandle = true;
        break;
    }

    const validActions: number[] = [];
    for (let i = 0; i < actionMask.length; i++) {
      if (actionMask[i]) {
        validActions.push(i);
      }
    }

    return { phase: decision, validActions, actionMask, metadata };
  }

  // ── Command Phase Mask ─────────────────────────────────────────────

  function buildCommandActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (!currentPhase?.is("CommandPhase")) {
      return mask;
    }

    const commandPhase = currentPhase as CommandPhase;
    const pokemon = commandPhase.getPokemon();
    if (!pokemon) {
      return mask;
    }

    const moveset = pokemon.getMoveset(false);
    const battle = globalScene.currentBattle;
    const isDouble = battle?.double ?? false;
    const allEnemies = globalScene.getEnemyField() ?? [];
    const enemy0Active = allEnemies.length > 0 && allEnemies[0]?.isActive();
    const enemy1Active = allEnemies.length > 1 && allEnemies[1]?.isActive();
    const playerParty = globalScene.getPlayerParty() ?? [];
    const playerField = globalScene.getPlayerField()?.filter(p => p?.isActive()) ?? [];

    metadata.fieldIndex = commandPhase.getFieldIndex();
    metadata.pokemonSpecies = pokemon.species?.name;
    metadata.isDouble = isDouble;

    const isTrainerBattle = battle?.battleType === BattleType.TRAINER;
    const isEndBiome = globalScene.arena?.biomeType === BiomeId.END;
    const trappedMessages: string[] = [];
    const isTrapped = pokemon.isTrapped(trappedMessages);

    // Move targets that require the player to choose a specific target
    const SINGLE_TARGET_ENEMY = new Set([MoveTarget.NEAR_ENEMY, MoveTarget.NEAR_OTHER, MoveTarget.OTHER]);

    const anyEnemyActive = enemy0Active || enemy1Active;

    // Moves in FIGHT_ENEMY (0-3) — the "default" move slot
    // Single-target moves: only if enemy slot 0 is alive
    // Multi-target/self-target/field moves: if ANY enemy is alive
    if (anyEnemyActive) {
      for (let i = 0; i < MAX_MOVES && i < moveset.length; i++) {
        const move = moveset[i];
        if (move) {
          const [usable] = move.isUsable(pokemon);
          if (usable) {
            const moveData = move.getMove();
            if (SINGLE_TARGET_ENEMY.has(moveData.moveTarget)) {
              // Single-target: only when enemy slot 0 is alive
              if (enemy0Active) {
                mask[ACTION_FIGHT_ENEMY_START + i] = true;
              }
            } else {
              // Multi-target, self-target, field effect: always available
              mask[ACTION_FIGHT_ENEMY_START + i] = true;
            }
          }
        }
      }
    }

    // Moves targeting ENEMY slot 1 (4-7) — only single-target moves, only if enemy slot 1 alive
    if (isDouble && enemy1Active) {
      for (let i = 0; i < MAX_MOVES && i < moveset.length; i++) {
        const move = moveset[i];
        if (move) {
          const [usable] = move.isUsable(pokemon);
          const moveData = move.getMove();
          if (usable && SINGLE_TARGET_ENEMY.has(moveData.moveTarget)) {
            mask[ACTION_FIGHT_ENEMY2_START + i] = true;
          }
        }
      }
    }

    // Moves targeting ally (8-11) - doubles only, single-target ally moves
    if (isDouble) {
      for (let i = 0; i < MAX_MOVES && i < moveset.length; i++) {
        const move = moveset[i];
        if (move) {
          const [usable] = move.isUsable(pokemon);
          const moveData = move.getMove();
          const canTargetAlly =
            moveData.moveTarget === MoveTarget.NEAR_ALLY
            || moveData.moveTarget === MoveTarget.USER_OR_NEAR_ALLY
            || moveData.moveTarget === MoveTarget.ALLY;
          if (usable && canTargetAlly) {
            mask[ACTION_FIGHT_ALLY_START + i] = true;
          }
        }
      }
    }

    // Switch (12-16): party slots 1-5
    if (!isTrapped) {
      const activeIds = new Set(playerField.map(p => p.id));
      let slotIdx = 0;
      for (let i = 1; i < playerParty.length && slotIdx < 5; i++) {
        const p = playerParty[i];
        if (!p.isFainted() && !activeIds.has(p.id)) {
          mask[ACTION_SWITCH_START + slotIdx] = true;
        }
        slotIdx++;
      }
    }

    // Ball (17-21)
    const canCatch = !isTrainerBattle && !isDouble && (enemy0Active || enemy1Active);
    if (canCatch) {
      const counts = globalScene.pokeballCounts ?? {};
      for (let i = 0; i < NUM_POKEBALL_TYPES; i++) {
        if ((counts[i] ?? 0) > 0) {
          mask[ACTION_BALL_START + i] = true;
        }
      }
    }

    // Run (22)
    if (!isTrainerBattle && !isEndBiome && !isTrapped) {
      mask[ACTION_RUN] = true;
    }

    // Tera (23-34): mirror the game's own command-time gate
    // (CommandUiHandler.canTera, command-ui-handler.ts:196) EXACTLY. It requires
    // a Tera Orb (TerastallizeAccessModifier, checked via canTerastallize) plus
    // an unused arena tera, and accounts for a tera already planned by the lead
    // in a double battle (plannedTera). Previously this only checked
    // isTerastallized, so the mask wrongly offered Tera from wave 1 with no orb.
    const fieldIndex = commandPhase.getFieldIndex();
    const currentTeras = globalScene.arena.playerTerasUsed;
    const plannedTera = Number(battle?.preTurnCommands?.[0]?.command === Command.TERA && fieldIndex > 0);
    const canTera = pokemon.isPlayer() && canTerastallize(pokemon);
    if (canTera && currentTeras + plannedTera < MAX_TERAS_PER_ARENA) {
      for (let i = 0; i < MAX_MOVES; i++) {
        if (mask[ACTION_FIGHT_ENEMY_START + i]) {
          mask[ACTION_TERA_ENEMY_START + i] = true;
        }
        if (mask[ACTION_FIGHT_ENEMY2_START + i]) {
          mask[ACTION_TERA_ENEMY2_START + i] = true;
        }
        if (mask[ACTION_FIGHT_ALLY_START + i]) {
          mask[ACTION_TERA_ALLY_START + i] = true;
        }
      }
    }

    return mask;
  }

  // ── Select Target Phase Mask ───────────────────────────────────────

  function buildSelectTargetActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (!currentPhase?.is("SelectTargetPhase")) {
      return mask;
    }

    const selectTargetPhase = currentPhase as SelectTargetPhase;
    const fieldIndex = (selectTargetPhase as unknown as { fieldIndex: number }).fieldIndex;
    const turnCommand = globalScene.currentBattle.turnCommands[fieldIndex];
    const moveId = turnCommand?.move?.move;

    if (moveId !== undefined) {
      const pokemon = globalScene.getPlayerField()[fieldIndex];
      if (pokemon) {
        const moveTargets = getMoveTargets(pokemon, moveId);
        metadata.moveTargets = moveTargets.targets;
        metadata.multiple = moveTargets.multiple;

        // For multi-target moves, any single target is valid
        // Map BattlerIndex values to action indices
        for (const target of moveTargets.targets) {
          switch (target) {
            case BattlerIndex.ENEMY:
              // Use action 0 (FIGHT_ENEMY move 0) as proxy for "target enemy 0"
              mask[ACTION_FIGHT_ENEMY_START] = true;
              break;
            case BattlerIndex.ENEMY_2:
              mask[ACTION_FIGHT_ENEMY2_START] = true;
              break;
            case BattlerIndex.PLAYER:
              mask[ACTION_FIGHT_ALLY_START] = true;
              break;
            case BattlerIndex.PLAYER_2:
              mask[ACTION_FIGHT_ALLY_START + 1] = true;
              break;
          }
        }
      }
    }

    return mask;
  }

  // ── Modifier Phase Mask ────────────────────────────────────────────

  function buildModifierActionMask(metadata: Record<string, unknown>): boolean[] {
    // Lazily install the show() monkey-patch on first modifier phase
    patchModifierHandler();

    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    const modifiers = getAvailableModifiers();
    if (!modifiers) {
      // Fallback: just allow skip
      mask[ACTION_SKIP] = true;
      return mask;
    }

    metadata.rewardCount = modifiers.rewards.length;
    metadata.shopCount = modifiers.shop.length;
    metadata.canReroll = modifiers.canReroll;
    metadata.money = modifiers.money;

    // Reward selection (35-37)
    for (let i = 0; i < MAX_REWARD_OPTIONS && i < modifiers.rewards.length; i++) {
      mask[ACTION_SELECT_REWARD_START + i] = true;
    }

    // Reroll (38)
    if (modifiers.canReroll) {
      mask[ACTION_REROLL] = true;
    }

    // Skip (39) - always available
    mask[ACTION_SKIP] = true;

    // Shop items (40-51)
    const money = globalScene.money ?? 0;
    for (let i = 0; i < MAX_SHOP_OPTIONS && i < modifiers.shop.length; i++) {
      if (money >= modifiers.shop[i].cost) {
        mask[ACTION_BUY_SHOP_START + i] = true;
      }
    }

    return mask;
  }

  // ── Modifier Target Phase Mask ──────────────────────────────────────

  function buildModifierTargetActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);

    if (!pendingModifierAction) {
      mask[ACTION_SKIP] = true;
      return mask;
    }

    const modifiers = getAvailableModifiers();
    if (!modifiers) {
      mask[ACTION_SKIP] = true;
      return mask;
    }

    const { source, index } = pendingModifierAction;
    const items = source === "reward" ? modifiers.rewards : modifiers.shop;
    if (index >= items.length) {
      mask[ACTION_SKIP] = true;
      return mask;
    }

    const modifierType = items[index].raw.type;
    const eligible = getEligiblePokemon(modifierType);

    for (let i = 0; i < MAX_PARTY_SIZE; i++) {
      if (eligible[i]) {
        mask[ACTION_PARTY_TARGET_START + i] = true;
      }
    }

    // Allow cancellation — return to modifier select
    mask[ACTION_SKIP] = true;

    metadata.source = source;
    metadata.modifierIndex = index;
    metadata.modifierName = modifierType.name;
    metadata.eligible = eligible;

    return mask;
  }

  // ── Switch Phase Mask ──────────────────────────────────────────────

  function buildSwitchActionMask(metadata: Record<string, unknown>, isFaintSwitch: boolean): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);

    const playerParty = globalScene.getPlayerParty() ?? [];
    const playerField = globalScene.getPlayerField()?.filter(p => p?.isActive()) ?? [];
    const activeIds = new Set(playerField.map(p => p.id));

    metadata.partySize = playerParty.length;
    metadata.isFaintSwitch = isFaintSwitch;

    // Map party slots to switch actions (12-16 for slots 1-5)
    // For switch phase, we allow selecting any non-fainted, non-active party member
    let slotIdx = 0;
    for (let i = 0; i < playerParty.length && slotIdx < 5; i++) {
      const p = playerParty[i];
      if (!p.isFainted() && !activeIds.has(p.id)) {
        // Use ACTION_SWITCH_START range: party slot i maps to switch action
        if (i > 0) {
          mask[ACTION_SWITCH_START + (i - 1)] = true;
        }
      }
      if (i > 0) {
        slotIdx++;
      }
    }

    return mask;
  }

  // ── CheckSwitch Phase Mask ─────────────────────────────────────────

  function buildCheckSwitchActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    // Binary choice: yes (accept switch) or no (decline)
    // Map to action 0 = yes, action 1 = no (using first two action slots)
    // Actually, for simplicity, use ACTION_SKIP (39) as "decline" and ACTION_FIGHT_ENEMY_START (0) as "accept"
    mask[ACTION_FIGHT_ENEMY_START] = true; // Yes (switch)
    mask[ACTION_SKIP] = true; // No (decline)
    metadata.isCheckSwitch = true;
    return mask;
  }

  // ── LearnMove Phase Mask ───────────────────────────────────────────

  function buildLearnMoveActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);

    // Extract move name metadata from the LearnMovePhase
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (currentPhase?.is("LearnMovePhase")) {
      const phase = currentPhase as unknown as { moveId: number; getPokemon(): any };
      const newMove = allMoves[phase.moveId];
      const pokemon = phase.getPokemon();
      const moveset = pokemon.getMoveset();

      metadata.newMoveName = newMove?.name ?? "???";
      metadata.currentMoveNames = moveset.map((m: any) => m?.getMove()?.name ?? "???");
    }

    // Actions 0-3: replace move at slot 0-3
    // ACTION_SKIP (39): don't learn the move
    for (let i = 0; i < MAX_MOVES; i++) {
      mask[i] = true;
    }
    mask[ACTION_SKIP] = true;
    metadata.isLearnMove = true;
    return mask;
  }

  // ── Revival Blessing Phase Mask ────────────────────────────────────

  function buildRevivalBlessingActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    const playerParty = globalScene.getPlayerParty() ?? [];

    // Allow selecting fainted party members
    for (let i = 0; i < playerParty.length && i < MAX_PARTY_SIZE; i++) {
      if (playerParty[i].isFainted()) {
        mask[ACTION_PARTY_TARGET_START + i] = true;
      }
    }

    metadata.isRevivalBlessing = true;
    return mask;
  }

  // ── SelectBiome Phase Mask ─────────────────────────────────────────

  function buildSelectBiomeActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    // Read the actual option count and labels from the UI handler (typically 2-3 biome links)
    let optionCount = 2; // safe default
    const biomeNames: string[] = [];
    // OptionSelectUiHandler stores options in a protected `config` property
    const handler = globalScene.ui.getHandler() as unknown as { config?: { options?: { label?: string }[] } };
    const handlerOptions = handler?.config?.options;
    if (handlerOptions?.length > 0) {
      optionCount = Math.min(handlerOptions.length, 4);
      for (let i = 0; i < optionCount; i++) {
        biomeNames.push(handlerOptions[i]?.label ?? `Biome ${i}`);
      }
    }
    for (let i = 0; i < optionCount; i++) {
      mask[i] = true;
    }
    metadata.isSelectBiome = true;
    metadata.optionCount = optionCount;
    metadata.biomeNames = biomeNames;
    return mask;
  }

  // ── GameOver Phase Mask ────────────────────────────────────────────

  function buildGameOverActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    // Binary: retry (0) or quit (1)
    mask[0] = true; // Retry
    mask[1] = true; // Quit
    metadata.isGameOver = true;
    return mask;
  }

  // ── MysteryEncounter Phase Mask ────────────────────────────────────

  function buildMysteryEncounterActionMask(metadata: Record<string, unknown>): boolean[] {
    const mask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    // Mystery encounters typically have 2-4 options
    const encounter = globalScene.currentBattle?.mysteryEncounter;
    const optionCount = encounter?.options?.length ?? 2;
    for (let i = 0; i < optionCount && i < 4; i++) {
      mask[i] = true;
    }
    metadata.isMysteryEncounter = true;
    metadata.optionCount = optionCount;
    return mask;
  }

  // ── Action Execution ───────────────────────────────────────────────

  async function executeActionInternal(action: number): Promise<void> {
    if (action < 0 || action >= ACTION_SPACE_SIZE) {
      console.error(`[PhaseRouter] Invalid action: ${action}`);
      return;
    }

    if (!currentPhaseState) {
      console.error("[PhaseRouter] No current phase state to execute action against");
      return;
    }

    // Validate action against mask
    if (!currentPhaseState.actionMask[action]) {
      console.warn(
        `[PhaseRouter] Action ${action} is not valid for phase ${currentPhaseState.phase}. `
          + `Valid actions: [${currentPhaseState.validActions.join(", ")}]`,
      );
      // Fall back to first valid action
      if (currentPhaseState.validActions.length > 0) {
        action = currentPhaseState.validActions[0];
        console.warn(`[PhaseRouter] Falling back to action ${action}`);
      } else {
        return;
      }
    }

    const phase = currentPhaseState.phase;

    if (verbose) {
      console.log(`[PhaseRouter] Execute: phase=${phase} action=${action}`);
    }

    switch (phase) {
      case DecisionPhase.COMMAND:
        executeCommandAction(action);
        break;
      case DecisionPhase.SELECT_TARGET:
        executeSelectTargetAction(action);
        break;
      case DecisionPhase.SELECT_MODIFIER:
        executeModifierAction(action);
        break;
      case DecisionPhase.MODIFIER_TARGET:
        executeModifierTargetAction(action);
        break;
      case DecisionPhase.SWITCH:
        executeSwitchAction(action);
        break;
      case DecisionPhase.CHECK_SWITCH:
        executeCheckSwitchAction(action);
        break;
      case DecisionPhase.LEARN_MOVE:
        executeLearnMoveAction(action);
        break;
      case DecisionPhase.REVIVAL_BLESSING:
        executeRevivalBlessingAction(action);
        break;
      case DecisionPhase.SELECT_BIOME:
        executeSelectBiomeAction(action);
        break;
      case DecisionPhase.GAME_OVER:
        executeGameOverAction(action);
        break;
      case DecisionPhase.MYSTERY_ENCOUNTER:
        executeMysteryEncounterAction(action);
        break;
      case DecisionPhase.EVOLUTION:
      case DecisionPhase.FORM_CHANGE:
        // Let evolution/form-change proceed naturally (no cancel)
        executeAutoAction();
        break;
      case DecisionPhase.TITLE:
        executeTitleAction();
        break;
      case DecisionPhase.SELECT_GENDER:
        executeAutoAction();
        break;
      case DecisionPhase.SELECT_STARTER:
        // Should not reach here -- TitlePhase handler triggers initBattle
        // which skips SelectStarterPhase entirely. But just in case:
        executeAutoAction();
        break;
      default:
        console.warn(`[PhaseRouter] Unhandled decision phase: ${phase}. Auto-completing.`);
        executeAutoAction();
        break;
    }

    // Clear current state after executing
    currentPhaseState = null;
  }

  // ── Command Action Execution ───────────────────────────────────────

  function executeCommandAction(action: number): void {
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (!currentPhase?.is("CommandPhase")) {
      console.error("[PhaseRouter] Not in CommandPhase for command action");
      return;
    }
    const commandPhase = currentPhase as CommandPhase;
    const pokemon = commandPhase.getPokemon();
    const moveset = pokemon.getMoveset(false);

    // Fight targeting ENEMY (0-3)
    if (action >= ACTION_FIGHT_ENEMY_START && action < ACTION_FIGHT_ENEMY_START + MAX_MOVES) {
      const moveIndex = action - ACTION_FIGHT_ENEMY_START;
      const moveData = moveset[moveIndex]?.getMove();
      const needsExplicitTarget =
        moveData
        && (moveData.moveTarget === MoveTarget.NEAR_ENEMY
          || moveData.moveTarget === MoveTarget.NEAR_OTHER
          || moveData.moveTarget === MoveTarget.OTHER);
      if (needsExplicitTarget) {
        // Single-target move: specify enemy slot 0
        const moveId = moveset[moveIndex]?.moveId ?? 0;
        commandPhase.handleCommand(Command.FIGHT, moveIndex, MoveUseMode.NORMAL, {
          move: moveId,
          targets: [BattlerIndex.ENEMY],
          useMode: MoveUseMode.NORMAL,
        });
      } else {
        // Multi-target/self-target/field: let the game compute targets
        commandPhase.handleCommand(Command.FIGHT, moveIndex, MoveUseMode.NORMAL);
      }
      return;
    }

    // Fight targeting ENEMY_2 (4-7)
    if (action >= ACTION_FIGHT_ENEMY2_START && action < ACTION_FIGHT_ENEMY2_START + MAX_MOVES) {
      const moveIndex = action - ACTION_FIGHT_ENEMY2_START;
      const moveId = moveset[moveIndex]?.moveId ?? 0;
      commandPhase.handleCommand(Command.FIGHT, moveIndex, MoveUseMode.NORMAL, {
        move: moveId,
        targets: [BattlerIndex.ENEMY_2],
        useMode: MoveUseMode.NORMAL,
      });
      return;
    }

    // Fight targeting ally (8-11)
    if (action >= ACTION_FIGHT_ALLY_START && action < ACTION_FIGHT_ALLY_START + MAX_MOVES) {
      const moveIndex = action - ACTION_FIGHT_ALLY_START;
      const moveId = moveset[moveIndex]?.moveId ?? 0;
      // Ally is the OTHER player slot: if we're fieldIndex 0, ally is PLAYER_2; if 1, ally is PLAYER
      const allyTarget = commandPhase.getFieldIndex() === 0 ? BattlerIndex.PLAYER_2 : BattlerIndex.PLAYER;
      commandPhase.handleCommand(Command.FIGHT, moveIndex, MoveUseMode.NORMAL, {
        move: moveId,
        targets: [allyTarget],
        useMode: MoveUseMode.NORMAL,
      });
      return;
    }

    // Switch (12-16)
    if (action >= ACTION_SWITCH_START && action < ACTION_SWITCH_START + 5) {
      const partySlot = action - ACTION_SWITCH_START + 1; // slots 1-5
      commandPhase.handleCommand(Command.POKEMON, partySlot, false);
      return;
    }

    // Ball (17-21)
    if (action >= ACTION_BALL_START && action < ACTION_BALL_START + NUM_POKEBALL_TYPES) {
      const ballType = action - ACTION_BALL_START;
      commandPhase.handleCommand(Command.BALL, ballType);
      return;
    }

    // Run (22)
    if (action === ACTION_RUN) {
      commandPhase.handleCommand(Command.RUN, 0);
      return;
    }

    // Tera targeting ENEMY (23-26)
    if (action >= ACTION_TERA_ENEMY_START && action < ACTION_TERA_ENEMY_START + MAX_MOVES) {
      const moveIndex = action - ACTION_TERA_ENEMY_START;
      const moveData = moveset[moveIndex]?.getMove();
      const needsExplicitTarget =
        moveData
        && (moveData.moveTarget === MoveTarget.NEAR_ENEMY
          || moveData.moveTarget === MoveTarget.NEAR_OTHER
          || moveData.moveTarget === MoveTarget.OTHER);
      if (needsExplicitTarget) {
        const moveId = moveset[moveIndex]?.moveId ?? 0;
        commandPhase.handleCommand(Command.TERA, moveIndex, MoveUseMode.NORMAL, {
          move: moveId,
          targets: [BattlerIndex.ENEMY],
          useMode: MoveUseMode.NORMAL,
        });
      } else {
        commandPhase.handleCommand(Command.TERA, moveIndex, MoveUseMode.NORMAL);
      }
      return;
    }

    // Tera targeting ENEMY_2 (27-30)
    if (action >= ACTION_TERA_ENEMY2_START && action < ACTION_TERA_ENEMY2_START + MAX_MOVES) {
      const moveIndex = action - ACTION_TERA_ENEMY2_START;
      const moveId = moveset[moveIndex]?.moveId ?? 0;
      commandPhase.handleCommand(Command.TERA, moveIndex, MoveUseMode.NORMAL, {
        move: moveId,
        targets: [BattlerIndex.ENEMY_2],
        useMode: MoveUseMode.NORMAL,
      });
      return;
    }

    // Tera targeting ally (31-34)
    if (action >= ACTION_TERA_ALLY_START && action < ACTION_TERA_ALLY_START + MAX_MOVES) {
      const moveIndex = action - ACTION_TERA_ALLY_START;
      const moveId = moveset[moveIndex]?.moveId ?? 0;
      const allyTarget = commandPhase.getFieldIndex() === 0 ? BattlerIndex.PLAYER_2 : BattlerIndex.PLAYER;
      commandPhase.handleCommand(Command.TERA, moveIndex, MoveUseMode.NORMAL, {
        move: moveId,
        targets: [allyTarget],
        useMode: MoveUseMode.NORMAL,
      });
      return;
    }
  }

  // ── Select Target Action Execution ─────────────────────────────────

  function executeSelectTargetAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for SelectTargetPhase");
      return;
    }

    // Map action back to BattlerIndex for the target selection.
    // Specific checks before range checks to avoid shadowing.
    let targetIndex: BattlerIndex = BattlerIndex.ENEMY;
    if (action === ACTION_FIGHT_ALLY_START) {
      targetIndex = BattlerIndex.PLAYER;
    } else if (action === ACTION_FIGHT_ALLY_START + 1) {
      targetIndex = BattlerIndex.PLAYER_2;
    } else if (action >= ACTION_FIGHT_ENEMY2_START && action < ACTION_FIGHT_ENEMY2_START + MAX_MOVES) {
      targetIndex = BattlerIndex.ENEMY_2;
    }

    // Use the TargetSelectUiHandler directly
    (handler as { setCursor(cursor: number): boolean }).setCursor(targetIndex);
    (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
  }

  // ── Modifier Action Execution ──────────────────────────────────────

  /**
   * Install a monkey-patch on ModifierSelectUiHandler.show() that kills
   * lingering tweens and timer events BEFORE the original show() runs.
   *
   * Root cause of the alternating-wave shop rendering bug:
   * show() creates a counter tween (1250ms) + delayedCall timer events
   * that continue running even after clear() is called. If we select an
   * action quickly (< 1250ms), the stale counter tween from wave N is
   * still active when wave N+1's show() starts. Both counter tweens
   * call option.show() on wave N+1's option objects, creating conflicting
   * sub-tweens that prevent the shop from rendering on alternating waves.
   *
   * Additionally, show()'s delayedCall events fire 1-2 seconds later and
   * set button containers visible + awaitingActionInput=true with a stale
   * callback, further corrupting handler state.
   *
   * The monkey-patch ensures every show() call starts with a clean slate
   * by killing all tweens/timers and destroying old UI children first.
   */
  function patchModifierHandler(): void {
    if (modifierHandlerPatched) {
      return;
    }
    try {
      const handler = (globalScene as any).ui?.handlers?.[UiMode.MODIFIER_SELECT];
      if (!handler) {
        return;
      }

      // The handler INSTANCE outlives this router (the scene is reused across
      // in-process episode resets), but `modifierHandlerPatched` is closure
      // state and resets with every new router. Without this marker each
      // episode would wrap the PREVIOUS wrapper — N stacked layers after N
      // episodes, every layer pinning its router's closures (observed as
      // steps/s decaying 109->43 and RSS tripling over a 100-episode soak).
      if ((handler.show as { __rlPatched?: boolean }).__rlPatched) {
        modifierHandlerPatched = true;
        return;
      }

      const originalShow = handler.show.bind(handler);
      handler.show = function (args: unknown[]): boolean {
        // Guard: only run cleanup and show if we're actually in a
        // SelectModifierPhase. show() can be called spuriously during
        // other phases (e.g., from delayed setMode transitions or mode
        // stack operations). Running killAll/removeAllEvents during those
        // phases would kill game-critical tweens (like ReturnPhase
        // pokemon return animations) and freeze the game.
        const currentPhase = globalScene.phaseManager?.getCurrentPhase();
        if (!currentPhase?.is("SelectModifierPhase")) {
          if (verbose) {
            console.log(
              `[PhaseRouter] ModifierSelectUiHandler.show() blocked — not in SelectModifierPhase (current: ${currentPhase?.phaseName})`,
            );
          }
          return false;
        }

        // Kill ALL running tweens. The stale counter tween from a previous
        // show() has no explicit target, so killTweensOf() can't find it.
        // killAll() is the only way. This is safe — we're entering a new
        // modifier phase, so any running tweens are stale.
        try {
          globalScene.tweens.killAll();
        } catch {
          /* ignore */
        }

        // Remove ALL pending timer events. show()'s delayedCall events
        // would fire later and corrupt handler state with stale callbacks.
        try {
          globalScene.time.removeAllEvents();
        } catch {
          /* ignore */
        }

        // Destroy ALL old children in modifierContainer (undestroyed options
        // from a previous show/clear cycle whose destroy-onComplete was
        // killed). The original show() will create fresh children.
        try {
          if (this.modifierContainer) {
            this.modifierContainer.removeAll(true);
          }
        } catch {
          /* ignore */
        }

        // Reset arrays that may have been repopulated by stale callbacks
        this.options = [];
        this.shopOptionsRows = [];

        // Ensure the handler is NOT marked active, so show() won't bail
        // out at the `if (this.active)` guard. clear() should have set
        // this to false, but stale callbacks could theoretically re-set it.
        this.active = false;

        if (verbose) {
          console.log("[PhaseRouter] ModifierSelectUiHandler.show() pre-cleanup done");
        }

        return originalShow(args);
      };
      (handler.show as { __rlPatched?: boolean }).__rlPatched = true;

      modifierHandlerPatched = true;
      if (verbose) {
        console.log("[PhaseRouter] ModifierSelectUiHandler patched");
      }
    } catch {
      // Non-critical — headless mode may not have the handler
    }
  }

  /**
   * Force-hide the shop overlay and modifier UI elements after an action.
   * This is a post-action cleanup that ensures visual elements are hidden
   * immediately (rather than waiting for clear()'s 250-750ms tweens).
   */
  function cleanupModifierUI(): void {
    try {
      const scene = globalScene as any;

      // Note: Do NOT call tweens.killAll() or time.removeAllEvents() here.
      // Those kill ALL tweens/events in the entire game, including ones needed
      // by subsequent phases (e.g., ReturnPhase pokemon animations, HP bar
      // update tweens). The monkey-patched show() in patchModifierHandler()
      // handles cleanup at the START of the next modifier phase instead.

      // Hide shop overlay instantly
      if (scene.shopOverlay) {
        scene.shopOverlay.setAlpha(0);
      }
      scene.shopOverlayShown = false;

      // Hide luck text
      if (scene.luckText) {
        scene.luckText.setAlpha(0);
      }
      if (scene.luckLabelText) {
        scene.luckLabelText.setAlpha(0);
      }

      const handler = scene.ui?.handlers?.[UiMode.MODIFIER_SELECT];
      if (!handler) {
        return;
      }

      // Destroy ALL children in modifierContainer
      if (handler.modifierContainer) {
        handler.modifierContainer.removeAll(true);
      }

      // Reset handler state
      handler.options = [];
      handler.shopOptionsRows = [];

      // Hide button containers
      const containers = [
        handler.rerollButtonContainer,
        handler.checkButtonContainer,
        handler.transferButtonContainer,
        handler.lockRarityButtonContainer,
        handler.continueButtonContainer,
      ];
      for (const c of containers) {
        if (c) {
          c.setVisible(false);
          c.setAlpha(0);
        }
      }

      handler.eraseCursor?.();
      handler.awaitingActionInput = false;
      handler.onActionInput = null;
    } catch {
      // Non-critical — ignore if elements don't exist (e.g. headless mode)
    }
  }

  function executeModifierAction(action: number): void {
    // Select reward (35-37)
    if (action >= ACTION_SELECT_REWARD_START && action < ACTION_SELECT_REWARD_START + MAX_REWARD_OPTIONS) {
      const rewardIndex = action - ACTION_SELECT_REWARD_START;
      const modifiers = getAvailableModifiers();
      if (modifiers && rewardIndex < modifiers.rewards.length) {
        const reward = modifiers.rewards[rewardIndex];
        if (reward.targetKind === "none") {
          selectRewardModifier(rewardIndex);
          cleanupModifierUI();
        } else {
          // Enter two-step targeting flow — agent picks which Pokemon next
          pendingModifierAction = { source: "reward", index: rewardIndex, cost: 0 };
        }
      }
      return;
    }

    // Reroll (38)
    if (action === ACTION_REROLL) {
      rerollModifiers();
      return;
    }

    // Skip (39)
    if (action === ACTION_SKIP) {
      skipModifiers();
      cleanupModifierUI();
      return;
    }

    // Buy shop item (40-51)
    // Shop purchases keep the phase active so the player can continue buying.
    // Don't call cleanupModifierUI() — the modifier select UI stays visible.
    // For targetKind "none": the phase callback handles it (returns false, handler stays active).
    // For targetKind "pokemon": applyModifierDirectly() calls resetModifierSelect() to re-show the shop.
    // The phase router will detect the still-active modifier phase as the next decision point.
    if (action >= ACTION_BUY_SHOP_START && action < ACTION_BUY_SHOP_START + MAX_SHOP_OPTIONS) {
      const shopIndex = action - ACTION_BUY_SHOP_START;
      const modifiers = getAvailableModifiers();
      if (modifiers && shopIndex < modifiers.shop.length) {
        const shopItem = modifiers.shop[shopIndex];
        if (shopItem.targetKind === "none") {
          selectShopModifier(shopIndex);
        } else {
          // Enter two-step targeting flow — agent picks which Pokemon next
          pendingModifierAction = { source: "shop", index: shopIndex, cost: shopItem.cost };
        }
      }
      return;
    }
  }

  // ── Modifier Target Execution ──────────────────────────────────────

  function executeModifierTargetAction(action: number): void {
    if (!pendingModifierAction) {
      console.error("[PhaseRouter] No pending modifier action for MODIFIER_TARGET");
      return;
    }

    // Skip/cancel: clear pending and return to modifier select
    if (action === ACTION_SKIP) {
      pendingModifierAction = null;
      return;
    }

    // Party target actions (52-57)
    if (action >= ACTION_PARTY_TARGET_START && action < ACTION_PARTY_TARGET_START + MAX_PARTY_SIZE) {
      const pokemonIndex = action - ACTION_PARTY_TARGET_START;
      const { source, index } = pendingModifierAction;
      pendingModifierAction = null;

      if (source === "reward") {
        selectRewardModifier(index, pokemonIndex);
        cleanupModifierUI();
      } else {
        // Shop purchase — don't cleanup, phase stays active for more shopping
        selectShopModifier(index, pokemonIndex);
      }
      return;
    }

    console.warn(`[PhaseRouter] Invalid MODIFIER_TARGET action: ${action}`);
  }

  // ── Switch Phase Execution ─────────────────────────────────────────

  function executeSwitchAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for SwitchPhase");
      return;
    }

    // Switch actions map: ACTION_SWITCH_START + i -> party slot i+1
    if (action >= ACTION_SWITCH_START && action < ACTION_SWITCH_START + 5) {
      const partySlotIndex = action - ACTION_SWITCH_START + 1;
      (handler as { setCursor(cursor: number): boolean }).setCursor(partySlotIndex);
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
      // May need a second ACTION to confirm "Send out" option
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
      return;
    }

    // Fallback: pick first non-fainted party member
    console.warn("[PhaseRouter] Invalid switch action, selecting first available");
    const party = globalScene.getPlayerParty() ?? [];
    const field = globalScene.getPlayerField()?.filter(p => p?.isActive()) ?? [];
    const activeIds = new Set(field.map(p => p.id));
    for (let i = 1; i < party.length; i++) {
      if (!party[i].isFainted() && !activeIds.has(party[i].id)) {
        (handler as { setCursor(cursor: number): boolean }).setCursor(i);
        (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
        (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
        return;
      }
    }
  }

  // ── CheckSwitch Phase Execution ────────────────────────────────────

  function executeCheckSwitchAction(action: number): void {
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (!currentPhase?.is("CheckSwitchPhase")) {
      console.error("[PhaseRouter] Not in CheckSwitchPhase");
      return;
    }

    // Bypass ConfirmUiHandler.processInput entirely to avoid a re-entrancy bug:
    // processInput calls option.handler() (which synchronously starts the next
    // CheckSwitchPhase and sets up a new CONFIRM dialog), then calls this.clear()
    // which destroys the newly created dialog state (sets active=false).
    // Instead, we directly replicate what the Yes/No callbacks do.
    if (action === ACTION_SKIP) {
      // Decline switch: replicate the "No" callback from CheckSwitchPhase
      globalScene.ui.setMode(UiMode.MESSAGE);
      currentPhase.end();
    } else {
      // Accept switch: replicate the "Yes" callback from CheckSwitchPhase
      const fieldIndex = (currentPhase as unknown as { fieldIndex: number }).fieldIndex;
      globalScene.phaseManager.unshiftNew("SwitchPhase", SwitchType.INITIAL_SWITCH, fieldIndex, false, true);
      globalScene.ui.setMode(UiMode.MESSAGE);
      currentPhase.end();
    }
  }

  // ── LearnMove Phase Execution ──────────────────────────────────────

  function executeLearnMoveAction(action: number): void {
    const currentPhase = globalScene.phaseManager.getCurrentPhase();
    if (!currentPhase?.is("LearnMovePhase")) {
      console.error("[PhaseRouter] Not in LearnMovePhase");
      return;
    }

    // Bypass the multi-step UI flow (CONFIRM → SUMMARY → callback) entirely.
    // Instead, directly call LearnMovePhase methods — same pattern as CheckSwitchPhase.
    const phase = currentPhase as unknown as {
      moveId: number;
      messageMode: UiMode;
      getPokemon(): any;
      learnMove(index: number, move: any, pokemon: any, textMessage?: string): void;
    };

    if (action === ACTION_SKIP) {
      // Decline to learn the move — just end the phase
      globalScene.ui.setMode(phase.messageMode);
      currentPhase.end();
    } else if (action >= 0 && action < MAX_MOVES) {
      // Replace move at the selected slot — call learnMove directly
      const move = allMoves[phase.moveId];
      const pokemon = phase.getPokemon();
      phase.learnMove(action, move, pokemon);
    }
  }

  // ── Revival Blessing Execution ─────────────────────────────────────

  function executeRevivalBlessingAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for RevivalBlessingPhase");
      return;
    }

    if (action >= ACTION_PARTY_TARGET_START && action < ACTION_PARTY_TARGET_START + MAX_PARTY_SIZE) {
      const partyIndex = action - ACTION_PARTY_TARGET_START;
      (handler as { setCursor(cursor: number): boolean }).setCursor(partyIndex);
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
    }
  }

  // ── SelectBiome Execution ──────────────────────────────────────────

  function executeSelectBiomeAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for SelectBiomePhase");
      return;
    }

    // action 0-3 maps to biome option 0-3
    const biomeIndex = Math.min(action, 3);
    (handler as { setCursor(cursor: number): boolean }).setCursor(biomeIndex);
    (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
  }

  // ── GameOver Execution ─────────────────────────────────────────────

  function executeGameOverAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for GameOverPhase");
      return;
    }

    if (action === 0) {
      // Retry - reset flags so the advance loop doesn't think the game is still over
      gameOverFlag = false;
      victoryFlag = false;
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
    } else {
      // Quit
      gameOverFlag = true;
      (handler as { processInput(button: Button): boolean }).processInput(Button.CANCEL);
    }
  }

  // ── MysteryEncounter Execution ─────────────────────────────────────

  function executeMysteryEncounterAction(action: number): void {
    const handler = globalScene.ui.getHandler();
    if (!handler) {
      console.error("[PhaseRouter] No UI handler for MysteryEncounterPhase");
      return;
    }

    const optionIndex = Math.min(action, 3);
    (handler as { setCursor(cursor: number): boolean }).setCursor(optionIndex);
    (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
  }

  // ── Title / Starter Action ────────────────────────────────────────

  /**
   * Handle TitlePhase programmatically: set game mode to CLASSIC,
   * generate default starters, and call initBattle() to skip the entire
   * starter selection UI. This mirrors test-utils/helpers/classic-mode-helper.ts.
   */
  function executeTitleAction(): void {
    // Re-entry guard: in the browser, initBattle() is async (loads real assets).
    // While assets load, TitlePhase stays current and the setup loop may
    // re-detect TITLE and call this function again. Prevent duplicating starters.
    if (titleActionExecuted) {
      if (verbose) {
        console.log("[PhaseRouter] executeTitleAction: already executed, skipping");
      }
      return;
    }
    titleActionExecuted = true;

    // Set game mode to Classic
    globalScene.gameMode = getGameMode(GameModes.CLASSIC);

    // Initialize the arena — TitlePhase.end() normally does this, but we
    // bypass that flow. Without it, the arena has no biome or BGM data.
    globalScene.newArena(globalScene.gameMode.getStartingBiome());

    // Set starting money — normally set by TitlePhase.end() (line 222) or
    // StarterSelectUiHandler (line 4466). We bypass both flows.
    globalScene.money = globalScene.gameMode.getStartingMoney();
    globalScene.updateMoneyText();

    // Clear any existing party (from a loaded session) before adding starters
    const existingParty = globalScene.getPlayerParty();
    if (existingParty.length > 0) {
      if (verbose) {
        console.log(`[PhaseRouter] Clearing existing party of ${existingParty.length} Pokemon`);
      }
      existingParty.splice(0);
    }

    // Generate default starters. generateStarters() is a test utility that
    // hardcodes scene.seed = "test" — save and restore the user's seed so
    // that initBattle() derives wave seeds from the user's seed, not "test".
    const userSeed = globalScene.seed;
    const starters = generateStarters(globalScene);
    globalScene.setSeed(userSeed);
    globalScene.resetSeed();

    // Create and init battle programmatically (bypasses SelectStarterPhase UI)
    const selectStarterPhase = new SelectStarterPhase();
    globalScene.phaseManager.pushPhase(new EncounterPhase(false));

    // Play the "menu" BGM so that initBattle()'s SoundFade.fadeOut() has a
    // valid audio object to fade. In headless mode (noAudio: true) this is a
    // harmless no-op. Without this, sound.get("menu") returns null and the
    // FadeOut plugin crashes trying to read .volume on null.
    globalScene.playBgm("menu");

    selectStarterPhase.initBattle(starters);

    // Force-populate movesets: in CLASSIC mode, PlayerPokemon constructor sets
    // this.moveset = [] and expects tryPopulateMoveset() to fill it. But that
    // silently fails when gameData.starterData has no egg move data (headless mode
    // has no save data). Call generateAndPopulateMoveset() directly as a fallback.
    const party = globalScene.getPlayerParty();
    for (const pokemon of party) {
      if (pokemon.moveset.length === 0) {
        pokemon.generateAndPopulateMoveset();
      }
    }
  }

  // ── Auto Action (for phases we don't control) ──────────────────────

  function executeAutoAction(): void {
    const handler = globalScene.ui.getHandler();
    if (handler) {
      // Press ACTION to proceed
      (handler as { processInput(button: Button): boolean }).processInput(Button.ACTION);
    }
  }

  // ── Advance to Next Decision ───────────────────────────────────────

  /**
   * Wait for the game to reach the next decision point.
   * Returns a promise that resolves with the PhaseState when a decision is needed.
   * Uses the setMode hook to detect decision points, with a polling fallback
   * in case the hook fires before we start waiting.
   */
  function waitForNextDecision(timeoutMs = 30000): Promise<PhaseState> {
    return new Promise<PhaseState>((resolve, reject) => {
      // Check if the setMode hook already detected a decision point before
      // advanceToNextDecision() was called. In the browser, the game runs
      // while the bridge waits for Python's "start" signal, so the hook
      // may have fired and set currentPhaseState before we get here.
      if (currentPhaseState) {
        resolve(currentPhaseState);
        return;
      }

      // Check if we're already at a decision point via fresh detection
      const existingState = detectCurrentDecision();
      if (existingState) {
        currentPhaseState = existingState;
        resolve(existingState);
        return;
      }

      // Check if game is already over
      if (gameOverFlag) {
        const overState: PhaseState = {
          phase: DecisionPhase.GAME_OVER,
          validActions: [],
          actionMask: new Array<boolean>(ACTION_SPACE_SIZE).fill(false),
          metadata: { gameOver: true, isVictory: victoryFlag },
        };
        currentPhaseState = overState;
        resolve(overState);
        return;
      }

      // Set up pending decision that will be resolved by the setMode hook
      pendingDecision = { resolve, reject };

      // Polling fallback: check periodically in case the hook missed a transition
      // This handles edge cases where the decision phase was already reached
      // before waitForNextDecision was called
      const pollInterval = setInterval(() => {
        if (destroyed) {
          clearInterval(pollInterval);
          if (pendingDecision) {
            pendingDecision = null;
            reject(new Error("PhaseRouter destroyed while waiting"));
          }
          return;
        }

        // Check for game over
        if (checkGameOver()) {
          clearInterval(pollInterval);
          if (pendingDecision) {
            const pd = pendingDecision;
            pendingDecision = null;
            const overState: PhaseState = {
              phase: DecisionPhase.GAME_OVER,
              validActions: [],
              actionMask: new Array<boolean>(ACTION_SPACE_SIZE).fill(false),
              metadata: { gameOver: true, isVictory: victoryFlag },
            };
            currentPhaseState = overState;
            pd.resolve(overState);
          }
          return;
        }

        // Auto-skip non-decision phases that block on UI input (PH2-PH13)
        tryAutoSkipPhase();

        // Check if a decision phase has appeared
        const state = detectCurrentDecision();
        if (state) {
          clearInterval(pollInterval);
          currentPhaseState = state;
          if (pendingDecision) {
            const pd = pendingDecision;
            pendingDecision = null;
            pd.resolve(state);
          }
        }
      }, 50); // 50ms poll interval

      // Timeout safety
      const timeoutHandle = setTimeout(() => {
        clearInterval(pollInterval);
        if (pendingDecision) {
          const pd = pendingDecision;
          pendingDecision = null;
          pd.reject(
            new Error(
              `[PhaseRouter] Timeout waiting for next decision after ${timeoutMs}ms. `
                + `Last phase: ${globalScene.phaseManager?.getCurrentPhase()?.phaseName ?? "none"}, `
                + `UI mode: ${UiMode[globalScene.ui?.getMode()] ?? "unknown"}`,
            ),
          );
        }
      }, timeoutMs);

      // Clean up timeout if resolved before it fires
      const originalResolve = resolve;
      pendingDecision = {
        resolve: (state: PhaseState) => {
          clearInterval(pollInterval);
          clearTimeout(timeoutHandle);
          originalResolve(state);
        },
        reject: (error: Error) => {
          clearInterval(pollInterval);
          clearTimeout(timeoutHandle);
          reject(error);
        },
      };
    });
  }

  /**
   * Auto-skip non-decision phases that block on UI input (PH2-PH13).
   * Called from the polling fallback. Checks if the current phase is in
   * AUTO_SKIP_PHASES and the UI handler is waiting for input, then presses
   * ACTION (or selects "yes/skip" for CONFIRM dialogs) to proceed.
   *
   * Returns true if an auto-skip was performed, false otherwise.
   */
  function tryAutoSkipPhase(): boolean {
    const currentPhase = globalScene.phaseManager?.getCurrentPhase();
    if (!currentPhase) {
      return false;
    }

    const phaseName = currentPhase.phaseName;
    if (!AUTO_SKIP_PHASES.has(phaseName)) {
      return false;
    }

    const handler = globalScene.ui.getHandler() as {
      active?: boolean;
      processInput?(button: Button): boolean;
      awaitingActionInput?: boolean;
      onActionInput?: (() => void) | null;
    };
    if (!handler) {
      return false;
    }

    // For CONFIRM dialogs:
    // - EggLapsePhase: "Skip hatching?" → Yes (cursor 0) = skip animations
    // - EvolutionPhase: "Pause evolutions?" → No (cursor 1) = DON'T pause
    const uiMode = globalScene.ui.getMode();

    // ── AttemptCapturePhase full-party release: poll-driven state machine ──
    // Catching with a full party opens: CONFIRM ("fullParty", 4 options) →
    // PARTY (RELEASE mode) → slot options submenu (RELEASE = option 0) →
    // goodbye text (prompt=true) → release done, phase continues. Each tick
    // reads the actual UI state and advances exactly one input.
    if (phaseName === "AttemptCapturePhase") {
      if (uiMode === UiMode.CONFIRM && handler.active) {
        // Option 2 = "Yes, release a party member"
        if (verbose) {
          console.log("[PhaseRouter] AttemptCapturePhase: CONFIRM → selecting release (option 2)");
        }
        (handler as { setCursor?(cursor: number): boolean }).setCursor?.(2);
        handler.processInput?.(Button.ACTION);
        return true;
      }
      if (uiMode === UiMode.PARTY) {
        const party = globalScene.getPlayerParty() ?? [];
        let lowestIdx = 0;
        let lowestLevel = Number.POSITIVE_INFINITY;
        for (let i = 0; i < party.length; i++) {
          if (party[i].level < lowestLevel) {
            lowestLevel = party[i].level;
            lowestIdx = i;
          }
        }
        const partyHandler = handler as {
          awaitingActionInput?: boolean;
          optionsMode?: boolean;
          cursor?: number;
          setCursor?(cursor: number): boolean;
          processInput?(button: Button): boolean;
        };
        if (partyHandler.awaitingActionInput) {
          // Goodbye text (or other prompt) inside the party UI — dismiss it
          if (verbose) {
            console.log("[PhaseRouter] AttemptCapturePhase: PARTY prompt → ACTION");
          }
          partyHandler.processInput?.(Button.ACTION);
          return true;
        }
        if (!partyHandler.optionsMode) {
          // Select the lowest-level slot (opens its options submenu)
          if (verbose) {
            console.log(
              `[PhaseRouter] AttemptCapturePhase: PARTY → selecting slot ${lowestIdx} (${party[lowestIdx]?.name}, Lv${lowestLevel})`,
            );
          }
          infoMessages.push(`AutoRelease: ${party[lowestIdx]?.name ?? "?"} Lv${lowestLevel} (slot ${lowestIdx})`);
          partyHandler.setCursor?.(lowestIdx);
          partyHandler.processInput?.(Button.ACTION);
          return true;
        }
        if (partyHandler.cursor === lowestIdx) {
          // Options submenu open on the right slot: RELEASE is option 0
          if (verbose) {
            console.log("[PhaseRouter] AttemptCapturePhase: PARTY options → RELEASE");
          }
          partyHandler.setCursor?.(0);
          partyHandler.processInput?.(Button.ACTION);
        } else {
          // Submenu open on the wrong slot (stale input) — back out and retry
          if (verbose) {
            console.log(
              `[PhaseRouter] AttemptCapturePhase: PARTY options on wrong slot ${partyHandler.cursor} → CANCEL`,
            );
          }
          partyHandler.processInput?.(Button.CANCEL);
        }
        return true;
      }
      // MESSAGE-mode prompts fall through to the generic handling below
    }

    if (uiMode === UiMode.CONFIRM && handler.active) {
      if (phaseName === "EvolutionPhase" || phaseName === "FormChangePhase") {
        // Select "No" (don't pause evolutions) — CANCEL sets cursor to last option (No)
        if (verbose) {
          console.log(`[PhaseRouter] Auto-skip: ${phaseName} CONFIRM dialog (pressing CANCEL = don't pause)`);
        }
        infoMessages.push(`AutoSkip: ${phaseName} (CONFIRM → don't pause)`);
        handler.processInput?.(Button.CANCEL);
      } else {
        // Default: select "Yes" (cursor 0) — e.g. EggLapsePhase "skip hatching?"
        if (verbose) {
          console.log(`[PhaseRouter] Auto-skip: ${phaseName} CONFIRM dialog (pressing ACTION = yes/skip)`);
        }
        infoMessages.push(`AutoSkip: ${phaseName} (CONFIRM)`);
        handler.processInput?.(Button.ACTION);
      }
      return true;
    }

    // For EGG_HATCH_SCENE: press ACTION to skip animation and dismiss text prompts
    if (uiMode === UiMode.EGG_HATCH_SCENE && handler.active) {
      if (verbose) {
        console.log(`[PhaseRouter] Auto-skip: ${phaseName} EGG_HATCH_SCENE (pressing ACTION)`);
      }
      infoMessages.push(`AutoSkip: ${phaseName} (EGG_HATCH_SCENE)`);
      handler.processInput?.(Button.ACTION);
      return true;
    }

    // For EGG_HATCH_SUMMARY: press CANCEL to dismiss (ACTION scrolls the grid)
    if (uiMode === UiMode.EGG_HATCH_SUMMARY && handler.active) {
      if (verbose) {
        console.log(`[PhaseRouter] Auto-skip: ${phaseName} EGG_HATCH_SUMMARY (pressing CANCEL)`);
      }
      infoMessages.push(`AutoSkip: ${phaseName} (EGG_HATCH_SUMMARY)`);
      handler.processInput?.(Button.CANCEL);
      return true;
    }

    // For EVOLUTION_SCENE mode (EvolutionPhase): pressing ACTION may speed up
    // or dismiss animations. Only press if handler is active.
    if (uiMode === UiMode.EVOLUTION_SCENE && handler.active) {
      if (verbose) {
        console.log(`[PhaseRouter] Auto-skip: ${phaseName} EVOLUTION_SCENE (pressing ACTION)`);
      }
      handler.processInput?.(Button.ACTION);
      return true;
    }

    // For showText prompts (MESSAGE mode with awaitingActionInput):
    // The message handler shows text and waits for ACTION. Check both the
    // handler from getHandler() and the dedicated message handler.
    if (uiMode === UiMode.MESSAGE) {
      const msgHandler = globalScene.ui.getMessageHandler() as {
        awaitingActionInput?: boolean;
        onActionInput?: (() => void) | null;
        processInput?(button: Button): boolean;
      };
      if (msgHandler?.awaitingActionInput) {
        if (verbose) {
          console.log(`[PhaseRouter] Auto-skip: ${phaseName} showText prompt (pressing ACTION)`);
        }
        infoMessages.push(`AutoSkip: ${phaseName} (prompt)`);
        msgHandler.processInput?.(Button.ACTION);
        return true;
      }
    }

    return false;
  }

  /**
   * Try to detect if we're currently at a decision point by inspecting
   * the current phase and UI mode.
   */
  function detectCurrentDecision(): PhaseState | null {
    // If a modifier target selection is pending, present MODIFIER_TARGET instead
    // of re-detecting the underlying SelectModifierPhase as SELECT_MODIFIER
    if (pendingModifierAction) {
      return buildPhaseState(DecisionPhase.MODIFIER_TARGET, "SelectModifierPhase", UiMode.MODIFIER_SELECT);
    }

    const currentPhase = globalScene.phaseManager?.getCurrentPhase();
    if (!currentPhase) {
      return null;
    }

    const phaseName = currentPhase.phaseName;
    const decision = PHASE_NAME_TO_DECISION[phaseName];
    if (decision === undefined) {
      return null;
    }

    const uiMode = globalScene.ui?.getMode();
    if (uiMode === undefined || uiMode === null) {
      return null;
    }

    // Check if the UI is in a decision mode for this phase
    const isDecisionMode = DECISION_UI_MODES.has(uiMode);
    if (!isDecisionMode) {
      return null;
    }

    // Verify the handler is active
    const handler = globalScene.ui?.getHandler();
    if (!handler || !handler.active) {
      return null;
    }

    return buildPhaseState(decision, phaseName, uiMode);
  }

  /**
   * Check if the game has ended by inspecting game state.
   */
  function checkGameOver(): boolean {
    if (gameOverFlag) {
      return true;
    }

    const currentPhase = globalScene.phaseManager?.getCurrentPhase();
    if (currentPhase?.is("GameOverPhase")) {
      gameOverFlag = true;
      return true;
    }

    // Check if all player pokemon are fainted
    const party = globalScene.getPlayerParty?.();
    if (party && party.length > 0 && party.every(p => p.isFainted())) {
      gameOverFlag = true;
      return true;
    }

    return false;
  }

  // ── Public API ─────────────────────────────────────────────────────

  const router: PhaseRouter = {
    isAtDecisionPoint(): boolean {
      return currentPhaseState !== null || detectCurrentDecision() !== null;
    },

    getCurrentPhaseState(): PhaseState | null {
      if (currentPhaseState) {
        return currentPhaseState;
      }
      const detected = detectCurrentDecision();
      if (detected) {
        currentPhaseState = detected;
      }
      return currentPhaseState;
    },

    async executeAction(action: number): Promise<void> {
      await executeActionInternal(action);
      // Flush microtask queue to allow phase transitions
      await new Promise<void>(r => setTimeout(r, 0));
    },

    async advanceToNextDecision(): Promise<PhaseState> {
      return waitForNextDecision();
    },

    onDecision(callback: (state: PhaseState) => void): void {
      decisionCallbacks.push(callback);
    },

    isGameOver(): boolean {
      return gameOverFlag || checkGameOver();
    },

    isVictory(): boolean {
      return victoryFlag;
    },

    drainInfoMessages(): string[] {
      const msgs = [...infoMessages];
      infoMessages = [];
      return msgs;
    },

    destroy(): void {
      destroyed = true;

      // Restore original prototypes
      UI.prototype.setMode = originalSetMode;
      Phase.prototype.end = originalPhaseEnd;

      // Clean up
      if (advanceTimeoutId !== null) {
        clearTimeout(advanceTimeoutId);
        advanceTimeoutId = null;
      }

      if (pendingDecision) {
        pendingDecision.reject(new Error("PhaseRouter destroyed"));
        pendingDecision = null;
      }

      currentPhaseState = null;
      decisionCallbacks = [];
    },
  };

  return router;
}

// ─── Convenience: Default Action Selection ──────────────────────────

/**
 * Given a PhaseState, pick a sensible default action.
 * Used for auto-handling phases the RL agent shouldn't control.
 */
export function pickDefaultAction(state: PhaseState): number {
  if (state.validActions.length === 0) {
    return 0;
  }

  switch (state.phase) {
    case DecisionPhase.COMMAND:
      // Prefer the first available fight action
      for (const a of state.validActions) {
        if (a >= ACTION_FIGHT_ENEMY_START && a < ACTION_FIGHT_ENEMY_START + MAX_MOVES) {
          return a;
        }
      }
      return state.validActions[0];

    case DecisionPhase.SELECT_MODIFIER:
      // Default: skip items
      if (state.actionMask[ACTION_SKIP]) {
        return ACTION_SKIP;
      }
      return state.validActions[0];

    case DecisionPhase.MODIFIER_TARGET:
      // Default: apply to first eligible pokemon
      return state.validActions.find(a => a >= ACTION_PARTY_TARGET_START) ?? state.validActions[0];

    case DecisionPhase.SWITCH:
      // Pick first available switch slot
      return state.validActions[0];

    case DecisionPhase.CHECK_SWITCH:
      // Default: decline
      if (state.actionMask[ACTION_SKIP]) {
        return ACTION_SKIP;
      }
      return state.validActions[0];

    case DecisionPhase.LEARN_MOVE:
      // Default: don't learn
      if (state.actionMask[ACTION_SKIP]) {
        return ACTION_SKIP;
      }
      return state.validActions[0];

    case DecisionPhase.GAME_OVER:
      // Default: don't retry
      return 1;

    case DecisionPhase.MYSTERY_ENCOUNTER:
      // Default: first option
      return 0;

    case DecisionPhase.SELECT_BIOME:
      // Default: first biome
      return 0;

    case DecisionPhase.REVIVAL_BLESSING:
      // Default: first fainted pokemon
      return state.validActions[0];

    default:
      return state.validActions[0];
  }
}
