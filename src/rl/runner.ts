/**
 * Dual-mode RL Runner for PokeRogue.
 *
 * Ties together headless-boot.ts (initialization/reset/destroy),
 * phase-router.ts (decision detection & action routing), spaces.ts
 * (observation encoding), and rewards.ts (reward computation) into a
 * clean RL environment API with init/reset/step/close lifecycle.
 *
 * Supports two modes:
 * - "headless": Full game logic with mocked rendering (fast, for training)
 * - "rendered": Real Phaser rendering (for visualization / human monitoring)
 *
 * Does NOT import from 'vitest' anywhere.
 */

import type { BattleScene } from "#app/battle-scene";
import {
  type PhaseRouter,
  type PhaseState,
  DecisionPhase as RouterDecisionPhase,
  createPhaseRouter,
  pickDefaultAction,
} from "#rl/phase-router";
import {
  encodeObservation,
  OBSERVATION_DIM,
  ACTION_SPACE_SIZE,
  ACTION_RUN,
  ACTION_SKIP,
  MAX_REWARD_OPTIONS,
  MAX_SHOP_OPTIONS,
} from "#rl/spaces";
import { buildGameState } from "#rl/state-builder";
import { RewardCalculator, type RewardConfig, type StateSnapshot } from "#rl/rewards";
import { getAvailableModifiers } from "#rl/modifier-api";
import { globalScene } from "#app/global-scene";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type RunnerMode = "headless" | "rendered";

export interface RunnerConfig {
  mode: RunnerMode;
  seed?: string;
  /** Number of starter Pokemon (default 3) */
  numStarters?: number;
  /** Species IDs for starters (if not provided, picks first 3 Gen I starters) */
  starterSpecies?: number[];
  /** Log decisions to console */
  logDecisions?: boolean;
  /** Reward function configuration overrides */
  rewardConfig?: Partial<RewardConfig>;
  /** Maximum decision log entries per episode (default 10000) */
  maxDecisionLogSize?: number;
  /** Timeout in ms for advancing to next decision (default 30000) */
  advanceTimeoutMs?: number;
}

export interface StepResult {
  observation: Float32Array;
  reward: number;
  done: boolean;
  truncated: boolean;
  info: {
    wave: number;
    turn: number;
    phase: string;
    action: number;
    actionValid: boolean;
    pokemonFainted: number;
    enemyFainted: number;
    [key: string]: unknown;
  };
}

export interface DecisionLogEntry {
  wave: number;
  turn: number;
  phase: string;
  action: number;
  actionName: string;
  metadata: Record<string, unknown>;
  timestamp: number;
}

// ---------------------------------------------------------------------------
// Resolved config with defaults
// ---------------------------------------------------------------------------

interface ResolvedConfig {
  mode: RunnerMode;
  seed: string;
  numStarters: number;
  starterSpecies: number[];
  logDecisions: boolean;
  rewardConfig: Partial<RewardConfig>;
  maxDecisionLogSize: number;
  advanceTimeoutMs: number;
}

function resolveConfig(config: RunnerConfig): ResolvedConfig {
  // Default starters: Bulbasaur(1), Charmander(4), Squirtle(7)
  const DEFAULT_STARTERS = [1, 4, 7];
  return {
    mode: config.mode,
    seed: config.seed ?? "rl-runner",
    numStarters: config.numStarters ?? 3,
    starterSpecies: config.starterSpecies ?? DEFAULT_STARTERS,
    logDecisions: config.logDecisions ?? false,
    rewardConfig: config.rewardConfig ?? {},
    maxDecisionLogSize: config.maxDecisionLogSize ?? 10_000,
    advanceTimeoutMs: config.advanceTimeoutMs ?? 30_000,
  };
}

// ---------------------------------------------------------------------------
// Action name mapping (for decision logging)
// ---------------------------------------------------------------------------

function getActionName(action: number): string {
  if (action >= 0 && action < 4) return `FIGHT_ENEMY_MOVE_${action}`;
  if (action >= 4 && action < 8) return `FIGHT_ENEMY2_MOVE_${action - 4}`;
  if (action >= 8 && action < 12) return `FIGHT_ALLY_MOVE_${action - 8}`;
  if (action >= 12 && action < 17) return `SWITCH_SLOT_${action - 12 + 1}`;
  if (action >= 17 && action < 22) return `BALL_TYPE_${action - 17}`;
  if (action === 22) return "RUN";
  if (action >= 23 && action < 27) return `TERA_ENEMY_MOVE_${action - 23}`;
  if (action >= 27 && action < 31) return `TERA_ENEMY2_MOVE_${action - 27}`;
  if (action >= 31 && action < 35) return `TERA_ALLY_MOVE_${action - 31}`;
  if (action >= 35 && action < 38) return `SELECT_REWARD_${action - 35}`;
  if (action === 38) return "REROLL";
  if (action === 39) return "SKIP";
  if (action >= 40 && action < 52) return `BUY_SHOP_${action - 40}`;
  if (action >= 52 && action < 58) return `PARTY_TARGET_${action - 52}`;
  return `UNKNOWN_${action}`;
}

// ---------------------------------------------------------------------------
// RLRunner Class
// ---------------------------------------------------------------------------

export class RLRunner {
  private config: ResolvedConfig;
  private scene: BattleScene | null = null;
  private router: PhaseRouter | null = null;
  private rewardCalc: RewardCalculator;
  private decisionLog: DecisionLogEntry[] = [];
  private initialized = false;
  private episodeActive = false;
  private stepCount = 0;
  private lastPhaseState: PhaseState | null = null;
  private preActionSnapshot: StateSnapshot | null = null;
  private fled = false;
  private lastModifierTier = -1;
  private cumulativePlayerFaints = 0;
  private cumulativeEnemyFaints = 0;

  constructor(config: RunnerConfig) {
    this.config = resolveConfig(config);
    this.rewardCalc = new RewardCalculator(this.config.rewardConfig);
  }

  // ─── Lifecycle ────────────────────────────────────────────────────

  /**
   * Initialize the game engine (call once).
   * For headless mode: calls initHeadless() from headless-boot.ts.
   * For rendered mode: throws (not yet implemented).
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (this.config.mode === "rendered") {
      throw new Error(
        "[RLRunner] Rendered mode is not yet implemented. Use 'headless' mode for now.",
      );
    }

    const { initHeadless } = await import("#rl/headless-boot");
    this.scene = await initHeadless({
      seed: this.config.seed,
      bypassLogin: true,
      quietConsole: true,
    });

    this.initialized = true;
  }

  /**
   * Reset to start a new episode. Returns initial observation.
   *
   * Handles the full game start sequence:
   * 1. Reset the game scene
   * 2. Create PhaseRouter
   * 3. Auto-handle TitlePhase, SelectGenderPhase, SelectStarterPhase
   * 4. Advance to the first real RL decision (CommandPhase of wave 1)
   */
  async reset(): Promise<{ observation: Float32Array; info: Record<string, unknown> }> {
    if (!this.initialized) {
      throw new Error("[RLRunner] Must call init() before reset()");
    }

    // Clean up previous router if any
    if (this.router) {
      this.router.destroy();
      this.router = null;
    }

    // Reset scene for new episode
    if (this.config.mode === "headless") {
      const { resetHeadless } = await import("#rl/headless-boot");
      this.scene = await resetHeadless({
        seed: this.config.seed,
        bypassLogin: true,
        quietConsole: true,
      });
    }

    // Reset episode state
    this.rewardCalc.reset();
    this.decisionLog = [];
    this.stepCount = 0;
    this.episodeActive = true;
    this.lastPhaseState = null;
    this.preActionSnapshot = null;
    this.fled = false;
    this.lastModifierTier = -1;
    this.cumulativePlayerFaints = 0;
    this.cumulativeEnemyFaints = 0;

    // Create new phase router
    this.router = createPhaseRouter();

    // Handle game start sequence: TitlePhase -> setup -> first CommandPhase
    await this.handleGameStartSequence();

    // Build initial observation
    const observation = this.getObservation();
    const state = this.router.getCurrentPhaseState();

    return {
      observation,
      info: {
        wave: globalScene.currentBattle?.waveIndex ?? 0,
        turn: globalScene.currentBattle?.turn ?? 0,
        phase: state?.phase ?? "unknown",
        actionMask: this.getActionMask(),
      },
    };
  }

  /**
   * Take one action. Returns step result.
   *
   * Flow:
   * 1. Get current PhaseState
   * 2. Validate action against mask
   * 3. Snapshot pre-action state
   * 4. Execute action via router
   * 5. Advance to next decision point
   * 6. Compute observation and reward
   * 7. Return StepResult
   */
  async step(action: number): Promise<StepResult> {
    if (!this.episodeActive) {
      throw new Error("[RLRunner] Episode is not active. Call reset() first.");
    }
    if (!this.router) {
      throw new Error("[RLRunner] Router not initialized. Call reset() first.");
    }
    if (action < 0 || action >= ACTION_SPACE_SIZE) {
      throw new Error(
        `[RLRunner] Invalid action: ${action}. Must be in range [0, ${ACTION_SPACE_SIZE - 1}].`,
      );
    }

    this.stepCount++;

    // 1. Get current phase state
    const phaseState = this.router.getCurrentPhaseState();
    if (!phaseState) {
      // If there's no decision point, try advancing first
      console.warn("[RLRunner] No current phase state at step() entry. Attempting to advance...");
      const advanced = await this.safeAdvance();
      if (!advanced || this.router.isGameOver()) {
        return this.buildTerminalResult(action);
      }
    }

    const currentState = this.router.getCurrentPhaseState();
    const phaseName = currentState?.phase ?? "unknown";

    // 2. Validate action against mask
    let actionValid = true;
    let effectiveAction = action;
    if (currentState && !currentState.actionMask[action]) {
      actionValid = false;
      if (currentState.validActions.length > 0) {
        effectiveAction = currentState.validActions[0];
        if (this.config.logDecisions) {
          console.warn(
            `[RLRunner] Action ${action} invalid for ${phaseName}. ` +
            `Falling back to ${effectiveAction} (${getActionName(effectiveAction)}).`,
          );
        }
      } else {
        // No valid actions -- this shouldn't happen but handle gracefully
        console.error(
          `[RLRunner] No valid actions in phase ${phaseName}. Attempting auto-advance.`,
        );
        const advanced = await this.safeAdvance();
        if (!advanced || this.router.isGameOver()) {
          return this.buildTerminalResult(action);
        }
        return this.buildStepResult(action, false, phaseName, 0);
      }
    }

    // 3. Snapshot pre-action state
    this.preActionSnapshot = this.takeSnapshot();
    this.rewardCalc.savePreActionSnapshot(this.preActionSnapshot);

    // Track whether this is a run action or modifier selection
    this.fled = effectiveAction === ACTION_RUN;
    this.lastModifierTier = -1;
    if (currentState?.phase === RouterDecisionPhase.SELECT_MODIFIER) {
      this.lastModifierTier = this.getModifierTierForAction(effectiveAction);
    }

    // 4. Execute action via router
    await this.router.executeAction(effectiveAction);

    // 5. Advance to next decision point (handle auto-phases along the way)
    const done = await this.advanceToNextRLDecision();

    // 6. Compute reward
    const postSnapshot = this.takeSnapshot();
    const isVictory = this.router.isVictory();
    const reward = this.rewardCalc.computeReward(
      postSnapshot,
      done,
      isVictory,
      this.fled,
      this.lastModifierTier,
    );

    // 7. Log decision
    this.logDecision(phaseName, effectiveAction, currentState?.metadata ?? {});

    // 8. Update cumulative counters
    this.cumulativePlayerFaints = this.countPlayerFaints();
    this.cumulativeEnemyFaints = this.countEnemyFaints();

    if (done) {
      this.episodeActive = false;
    }

    // 9. Build and return StepResult
    return this.buildStepResult(action, actionValid, phaseName, reward, done);
  }

  /**
   * Get current observation without stepping.
   */
  getObservation(): Float32Array {
    if (!this.scene) {
      return new Float32Array(OBSERVATION_DIM);
    }

    try {
      return this.buildCurrentObservation();
    } catch (e) {
      console.error("[RLRunner] Error building observation:", e);
      return new Float32Array(OBSERVATION_DIM);
    }
  }

  /**
   * Get current action mask (true = valid action).
   */
  getActionMask(): boolean[] {
    if (!this.router) {
      return new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    }

    const state = this.router.getCurrentPhaseState();
    if (!state) {
      return new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
    }

    return state.actionMask;
  }

  /**
   * Check if current episode is done.
   */
  isDone(): boolean {
    if (!this.episodeActive) return true;
    if (!this.router) return true;
    return this.router.isGameOver();
  }

  /**
   * Clean shutdown.
   */
  async close(): Promise<void> {
    if (this.router) {
      this.router.destroy();
      this.router = null;
    }

    if (this.config.mode === "headless") {
      const { destroyHeadless } = await import("#rl/headless-boot");
      await destroyHeadless();
    }

    this.scene = null;
    this.initialized = false;
    this.episodeActive = false;
  }

  /**
   * Get decision log for current episode.
   */
  getDecisionLog(): DecisionLogEntry[] {
    return [...this.decisionLog];
  }

  /**
   * Flush decision log (free memory during long episodes).
   */
  flushDecisionLog(): DecisionLogEntry[] {
    const log = this.decisionLog;
    this.decisionLog = [];
    return log;
  }

  /**
   * Get episode statistics.
   */
  getEpisodeStats(): Record<string, unknown> {
    return {
      stepCount: this.stepCount,
      wave: globalScene?.currentBattle?.waveIndex ?? 0,
      turn: globalScene?.currentBattle?.turn ?? 0,
      playerFaints: this.cumulativePlayerFaints,
      enemyFaints: this.cumulativeEnemyFaints,
      episodeActive: this.episodeActive,
      decisionLogSize: this.decisionLog.length,
    };
  }

  // ─── Game Start Sequence ──────────────────────────────────────────

  /**
   * Handle the game startup phases before the first battle:
   * TitlePhase -> SelectGenderPhase -> SelectStarterPhase -> EncounterPhase -> CommandPhase
   *
   * This mirrors the test harness's classicMode.startBattle() approach:
   * - At TitlePhase: set game mode to CLASSIC, generate starters, call initBattle()
   * - Skip SelectGenderPhase, SelectStarterPhase (handled automatically)
   * - Advance through EncounterPhase to first CommandPhase
   */
  private async handleGameStartSequence(): Promise<void> {
    if (!this.router) {
      throw new Error("[RLRunner] Router not available for game start sequence");
    }

    // Wait for the first decision point (usually TitlePhase)
    let state = await this.router.advanceToNextDecision();

    // Handle pre-battle setup phases automatically
    const MAX_SETUP_PHASES = 20; // Safety limit to avoid infinite loops
    let setupPhaseCount = 0;

    while (setupPhaseCount < MAX_SETUP_PHASES) {
      setupPhaseCount++;

      if (this.router.isGameOver()) {
        throw new Error("[RLRunner] Game ended during setup sequence");
      }

      const phase = state.phase;

      // If we've reached a "real" RL decision phase, we're done with setup
      if (this.isRLDecisionPhase(phase)) {
        this.lastPhaseState = state;
        return;
      }

      // Handle setup/auto phases
      if (phase === RouterDecisionPhase.TITLE) {
        await this.handleTitlePhase();
      } else if (phase === RouterDecisionPhase.SELECT_GENDER) {
        await this.handleSelectGenderPhase();
      } else if (phase === RouterDecisionPhase.SELECT_STARTER) {
        await this.handleSelectStarterPhase();
      } else if (
        phase === RouterDecisionPhase.EVOLUTION ||
        phase === RouterDecisionPhase.FORM_CHANGE
      ) {
        // Let evolution/form-change proceed
        const defaultAction = pickDefaultAction(state);
        await this.router.executeAction(defaultAction);
      } else if (phase === RouterDecisionPhase.CHECK_SWITCH) {
        // Decline the check-switch during setup (not an RL decision yet)
        if (state.actionMask[ACTION_SKIP]) {
          await this.router.executeAction(ACTION_SKIP);
        } else {
          const defaultAction = pickDefaultAction(state);
          await this.router.executeAction(defaultAction);
        }
      } else {
        // Unknown setup phase -- use default action
        const defaultAction = pickDefaultAction(state);
        await this.router.executeAction(defaultAction);
      }

      // Advance to next decision
      state = await this.router.advanceToNextDecision();
    }

    // If we exit the loop without reaching an RL decision, something is wrong
    throw new Error(
      `[RLRunner] Failed to reach first RL decision after ${MAX_SETUP_PHASES} setup phases. ` +
      `Last phase: ${state?.phase ?? "unknown"}`,
    );
  }

  /**
   * Handle TitlePhase: set game mode to CLASSIC and bypass with starter generation.
   *
   * This follows the same approach as ClassicModeHelper.runToSummon():
   * 1. Set game mode to CLASSIC
   * 2. Generate starters using generateStarters()
   * 3. Call selectStarterPhase.initBattle(starters)
   * 4. Push EncounterPhase
   */
  private async handleTitlePhase(): Promise<void> {
    // Dynamic imports to avoid loading game code before init
    const { getGameMode } = await import("#app/game-mode");
    const { GameModes } = await import("#enums/game-modes");
    const { SelectStarterPhase } = await import("#phases/select-starter-phase");
    const { EncounterPhase } = await import("#phases/encounter-phase");
    const { generateStarters } = await import("#test/test-utils/game-manager-utils");
    type SpeciesId = import("#enums/species-id").SpeciesId;

    const scene = globalScene;

    // Set game mode
    scene.gameMode = getGameMode(GameModes.CLASSIC);

    // Slice to requested number of starters and cast to SpeciesId[]
    // (SpeciesId is a numeric enum, and our config stores raw numbers)
    const speciesIds = this.config.starterSpecies.slice(0, this.config.numStarters) as SpeciesId[];

    // Generate starters (same function used by test harness)
    const starters = generateStarters(scene as BattleScene, speciesIds);

    // Create starter phase and push encounter
    const selectStarterPhase = new SelectStarterPhase();
    scene.phaseManager.pushPhase(new EncounterPhase(false));
    selectStarterPhase.initBattle(starters);
  }

  /**
   * Handle SelectGenderPhase: auto-select male gender.
   */
  private async handleSelectGenderPhase(): Promise<void> {
    if (!this.router) return;
    // Press ACTION to select the default gender
    const state = this.router.getCurrentPhaseState();
    if (state && state.validActions.length > 0) {
      await this.router.executeAction(state.validActions[0]);
    }
  }

  /**
   * Handle SelectStarterPhase: starters were already set up in TitlePhase handler,
   * so just proceed.
   */
  private async handleSelectStarterPhase(): Promise<void> {
    if (!this.router) return;
    const state = this.router.getCurrentPhaseState();
    if (state && state.validActions.length > 0) {
      await this.router.executeAction(state.validActions[0]);
    }
  }

  // ─── Decision Phase Classification ────────────────────────────────

  /**
   * Check if a phase is a "real" RL decision phase (not setup/auto).
   */
  private isRLDecisionPhase(phase: RouterDecisionPhase): boolean {
    switch (phase) {
      case RouterDecisionPhase.COMMAND:
      case RouterDecisionPhase.SELECT_TARGET:
      case RouterDecisionPhase.SELECT_MODIFIER:
      case RouterDecisionPhase.MODIFIER_TARGET:
      case RouterDecisionPhase.SWITCH:
      case RouterDecisionPhase.LEARN_MOVE:
      case RouterDecisionPhase.MYSTERY_ENCOUNTER:
      case RouterDecisionPhase.SELECT_BIOME:
      case RouterDecisionPhase.REVIVAL_BLESSING:
      case RouterDecisionPhase.GAME_OVER:
        return true;
      default:
        return false;
    }
  }

  // ─── Advance Logic ────────────────────────────────────────────────

  /**
   * Advance the game to the next RL decision point, auto-handling
   * non-RL phases (evolution, check-switch, form-change, etc.) along the way.
   *
   * @returns true if the game is over (done), false if at a new decision point.
   */
  private async advanceToNextRLDecision(): Promise<boolean> {
    if (!this.router) return true;

    const MAX_AUTO_PHASES = 50; // Safety limit
    let autoPhaseCount = 0;

    while (autoPhaseCount < MAX_AUTO_PHASES) {
      autoPhaseCount++;

      // Advance to next decision
      let state: PhaseState;
      try {
        state = await this.router.advanceToNextDecision();
      } catch (e) {
        console.error("[RLRunner] Error advancing to next decision:", e);
        return true; // Treat as done
      }

      // Check for game over
      if (this.router.isGameOver()) {
        return true;
      }

      // If it's an RL decision phase, we're at the next step
      if (this.isRLDecisionPhase(state.phase)) {
        this.lastPhaseState = state;
        return false;
      }

      // Auto-handle non-RL phases
      if (this.config.logDecisions) {
        console.log(`[RLRunner] Auto-handling phase: ${state.phase}`);
      }

      const defaultAction = pickDefaultAction(state);
      await this.router.executeAction(defaultAction);

      // Log auto-handled decisions
      this.logDecision(state.phase, defaultAction, {
        ...state.metadata,
        autoHandled: true,
      });
    }

    console.error(`[RLRunner] Exceeded ${MAX_AUTO_PHASES} auto-handled phases without reaching RL decision.`);
    return true; // Treat as done to prevent infinite loops
  }

  /**
   * Safely attempt to advance without throwing.
   */
  private async safeAdvance(): Promise<boolean> {
    try {
      if (!this.router) return false;
      const state = await this.router.advanceToNextDecision();
      this.lastPhaseState = state;
      return !this.router.isGameOver();
    } catch (e) {
      console.error("[RLRunner] Error during safe advance:", e);
      return false;
    }
  }

  // ─── Observation Building ─────────────────────────────────────────

  /**
   * Build the full observation vector from current game state.
   * Uses buildGameState() to produce a JSON dict, then encodes it.
   */
  private buildCurrentObservation(): Float32Array {
    if (!globalScene) {
      return new Float32Array(OBSERVATION_DIM);
    }

    const battle = globalScene.currentBattle;
    const arena = globalScene.arena;

    if (!battle || !arena) {
      return new Float32Array(OBSERVATION_DIM);
    }

    const phaseState = this.router?.getCurrentPhaseState() ?? null;
    const gameState = buildGameState(phaseState, this.stepCount);
    return encodeObservation(gameState);
  }

  // ─── State Helpers ────────────────────────────────────────────────

  private takeSnapshot(): StateSnapshot {
    const playerParty = globalScene?.getPlayerParty?.() ?? [];
    const enemyParty = globalScene?.getEnemyParty?.() ?? [];
    const battle = globalScene?.currentBattle;

    return this.rewardCalc.snapshot(
      playerParty,
      enemyParty,
      this.countEnemyFaints(),
      this.countPlayerFaints(),
      battle?.waveIndex ?? 0,
      globalScene?.money ?? 0,
    );
  }

  private countPlayerFaints(): number {
    const party = globalScene?.getPlayerParty?.() ?? [];
    return party.filter((p: { isFainted: () => boolean }) => p.isFainted()).length;
  }

  private countEnemyFaints(): number {
    return globalScene?.currentBattle?.enemyFaints ?? 0;
  }

  private getModifierTierForAction(action: number): number {
    const modifiers = getAvailableModifiers();
    if (!modifiers) return -1;

    // Reward selection (35-37)
    if (action >= 35 && action < 35 + MAX_REWARD_OPTIONS) {
      const rewardIndex = action - 35;
      if (rewardIndex < modifiers.rewards.length) {
        return modifiers.rewards[rewardIndex].tier;
      }
    }

    // Shop items (40-51)
    if (action >= 40 && action < 40 + MAX_SHOP_OPTIONS) {
      const shopIndex = action - 40;
      if (shopIndex < modifiers.shop.length) {
        return modifiers.shop[shopIndex].tier;
      }
    }

    // Skip or reroll return -1 (no modifier selected)
    return -1;
  }

  // ─── Result Building ──────────────────────────────────────────────

  private buildStepResult(
    action: number,
    actionValid: boolean,
    phaseName: string,
    reward: number,
    done = false,
  ): StepResult {
    const observation = this.getObservation();
    const truncated = false; // Can be extended for max-wave truncation

    return {
      observation,
      reward,
      done,
      truncated,
      info: {
        wave: globalScene?.currentBattle?.waveIndex ?? 0,
        turn: globalScene?.currentBattle?.turn ?? 0,
        phase: phaseName,
        action,
        actionValid,
        pokemonFainted: this.countPlayerFaints(),
        enemyFainted: this.countEnemyFaints(),
        stepCount: this.stepCount,
        actionMask: this.getActionMask(),
      },
    };
  }

  private buildTerminalResult(action: number): StepResult {
    const postSnapshot = this.takeSnapshot();
    const isVictory = this.router?.isVictory() ?? false;

    // Compute terminal reward
    let reward = 0;
    if (this.preActionSnapshot) {
      this.rewardCalc.savePreActionSnapshot(this.preActionSnapshot);
      reward = this.rewardCalc.computeReward(postSnapshot, true, isVictory, false, -1);
    }

    this.episodeActive = false;

    return {
      observation: this.getObservation(),
      reward,
      done: true,
      truncated: false,
      info: {
        wave: globalScene?.currentBattle?.waveIndex ?? 0,
        turn: globalScene?.currentBattle?.turn ?? 0,
        phase: "game_over",
        action,
        actionValid: false,
        pokemonFainted: this.countPlayerFaints(),
        enemyFainted: this.countEnemyFaints(),
        stepCount: this.stepCount,
        isVictory,
      },
    };
  }

  // ─── Decision Logging ─────────────────────────────────────────────

  private logDecision(phase: string, action: number, metadata: Record<string, unknown>): void {
    // Enforce max log size
    if (this.decisionLog.length >= this.config.maxDecisionLogSize) {
      // Drop oldest 10% to avoid constant array manipulation
      const dropCount = Math.floor(this.config.maxDecisionLogSize * 0.1);
      this.decisionLog.splice(0, dropCount);
    }

    const entry: DecisionLogEntry = {
      wave: globalScene?.currentBattle?.waveIndex ?? 0,
      turn: globalScene?.currentBattle?.turn ?? 0,
      phase,
      action,
      actionName: getActionName(action),
      metadata,
      timestamp: Date.now(),
    };

    this.decisionLog.push(entry);

    if (this.config.logDecisions) {
      const wave = entry.wave;
      const turn = entry.turn;
      console.log(
        `[RLRunner] Wave ${wave} Turn ${turn} | ${phase} -> ${entry.actionName} (${action})`,
      );
    }
  }
}
