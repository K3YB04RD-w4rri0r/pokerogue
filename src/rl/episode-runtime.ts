/**
 * Shared per-episode orchestration for the two RL transports — the headless
 * CLI (cli.ts, JSON-lines over stdio) and the rendered browser bridge
 * (browser-bridge.ts, WebSocket). One implementation of:
 *
 *   - the setup-phase set both loops auto-play,
 *   - invalid-action fallback (mask-validated execution),
 *   - reward bookkeeping (RewardCalculator snapshots, fled/tier tracking),
 *   - the terminal gameState patch (at game over the live scene is already
 *     post-reset — cleared party, next battle from an unseeded RNG — so the
 *     last decision state is reused and its phase sub-dict patched to
 *     terminal truth: game_over one-hot, all-false mask).
 *
 * Keeping these in one module is what makes a rendered episode report the
 * same rewards a headless training episode would for the same trajectory.
 *
 * NOTE: touches live game state (globalScene) — headless code must import
 * this dynamically AFTER initHeadless(), same rule as state-builder.ts.
 */

import { globalScene } from "#app/global-scene";
import { getAvailableModifiers } from "#rl/modifier-api";
import type { PhaseState } from "#rl/phase-router";
import { DecisionPhase } from "#rl/phase-router";
import type { RewardConfig, StateSnapshot } from "#rl/rewards";
import { RewardCalculator } from "#rl/rewards";
import {
  ACTION_BUY_SHOP_START,
  ACTION_RUN,
  ACTION_SELECT_REWARD_START,
  ACTION_SPACE_SIZE,
  MAX_REWARD_OPTIONS,
  MAX_SHOP_OPTIONS,
} from "#rl/spaces";

/** Setup phases auto-played by the drivers (never surfaced to the agent as decisions). */
export const SETUP_PHASES: ReadonlySet<string> = new Set([
  DecisionPhase.TITLE,
  DecisionPhase.SELECT_GENDER,
  DecisionPhase.SELECT_STARTER,
  DecisionPhase.EVOLUTION,
  DecisionPhase.FORM_CHANGE,
]);

/**
 * Validate an agent action against the state's mask; invalid actions fall
 * back to the first valid action (both transports behave identically).
 */
export function resolveExecutedAction(state: PhaseState, action: number): { executed: number; wasValid: boolean } {
  const wasValid = !!state.actionMask[action];
  return { executed: wasValid ? action : (state.validActions[0] ?? 0), wasValid };
}

/**
 * Patch a decision gameState into a self-consistent TERMINAL state:
 * game_over phase one-hot, all-false action mask, victory verdict.
 */
export function buildTerminalGameState(base: Record<string, unknown>, isVictory: boolean): Record<string, unknown> {
  const terminalMask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
  return {
    ...base,
    phase: {
      ...((base.phase as Record<string, unknown>) ?? {}),
      current_phase: "game_over",
      action_mask: terminalMask,
      valid_actions: [],
      is_game_over: true,
      is_victory: isVictory,
    },
  };
}

/** Tier of the modifier a reward/shop action would select, or -1. */
function getModifierTier(action: number): number {
  try {
    const modifiers = getAvailableModifiers();
    if (!modifiers) {
      return -1;
    }
    if (
      action >= ACTION_SELECT_REWARD_START
      && action < ACTION_SELECT_REWARD_START + MAX_REWARD_OPTIONS
      && action - ACTION_SELECT_REWARD_START < modifiers.rewards.length
    ) {
      return modifiers.rewards[action - ACTION_SELECT_REWARD_START].tier;
    }
    if (
      action >= ACTION_BUY_SHOP_START
      && action < ACTION_BUY_SHOP_START + MAX_SHOP_OPTIONS
      && action - ACTION_BUY_SHOP_START < modifiers.shop.length
    ) {
      return modifiers.shop[action - ACTION_BUY_SHOP_START].tier;
    }
  } catch {
    /* not in a modifier phase */
  }
  return -1;
}

/**
 * Reward bookkeeping for one episode. Mirrors what headless training sees:
 * call {@linkcode rewardOnArrival} when a decision (or terminal) state
 * arrives, {@linkcode noteDecisionState} with the gameState sent for it, and
 * {@linkcode notePreAction} just before executing the chosen action (the
 * modifier phase is gone once the action resolves, so tier/snapshot must be
 * taken pre-execution).
 */
export class EpisodeRewardTracker {
  private readonly calc: RewardCalculator;
  private lastFled = false;
  private lastTier = -1;
  /**
   * Pre-action snapshot of the final step, kept for the terminal reward: at
   * game over the live scene is already post-reset (cleared party, starting
   * money), so snapshotting it would inject spurious deltas.
   */
  private lastSnapshot: StateSnapshot | null = null;
  /** Last decision-point gameState — reused as the terminal state's base. */
  private lastGameState: Record<string, unknown> | null = null;

  constructor(rewardConfig?: Partial<RewardConfig>) {
    this.calc = new RewardCalculator(rewardConfig);
  }

  private takeSnapshot(): StateSnapshot {
    const scene = globalScene;
    const playerParty = scene?.getPlayerParty?.() ?? [];
    const enemyParty = scene?.getEnemyParty?.() ?? [];
    return this.calc.snapshot(
      playerParty,
      enemyParty,
      scene?.currentBattle?.enemyFaints ?? 0,
      // arena.playerFaints is the game's CUMULATIVE player-faint counter (the
      // player analog to currentBattle.enemyFaints), monotonic within an
      // arena. The old point-in-time filter(isFainted) under-counted deaths
      // after a revive — see rewards.ts [RB2].
      (scene as { arena?: { playerFaints?: number } })?.arena?.playerFaints ?? 0,
      scene?.currentBattle?.waveIndex ?? 0,
      scene?.money ?? 0,
    );
  }

  /**
   * Reward earned by the PREVIOUS action (0 on the very first state). At
   * terminal the pre-action snapshot stands in for the post-reset scene:
   * all deltas zero, only terminal/fled/tier components apply.
   */
  rewardOnArrival(step: number, terminal: boolean, victory: boolean): number {
    const postSnap = terminal && this.lastSnapshot ? this.lastSnapshot : this.takeSnapshot();
    return step > 0 ? this.calc.computeReward(postSnap, terminal, victory, this.lastFled, this.lastTier) : 0;
  }

  /** Remember the gameState sent for this decision (terminal base). */
  noteDecisionState(gameState: Record<string, unknown>): void {
    this.lastGameState = gameState;
  }

  /** The last decision gameState, or null before the first decision. */
  getLastGameState(): Record<string, unknown> | null {
    return this.lastGameState;
  }

  /** Pre-action bookkeeping for the NEXT step's reward. */
  notePreAction(state: PhaseState, executed: number): void {
    this.lastSnapshot = this.takeSnapshot();
    this.calc.savePreActionSnapshot(this.lastSnapshot);
    this.lastFled = executed === ACTION_RUN;
    this.lastTier = state.phase === DecisionPhase.SELECT_MODIFIER ? getModifierTier(executed) : -1;
  }
}
