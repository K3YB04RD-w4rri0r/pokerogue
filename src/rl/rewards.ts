/**
 * Configurable reward function for the RL environment.
 *
 * Computes reward as a weighted sum of independent components,
 * each measuring the delta between pre- and post-action game state.
 */

import type { Pokemon } from "#field/pokemon";

// ─── Reward Configuration ─────────────────────────────────────────────

export interface RewardConfig {
  /** Reward per unit of HP ratio dealt to enemies (default: 1.0) */
  hpDamageDealt: number;
  /** Penalty per unit of HP ratio taken by player (default: -1.0) */
  hpDamageTaken: number;
  /** Reward per enemy KO (default: 5.0) */
  enemyKo: number;
  /** Penalty per player KO (default: -5.0) */
  playerKo: number;
  /** Reward for clearing a wave (default: 10.0) */
  waveCleared: number;
  /** Reward for clearing a boss wave (every 10th, default: 25.0) */
  bossWaveCleared: number;
  /** Terminal reward for winning the run (default: 100.0) */
  runWon: number;
  /** Terminal penalty for losing the run (default: -50.0) */
  runLost: number;
  /** Penalty for fleeing from battle (default: -2.0) */
  ranAway: number;
  /** Reward per unit of money gained (default: 0.01) */
  moneyGained: number;
  /** Reward for catching a Pokemon (default: 3.0) */
  pokemonCaught: number;
  /** Reward for selecting a modifier vs skipping (default: 0.5) */
  modifierSelected: number;
  /** Bonus per modifier tier (multiplied by tier 0-5, default: 0.1) */
  modifierTierBonus: number;
  /** Per-step time penalty to discourage stalling (default: -0.01) */
  turnPenalty: number;
  /** Reward per positive stat stage gained by player Pokemon (default: 0, shaped) */
  statBoostReward: number;
  /** Reward per new non-NONE status inflicted on enemy (default: 0, shaped) */
  statusInflictionReward: number;
}

export const DEFAULT_REWARD_CONFIG: Readonly<RewardConfig> = {
  hpDamageDealt: 1.0,
  hpDamageTaken: -1.0,
  enemyKo: 5.0,
  playerKo: -5.0,
  waveCleared: 10.0,
  bossWaveCleared: 25.0,
  runWon: 100.0,
  runLost: -50.0,
  ranAway: -2.0,
  moneyGained: 0.01,
  pokemonCaught: 3.0,
  modifierSelected: 0.5,
  modifierTierBonus: 0.1,
  turnPenalty: -0.01,
  statBoostReward: 0,
  statusInflictionReward: 0,
};

// ─── State Snapshot ───────────────────────────────────────────────────

/** Snapshot of game state for computing reward deltas */
export interface StateSnapshot {
  /** HP ratios of player party [0-1] indexed by position */
  playerHpRatios: number[];
  /** HP ratios of enemy party [0-1] indexed by position */
  enemyHpRatios: number[];
  /** Pokemon ids parallel to playerHpRatios — HP deltas match by id, not
   *  slot (switches reorder slots and fabricated deltas [RD3]); optional so
   *  hand-built test snapshots keep index semantics. */
  playerIds?: number[];
  /** Pokemon ids parallel to enemyHpRatios (see playerIds). */
  enemyIds?: number[];
  /** Remaining free-reward picks at snapshot time; -1 = shop not open.
   *  Gates the modifier bonus on the pick actually APPLYING [RD10]. */
  rewardsLeft?: number;
  /** Total enemy faints so far */
  enemyFaints: number;
  /** Total player faints so far */
  playerFaints: number;
  /** Current wave index */
  waveIndex: number;
  /** Player money */
  money: number;
  /** Player party size */
  partySize: number;
  /** Sum of all positive stat stages across player party (for shaped reward) */
  playerStatStageSum: number;
  /** Count of enemies with non-NONE status effect (for shaped reward) */
  enemyStatusCount: number;
}

// ─── Reward Calculator ────────────────────────────────────────────────

/** Sum of positive per-mon HP-ratio drops from pre to post, id-matched when
 *  both sides provide ids (falls back to index alignment otherwise). */
function positiveHpDelta(preRatios: number[], postRatios: number[], preIds?: number[], postIds?: number[]): number {
  let sum = 0;
  const canMatchById = preIds && postIds && preIds.length === preRatios.length && postIds.length === postRatios.length;
  if (canMatchById) {
    const preById = new Map<number, number>();
    for (let i = 0; i < preIds.length; i++) {
      if (preIds[i] !== -1) {
        preById.set(preIds[i], preRatios[i]);
      }
    }
    for (let i = 0; i < postIds.length; i++) {
      const preRatio = preById.get(postIds[i]);
      if (preRatio !== undefined) {
        const delta = preRatio - postRatios[i];
        if (delta > 0) {
          sum += delta;
        }
      }
    }
    return sum;
  }
  const n = Math.min(preRatios.length, postRatios.length);
  for (let i = 0; i < n; i++) {
    const delta = preRatios[i] - postRatios[i];
    if (delta > 0) {
      sum += delta;
    }
  }
  return sum;
}

export class RewardCalculator {
  private config: RewardConfig;
  private prevSnapshot: StateSnapshot | null = null;
  /** Monotonically increasing catch counter (RB3: survives party size decreases) */
  private cumulativeCatches = 0;

  constructor(config?: Partial<RewardConfig>) {
    this.config = { ...DEFAULT_REWARD_CONFIG, ...config };
  }

  /**
   * Take a snapshot of the current game state.
   * Call this before executing an action.
   */
  snapshot(
    playerParty: Pokemon[],
    enemyParty: Pokemon[],
    enemyFaints: number,
    playerFaints: number,
    waveIndex: number,
    money: number,
  ): StateSnapshot {
    // Sum of positive stat stages across all player Pokemon
    let playerStatStageSum = 0;
    for (const p of playerParty) {
      for (const stage of p.getStatStages()) {
        if (stage > 0) {
          playerStatStageSum += stage;
        }
      }
    }

    // Count of enemies with a non-NONE status effect
    let enemyStatusCount = 0;
    for (const p of enemyParty) {
      if (p.status && p.status.effect !== 0) {
        enemyStatusCount++;
      }
    }

    return {
      playerHpRatios: playerParty.map(p => {
        const max = p.getMaxHp();
        return max > 0 ? p.hp / max : 0;
      }),
      enemyHpRatios: enemyParty.map(p => {
        const max = p.getMaxHp();
        return max > 0 ? p.hp / max : 0;
      }),
      // Pokemon ids parallel to the ratio arrays: HP deltas are matched by id,
      // not slot — switches physically reorder party slots (SwitchSummonPhase
      // swaps party[slot] and party[fieldIndex]), and slot-indexed diffs
      // fabricated damage-dealt/taken reward on every unequal-HP switch [RD3].
      playerIds: playerParty.map(p => p.id ?? -1),
      enemyIds: enemyParty.map(p => p.id ?? -1),
      enemyFaints,
      playerFaints,
      waveIndex,
      money,
      partySize: playerParty.length,
      playerStatStageSum,
      enemyStatusCount,
    };
  }

  /**
   * Store a pre-action snapshot.
   */
  savePreActionSnapshot(snap: StateSnapshot): void {
    this.prevSnapshot = snap;
  }

  /**
   * Compute the reward for the transition from the saved pre-action snapshot
   * to the given post-action snapshot.
   *
   * @param postSnap - State after action execution
   * @param terminated - Whether the game ended
   * @param victory - Whether the run was won (only relevant if terminated)
   * @param fled - Whether the player fled from battle
   * @param modifierTier - Tier of selected modifier, or -1 if none / skipped
   */
  computeReward(
    postSnap: StateSnapshot,
    terminated: boolean,
    victory: boolean,
    fled: boolean,
    modifierTier: number,
  ): number {
    const pre = this.prevSnapshot;
    if (!pre) {
      // No previous snapshot (first step), return just the turn penalty
      return this.config.turnPenalty;
    }

    let reward = 0;

    // [RB1] Skip HP deltas across wave transitions — enemies are different Pokemon
    const sameWave = postSnap.waveIndex === pre.waveIndex;

    if (sameWave) {
      // HP deltas matched BY POKEMON ID when both snapshots carry ids (live
      // path); index-matched otherwise (hand-built test snapshots). Slot
      // matching fabricated deltas whenever a switch reordered the party [RD3].
      reward +=
        this.config.hpDamageDealt
        * positiveHpDelta(pre.enemyHpRatios, postSnap.enemyHpRatios, pre.enemyIds, postSnap.enemyIds);
      reward +=
        this.config.hpDamageTaken
        * positiveHpDelta(pre.playerHpRatios, postSnap.playerHpRatios, pre.playerIds, postSnap.playerIds);
    }

    // Enemy KOs
    const enemyKoDelta = postSnap.enemyFaints - pre.enemyFaints;
    if (enemyKoDelta > 0) {
      reward += this.config.enemyKo * enemyKoDelta;
    }

    // [RB2] Player KOs — per-step delta of the game's cumulative faint
    // counter (arena.playerFaints, snapshot source), mirroring enemyKo. The
    // old approach counted currently-fainted party members and max-tracked
    // them, so a faint→revive→faint of the same mon was NOT penalized the
    // second time (the max never rose). arena.playerFaints is monotonic
    // within an arena, so a genuine re-faint now scores; a biome-reset
    // decrement is a negative delta and is ignored, exactly like enemyKo.
    const playerKoDelta = postSnap.playerFaints - pre.playerFaints;
    if (playerKoDelta > 0) {
      reward += this.config.playerKo * playerKoDelta;
    }

    // [RB4] Wave cleared — reward per wave advanced, not just once.
    // GATED ON NOT-FLED [RD1]: a successful RUN also advances waveIndex
    // (AttemptRunPhase pushes NewBattlePhase), and paying waveCleared (+10)
    // against ranAway (-2) made flee-spam a risk-free +7.99/wave exploit —
    // a trained agent's most discoverable local optimum. Fleeing PAST a wave
    // is not clearing it.
    if (!fled && postSnap.waveIndex > pre.waveIndex) {
      const wavesAdvanced = postSnap.waveIndex - pre.waveIndex;
      // Boss wave check on the wave that was CLEARED, not the one arrived at.
      // postSnap.waveIndex is the new wave; the boss lives AT waves %10==0, so
      // beating it means arriving at %10==1. Keying on postSnap.waveIndex
      // paid the boss bonus for clearing the trivial wave 9/19/... and only
      // the regular reward for actually beating the boss (bug fixed 2026-07-08).
      const isBossWave = (postSnap.waveIndex - 1) % 10 === 0;
      if (wavesAdvanced === 1) {
        reward += isBossWave ? this.config.bossWaveCleared : this.config.waveCleared;
      } else {
        // Multiple waves: boss reward for the final wave if boss, regular for the rest
        reward += this.config.waveCleared * (wavesAdvanced - 1);
        reward += isBossWave ? this.config.bossWaveCleared : this.config.waveCleared;
      }
    }

    // Terminal rewards
    if (terminated) {
      reward += victory ? this.config.runWon : this.config.runLost;
    }

    // Fled from battle
    if (fled) {
      reward += this.config.ranAway;
    }

    // Money gained
    const moneyDelta = postSnap.money - pre.money;
    if (moneyDelta > 0) {
      reward += this.config.moneyGained * moneyDelta;
    }

    // [RB3] Pokemon caught — cumulative counter, never let delta go negative
    const partySizeDelta = postSnap.partySize - pre.partySize;
    const newCatches = Math.max(0, partySizeDelta);
    if (newCatches > 0) {
      this.cumulativeCatches += newCatches;
      reward += this.config.pokemonCaught * newCatches;
    }

    // Modifier selection — paid only when the pick APPLIED [RD10]: a pick
    // that enters the two-step target flow and gets CANCELLED leaves the
    // reward inventory and money untouched, and paying on selection made
    // pick-cancel loops farm +modifierSelected per bounce (reproduced
    // empirically with a first-legal policy). Applied means: a free pick was
    // consumed, money was spent (shop buy), or the shop closed with the pick
    // (targetless picks that end the phase). Snapshots without rewardsLeft
    // (hand-built tests) keep the old semantics.
    if (modifierTier >= 0) {
      const preRewards = pre.rewardsLeft;
      const postRewards = postSnap.rewardsLeft;
      const applied =
        preRewards === undefined
        || postRewards === undefined
        || postRewards === -1
        || (postRewards >= 0 && preRewards > postRewards)
        || postSnap.money < pre.money;
      if (applied) {
        reward += this.config.modifierSelected + this.config.modifierTierBonus * modifierTier;
      }
    }

    // Shaped reward: stat boosts gained by player
    if (this.config.statBoostReward !== 0) {
      const statDelta = postSnap.playerStatStageSum - pre.playerStatStageSum;
      if (statDelta > 0) {
        reward += this.config.statBoostReward * statDelta;
      }
    }

    // Shaped reward: status inflicted on enemy
    if (this.config.statusInflictionReward !== 0) {
      const statusDelta = postSnap.enemyStatusCount - pre.enemyStatusCount;
      if (statusDelta > 0) {
        reward += this.config.statusInflictionReward * statusDelta;
      }
    }

    // Turn penalty
    reward += this.config.turnPenalty;

    return reward;
  }

  /** Reset the calculator for a new episode */
  reset(): void {
    this.prevSnapshot = null;
    this.cumulativeCatches = 0;
  }

  /** Get current config (read-only) */
  getConfig(): Readonly<RewardConfig> {
    return this.config;
  }

  /** Update config */
  updateConfig(partial: Partial<RewardConfig>): void {
    this.config = { ...this.config, ...partial };
  }
}
