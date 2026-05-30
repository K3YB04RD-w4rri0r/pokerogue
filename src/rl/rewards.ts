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

export class RewardCalculator {
  private config: RewardConfig;
  private prevSnapshot: StateSnapshot | null = null;
  /** Monotonically increasing player faint counter (RB2: survives revives) */
  private cumulativePlayerFaints = 0;
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
      // HP damage dealt to enemies
      const minEnemyLen = Math.min(pre.enemyHpRatios.length, postSnap.enemyHpRatios.length);
      for (let i = 0; i < minEnemyLen; i++) {
        const delta = pre.enemyHpRatios[i] - postSnap.enemyHpRatios[i];
        if (delta > 0) {
          reward += this.config.hpDamageDealt * delta;
        }
      }

      // HP damage taken by player
      const minPlayerLen = Math.min(pre.playerHpRatios.length, postSnap.playerHpRatios.length);
      for (let i = 0; i < minPlayerLen; i++) {
        const delta = pre.playerHpRatios[i] - postSnap.playerHpRatios[i];
        if (delta > 0) {
          reward += this.config.hpDamageTaken * delta;
        }
      }
    }

    // Enemy KOs
    const enemyKoDelta = postSnap.enemyFaints - pre.enemyFaints;
    if (enemyKoDelta > 0) {
      reward += this.config.enemyKo * enemyKoDelta;
    }

    // [RB2] Player KOs — use cumulative counter that never decrements on revive
    if (postSnap.playerFaints > this.cumulativePlayerFaints) {
      const newFaints = postSnap.playerFaints - this.cumulativePlayerFaints;
      reward += this.config.playerKo * newFaints;
      this.cumulativePlayerFaints = postSnap.playerFaints;
    }

    // [RB4] Wave cleared — reward per wave advanced, not just once
    if (postSnap.waveIndex > pre.waveIndex) {
      const wavesAdvanced = postSnap.waveIndex - pre.waveIndex;
      // Boss wave check on the final wave cleared
      const isBossWave = postSnap.waveIndex % 10 === 0;
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

    // Modifier selection
    if (modifierTier >= 0) {
      reward += this.config.modifierSelected + this.config.modifierTierBonus * modifierTier;
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
    this.cumulativePlayerFaints = 0;
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
