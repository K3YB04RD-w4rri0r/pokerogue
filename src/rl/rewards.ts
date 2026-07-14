/**
 * Configurable reward function for the RL environment.
 *
 * Computes reward as a weighted sum of independent components,
 * each measuring the delta between pre- and post-action game state.
 */

import { StatusEffect } from "#enums/status-effect";
import type { Pokemon } from "#field/pokemon";

// ─── Reward Configuration ─────────────────────────────────────────────

/** How a capped/livelocked episode ended — drives {@linkcode RewardCalculator.episodeEndAdjustment}. */
export type EpisodeEndReason = "livelock" | "wave_cap" | "step_cap";

export interface RewardConfig {
  /** Reward per unit of NEW-LOW HP ratio dealt to enemies (default: 1.0).
   *  Pays only when an enemy's HP ratio drops below its per-episode minimum
   *  [RD6]: re-dealt HP after an enemy heal earned unbounded reward. */
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
  /** Terminal reward for winning the run (default: 100.0). Fires only on a
   *  true wave-200 classic victory — under a wave cap < 200 the live win
   *  signal is {@linkcode waveCapReached} [RD5]. */
  runWon: number;
  /** Terminal penalty for losing the run (default: -50.0) */
  runLost: number;
  /** Penalty for fleeing from battle (default: -2.0) */
  ranAway: number;
  /** Money-gain weight on a CUMULATIVE log scale (default: 0.2):
   *  w * (ln(1+G_post) - ln(1+G_pre)) where G is total money gained this
   *  episode. Telescopes to w*ln(1+total) no matter how income is split —
   *  per-delta log was superadditive (ln concavity), letting per-hit money
   *  items (Golden Punch) mint +0.5..1.5/step indefinitely [RD4].
   *  Replaces the linear `moneyGained` (a wave-50 Relic Gold paid +77). */
  moneyGainedLog: number;
  /** Money-spend penalty per ln(1+spent) of each negative delta
   *  (default: -0.05). Per-delta on purpose: splitting a spend only raises
   *  the charge. Prices reroll-fishing (250*ceil(wave/10)*2^n cost curve
   *  => -0.28..-0.39/reroll vs <= +0.4 tier-bonus upside) [RD10]. */
  moneySpentLog: number;
  /** Reward for catching a Pokemon (default: 3.0) */
  pokemonCaught: number;
  /** Reward for selecting a modifier vs skipping (default: 0.5) */
  modifierSelected: number;
  /** Bonus per modifier tier (multiplied by tier 0-5, default: 0.1) */
  modifierTierBonus: number;
  /** Per-step time penalty to discourage stalling (default: -0.01) */
  turnPenalty: number;
  /** Extra per-decision penalty while the no-progress counter exceeds
   *  STALL_GRACE_STEPS (default: -2.5). Charged AT the stalling steps so
   *  discounting cannot erode it — a lump 40 steps out is only x0.669 at
   *  gamma=0.99, which left stalling preferable to a worst-case loss [RD2]. */
  stallStepPenalty: number;
  /** Terminal lump when the episode ends by livelock (default: -50 = runLost:
   *  stalling IS losing). Combined with stallStepPenalty the detected-stall
   *  return is ~-138 undiscounted / ~-104 PV, strictly below any fight [RD2]. */
  stallPenalty: number;
  /** Surrogate win paid when the episode ends by wave cap (default: 100),
   *  scaled by cleanWaveAdvances/totalWaveAdvances — flee also advances
   *  waveIndex, and an unscaled bonus paid flee-rushing the endgame +98.
   *  Pure flee-rush earns ~0; a late survival-flee keeps most of it (a
   *  priced tactical option, not an exploit) [RD5]. */
  waveCapReached: number;
  /** SIGNED reward per player stat stage delta (default: 0, shaped).
   *  Signed so boost -> switch-out (stages reset) -> re-boost nets ~0 [RD7].
   *  Residual hold bias is w*(1-gamma)*Phi per step — keep
   *  w*(1-gamma)*Phi_max < |turnPenalty|. */
  statBoostReward: number;
  /** SIGNED reward per statused-enemy-count delta (default: 0, shaped).
   *  Fainted mons are excluded (FAINT is technically a status) [RD7]. */
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
  moneyGainedLog: 0.2,
  moneySpentLog: -0.05,
  pokemonCaught: 3.0,
  modifierSelected: 0.5,
  modifierTierBonus: 0.1,
  turnPenalty: -0.01,
  stallStepPenalty: -2.5,
  stallPenalty: -50.0,
  waveCapReached: 100.0,
  statBoostReward: 0,
  statusInflictionReward: 0,
};

/** Reward keys REMOVED by the v2 redesign — configs still setting them must
 *  fail loudly, not silently train at the new defaults. */
export const REMOVED_REWARD_KEYS: Readonly<Record<string, string>> = {
  moneyGained: "renamed to moneyGainedLog (cumulative log scale — retune the weight, linear 0.01 ≈ log 0.2)",
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
  /** Max HP parallel to enemyHpRatios: a maxHp change (form change, boss
   *  transition) drops the RATIO without any damage dealt — the new-low
   *  tracker rebases instead of paying on those steps [RD6]. */
  enemyMaxHps?: number[];
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
  /** Per-enemy-id lowest HP ratio seen (+ maxHp for the form-change rebase):
   *  hpDamageDealt pays only NEW LOWS, so total payable damage per enemy is
   *  1.0 for its life and re-dealt HP after heals pays nothing [RD6]. Ids are
   *  unique per run w.h.p. (seeded 32-bit draws), so the map is never pruned
   *  mid-episode (up to ~6 entries per trainer wave). */
  private minEnemyHp = new Map<number, { ratio: number; maxHp: number }>();
  /** Cumulative money gained this episode — the money reward telescopes over
   *  ln(1+G) so splitting income across steps cannot inflate it [RD4]. */
  private cumulativeMoneyGained = 0;
  /** Wave advances total / not-fled, for the waveCapReached clean-ratio [RD5]. */
  private waveAdvancesTotal = 0;
  private waveAdvancesClean = 0;

  constructor(config?: Partial<RewardConfig>) {
    this.config = { ...DEFAULT_REWARD_CONFIG, ...config };
  }

  /** New-low enemy HP payment: sum of per-id drops below the tracked
   *  per-episode minimum. Falls back to v1 positive-delta semantics when
   *  either snapshot lacks ids (hand-built test snapshots). */
  private newLowEnemyHpDelta(pre: StateSnapshot, post: StateSnapshot): number {
    const preIds = pre.enemyIds;
    const postIds = post.enemyIds;
    const canMatchById =
      preIds && postIds && preIds.length === pre.enemyHpRatios.length && postIds.length === post.enemyHpRatios.length;
    if (!canMatchById) {
      return positiveHpDelta(pre.enemyHpRatios, post.enemyHpRatios, preIds, postIds);
    }
    const preById = new Map<number, { ratio: number; maxHp: number }>();
    for (let i = 0; i < preIds.length; i++) {
      if (preIds[i] !== -1) {
        preById.set(preIds[i], { ratio: pre.enemyHpRatios[i], maxHp: pre.enemyMaxHps?.[i] ?? -1 });
      }
    }
    let sum = 0;
    for (let i = 0; i < postIds.length; i++) {
      const id = postIds[i];
      if (id === -1) {
        continue;
      }
      const ratio = post.enemyHpRatios[i];
      const maxHp = post.enemyMaxHps?.[i] ?? -1;
      // First sighting seeds from the PRE ratio when available (pays for this
      // step's damage), else from the post ratio (no pay on entry).
      const entry = this.minEnemyHp.get(id) ?? preById.get(id) ?? { ratio, maxHp };
      if (maxHp !== entry.maxHp) {
        // maxHp changed (form change / boss transition): the ratio moved
        // without damage — rebase the minimum without paying.
        this.minEnemyHp.set(id, { ratio, maxHp });
        continue;
      }
      const drop = entry.ratio - ratio;
      if (drop > 0) {
        sum += drop;
        this.minEnemyHp.set(id, { ratio, maxHp });
      } else if (!this.minEnemyHp.has(id)) {
        this.minEnemyHp.set(id, entry);
      }
    }
    return sum;
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

    // Count of LIVING enemies with a real non-volatile status. FAINT is
    // technically a StatusEffect, and counting it made the status potential
    // track KO timing instead of status play (every faint paid +w, then the
    // wave-transition party swap charged it back in a lump) [RD7].
    let enemyStatusCount = 0;
    for (const p of enemyParty) {
      if (p.status && p.status.effect !== StatusEffect.NONE && p.status.effect !== StatusEffect.FAINT && p.hp > 0) {
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
      // maxHp alongside the ratios: lets the new-low tracker tell a form
      // change (maxHp moved, ratio dropped without damage) from real damage.
      enemyMaxHps: enemyParty.map(p => p.getMaxHp()),
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
      // Enemy damage pays NEW LOWS only (per-episode min per id) so healing
      // enemies cannot be farmed for unbounded damage reward [RD6].
      reward += this.config.hpDamageDealt * this.newLowEnemyHpDelta(pre, postSnap);
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

    // Wave-advance bookkeeping for the waveCapReached clean-ratio [RD5]:
    // every advance counts toward the denominator, only not-fled advances
    // toward the numerator.
    if (postSnap.waveIndex > pre.waveIndex) {
      const advanced = postSnap.waveIndex - pre.waveIndex;
      this.waveAdvancesTotal += advanced;
      if (!fled) {
        this.waveAdvancesClean += advanced;
      }
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

    // Terminal rewards. The shaping potentials are charged back at every
    // true terminal (Phi(absorbing) = 0, Ng et al. 1999) so ending an
    // episode holding boosts/status is not free credit [RD7].
    if (terminated) {
      reward += victory ? this.config.runWon : this.config.runLost;
      reward += this.shapingChargeback(postSnap);
    }

    // Fled from battle
    if (fled) {
      reward += this.config.ranAway;
    }

    // Money [RD4/RD10]: gains telescope over ln(1+cumulative) — splitting
    // income across steps (per-hit money items) cannot inflate the total,
    // which is bounded by moneyGainedLog * ln(1+total_gained) ~ +2.6.
    // Spends are priced PER DELTA (splitting a spend only raises the charge),
    // keeping reroll-fishing EV negative from the first reroll.
    const moneyDelta = postSnap.money - pre.money;
    if (moneyDelta > 0) {
      const logBefore = Math.log1p(this.cumulativeMoneyGained);
      this.cumulativeMoneyGained += moneyDelta;
      reward += this.config.moneyGainedLog * (Math.log1p(this.cumulativeMoneyGained) - logBefore);
    } else if (moneyDelta < 0) {
      reward += this.config.moneySpentLog * Math.log1p(-moneyDelta);
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

    // Shaped rewards: SIGNED potential deltas [RD7]. The old positive-only
    // gates made boost -> switch-out (stages reset) -> re-boost and
    // status -> cure -> re-inflict infinite farm loops; signed deltas make
    // each cycle net ~0 (approximate potential-based shaping; the gamma<1
    // residual is a hold bias of w*(1-gamma)*Phi per step — keep weights
    // under |turnPenalty| / ((1-gamma)*Phi_max)).
    if (this.config.statBoostReward !== 0) {
      reward += this.config.statBoostReward * (postSnap.playerStatStageSum - pre.playerStatStageSum);
    }
    if (this.config.statusInflictionReward !== 0) {
      reward += this.config.statusInflictionReward * (postSnap.enemyStatusCount - pre.enemyStatusCount);
    }

    // Turn penalty
    reward += this.config.turnPenalty;

    return reward;
  }

  /** -w * Phi(s) for both shaping potentials: closes the telescoping sum at
   *  episode end (otherwise ending while holding boosts/status keeps
   *  unearned credit). No-op at the default zero weights. */
  private shapingChargeback(finalSnap: StateSnapshot): number {
    let adj = 0;
    if (this.config.statBoostReward !== 0) {
      adj -= this.config.statBoostReward * finalSnap.playerStatStageSum;
    }
    if (this.config.statusInflictionReward !== 0) {
      adj -= this.config.statusInflictionReward * finalSnap.enemyStatusCount;
    }
    return adj;
  }

  /**
   * Adjustment added to the FINAL reward when the episode ends by a cap or
   * livelock (game over goes through {@linkcode computeReward} instead).
   *
   * - livelock: stallPenalty, reported as TERMINATED — under truncation SB3
   *   bootstraps V(s_final) where s_final IS the repeated observation, and
   *   V(stall) self-consistently diverges toward stallPenalty/(1-gamma) [RD2].
   * - wave_cap: waveCapReached x (clean/total wave advances), TERMINATED —
   *   the bonus stands in for continuation value, bootstrapping on top would
   *   double-count; the clean-ratio keeps flee-rushing worthless [RD5].
   * - step_cap: no adjustment, stays TRUNCATED — the step budget is a compute
   *   artifact invisible to the observation; penalizing it punished honest
   *   slow play below a genuine loss (and the shaping potentials stay open
   *   because the bootstrap absorbs them).
   */
  episodeEndAdjustment(reason: EpisodeEndReason, finalSnap: StateSnapshot | null): number {
    if (reason === "step_cap") {
      return 0;
    }
    let adj = 0;
    if (reason === "livelock") {
      adj += this.config.stallPenalty;
    } else {
      const ratio = this.waveAdvancesTotal > 0 ? this.waveAdvancesClean / this.waveAdvancesTotal : 1;
      adj += this.config.waveCapReached * ratio;
    }
    if (finalSnap) {
      adj += this.shapingChargeback(finalSnap);
    }
    return adj;
  }

  /** Extra penalty for a decision whose observation repeats an already-seen
   *  one: 0 within the grace window, stallStepPenalty per step beyond it.
   *  Charged where the stall happens so gamma cannot erode it [RD2]. */
  stallStepAdjustment(consecutiveRepeats: number, graceSteps: number): number {
    return consecutiveRepeats > graceSteps ? this.config.stallStepPenalty : 0;
  }

  /** Reset the calculator for a new episode */
  reset(): void {
    this.prevSnapshot = null;
    this.cumulativeCatches = 0;
    this.minEnemyHp.clear();
    this.cumulativeMoneyGained = 0;
    this.waveAdvancesTotal = 0;
    this.waveAdvancesClean = 0;
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
