import type { Pokemon } from "#field/pokemon";
import { DEFAULT_REWARD_CONFIG, RewardCalculator, type StateSnapshot } from "#rl/rewards";
import { beforeEach, describe, expect, it } from "vitest";

const TP = DEFAULT_REWARD_CONFIG.turnPenalty;

function snap(over: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    playerHpRatios: [1, 1],
    enemyHpRatios: [1],
    enemyFaints: 0,
    playerFaints: 0,
    waveIndex: 1,
    money: 1000,
    partySize: 2,
    playerStatStageSum: 0,
    enemyStatusCount: 0,
    ...over,
  };
}

/** Step the calculator: save `pre`, compute reward against `post`. */
function step(
  calc: RewardCalculator,
  pre: StateSnapshot,
  post: StateSnapshot,
  opts: { terminated?: boolean; victory?: boolean; fled?: boolean; modifierTier?: number } = {},
): number {
  calc.savePreActionSnapshot(pre);
  return calc.computeReward(
    post,
    opts.terminated ?? false,
    opts.victory ?? false,
    opts.fled ?? false,
    opts.modifierTier ?? -1,
  );
}

describe("RewardCalculator", () => {
  let calc: RewardCalculator;

  beforeEach(() => {
    calc = new RewardCalculator();
  });

  it("returns only the turn penalty when no pre-action snapshot exists", () => {
    expect(calc.computeReward(snap(), false, false, false, -1)).toBe(TP);
  });

  it("rewards HP damage dealt to enemies and ignores enemy healing", () => {
    const r = step(calc, snap({ enemyHpRatios: [1] }), snap({ enemyHpRatios: [0.6] }));
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.hpDamageDealt * 0.4 + TP, 9);

    const heal = step(calc, snap({ enemyHpRatios: [0.6] }), snap({ enemyHpRatios: [1] }));
    expect(heal).toBeCloseTo(TP, 9);
  });

  it("penalizes HP damage taken by the player and ignores player healing", () => {
    const r = step(calc, snap({ playerHpRatios: [1, 1] }), snap({ playerHpRatios: [0.5, 1] }));
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.hpDamageTaken * 0.5 + TP, 9);

    const heal = step(calc, snap({ playerHpRatios: [0.5, 1] }), snap({ playerHpRatios: [1, 1] }));
    expect(heal).toBeCloseTo(TP, 9);
  });

  it("[RB1] skips HP deltas across wave transitions", () => {
    const r = step(
      calc,
      snap({ waveIndex: 1, enemyHpRatios: [1], playerHpRatios: [1, 1] }),
      snap({ waveIndex: 2, enemyHpRatios: [0.1], playerHpRatios: [0.2, 1] }),
    );
    // No HP terms — only the wave-clear reward and turn penalty
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.waveCleared + TP, 9);
  });

  it("rewards enemy KOs from the faint-counter delta", () => {
    const r = step(calc, snap({ enemyFaints: 0 }), snap({ enemyFaints: 2 }));
    expect(r).toBeCloseTo(2 * DEFAULT_REWARD_CONFIG.enemyKo + TP, 9);
  });

  it("[RB2] penalizes player faints once per cumulative faint", () => {
    const first = step(calc, snap({ playerFaints: 0 }), snap({ playerFaints: 1 }));
    expect(first).toBeCloseTo(DEFAULT_REWARD_CONFIG.playerKo + TP, 9);

    // Same fainted count next step — no repeated penalty
    const repeat = step(calc, snap({ playerFaints: 1 }), snap({ playerFaints: 1 }));
    expect(repeat).toBeCloseTo(TP, 9);
  });

  it("[RB2] every genuine faint is penalized, including after a revive", () => {
    // playerFaints is now the game's CUMULATIVE faint counter
    // (arena.playerFaints), monotonic within an arena. Reward is the
    // positive per-step delta, mirroring enemyKo. A faint→revive→faint of
    // the same mon therefore scores TWICE (the counter goes 1 then 2); the
    // old point-in-time-count + max-tracking approach silently dropped the
    // second faint (bug fixed 2026-07-08).
    const first = step(calc, snap({ playerFaints: 0 }), snap({ playerFaints: 1 }));
    expect(first).toBeCloseTo(DEFAULT_REWARD_CONFIG.playerKo + TP, 9);
    // revive does not change the cumulative counter -> no reward change
    const revive = step(calc, snap({ playerFaints: 1 }), snap({ playerFaints: 1 }));
    expect(revive).toBeCloseTo(TP, 9);
    // second genuine faint: cumulative 1 -> 2, penalized again
    const again = step(calc, snap({ playerFaints: 1 }), snap({ playerFaints: 2 }));
    expect(again).toBeCloseTo(DEFAULT_REWARD_CONFIG.playerKo + TP, 9);
  });

  it("rewards wave clears, boss bonus keyed on the wave CLEARED (not arrived at)", () => {
    // Clearing wave 1 -> arrive at 2: a normal wave.
    const normal = step(calc, snap({ waveIndex: 1 }), snap({ waveIndex: 2 }));
    expect(normal).toBeCloseTo(DEFAULT_REWARD_CONFIG.waveCleared + TP, 9);

    // Clearing wave 9 -> arrive at 10: still a NORMAL clear (wave 9 is not a
    // boss); the old code wrongly paid the boss bonus here.
    const preBoss = step(calc, snap({ waveIndex: 9 }), snap({ waveIndex: 10 }));
    expect(preBoss).toBeCloseTo(DEFAULT_REWARD_CONFIG.waveCleared + TP, 9);

    // Clearing wave 10 (the boss) -> arrive at 11: the boss bonus.
    const boss = step(calc, snap({ waveIndex: 10 }), snap({ waveIndex: 11 }));
    expect(boss).toBeCloseTo(DEFAULT_REWARD_CONFIG.bossWaveCleared + TP, 9);
  });

  it("[RB4] rewards every wave in a multi-wave skip, boss on the cleared boss wave", () => {
    // 8 -> 11: cleared waves 8, 9, 10 — wave 10 is the boss (arrive at 11).
    const toBoss = step(calc, snap({ waveIndex: 8 }), snap({ waveIndex: 11 }));
    expect(toBoss).toBeCloseTo(2 * DEFAULT_REWARD_CONFIG.waveCleared + DEFAULT_REWARD_CONFIG.bossWaveCleared + TP, 9);

    // 1 -> 3: two regular waves, no boss
    const twoWaves = step(calc, snap({ waveIndex: 1 }), snap({ waveIndex: 3 }));
    expect(twoWaves).toBeCloseTo(2 * DEFAULT_REWARD_CONFIG.waveCleared + TP, 9);
  });

  it("applies terminal rewards for victory and defeat", () => {
    const won = step(calc, snap(), snap(), { terminated: true, victory: true });
    expect(won).toBeCloseTo(DEFAULT_REWARD_CONFIG.runWon + TP, 9);

    const lost = step(calc, snap(), snap(), { terminated: true, victory: false });
    expect(lost).toBeCloseTo(DEFAULT_REWARD_CONFIG.runLost + TP, 9);
  });

  it("penalizes fleeing", () => {
    const r = step(calc, snap(), snap(), { fled: true });
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.ranAway + TP, 9);
  });

  it("[RD4] money gains pay on the CUMULATIVE log scale and telescope", () => {
    // First gain: G goes 0 -> 500
    const first = step(calc, snap({ money: 1000 }), snap({ money: 1500 }));
    expect(first).toBeCloseTo(DEFAULT_REWARD_CONFIG.moneyGainedLog * Math.log1p(500) + TP, 9);

    // Second gain: G goes 500 -> 1000 — pays only the log INCREMENT
    const second = step(calc, snap({ money: 1500 }), snap({ money: 2500 }));
    expect(second).toBeCloseTo(DEFAULT_REWARD_CONFIG.moneyGainedLog * (Math.log1p(1500) - Math.log1p(500)) + TP, 9);
  });

  it("[RD4] splitting income across steps pays exactly the same as one lump", () => {
    // Five gains of 100 (the Golden-Punch per-hit pattern)...
    let split = 0;
    for (let i = 0; i < 5; i++) {
      split += step(calc, snap({ money: 1000 + 100 * i }), snap({ money: 1000 + 100 * (i + 1) }));
    }
    // ...equal one gain of 500 (per-delta log paid the split ~3.7x more)
    const lumpCalc = new RewardCalculator();
    const lump = step(lumpCalc, snap({ money: 1000 }), snap({ money: 1500 }));
    expect(split - 5 * TP).toBeCloseTo(lump - TP, 9);
    expect(split - 5 * TP).toBeCloseTo(DEFAULT_REWARD_CONFIG.moneyGainedLog * Math.log1p(500), 9);
  });

  it("[RD10] money spent is priced per delta on the log scale", () => {
    const spent = step(calc, snap({ money: 1500 }), snap({ money: 800 }));
    expect(spent).toBeCloseTo(DEFAULT_REWARD_CONFIG.moneySpentLog * Math.log1p(700) + TP, 9);
  });

  it("[RD10] pins the wave-50 first-reroll penalty above the tier-bonus fishing upside", () => {
    // Real reroll cost curve: 250 * ceil(wave/10) * 2^rerollCount
    // (select-modifier-phase.ts getRerollCost) -> wave 41-50 first reroll = 1250.
    const reroll = step(calc, snap({ money: 2000 }), snap({ money: 750 }));
    const penalty = DEFAULT_REWARD_CONFIG.moneySpentLog * Math.log1p(1250);
    expect(reroll).toBeCloseTo(penalty + TP, 9);
    expect(penalty).toBeCloseTo(-0.3565, 3);
    // A reroll's EXPECTED upside is E[best-tier improvement] * modifierTierBonus,
    // well under 0.1 with common-heavy tier weights (modifierSelected pays
    // regardless of tier, so only the tier-bonus delta is at stake). Pin the
    // penalty above that so reroll-fishing EV stays negative from the first
    // reroll at every wave — this stops the coefficient drifting under it [RD10].
    expect(Math.abs(penalty)).toBeGreaterThan(0.1);
  });

  it("[RB3] rewards catches from party growth and ignores party shrinkage", () => {
    const caught = step(calc, snap({ partySize: 2 }), snap({ partySize: 3 }));
    expect(caught).toBeCloseTo(DEFAULT_REWARD_CONFIG.pokemonCaught + TP, 9);

    const released = step(calc, snap({ partySize: 3 }), snap({ partySize: 2 }));
    expect(released).toBeCloseTo(TP, 9);
  });

  it("rewards modifier selection with tier bonus; tier -1 gives nothing", () => {
    const tier0 = step(calc, snap(), snap(), { modifierTier: 0 });
    expect(tier0).toBeCloseTo(DEFAULT_REWARD_CONFIG.modifierSelected + TP, 9);

    const tier4 = step(calc, snap(), snap(), { modifierTier: 4 });
    expect(tier4).toBeCloseTo(
      DEFAULT_REWARD_CONFIG.modifierSelected + 4 * DEFAULT_REWARD_CONFIG.modifierTierBonus + TP,
      9,
    );

    const skipped = step(calc, snap(), snap(), { modifierTier: -1 });
    expect(skipped).toBeCloseTo(TP, 9);
  });

  it("shaped rewards contribute nothing at default (zero) config", () => {
    const r = step(
      calc,
      snap({ playerStatStageSum: 0, enemyStatusCount: 0 }),
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
    );
    expect(r).toBeCloseTo(TP, 9);
  });

  it("[RD7] shaped rewards apply SIGNED deltas when configured", () => {
    const shaped = new RewardCalculator({ statBoostReward: 0.1, statusInflictionReward: 0.5 });
    const r = step(
      shaped,
      snap({ playerStatStageSum: 0, enemyStatusCount: 0 }),
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
    );
    expect(r).toBeCloseTo(0.1 * 4 + 0.5 * 1 + TP, 9);

    // Negative deltas CHARGE BACK (potential-based): losing the boosts /
    // status being cured undoes the credit, so boost -> switch -> re-boost
    // and status -> cure -> re-inflict cycles net ~0 instead of farming.
    const drop = step(
      shaped,
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
      snap({ playerStatStageSum: 0, enemyStatusCount: 0 }),
    );
    expect(drop).toBeCloseTo(-(0.1 * 4 + 0.5 * 1) + TP, 9);
  });

  it("[RD7] terminal transitions charge back the held shaping potentials", () => {
    const shaped = new RewardCalculator({ statBoostReward: 0.1, statusInflictionReward: 0.5 });
    // Phi is unchanged across the transition (delta 0) but the episode ends —
    // Phi(absorbing)=0, so the held potential is charged back.
    const r = step(
      shaped,
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
      { terminated: true, victory: true },
    );
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.runWon - (0.1 * 4 + 0.5 * 1) + TP, 9);
  });

  it("compares HP arrays over the shared prefix when party sizes differ", () => {
    const r = step(calc, snap({ enemyHpRatios: [1, 1] }), snap({ enemyHpRatios: [0.5] }));
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.hpDamageDealt * 0.5 + TP, 9);
  });

  it("reset() clears cumulative counters and the previous snapshot", () => {
    step(calc, snap({ playerFaints: 0 }), snap({ playerFaints: 1 })); // cumulative=1
    calc.reset();

    // prevSnapshot cleared: first compute returns bare turn penalty
    expect(calc.computeReward(snap({ playerFaints: 1 }), false, false, false, -1)).toBe(TP);

    // cumulative cleared: the same faint count is penalized again
    const r = step(calc, snap({ playerFaints: 0 }), snap({ playerFaints: 1 }));
    expect(r).toBeCloseTo(DEFAULT_REWARD_CONFIG.playerKo + TP, 9);
  });

  it("snapshot() derives ratios, positive stat stages, and status counts from Pokemon objects", () => {
    const mon = (hp: number, maxHp: number, stages: number[], statusEffect: number | null): Pokemon =>
      ({
        hp,
        getMaxHp: () => maxHp,
        getStatStages: () => stages,
        status: statusEffect === null ? null : { effect: statusEffect },
      }) as unknown as Pokemon;

    const s = calc.snapshot(
      [mon(50, 100, [2, -1, 0, 3], null), mon(10, 0, [0, 0, 0, 0], null)], // maxHp 0 -> ratio 0
      [
        mon(80, 100, [0, 0], 2),
        mon(100, 100, [0, 0], 0),
        mon(100, 100, [0, 0], null),
        // FAINT (7) is technically a StatusEffect and fainted mons linger in
        // trainer parties — neither may count as "statused" [RD7]:
        mon(0, 100, [0, 0], 7),
        mon(0, 100, [0, 0], 2), // statused but dead — dead mons don't count
      ],
      3,
      1,
      12,
      2500,
    );

    expect(s.playerHpRatios).toEqual([0.5, 0]);
    expect(s.enemyHpRatios).toEqual([0.8, 1, 1, 0, 0]);
    expect(s.enemyMaxHps).toEqual([100, 100, 100, 100, 100]);
    expect(s.playerStatStageSum).toBe(5); // only positive stages: 2 + 3
    expect(s.enemyStatusCount).toBe(1); // effect 0/null/FAINT/dead don't count
    expect(s.enemyFaints).toBe(3);
    expect(s.playerFaints).toBe(1);
    expect(s.waveIndex).toBe(12);
    expect(s.money).toBe(2500);
    expect(s.partySize).toBe(2);
  });

  it("updateConfig merges partial config", () => {
    calc.updateConfig({ turnPenalty: -1 });
    expect(calc.getConfig().turnPenalty).toBe(-1);
    expect(calc.getConfig().enemyKo).toBe(DEFAULT_REWARD_CONFIG.enemyKo);
    expect(calc.computeReward(snap(), false, false, false, -1)).toBe(-1);
  });
});

describe("RewardCalculator — new-low damage tracking [RD6]", () => {
  const D = DEFAULT_REWARD_CONFIG.hpDamageDealt;
  /** Id-carrying snapshot: enemy i keeps id 100+i across snapshots. */
  const idSnap = (ratios: number[], maxHps: number[], over: Partial<StateSnapshot> = {}): StateSnapshot =>
    snap({
      enemyHpRatios: ratios,
      enemyIds: ratios.map((_, i) => 100 + i),
      enemyMaxHps: maxHps,
      ...over,
    });

  let calc: RewardCalculator;
  beforeEach(() => {
    calc = new RewardCalculator();
  });

  it("pays new damage but never re-pays HP re-dealt after an enemy heal", () => {
    // 1.0 -> 0.6: new low, pays 0.4
    expect(step(calc, idSnap([1], [100]), idSnap([0.6], [100]))).toBeCloseTo(0.4 * D + TP, 9);
    // heal to full: nothing (and the 0.6 minimum is remembered)
    expect(step(calc, idSnap([0.6], [100]), idSnap([1], [100]))).toBeCloseTo(TP, 9);
    // re-damage to 0.7 — ABOVE the tracked minimum: v1 paid +0.3 here again
    // (the chip-heal farm); v2 pays nothing.
    expect(step(calc, idSnap([1], [100]), idSnap([0.7], [100]))).toBeCloseTo(TP, 9);
    // below the minimum: pays only the new-low part (0.6 -> 0.5)
    expect(step(calc, idSnap([0.7], [100]), idSnap([0.5], [100]))).toBeCloseTo(0.1 * D + TP, 9);
  });

  it("rebases without paying when maxHp changes (form change, not damage)", () => {
    // maxHp doubles, hp preserved: ratio halves with no damage dealt
    expect(step(calc, idSnap([0.5], [100]), idSnap([0.25], [200]))).toBeCloseTo(TP, 9);
    // subsequent real damage pays from the rebased minimum
    expect(step(calc, idSnap([0.25], [200]), idSnap([0.15], [200]))).toBeCloseTo(0.1 * D + TP, 9);
  });

  it("reset() clears the min-map and the cumulative money total", () => {
    step(calc, idSnap([1], [100]), idSnap([0.5], [100]));
    step(calc, snap({ money: 0 }), snap({ money: 100 }));
    calc.reset();
    // Same enemy id pays fresh from its pre ratio after reset
    expect(step(calc, idSnap([1], [100]), idSnap([0.6], [100]))).toBeCloseTo(0.4 * D + TP, 9);
    // Money telescoping restarts at G=0
    expect(step(calc, snap({ money: 0 }), snap({ money: 100 }))).toBeCloseTo(
      DEFAULT_REWARD_CONFIG.moneyGainedLog * Math.log1p(100) + TP,
      9,
    );
  });
});

describe("RewardCalculator — episode-end adjustments (reward v2)", () => {
  let calc: RewardCalculator;
  beforeEach(() => {
    calc = new RewardCalculator();
  });

  it("[RD2] livelock ends pay stallPenalty; step_cap pays nothing (plain truncation)", () => {
    expect(calc.episodeEndAdjustment("livelock", null)).toBeCloseTo(DEFAULT_REWARD_CONFIG.stallPenalty, 9);
    expect(calc.episodeEndAdjustment("step_cap", null)).toBe(0);
  });

  it("[RD5] wave_cap scales waveCapReached by the clean advance ratio", () => {
    // 4 genuine clears + 1 fled advance = 4/5 of the bonus
    for (let w = 1; w <= 4; w++) {
      step(calc, snap({ waveIndex: w }), snap({ waveIndex: w + 1 }));
    }
    step(calc, snap({ waveIndex: 5 }), snap({ waveIndex: 6 }), { fled: true });
    expect(calc.episodeEndAdjustment("wave_cap", null)).toBeCloseTo(DEFAULT_REWARD_CONFIG.waveCapReached * 0.8, 9);
  });

  it("[RD5] wave_cap with no recorded advances pays the full bonus", () => {
    expect(calc.episodeEndAdjustment("wave_cap", null)).toBeCloseTo(DEFAULT_REWARD_CONFIG.waveCapReached, 9);
  });

  it("[RD7] terminated ends charge back held shaping potential from the final snapshot", () => {
    const shaped = new RewardCalculator({ statBoostReward: 0.1, statusInflictionReward: 0.5 });
    expect(shaped.episodeEndAdjustment("livelock", snap({ playerStatStageSum: 6, enemyStatusCount: 2 }))).toBeCloseTo(
      DEFAULT_REWARD_CONFIG.stallPenalty - 0.1 * 6 - 0.5 * 2,
      9,
    );
  });

  it("[RD2] stallStepAdjustment charges only past the grace window", () => {
    expect(calc.stallStepAdjustment(0, 5)).toBe(0);
    expect(calc.stallStepAdjustment(5, 5)).toBe(0);
    expect(calc.stallStepAdjustment(6, 5)).toBeCloseTo(DEFAULT_REWARD_CONFIG.stallStepPenalty, 9);
  });

  it("[RD2] ordering invariant: a detected stall is strictly worse than a worst-case wipe", () => {
    const cfg = DEFAULT_REWARD_CONFIG;
    // Detected stall: 35 charged repeat-steps (grace 5 of 40) + terminal lump
    const stall = 35 * cfg.stallStepPenalty + cfg.stallPenalty + 40 * cfg.turnPenalty;
    // Worst fighting loss: full 6-mon wipe from full HP
    const worstLoss = cfg.runLost + 6 * cfg.playerKo + 6 * cfg.hpDamageTaken;
    expect(stall).toBeLessThan(worstLoss);
    // And the surrogate win must dominate both
    expect(cfg.waveCapReached).toBeGreaterThan(0);
    expect(worstLoss).toBeLessThan(0);
  });
});
