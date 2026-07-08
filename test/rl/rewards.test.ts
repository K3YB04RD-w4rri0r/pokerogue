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

  it("rewards money gained but not money spent", () => {
    const gained = step(calc, snap({ money: 1000 }), snap({ money: 1500 }));
    expect(gained).toBeCloseTo(DEFAULT_REWARD_CONFIG.moneyGained * 500 + TP, 9);

    const spent = step(calc, snap({ money: 1500 }), snap({ money: 800 }));
    expect(spent).toBeCloseTo(TP, 9);
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

  it("shaped rewards apply positive deltas when configured", () => {
    const shaped = new RewardCalculator({ statBoostReward: 0.1, statusInflictionReward: 0.5 });
    const r = step(
      shaped,
      snap({ playerStatStageSum: 0, enemyStatusCount: 0 }),
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
    );
    expect(r).toBeCloseTo(0.1 * 4 + 0.5 * 1 + TP, 9);

    // Negative deltas (stat drops, status cured) contribute nothing
    const drop = step(
      shaped,
      snap({ playerStatStageSum: 4, enemyStatusCount: 1 }),
      snap({ playerStatStageSum: 0, enemyStatusCount: 0 }),
    );
    expect(drop).toBeCloseTo(TP, 9);
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
      [mon(80, 100, [0, 0], 2), mon(100, 100, [0, 0], 0), mon(100, 100, [0, 0], null)],
      3,
      1,
      12,
      2500,
    );

    expect(s.playerHpRatios).toEqual([0.5, 0]);
    expect(s.enemyHpRatios).toEqual([0.8, 1, 1]);
    expect(s.playerStatStageSum).toBe(5); // only positive stages: 2 + 3
    expect(s.enemyStatusCount).toBe(1); // effect 0 and null don't count
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
