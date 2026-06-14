/*
 * SPDX-License-Identifier: AGPL-3.0-only
 *
 * RL cross-process determinism guard for the evil-team-grunt double-battle roll.
 *
 * `getRandomTrainerFunc` (src/battle.ts) decides whether an evil-team grunt is a
 * single or double battle. That outcome is OBSERVED by the RL env (it changes the
 * enemy count / `is_double` / `enemy_alive_count`) and changes the number of
 * phases, so it must be a pure function of the game seed. It used to roll with
 * `randInt(3)` (= `Math.random`), which the headless RL env seeds per-process, so
 * two fresh processes with the same game seed could pick different single/double
 * outcomes and diverge — the root cause of the intermittent cross-process auto
 * determinism flake.
 *
 * The fix seeds the roll via an isolated `executeWithSeedOffset`. This test proves
 * the property semantically against the real game function: under a fixed game
 * seed, the chosen `TrainerVariant` is identical no matter how `Math.random`'s
 * internal state is perturbed between calls. The pre-fix code fails this; the
 * fixed code passes.
 */
import { getRandomTrainerFunc } from "#app/battle";
import { globalScene } from "#app/global-scene";
import { TrainerType } from "#enums/trainer-type";
import { TrainerVariant } from "#enums/trainer-variant";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

describe("RL determinism - evil-grunt double-battle roll is seeded (not Math.random)", () => {
  let phaserGame: Phaser.Game;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    new GameManager(phaserGame);
  });

  /**
   * Generate a Rocket Grunt exactly the way battle-scene.ts does — inside an
   * `executeWithSeedOffset(waveIndex << 8)` context — and return the rolled
   * variant. Rocket Grunt has `hasDouble` and is an evil-team grunt, so the
   * 1/3 double roll is exercised.
   */
  function genVariant(seed: string, waveIndex: number): TrainerVariant {
    globalScene.seed = seed;
    let variant: TrainerVariant = TrainerVariant.DEFAULT;
    globalScene.executeWithSeedOffset(() => {
      variant = getRandomTrainerFunc([TrainerType.ROCKET_GRUNT])().variant;
    }, waveIndex << 8);
    return variant;
  }

  /** Advance Math.random's internal state by `n` draws. */
  function churnMathRandom(n: number): void {
    for (let k = 0; k < n; k++) {
      Math.random();
    }
  }

  it("yields the same single/double outcome under one seed regardless of Math.random state", () => {
    const seed = "evil-grunt-det";
    const wave = 5;
    const first = genVariant(seed, wave);
    churnMathRandom(5000);
    const second = genVariant(seed, wave);
    expect(second).toBe(first);
  });

  it("is reproducible across many independent Math.random perturbations", () => {
    const seed = "evil-grunt-det-2";
    const wave = 7;
    const baseline = genVariant(seed, wave);
    for (let trial = 1; trial <= 10; trial++) {
      churnMathRandom(trial * 777);
      expect(genVariant(seed, wave)).toBe(baseline);
    }
  });

  it("actually exercises the double-battle branch for at least one seed (roll is live, not constant)", () => {
    // Sanity: confirm the seeded roll can produce DOUBLE for some seed, so the
    // test above isn't trivially passing because the variant is always DEFAULT.
    const wave = 3;
    let sawDouble = false;
    let sawDefault = false;
    for (let s = 0; s < 40 && !(sawDouble && sawDefault); s++) {
      const v = genVariant(`grunt-spread-${s}`, wave);
      sawDouble ||= v === TrainerVariant.DOUBLE;
      sawDefault ||= v === TrainerVariant.DEFAULT;
    }
    expect(sawDouble).toBe(true);
    expect(sawDefault).toBe(true);
  });
});
