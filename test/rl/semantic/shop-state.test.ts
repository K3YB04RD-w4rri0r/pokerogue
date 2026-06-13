/**
 * Semantic audit: shop/reward state during SelectModifierPhase.
 * Reuses the phase-injection pattern from test/rl/modifier-api.test.ts.
 */
import type { BattleScene } from "#app/battle-scene";
import { modifierTypes } from "#data/data-lists";
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import type { CustomModifierSettings } from "#modifiers/modifier-type";
import { SelectModifierPhase } from "#phases/select-modifier-phase";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs, MODPHASE_BASE } from "./obs-layout";

describe("RL Semantic - Shop State", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;
  let scene: BattleScene;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    scene = game.scene;
    game.override
      .moveset([MoveId.FISSURE, MoveId.SPLASH])
      .ability(AbilityId.NO_GUARD)
      .startingLevel(200)
      .enemySpecies(SpeciesId.MAGIKARP);
  });

  it("is inactive outside SelectModifierPhase", async () => {
    await game.classicMode.startBattle(SpeciesId.ABRA);

    const state = gs();
    expect(state.shop).toBeNull();

    const obs = encodeObservation(state);
    expect(obs[MODPHASE_BASE + 0]).toBe(0); // modifier_active
  });

  it("serializes reward options with tier/target metadata during the phase", async () => {
    await game.classicMode.startBattle(SpeciesId.ABRA);
    const customModifiers: CustomModifierSettings = {
      guaranteedModifierTypeFuncs: [modifierTypes.AMULET_COIN, modifierTypes.LEFTOVERS, modifierTypes.TM_ULTRA],
    };
    scene.phaseManager.unshiftPhase(new SelectModifierPhase(0, undefined, customModifiers));
    game.move.select(MoveId.SPLASH);
    await game.phaseInterceptor.to("SelectModifierPhase");

    const state = gs();
    expect(state.shop).not.toBeNull();
    expect(state.shop.reward_options.length).toBe(3);

    const [coin, leftovers, tm] = state.shop.reward_options;
    expect(coin.modifier_id).toBe("AMULET_COIN");
    expect(coin.target_kind).toBe("none");
    expect(leftovers.modifier_id).toBe("LEFTOVERS");
    expect(leftovers.target_kind).toBe("pokemon");
    expect(tm.target_kind).toBe("pokemon");
    expect(tm.move_id).toBeGreaterThan(0); // TMs expose the taught move

    expect(state.shop.money).toBe(scene.money);
    expect(state.shop.reroll_cost).toBeGreaterThan(0);

    const obs = encodeObservation(state);
    expect(obs[MODPHASE_BASE + 0]).toBe(1); // modifier_active
    expect(obs[MODPHASE_BASE + 3]).toBe(1); // reward option 0 valid flag
  });
});
