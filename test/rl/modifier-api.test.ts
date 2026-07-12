import type { BattleScene } from "#app/battle-scene";
import { modifierTypes } from "#data/data-lists";
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import type { CustomModifierSettings } from "#modifiers/modifier-type";
import { SelectModifierPhase } from "#phases/select-modifier-phase";
import { getAvailableModifiers, rerollModifiers, selectRewardModifier, skipModifiers } from "#rl/modifier-api";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

describe("RL Modifier API", () => {
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

  describe("getAvailableModifiers", () => {
    it("should return null when not in SelectModifierPhase", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      expect(getAvailableModifiers()).toBeNull();
    });

    it("should return reward modifiers during SelectModifierPhase", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      game.move.select(MoveId.FISSURE);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = getAvailableModifiers();
      expect(result).not.toBeNull();
      expect(result!.rewards.length).toBe(3);
      expect(result!.money).toBeGreaterThanOrEqual(0);
      expect(result!.rerollCost).toBeGreaterThan(0);

      for (const reward of result!.rewards) {
        expect(reward.source).toBe("reward");
        expect(reward.id).toBeTruthy();
        expect(reward.cost).toBe(0);
        expect(["none", "pokemon", "move", "pokemon_pair"]).toContain(reward.targetKind);
      }
    });

    it("should correctly identify targetKind for known modifier types", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      const customModifiers: CustomModifierSettings = {
        guaranteedModifierTypeFuncs: [modifierTypes.AMULET_COIN, modifierTypes.LEFTOVERS, modifierTypes.TM_ULTRA],
      };
      const selectModifierPhase = new SelectModifierPhase(0, undefined, customModifiers);
      scene.phaseManager.unshiftPhase(selectModifierPhase);
      game.move.select(MoveId.SPLASH);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = getAvailableModifiers();
      expect(result).not.toBeNull();
      expect(result!.rewards.length).toBe(3);
      // AMULET_COIN is a non-pokemon modifier
      expect(result!.rewards[0].targetKind).toBe("none");
      // LEFTOVERS is a PokemonHeldItemModifierType (extends PokemonModifierType)
      expect(result!.rewards[1].targetKind).toBe("pokemon");
      // TM_ULTRA is a TmModifierType (extends PokemonModifierType)
      expect(result!.rewards[2].targetKind).toBe("pokemon");
    });
  });

  describe("selectRewardModifier", () => {
    it("should apply a non-pokemon modifier and end the phase", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      const customModifiers: CustomModifierSettings = {
        guaranteedModifierTypeFuncs: [modifierTypes.AMULET_COIN],
      };
      const selectModifierPhase = new SelectModifierPhase(0, undefined, customModifiers);
      scene.phaseManager.unshiftPhase(selectModifierPhase);
      game.move.select(MoveId.SPLASH);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = selectRewardModifier(0);
      expect(result.success).toBe(true);
    });

    it("should apply a pokemon modifier with pokemonIndex", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      const customModifiers: CustomModifierSettings = {
        guaranteedModifierTypeFuncs: [modifierTypes.LEFTOVERS],
      };
      const selectModifierPhase = new SelectModifierPhase(0, undefined, customModifiers);
      scene.phaseManager.unshiftPhase(selectModifierPhase);
      game.move.select(MoveId.SPLASH);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = selectRewardModifier(0, 0);
      expect(result.success).toBe(true);
    });

    it("should return error for invalid index", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      game.move.select(MoveId.FISSURE);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = selectRewardModifier(99);
      expect(result.success).toBe(false);
      expect(result.error).toContain("Invalid reward index");
    });

    it("should return error when not in phase", () => {
      const result = selectRewardModifier(0);
      expect(result.success).toBe(false);
      expect(result.error).toBe("Not in SelectModifierPhase");
    });
  });

  describe("skipModifiers", () => {
    it("should end the SelectModifierPhase", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      game.move.select(MoveId.FISSURE);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = skipModifiers();
      expect(result.success).toBe(true);
    });

    it("should return error when not in phase", () => {
      const result = skipModifiers();
      expect(result.success).toBe(false);
    });
  });

  describe("rerollModifiers", () => {
    it("should deduct money on reroll", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      scene.money = 100000;
      game.move.select(MoveId.FISSURE);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const modsBefore = getAvailableModifiers();
      expect(modsBefore).not.toBeNull();
      const costBefore = modsBefore!.rerollCost;

      const result = rerollModifiers();
      expect(result.success).toBe(true);
      expect(scene.money).toBe(100000 - costBefore);
    });

    it("should fail when insufficient money", async () => {
      await game.classicMode.startBattle(SpeciesId.ABRA);
      scene.money = 0;
      game.move.select(MoveId.FISSURE);
      await game.phaseInterceptor.to("SelectModifierPhase");

      const result = rerollModifiers();
      expect(result.success).toBe(false);
      expect(result.error).toContain("Insufficient money");
    });

    it("should return error when not in phase", () => {
      const result = rerollModifiers();
      expect(result.success).toBe(false);
    });
  });
});
