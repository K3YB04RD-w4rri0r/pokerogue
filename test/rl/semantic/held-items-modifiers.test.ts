/**
 * Semantic audit: held items + modifier inventory in buildGameState().
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs, PKMN, pokemonDim } from "./obs-layout";

describe("RL Semantic - Held Items & Modifiers", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .battleStyle("single")
      .moveset([MoveId.SPLASH])
      .enemySpecies(SpeciesId.SHUCKLE)
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("mirrors starting held items (Leftovers) on the active Pokemon", async () => {
    game.override.startingHeldItems([{ name: "LEFTOVERS", count: 1 }]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const player = game.field.getPlayerPokemon();
    const liveItems = player.getHeldItems();
    expect(liveItems.length).toBe(1);

    const state = gs();
    const held = state.player_0.held_items;
    expect(held.length).toBe(1);
    expect(held[0].modifier_id).toBe("LEFTOVERS");
    expect(held[0].stack_count).toBe(liveItems[0].stackCount);
    expect(held[0].max_stack_count).toBe(liveItems[0].getMaxStackCount());

    // modifiers.held_items mirrors the same data keyed by party slot
    expect(state.modifiers.held_items["0"]).toBeDefined();
    expect(state.modifiers.held_items["0"][0].modifier_id).toBe("LEFTOVERS");

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.HELD_ITEM_COUNT)]).toBeCloseTo(1 / 10, 6);
  });

  it("serializes berry held items with berry_type", async () => {
    game.override.startingHeldItems([{ name: "BERRY", type: 0, count: 2 }]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const player = game.field.getPlayerPokemon();
    expect(player.getHeldItems().length).toBeGreaterThan(0);

    const state = gs();
    const held = state.player_0.held_items;
    expect(held.length).toBeGreaterThan(0);
    expect(held[0].berry_type).not.toBeNull();
    expect(held[0].stack_count).toBe(2);
  });

  it("stacks multiple copies of a stackable item", async () => {
    game.override.startingHeldItems([{ name: "ATTACK_TYPE_BOOSTER", type: 0, count: 3 }]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const player = game.field.getPlayerPokemon();
    const live = player.getHeldItems()[0];

    const state = gs();
    const held = state.player_0.held_items[0];
    expect(held.stack_count).toBe(live.stackCount);
    expect(held.stack_count).toBe(3);
  });
});
