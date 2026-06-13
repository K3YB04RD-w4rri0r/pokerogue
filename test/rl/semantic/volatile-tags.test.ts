/**
 * Semantic audit: curated volatile battler tags in buildGameState() + encoding.
 *
 * Tier A mirrors live `pokemon.getTag(...)` objects (turn counts, subclass
 * extras); Tier B pins game rules (Substitute HP = floor(maxHp/4)).
 */

import { AbilityId } from "#enums/ability-id";
import { BattlerTagType } from "#enums/battler-tag-type";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs, PKMN, pokemonDim, volatileTagIndex } from "./obs-layout";

describe("RL Semantic - Volatile Tags", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .battleStyle("single")
      .enemySpecies(SpeciesId.SHUCKLE)
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("serializes enemy-cast Taunt on the player with live turn count", async () => {
    game.override.moveset([MoveId.SPLASH]).enemyMoveset(MoveId.TAUNT);
    await game.classicMode.startBattle(SpeciesId.REGIELEKI);

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    const player = game.field.getPlayerPokemon();
    const live = player.getTag(BattlerTagType.TAUNT) as any;
    expect(live).toBeDefined();

    const state = gs();
    const taunt = state.player_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.TAUNT);
    expect(taunt).toBeDefined();
    expect(taunt.turn_count).toBe(live.turnCount);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.TAUNT))]).toBe(1);
  });

  it("serializes Leech Seed (SEEDED) on the enemy", async () => {
    await game.classicMode.startBattle(SpeciesId.BULBASAUR);

    game.move.use(MoveId.LEECH_SEED);
    await game.move.forceHit(); // 90%-accuracy move; the mocked RNG rolls a miss
    await game.toNextTurn();

    const enemy = game.field.getEnemyPokemon();
    expect(enemy.getTag(BattlerTagType.SEEDED)).toBeDefined();

    const state = gs();
    const seeded = state.enemy_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.SEEDED);
    expect(seeded).toBeDefined();

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("enemy_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.SEEDED))]).toBe(1);
    expect(obs[pokemonDim("player_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.SEEDED))]).toBe(0);
  });

  it("stores Substitute remaining HP (game rule: floor(maxHp/4))", async () => {
    game.override.moveset([MoveId.SUBSTITUTE]);
    await game.classicMode.startBattle(SpeciesId.BLISSEY);

    game.move.select(MoveId.SUBSTITUTE);
    await game.toEndOfTurn();

    const player = game.field.getPlayerPokemon();
    const live = player.getTag(BattlerTagType.SUBSTITUTE) as any;
    expect(live).toBeDefined();

    const state = gs();
    const sub = state.player_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.SUBSTITUTE);
    expect(sub).toBeDefined();
    expect(sub.substitute_hp).toBe(live.hp); // Tier A mirror
    expect(sub.substitute_hp).toBe(Math.floor(player.getMaxHp() / 4)); // Tier B rule
  });

  it("stores the Encore-locked move id", async () => {
    game.override.moveset([MoveId.TACKLE]).enemyMoveset(MoveId.ENCORE);
    await game.classicMode.startBattle(SpeciesId.REGIELEKI);

    game.move.select(MoveId.TACKLE);
    await game.toNextTurn();

    const player = game.field.getPlayerPokemon();
    const live = player.getTag(BattlerTagType.ENCORE) as any;
    expect(live).toBeDefined();

    const state = gs();
    const encore = state.player_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.ENCORE);
    expect(encore).toBeDefined();
    expect(encore.encore_move_id).toBe(live.moveId);
    expect(encore.encore_move_id).toBe(MoveId.TACKLE);
  });

  it("serializes Salt Cure on the enemy", async () => {
    game.override.enemyLevel(100); // survive the initial Salt Cure hit
    await game.classicMode.startBattle(SpeciesId.GARGANACL);

    game.move.use(MoveId.SALT_CURE);
    await game.toNextTurn();

    const enemy = game.field.getEnemyPokemon();
    expect(enemy.getTag(BattlerTagType.SALT_CURED)).toBeDefined();

    const state = gs();
    const tag = state.enemy_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.SALT_CURED);
    expect(tag).toBeDefined();

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("enemy_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.SALT_CURED))]).toBe(1);
  });

  // ── v8: new curated tags ──
  it("serializes Bind (BIND partial-trap) on the enemy", async () => {
    await game.classicMode.startBattle(SpeciesId.SHUCKLE);

    game.move.use(MoveId.BIND);
    await game.move.forceHit(); // Bind is 85% accuracy
    await game.toNextTurn();

    const enemy = game.field.getEnemyPokemon();
    expect(enemy.getTag(BattlerTagType.BIND)).toBeDefined();

    const state = gs();
    const tag = state.enemy_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.BIND);
    expect(tag).toBeDefined();

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("enemy_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.BIND))]).toBe(1);
  });

  it("serializes Focus Energy (CRIT_BOOST) on the user", async () => {
    game.override.moveset([MoveId.FOCUS_ENERGY]);
    await game.classicMode.startBattle(SpeciesId.MEW);

    game.move.select(MoveId.FOCUS_ENERGY);
    await game.toEndOfTurn();

    const player = game.field.getPlayerPokemon();
    expect(player.getTag(BattlerTagType.CRIT_BOOST)).toBeDefined();

    const state = gs();
    const tag = state.player_0.volatile_tags.find((t: any) => t.tag_type === BattlerTagType.CRIT_BOOST);
    expect(tag).toBeDefined();

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.CRIT_BOOST))]).toBe(1);
  });
});
