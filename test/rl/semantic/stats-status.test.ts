/**
 * Semantic audit: stat stages, effective stats, status effects + counters.
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { Stat } from "#enums/stat";
import { StatusEffect } from "#enums/status-effect";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs, PKMN, pokemonDim } from "./obs-layout";

describe("RL Semantic - Stats & Status", () => {
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

  it("mirrors stat stages after Swords Dance and enemy Growl", async () => {
    game.override.moveset([MoveId.SWORDS_DANCE, MoveId.SPLASH]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);
    const player = game.field.getPlayerPokemon();

    game.move.select(MoveId.SWORDS_DANCE);
    await game.toNextTurn();

    expect(player.getStatStage(Stat.ATK)).toBe(2);

    let state = gs();
    expect(state.player_0.stat_stages[0]).toBe(2); // [atk, def, spatk, spdef, spd, acc, eva]

    let obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.STAT_STAGES + 0)]).toBeCloseTo(2 / 6, 6);

    game.override.enemyMoveset(MoveId.GROWL);
    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    state = gs();
    expect(state.player_0.stat_stages[0]).toBe(player.getStatStage(Stat.ATK));
    expect(player.getStatStage(Stat.ATK)).toBe(1); // 2 from SD, -1 from Growl

    obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.STAT_STAGES + 0)]).toBeCloseTo(1 / 6, 6);
  });

  it("mirrors live effective stats", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);
    const player = game.field.getPlayerPokemon();

    const state = gs();
    const liveStats = player.getStats(true);
    expect(state.player_0.stats).toEqual(liveStats);
  });

  it("tracks toxic turn count across turns", async () => {
    game.override.moveset([MoveId.SPLASH]).statusEffect(StatusEffect.TOXIC);
    await game.classicMode.startBattle(SpeciesId.BLISSEY);
    const player = game.field.getPlayerPokemon();

    let state = gs();
    expect(state.player_0.status_effect).toBe(StatusEffect.TOXIC);
    const t0 = state.player_0.toxic_turn_count;
    expect(t0).toBe(player.status?.toxicTurnCount ?? -1);

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    state = gs();
    expect(state.player_0.toxic_turn_count).toBe(player.status?.toxicTurnCount);
    expect(state.player_0.toxic_turn_count).toBeGreaterThan(t0);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.STATUS + StatusEffect.TOXIC)]).toBe(1);
    expect(obs[pokemonDim("player_0", PKMN.TOXIC_TURNS)]).toBeCloseTo(state.player_0.toxic_turn_count / 16, 6);
  });

  it("mirrors sleep turns remaining", async () => {
    game.override.moveset([MoveId.SPLASH]).statusEffect(StatusEffect.SLEEP);
    await game.classicMode.startBattle(SpeciesId.BLISSEY);
    const player = game.field.getPlayerPokemon();

    const state = gs();
    expect(state.player_0.status_effect).toBe(StatusEffect.SLEEP);
    expect(state.player_0.sleep_turns_remaining).toBe(player.status?.sleepTurnsRemaining ?? -1);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.STATUS + StatusEffect.SLEEP)]).toBe(1);
  });

  it("mirrors HP ratio after taking damage", async () => {
    game.override.moveset([MoveId.SPLASH]).enemyMoveset(MoveId.TACKLE).enemySpecies(SpeciesId.MAGIKARP);
    await game.classicMode.startBattle(SpeciesId.BLISSEY);
    const player = game.field.getPlayerPokemon();

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    const state = gs();
    expect(player.hp).toBeLessThan(player.getMaxHp());
    expect(state.player_0.hp).toBe(player.hp);
    expect(state.player_0.hp_ratio).toBeCloseTo(player.hp / player.getMaxHp(), 6);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.HP_RATIO)]).toBeCloseTo(player.hp / player.getMaxHp(), 5);
  });
});
