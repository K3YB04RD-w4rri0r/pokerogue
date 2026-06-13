/**
 * Semantic audit: weather/terrain state + one-hot encoding + turn counters.
 */
import { TerrainType } from "#data/terrain";
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { WeatherType } from "#enums/weather-type";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { FIELD, fieldDim, gs } from "./obs-layout";

describe("RL Semantic - Weather & Terrain", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .battleStyle("single")
      .moveset([MoveId.SPLASH, MoveId.RAIN_DANCE, MoveId.MISTY_TERRAIN])
      .enemySpecies(SpeciesId.SHUCKLE)
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("serializes move-cast Rain with live turn counter that decrements", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.RAIN_DANCE);
    await game.toNextTurn();

    const liveWeather = game.scene.arena.weather as any;
    expect(liveWeather?.weatherType).toBe(WeatherType.RAIN);

    let state = gs();
    expect(state.field.weather_type).toBe(WeatherType.RAIN);
    expect(state.field.weather_turns_left).toBe(liveWeather.turnsLeft);
    const afterCast = state.field.weather_turns_left;

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.WEATHER_OH + WeatherType.RAIN)]).toBe(1);
    expect(obs[fieldDim(FIELD.WEATHER_OH + WeatherType.SUNNY)]).toBe(0);
    expect(obs[fieldDim(FIELD.WEATHER_TURNS)]).toBeCloseTo(afterCast / 8, 6);

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    state = gs();
    expect(state.field.weather_turns_left).toBe(afterCast - 1);
  });

  it("serializes override weather (Sandstorm) into the one-hot", async () => {
    game.override.weather(WeatherType.SANDSTORM);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const state = gs();
    expect(state.field.weather_type).toBe(WeatherType.SANDSTORM);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.WEATHER_OH + WeatherType.SANDSTORM)]).toBe(1);
    // exactly one weather dim hot
    let hot = 0;
    for (let i = 0; i < 10; i++) {
      hot += obs[fieldDim(FIELD.WEATHER_OH + i)];
    }
    expect(hot).toBe(1);
  });

  it("serializes move-cast Misty Terrain with live turn counter", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.MISTY_TERRAIN);
    await game.toNextTurn();

    const liveTerrain = game.scene.arena.terrain as any;
    expect(liveTerrain?.terrainType).toBe(TerrainType.MISTY);

    const state = gs();
    expect(state.field.terrain_type).toBe(TerrainType.MISTY);
    expect(state.field.terrain_turns_left).toBe(liveTerrain.turnsLeft);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.TERRAIN_OH + TerrainType.MISTY)]).toBe(1);
    expect(obs[fieldDim(FIELD.TERRAIN_TURNS)]).toBeCloseTo(liveTerrain.turnsLeft / 8, 6);
  });

  it("sets weather via Drought at battle start", async () => {
    game.override.ability(AbilityId.DROUGHT);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const state = gs();
    expect(state.field.weather_type).toBe(WeatherType.SUNNY);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.WEATHER_OH + WeatherType.SUNNY)]).toBe(1);
  });
});
