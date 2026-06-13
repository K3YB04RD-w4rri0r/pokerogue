/**
 * Semantic audit: ABILITY_FEATURES lookup table spot checks + suppression.
 *
 * Feature schema (ability-features.ts header):
 *   [0] immune_ground  [9] <varies>  [12] sets_weather (coded 0.1=sun, 0.2=rain, 0.3=sand...)
 *   [14] atk_multiplier (0.333 = neutral 1x baseline, 0.667 = 2x)
 *   [22] on_switch_in_stat_drop (Intimidate)  [33] stat_stage_inversion (Contrary -1)
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { ABILITY_FEATURES } from "#rl/ability-features";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs, PKMN, pokemonDim } from "./obs-layout";

describe("RL Semantic - Ability Features", () => {
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

  it("table spot checks: known ability rows match game semantics", () => {
    // Levitate: ground immunity
    expect(ABILITY_FEATURES[AbilityId.LEVITATE][0]).toBe(1.0);
    // Intimidate: on-switch-in enemy stat drop
    expect(ABILITY_FEATURES[AbilityId.INTIMIDATE][22]).toBe(1.0);
    // Drought: sets weather (sun code 0.1)
    expect(ABILITY_FEATURES[AbilityId.DROUGHT][12]).toBeCloseTo(0.1, 6);
    // Huge Power: doubled attack (0.667 = 2x on the multiplier scale)
    expect(ABILITY_FEATURES[AbilityId.HUGE_POWER][14]).toBeCloseTo(0.667, 6);
    // Contrary: inverted stat stages
    expect(ABILITY_FEATURES[AbilityId.CONTRARY][33]).toBe(-1.0);
    // A defaults-row ability shares the neutral baseline at the multiplier cells
    expect(ABILITY_FEATURES[AbilityId.RUN_AWAY][14]).toBeCloseTo(0.333, 6);
  });

  it("encodes the active ability's feature vector into the pokemon block", async () => {
    game.override.ability(AbilityId.LEVITATE);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    const state = gs();
    expect(state.player_0.ability_id).toBe(AbilityId.LEVITATE);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("player_0", PKMN.ABILITY_FEAT + 0)]).toBe(1.0); // immune_ground
  });

  it("zeroes both 40-dim ability blocks when the ability is suppressed (Gastro Acid)", async () => {
    game.override.ability(AbilityId.LEVITATE).enemyMoveset(MoveId.GASTRO_ACID);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    const player = game.field.getPlayerPokemon();
    expect(player.summonData.abilitySuppressed).toBe(true);

    const state = gs();
    expect(state.player_0.ability_suppressed).toBe(true);

    const obs = encodeObservation(state);
    for (let i = 0; i < 40; i++) {
      expect(obs[pokemonDim("player_0", PKMN.ABILITY_FEAT + i)]).toBe(0);
      expect(obs[pokemonDim("player_0", PKMN.PASSIVE_FEAT + i)]).toBe(0);
    }
    expect(obs[pokemonDim("player_0", PKMN.ABILITY_SUPPRESSED)]).toBe(1);
  });
});
