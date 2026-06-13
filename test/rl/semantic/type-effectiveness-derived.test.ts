/**
 * Semantic audit: derived type-effectiveness/STAB dims + turn_data.move_effectiveness.
 *
 * Derived block layout (spaces.ts encodeDerivedFields):
 *   type_eff[p][m][e] at DERIVED_BASE + p*8 + m*2 + e   (value = eff/4)
 *   stab[p][m]        at DERIVED_BASE + 16 + p*4 + m
 *   speed_rank[slot]  at DERIVED_BASE + 24 + slot
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { DERIVED_BASE, gs, PKMN, pokemonDim } from "./obs-layout";

describe("RL Semantic - Type Effectiveness & Derived", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .battleStyle("single")
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("derives 4x effectiveness for Thunderbolt vs Gyarados (Water/Flying)", async () => {
    game.override.moveset([MoveId.THUNDERBOLT]).enemySpecies(SpeciesId.GYARADOS);
    await game.classicMode.startBattle(SpeciesId.PIKACHU);

    const obs = encodeObservation(gs());
    // player_0 (p=0), move 0 (m=0), enemy_0 (e=0)
    expect(obs[DERIVED_BASE + 0]).toBe(1.0); // 4x / 4
    // STAB: Pikachu is Electric, Thunderbolt is Electric
    expect(obs[DERIVED_BASE + 16 + 0]).toBe(1.0);
  });

  it("derives 0x effectiveness for Thunderbolt vs Ground-type", async () => {
    game.override.moveset([MoveId.THUNDERBOLT]).enemySpecies(SpeciesId.DIGLETT);
    await game.classicMode.startBattle(SpeciesId.PIKACHU);

    const obs = encodeObservation(gs());
    expect(obs[DERIVED_BASE + 0]).toBe(0.0); // immune
  });

  it("derives no STAB for off-type moves", async () => {
    game.override.moveset([MoveId.TACKLE]).enemySpecies(SpeciesId.SHUCKLE);
    await game.classicMode.startBattle(SpeciesId.PIKACHU);

    const obs = encodeObservation(gs());
    expect(obs[DERIVED_BASE + 16 + 0]).toBe(0.0); // Normal move on Electric mon
  });

  it("documents turn_data.move_effectiveness as transient (cleared before decision points)", async () => {
    // GAME TRUTH: the game nulls target.turnData.moveEffectiveness for every
    // target at the END of MoveEffectPhase (move-effect-phase.ts ~900). It is
    // a transient damage-calc cache, NOT a persistent "last hit effectiveness".
    // By the time the RL builder runs at any decision point the value is
    // already cleared, so this observation dim is ~always 0. Tracked as a
    // limitation (the derived type-effectiveness block is the usable signal).
    game.override.moveset([MoveId.THUNDERBOLT]).enemySpecies(SpeciesId.GYARADOS).enemyLevel(100);
    await game.classicMode.startBattle(SpeciesId.PIKACHU);

    game.move.select(MoveId.THUNDERBOLT);
    await game.phaseInterceptor.to("MoveEndPhase");

    const state = gs();
    expect(state.enemy_0.turn_data.move_effectiveness).toBe(0); // cleared by the game

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("enemy_0", PKMN.MOVE_EFF)]).toBe(0);
  });

  it("ranks active-slot speeds in the derived block", async () => {
    game.override.moveset([MoveId.SPLASH]).enemySpecies(SpeciesId.SHUCKLE);
    await game.classicMode.startBattle(SpeciesId.REGIELEKI); // very fast vs very slow

    const obs = encodeObservation(gs());
    const playerRank = obs[DERIVED_BASE + 24 + 0];
    const enemyRank = obs[DERIVED_BASE + 24 + 2];
    expect(playerRank).toBeGreaterThan(enemyRank); // faster mon gets the higher rank value
  });
});
