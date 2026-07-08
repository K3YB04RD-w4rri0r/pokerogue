/**
 * Semantic audit: slot mapping in double battles + KO/switch slot integrity.
 *
 * Slot layout: player_0/player_1 + enemy_0/enemy_1 are FIELD positions;
 * player_2..5 / enemy_2..5 are bench (party order minus field members).
 * state-builder.ts has an explicit "do NOT filter" warning here — this
 * mapping already regressed once.
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { FIELD, fieldDim, gs, PKMN, pokemonDim } from "./obs-layout";

describe("RL Semantic - Double Battle Slots & KO/Switch", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .moveset([MoveId.SPLASH])
      .enemySpecies(SpeciesId.SHUCKLE)
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("maps field positions and bench in a double battle", async () => {
    game.override.battleStyle("double");
    await game.classicMode.startBattle(SpeciesId.MAGIKARP, SpeciesId.MEOWTH, SpeciesId.ABRA);

    const [active0, active1] = game.scene.getPlayerField();
    const state = gs();

    expect(state.field.is_double_battle).toBe(true);
    expect(state.battle.is_double).toBe(true);

    expect(state.player_0.valid).toBe(true);
    expect(state.player_0.species_id).toBe(active0.species.speciesId);
    expect(state.player_1.valid).toBe(true);
    expect(state.player_1.species_id).toBe(active1.species.speciesId);

    // Bench: ABRA is the only non-field party member -> player_2
    expect(state.player_2.valid).toBe(true);
    expect(state.player_2.species_id).toBe(SpeciesId.ABRA);
    expect(state.player_3.valid).toBe(false);

    // Enemy doubles
    const [enemy0, enemy1] = game.scene.getEnemyField();
    expect(state.enemy_0.species_id).toBe(enemy0.species.speciesId);
    expect(state.enemy_1.species_id).toBe(enemy1.species.speciesId);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.IS_DOUBLE)]).toBe(1);
    expect(obs[pokemonDim("player_2", PKMN.VALID)]).toBe(1);
    expect(obs[pokemonDim("player_3", PKMN.VALID)]).toBe(0);
  });

  it("remaps slots after a switch: new active in player_0, old active on bench", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP, SpeciesId.ABRA);

    // v9 singles slot mapping: the (empty-in-singles) ally slot doubles as
    // FIRST BENCH — slot 1 = second active in doubles, first bench in
    // singles. This is what makes a 6th party member observable at all.
    const before = gs();
    expect(before.player_0.species_id).toBe(SpeciesId.MAGIKARP);
    expect(before.player_1.species_id).toBe(SpeciesId.ABRA);
    expect(before.player_2.valid).toBe(false);

    game.doSwitchPokemon(1);
    await game.toNextTurn();

    const after = gs();
    expect(after.player_0.species_id).toBe(SpeciesId.ABRA);
    // is_on_field is field membership; is_active means "alive and allowed in
    // battle" (pokemon.isActive()) and stays true for a healthy benched mon
    expect(after.player_0.is_on_field).toBe(true);
    expect(after.player_1.species_id).toBe(SpeciesId.MAGIKARP);
    expect(after.player_1.is_on_field).toBe(false);
    expect(after.player_1.is_active).toBe(true);
  });

  it("marks a KO'd enemy as fainted with zero HP ratio (doubles, one survivor)", async () => {
    game.override
      .battleStyle("double")
      .moveset([MoveId.FISSURE, MoveId.SPLASH])
      .ability(AbilityId.NO_GUARD)
      .startingLevel(200);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP, SpeciesId.MEOWTH);

    const [enemy0] = game.scene.getEnemyField();
    game.move.select(MoveId.FISSURE, 0, enemy0.getBattlerIndex());
    game.move.select(MoveId.SPLASH, 1);
    // toEndOfTurn, not the first MoveEndPhase: the faster ally's Splash
    // resolves before Fissure does
    await game.toEndOfTurn();

    expect(enemy0.isFainted()).toBe(true);

    const state = gs();
    expect(state.enemy_0.is_fainted).toBe(true);
    expect(state.enemy_0.hp).toBe(0);
    expect(state.enemy_0.hp_ratio).toBe(0);

    const obs = encodeObservation(state);
    expect(obs[pokemonDim("enemy_0", PKMN.IS_FAINTED)]).toBe(1);
    expect(obs[pokemonDim("enemy_0", PKMN.HP_RATIO)]).toBe(0);
  });

  it("counts battle faints after a KO", async () => {
    game.override
      .battleStyle("double")
      .moveset([MoveId.FISSURE, MoveId.SPLASH])
      .ability(AbilityId.NO_GUARD)
      .startingLevel(200);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP, SpeciesId.MEOWTH);

    const [enemy0] = game.scene.getEnemyField();
    game.move.select(MoveId.FISSURE, 0, enemy0.getBattlerIndex());
    game.move.select(MoveId.SPLASH, 1);
    await game.toEndOfTurn();

    const state = gs();
    expect(state.battle.enemy_faints_battle).toBe(game.scene.currentBattle.enemyFaints);
    expect(state.battle.enemy_faints_battle).toBe(1);
  });
});
