/**
 * Semantic teeth-test: buildGameState must emit a frozen SNAPSHOT, not live
 * references into the live game objects.
 *
 * Root-cause regression guard (rl-framework). pokemon.getStats(true) returns the
 * LIVE `this.stats` array BY REFERENCE. state-builder stored it directly, so the
 * dumped game state aliased it. When stats were recalculated in place between the
 * moment the observation is encoded and the moment the dict is serialized to JSON
 * (e.g. at an evolution boundary), the serialized stats diverged from the frozen
 * observation — a TS<->Python parity mismatch, with the dict left internally
 * inconsistent (pre-evolution species + post-evolution stats). The builder now
 * snapshots the array (.slice()).
 *
 * Teeth: on the old aliasing builder the post-mutation assertion FAILS, because
 * state.player_0.stats IS pokemon.stats.
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs } from "./obs-layout";

describe("RL Semantic - game-state snapshot (no live references)", () => {
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

  it("snapshots player stats so a later in-place mutation does not leak into the built state", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);
    const pokemon = game.field.getPlayerPokemon();

    const state = gs();
    const captured = [...state.player_0.stats];
    expect(captured.length).toBe(6);

    // Mutate the live this.stats array in place, exactly as an evolution's stat
    // recalculation does. getStats(true) returns that very array by reference.
    const liveStats = pokemon.getStats(true);
    liveStats[0] += 50;
    liveStats[1] += 50;

    // The already-built dict must NOT reflect the later mutation: it is a snapshot.
    expect(state.player_0.stats).toEqual(captured);
    expect(state.player_0.stats[0]).not.toBe(pokemon.stats[0]);
  });
});
