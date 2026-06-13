/**
 * Semantic audit: positional tags (Future Sight, Wish) — was bug B1
 * (positional_tags hardcoded to []), so this guards the fix.
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { gs } from "./obs-layout";

describe("RL Semantic - Positional Tags", () => {
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
      .ability(AbilityId.BALL_FETCH)
      .enemyLevel(100); // Future Sight's delayed hit must not end the wave
  });

  it("serializes a pending Future Sight with countdown that decrements", async () => {
    await game.classicMode.startBattle(SpeciesId.MEW);

    game.move.use(MoveId.FUTURE_SIGHT);
    await game.toNextTurn();

    const liveTags = (game.scene.arena as any).positionalTagManager?.tags ?? [];
    expect(liveTags.length).toBe(1);

    let state = gs();
    expect(state.field.positional_tags.length).toBe(1);
    const tag = state.field.positional_tags[0];
    expect(tag.tag_type).toBe(liveTags[0].tagType);
    expect(tag.countdown).toBe(liveTags[0].turnCount);
    expect(tag.move_id).toBe(MoveId.FUTURE_SIGHT);
    const c0 = tag.countdown;

    game.move.use(MoveId.SPLASH);
    await game.toNextTurn();

    state = gs();
    expect(state.field.positional_tags.length).toBe(1);
    expect(state.field.positional_tags[0].countdown).toBe(c0 - 1);
  });

  it("serializes a pending Wish with heal_hp = floor(maxHp/2)", async () => {
    await game.classicMode.startBattle(SpeciesId.BLISSEY);
    const player = game.field.getPlayerPokemon();

    game.move.use(MoveId.WISH);
    await game.toNextTurn();

    const state = gs();
    expect(state.field.positional_tags.length).toBe(1);
    const tag = state.field.positional_tags[0];
    const liveTags = (game.scene.arena as any).positionalTagManager?.tags ?? [];
    expect(tag.heal_hp).toBe(liveTags[0].healHp); // Tier A mirror
    expect(tag.heal_hp).toBe(Math.floor(player.getMaxHp() / 2)); // Tier B rule
  });
});
