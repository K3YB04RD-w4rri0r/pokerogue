/**
 * Semantic audit: arena tag SIDE attribution in buildGameState() + encoding.
 *
 * Side semantics are the easiest thing for a serializer to get wrong:
 * a player-cast hazard lives on the ENEMY side (it hits enemies switching in),
 * while a player-cast screen lives on the PLAYER side. Tier A assertions
 * mirror the live arena; Tier B assertions pin known game rules.
 */
import { AbilityId } from "#enums/ability-id";
import { ArenaTagSide } from "#enums/arena-tag-side";
import { ArenaTagType } from "#enums/arena-tag-type";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { encodeObservation } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { arenaTagIndex, FIELD, fieldDim, gs, keyTagTurnsDim } from "./obs-layout";

describe("RL Semantic - Arena Tag Sides", () => {
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

  it("attributes player-cast Stealth Rock to the ENEMY side", async () => {
    game.override.moveset([MoveId.STEALTH_ROCK]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.STEALTH_ROCK);
    await game.toNextTurn();

    // Live-scene ground truth
    expect(game.scene.arena.getTagOnSide(ArenaTagType.STEALTH_ROCK, ArenaTagSide.ENEMY)).toBeDefined();

    const state = gs();
    expect(state.field.enemy_stealth_rock).toBe(true);
    expect(state.field.player_stealth_rock).toBe(false);
    const tag = state.field.arena_tags.find((t: any) => t.tag_type === ArenaTagType.STEALTH_ROCK);
    expect(tag).toBeDefined();
    expect(tag.side).toBe(ArenaTagSide.ENEMY);

    const obs = encodeObservation(state);
    const i = arenaTagIndex(ArenaTagType.STEALTH_ROCK);
    expect(obs[fieldDim(FIELD.ENEMY_TAGS + i)]).toBe(1);
    expect(obs[fieldDim(FIELD.PLAYER_TAGS + i)]).toBe(0);
  });

  it("counts player-cast Spikes layers on the enemy side", async () => {
    game.override.moveset([MoveId.SPIKES]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.SPIKES);
    await game.toNextTurn();
    game.move.select(MoveId.SPIKES);
    await game.toNextTurn();

    const liveTag = game.scene.arena.getTagOnSide(ArenaTagType.SPIKES, ArenaTagSide.ENEMY) as any;
    expect(liveTag).toBeDefined();
    expect(liveTag.layers).toBe(2);

    const state = gs();
    expect(state.field.enemy_spikes_layers).toBe(2);
    expect(state.field.player_spikes_layers).toBe(0);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.ENEMY_SPIKES_LAYERS)]).toBeCloseTo(2 / 3, 6);
    expect(obs[fieldDim(FIELD.PLAYER_SPIKES_LAYERS)]).toBe(0);
  });

  it("attributes enemy-cast Spikes to the player side", async () => {
    game.override.moveset([MoveId.SPLASH]).enemyMoveset(MoveId.SPIKES);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    expect(game.scene.arena.getTagOnSide(ArenaTagType.SPIKES, ArenaTagSide.PLAYER)).toBeDefined();

    const state = gs();
    expect(state.field.player_spikes_layers).toBe(1);
    expect(state.field.enemy_spikes_layers).toBe(0);

    const obs = encodeObservation(state);
    expect(obs[fieldDim(FIELD.PLAYER_SPIKES_LAYERS)]).toBeCloseTo(1 / 3, 6);
  });

  it("keeps player-cast Reflect on the PLAYER side with live turn count", async () => {
    game.override.moveset([MoveId.REFLECT]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.REFLECT);
    await game.toNextTurn();

    const liveTag = game.scene.arena.getTagOnSide(ArenaTagType.REFLECT, ArenaTagSide.PLAYER) as any;
    expect(liveTag).toBeDefined();

    const state = gs();
    const tag = state.field.arena_tags.find((t: any) => t.tag_type === ArenaTagType.REFLECT);
    expect(tag).toBeDefined();
    expect(tag.side).toBe(ArenaTagSide.PLAYER);
    expect(tag.turn_count).toBe(liveTag.turnCount);

    const obs = encodeObservation(state);
    const i = arenaTagIndex(ArenaTagType.REFLECT);
    expect(obs[fieldDim(FIELD.PLAYER_TAGS + i)]).toBe(1);
    expect(obs[fieldDim(FIELD.ENEMY_TAGS + i)]).toBe(0);
    // Key-tag remaining turns: player Reflect slot populated, enemy slot zero
    expect(obs[keyTagTurnsDim("player", ArenaTagType.REFLECT)]).toBeCloseTo(liveTag.turnCount / 8, 6);
    expect(obs[keyTagTurnsDim("enemy", ArenaTagType.REFLECT)]).toBe(0);
  });

  it("decrements Tailwind turn count between turns", async () => {
    game.override.moveset([MoveId.TAILWIND, MoveId.SPLASH]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.TAILWIND);
    await game.toNextTurn();
    const turnsAfterCast = gs().field.arena_tags.find((t: any) => t.tag_type === ArenaTagType.TAILWIND)?.turn_count;

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();
    const turnsNext = gs().field.arena_tags.find((t: any) => t.tag_type === ArenaTagType.TAILWIND)?.turn_count;

    expect(turnsAfterCast).toBeGreaterThan(0);
    expect(turnsNext).toBe(turnsAfterCast - 1);
  });

  it("fans BOTH-side Trick Room into both encoded tag banks", async () => {
    game.override.moveset([MoveId.TRICK_ROOM]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    game.move.select(MoveId.TRICK_ROOM);
    await game.toNextTurn();

    const state = gs();
    const tag = state.field.arena_tags.find((t: any) => t.tag_type === ArenaTagType.TRICK_ROOM);
    expect(tag).toBeDefined();
    expect(tag.side).toBe(ArenaTagSide.BOTH);
    expect(state.field.trick_room_active).toBe(true);

    const obs = encodeObservation(state);
    const i = arenaTagIndex(ArenaTagType.TRICK_ROOM);
    expect(obs[fieldDim(FIELD.PLAYER_TAGS + i)]).toBe(1);
    expect(obs[fieldDim(FIELD.ENEMY_TAGS + i)]).toBe(1);
    expect(obs[fieldDim(FIELD.TRICK_ROOM)]).toBe(1);
  });
});
