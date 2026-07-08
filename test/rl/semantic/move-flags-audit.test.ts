/**
 * Semantic audit: v6/v7 runtime MoveAttr flag extraction.
 *
 * state-builder.ts derives ~80 boolean move flags via string-named
 * `move.hasAttr("SomeAttr")` lookups. If the game renames an attr class the
 * flag silently becomes false for every move — no crash, no warning. This
 * table pins known-true (and known-false) flags for moves whose semantics
 * are stable game rules.
 *
 * One battle for the whole table: the moveset is swapped per row with
 * game.move.changeMoveset, then buildGameState() re-reads the live moveset.
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
import { gs, MOVE, moveDim } from "./obs-layout";

const MOVE_FLAG_TRUTH: [MoveId, Record<string, unknown>][] = [
  [
    MoveId.STEALTH_ROCK,
    { sets_arena_tag: true, sets_hazard: true, sets_screen: false, arena_tag_self_side: false, is_reflectable: true },
  ],
  [MoveId.REFLECT, { sets_screen: true, arena_tag_self_side: true, sets_hazard: false }],
  [MoveId.U_TURN, { self_switch: true, force_switch: false }],
  [MoveId.WHIRLWIND, { force_switch: true, self_switch: false }],
  [MoveId.KNOCK_OFF, { removes_item: true, steals_item: false }],
  [MoveId.THIEF, { steals_item: true, removes_item: false }],
  [MoveId.BUG_BITE, { steals_berry: true }],
  [MoveId.RAIN_DANCE, { weather_change: WeatherType.RAIN }],
  [MoveId.GRASSY_TERRAIN, { terrain_change: TerrainType.GRASSY }],
  [MoveId.LEECH_SEED, { applies_battler_tag: true, applies_continuous_damage: true }],
  [MoveId.TAUNT, { applies_battler_tag: true, applies_move_restriction: true }],
  [MoveId.SEISMIC_TOSS, { is_level_damage: true, fixed_damage: 0 }],
  [MoveId.SUPER_FANG, { is_target_half_hp: true }],
  [MoveId.COUNTER, { is_counter_damage: true }],
  [MoveId.FUTURE_SIGHT, { is_delayed_attack: true }],
  [MoveId.WISH, { is_wish: true }],
  [MoveId.PROTECT, { is_protect: true }],
  [MoveId.TRANSFORM, { transforms_into_target: true }],
  [MoveId.SKETCH, { copies_move_perm: true }],
  [MoveId.DEFOG, { removes_arena_tags: true }],
  [MoveId.COURT_CHANGE, { swaps_arena_tags: true }],
  [MoveId.EXPLOSION, { is_sacrifice: true }],
  [MoveId.DOUBLE_EDGE, { is_reckless_move: true, makes_contact: true }],
  // v8 (Group 18): survival / HP-relative semantics
  [MoveId.FALSE_SWIPE, { survives_at_1hp: true }],
  [MoveId.ENDEAVOR, { matches_user_hp: true }],
  [MoveId.BELLY_DRUM, { hp_cost_stat_boost: true }],
  [MoveId.STOMP, { hits_semi_invulnerable: true }], // HitsTagForDoubleDamageAttr extends HitsTagAttr
];

describe("RL Semantic - Move Flag Audit", () => {
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

  it("extracts the expected runtime MoveAttr flags for known moves", async () => {
    await game.classicMode.startBattle(SpeciesId.MEW);
    const player = game.field.getPlayerPokemon();

    for (const [moveId, expected] of MOVE_FLAG_TRUTH) {
      game.move.changeMoveset(player, [moveId]);
      const slot = gs().player_0.moves[0];
      expect(slot.move_id, `${MoveId[moveId]}.move_id`).toBe(moveId);
      for (const [key, value] of Object.entries(expected)) {
        expect(slot[key], `${MoveId[moveId]}.${key}`).toBe(value);
      }
    }
  });

  it("tracks PP usage in state and encoding", async () => {
    game.override.moveset([MoveId.SPLASH]);
    await game.classicMode.startBattle(SpeciesId.MEW);
    const player = game.field.getPlayerPokemon();

    game.move.select(MoveId.SPLASH);
    await game.toNextTurn();

    const liveSlot = player.getMoveset()[0];
    expect(liveSlot.ppUsed).toBe(1);

    const state = gs();
    const move = state.player_0.moves[0];
    expect(move.pp_used).toBe(1);
    expect(move.pp_max).toBe(liveSlot.getMovePp());
    expect(move.pp_remaining).toBe(move.pp_max - 1);

    const obs = encodeObservation(state);
    expect(obs[moveDim("player_0", 0, MOVE.PP_RATIO)]).toBeCloseTo((move.pp_max - 1) / move.pp_max, 6);
  });

  it("encodes kept flags + has_other_effect for Stealth Rock and Knock Off (v9)", async () => {
    game.override.moveset([MoveId.STEALTH_ROCK, MoveId.KNOCK_OFF]);
    await game.classicMode.startBattle(SpeciesId.MEW);

    const obs = encodeObservation(gs());
    // Stealth Rock: sets_arena_tag is a KEPT flag; sets_hazard folded into
    // the catch-all
    expect(obs[moveDim("player_0", 0, MOVE.SETS_ARENA_TAG)]).toBe(1);
    expect(obs[moveDim("player_0", 0, MOVE.HAS_OTHER_EFFECT)]).toBe(1);
    // Knock Off: removes_item is a CUT flag → catch-all; steals_item kept
    // and false for Knock Off
    expect(obs[moveDim("player_0", 1, MOVE.STEALS_ITEM)]).toBe(0);
    expect(obs[moveDim("player_0", 1, MOVE.HAS_OTHER_EFFECT)]).toBe(1);
    // A vanilla damaging move with no folded effects keeps the catch-all 0
    // (Stealth Rock's slot-2 empty move stays all-zero)
    expect(obs[moveDim("player_0", 2, MOVE.HAS_OTHER_EFFECT)]).toBe(0);
  });
});
