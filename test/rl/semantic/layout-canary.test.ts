/**
 * Layout canary — proves the offsets in obs-layout.ts against the real
 * encoder using pure synthetic GameState dicts (no battle, no Phaser game).
 *
 * If any test here fails, the offsets in obs-layout.ts no longer match
 * spaces.ts and EVERY semantic audit test is suspect — fix this first.
 */

import { AbilityId } from "#enums/ability-id";
import { ArenaTagType } from "#enums/arena-tag-type";
import { BattlerTagType } from "#enums/battler-tag-type";
import { ABILITY_FEATURES } from "#rl/ability-features";
import {
  ABILITY_FEATURE_DIM,
  ARENA_TAG_ORDER,
  BATTLE_META_DIM,
  CURATED_VOLATILE_TAGS,
  encodeObservation,
  FIELD_STATE_DIM,
  MOVE_BLOCK_DIM,
  NUM_CURATED_TAGS,
  OBSERVATION_DIM,
  PHASE_INDICATOR_DIM,
  POKEMON_BLOCK_DIM,
} from "#rl/spaces";
import { describe, expect, it } from "vitest";
import {
  arenaTagIndex,
  BATTLE,
  BATTLE_BASE,
  battleDim,
  DERIVED_BASE,
  FIELD,
  FIELD_BASE,
  fieldDim,
  keyTagTurnsDim,
  LEARN_MOVE_BASE,
  MODINV_BASE,
  MODPHASE_BASE,
  MOVE,
  moveDim,
  PHASE_BASE,
  PKMN,
  pokemonDim,
  SLOT_ORDER,
  volatileTagIndex,
} from "./obs-layout";

describe("RL Semantic Audit — Layout Canary", () => {
  describe("block-base arithmetic", () => {
    it("block bases land on the expected absolute indices", () => {
      expect(POKEMON_BLOCK_DIM).toBe(513);
      expect(MOVE_BLOCK_DIM).toBe(60);
      expect(FIELD_BASE).toBe(6156);
      expect(BATTLE_BASE).toBe(6258);
      expect(MODPHASE_BASE).toBe(6298);
      expect(MODINV_BASE).toBe(6661);
      expect(DERIVED_BASE).toBe(6881);
      expect(LEARN_MOVE_BASE).toBe(6909);
      expect(PHASE_BASE).toBe(6975);
      expect(PHASE_BASE + PHASE_INDICATOR_DIM).toBe(OBSERVATION_DIM);
      expect(OBSERVATION_DIM).toBe(6991);
    });

    it("intra-block offsets tile their blocks exactly", () => {
      // Pokemon non-move segment + 4 move blocks fill the pokemon block
      expect(PKMN.MOVES).toBe(273);
      expect(PKMN.MOVES + 4 * MOVE_BLOCK_DIM).toBe(POKEMON_BLOCK_DIM);
      // Last named offsets are the last dims of their blocks
      expect(MOVE.HAS_OTHER_EFFECT).toBe(MOVE_BLOCK_DIM - 1);
      expect(FIELD.ENEMY_FUTURE_SIGHT_TURNS).toBe(FIELD_STATE_DIM - 1);
      expect(BATTLE.INVERSE_BATTLE).toBe(BATTLE_META_DIM - 1);
      // Last dim of move slot 3 is the last dim of the pokemon block
      expect(moveDim("player_0", 3, MOVE.HAS_OTHER_EFFECT)).toBe(POKEMON_BLOCK_DIM - 1);
      // 12 slots, no gaps
      expect(SLOT_ORDER.length).toBe(12);
      expect(pokemonDim("enemy_5", PKMN.VALID) + POKEMON_BLOCK_DIM).toBe(FIELD_BASE);
    });
  });

  describe("constant table sizes", () => {
    it("CURATED_VOLATILE_TAGS has 69 entries (v9)", () => {
      expect(CURATED_VOLATILE_TAGS.length).toBe(69);
      expect(NUM_CURATED_TAGS).toBe(69);
    });

    it("ARENA_TAG_ORDER has 28 entries", () => {
      expect(ARENA_TAG_ORDER.length).toBe(28);
    });

    it("ABILITY_FEATURES covers every AbilityId with 40-dim rows", () => {
      const maxAbilityId = AbilityId.POISON_PUPPETEER;
      expect(ABILITY_FEATURES.length).toBeGreaterThanOrEqual(maxAbilityId + 1);
      for (let i = 0; i < ABILITY_FEATURES.length; i++) {
        expect(ABILITY_FEATURES[i].length, `ABILITY_FEATURES[${i}] row width`).toBe(ABILITY_FEATURE_DIM);
      }
    });
  });

  describe("synthetic-dict probes", () => {
    it("pokemon scalar offsets (active + bench slot)", () => {
      const obs = encodeObservation({
        enemy_0: { valid: true, hp_ratio: 0.5, level: 50 },
        // bench slot probe — proves SLOT_ORDER places player_3 at index 5
        player_3: {
          valid: true,
          is_mega: true,
          is_max: true,
          stats: [999, 250, 0, 0, 0, 100],
          volatile_tags: [{ tag_type: BattlerTagType.SALT_CURED }],
        },
      });
      expect(obs.length).toBe(OBSERVATION_DIM);

      expect(obs[pokemonDim("enemy_0", PKMN.VALID)]).toBe(1);
      expect(obs[pokemonDim("enemy_0", PKMN.HP_RATIO)]).toBe(0.5);
      expect(obs[pokemonDim("enemy_0", PKMN.LEVEL)]).toBe(0.5);
      // nature multipliers default-fill with 1.0 when input missing
      expect(obs[pokemonDim("enemy_0", PKMN.NATURE)]).toBe(1);
      expect(obs[pokemonDim("enemy_0", PKMN.NATURE + 4)]).toBe(1);

      expect(obs[pokemonDim("player_3", PKMN.IS_MEGA)]).toBe(1);
      expect(obs[pokemonDim("player_3", PKMN.IS_MAX)]).toBe(1);
      // computed stats use indices 1..5 of `stats`, /500
      expect(obs[pokemonDim("player_3", PKMN.COMPUTED_STATS)]).toBe(0.5);
      expect(obs[pokemonDim("player_3", PKMN.COMPUTED_STATS + 4)]).toBeCloseTo(0.2, 6);
      // curated volatile tag bank
      expect(obs[pokemonDim("player_3", PKMN.VOLATILE_TAGS + volatileTagIndex(BattlerTagType.SALT_CURED))]).toBe(1);
      expect(obs[pokemonDim("player_3", PKMN.OTHER_TAG_COUNT)]).toBe(0);
      // invalid slots stay zero
      expect(obs[pokemonDim("player_0", PKMN.VALID)]).toBe(0);
      expect(obs[pokemonDim("enemy_5", PKMN.VALID)]).toBe(0);
    });

    it("move block offsets (head, kept flags, catch-all)", () => {
      const obs = encodeObservation({
        player_0: {
          valid: true,
          moves: [
            {
              move_id: 33,
              type: 4, // GROUND
              category: 1, // SPECIAL
              power: 125, // → /250 = 0.5
              accuracy: 50, // → /100 = 0.5
              pp_max: 10,
              pp_remaining: 5, // → 0.5
              priority: 7, // → /7 = 1.0
              self_switch: true,
              weather_change: 2, // → /9
              terrain_change: 3, // → /4 = 0.75
              sets_arena_tag: true,
              sets_hazard: true,
              sets_screen: true,
              arena_tag_self_side: true,
              applies_battler_tag: true,
              applies_move_restriction: true,
              applies_continuous_damage: true,
              is_level_damage: true,
              is_delayed_attack: true,
              steals_item: true,
              removes_item: true,
              scatters_money: true, // last move dim
            },
          ],
        },
      });

      const d = (off: number) => obs[moveDim("player_0", 0, off)];
      expect(d(MOVE.VALID)).toBe(1);
      expect(d(MOVE.TYPE + 4)).toBe(1);
      expect(d(MOVE.TYPE + 0)).toBe(0);
      expect(d(MOVE.CATEGORY + 1)).toBe(1);
      expect(d(MOVE.POWER)).toBe(0.5);
      expect(d(MOVE.ACCURACY)).toBe(0.5);
      expect(d(MOVE.PP_RATIO)).toBe(0.5);
      expect(d(MOVE.PRIORITY)).toBe(1);
      expect(d(MOVE.FORCE_SWITCH)).toBe(0);
      expect(d(MOVE.WEATHER_CHANGE)).toBeCloseTo(2 / 9, 6);
      expect(d(MOVE.SETS_ARENA_TAG)).toBe(1);
      expect(d(MOVE.APPLIES_BATTLER_TAG)).toBe(1);
      expect(d(MOVE.APPLIES_MOVE_RESTRICTION)).toBe(1);
      expect(d(MOVE.STEALS_ITEM)).toBe(1);
      // self_switch, terrain_change, sets_hazard/screen, continuous damage,
      // level damage, delayed attack, removes_item — all cut flags — fold
      // into the catch-all
      expect(d(MOVE.HAS_OTHER_EFFECT)).toBe(1);
      // empty move slot 1 stays invalid
      expect(obs[moveDim("player_0", 1, MOVE.VALID)]).toBe(0);
    });

    it("field weather/terrain offsets", () => {
      const obs = encodeObservation({
        field: {
          weather_type: 2, // RAIN
          weather_turns_left: 4, // → /8 = 0.5
          terrain_type: 3, // GRASSY
          terrain_turns_left: 8, // → /8 = 1.0
        },
      });
      expect(obs[fieldDim(FIELD.WEATHER_OH + 2)]).toBe(1);
      expect(obs[fieldDim(FIELD.WEATHER_OH + 0)]).toBe(0);
      expect(obs[fieldDim(FIELD.WEATHER_TURNS)]).toBe(0.5);
      expect(obs[fieldDim(FIELD.TERRAIN_OH + 3)]).toBe(1);
      expect(obs[fieldDim(FIELD.TERRAIN_TURNS)]).toBe(1);
    });

    it("arena tag banks, hazard layers and key-tag turns", () => {
      const obs = encodeObservation({
        field: {
          arena_tags: [
            { tag_type: ArenaTagType.STEALTH_ROCK, side: 2, turn_count: 0 }, // ENEMY
            { tag_type: ArenaTagType.REFLECT, side: 1, turn_count: 4 }, // PLAYER
            { tag_type: ArenaTagType.TRICK_ROOM, side: 0, turn_count: 3 }, // BOTH
          ],
          player_spikes_layers: 2,
          enemy_toxic_spikes_layers: 1,
        },
      });
      const sr = arenaTagIndex(ArenaTagType.STEALTH_ROCK);
      const refl = arenaTagIndex(ArenaTagType.REFLECT);
      const tr = arenaTagIndex(ArenaTagType.TRICK_ROOM);

      expect(obs[fieldDim(FIELD.ENEMY_TAGS + sr)]).toBe(1);
      expect(obs[fieldDim(FIELD.PLAYER_TAGS + sr)]).toBe(0);
      expect(obs[fieldDim(FIELD.PLAYER_TAGS + refl)]).toBe(1);
      expect(obs[fieldDim(FIELD.ENEMY_TAGS + refl)]).toBe(0);
      // BOTH-side tags fan into both banks
      expect(obs[fieldDim(FIELD.PLAYER_TAGS + tr)]).toBe(1);
      expect(obs[fieldDim(FIELD.ENEMY_TAGS + tr)]).toBe(1);

      expect(obs[fieldDim(FIELD.PLAYER_SPIKES_LAYERS)]).toBeCloseTo(2 / 3, 6);
      expect(obs[fieldDim(FIELD.ENEMY_SPIKES_LAYERS)]).toBe(0);
      expect(obs[fieldDim(FIELD.ENEMY_TSPIKES_LAYERS)]).toBe(0.5);

      // key-tag turns: player side first, /8
      expect(obs[keyTagTurnsDim("player", ArenaTagType.REFLECT)]).toBe(0.5);
      expect(obs[keyTagTurnsDim("enemy", ArenaTagType.REFLECT)]).toBe(0);
      expect(obs[keyTagTurnsDim("player", ArenaTagType.TRICK_ROOM)]).toBe(3 / 8);
      expect(obs[keyTagTurnsDim("enemy", ArenaTagType.TRICK_ROOM)]).toBe(3 / 8);
    });

    it("battle meta offsets", () => {
      const obs = encodeObservation({
        battle: { wave_index: 100, turn: 25, battle_type: 1, inverse_battle: true },
      });
      expect(obs[battleDim(BATTLE.WAVE)]).toBe(0.5);
      expect(obs[battleDim(BATTLE.TURN)]).toBe(0.5);
      expect(obs[battleDim(BATTLE.BATTLE_TYPE_OH + 1)]).toBe(1);
      expect(obs[battleDim(BATTLE.BATTLE_TYPE_OH + 0)]).toBe(0);
      expect(obs[battleDim(BATTLE.INVERSE_BATTLE)]).toBe(1);
    });

    it("phase indicator one-hot", () => {
      const obsCommand = encodeObservation({ phase: { current_phase: "command" } });
      expect(obsCommand[PHASE_BASE + 0]).toBe(1);

      const obsSwitch = encodeObservation({ phase: { current_phase: "switch" } });
      expect(obsSwitch[PHASE_BASE + 4]).toBe(1);
      // exactly one bit set in the phase block
      let sum = 0;
      for (let i = 0; i < PHASE_INDICATOR_DIM; i++) {
        sum += obsSwitch[PHASE_BASE + i];
      }
      expect(sum).toBe(1);
    });
  });
});
