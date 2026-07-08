/**
 * Mirror of the observation encoder's internal layout — test-side source of truth.
 *
 * Every offset below was derived by hand-counting the writes in
 * `src/rl/spaces.ts` (`encodePokemonFromDict`, `encodeMoveFromDict`,
 * `encodeFieldFromDict`, `encodeBattleFromDict`, `encodeObservation`).
 * The canary suite (`layout-canary.test.ts`) proves these offsets against
 * the real encoder with synthetic-dict probes; if spaces.ts is ever
 * re-ordered or re-sized, the canary fails before any semantic test runs.
 *
 * NOT a test file — helper module only.
 */

import { ArenaTagType } from "#enums/arena-tag-type";
import type { BattlerTagType } from "#enums/battler-tag-type";
import {
  ARENA_TAG_ORDER,
  BATTLE_META_DIM,
  CURATED_VOLATILE_TAGS,
  DERIVED_FIELDS_DIM,
  FIELD_STATE_DIM,
  LEARN_MOVE_BLOCK_DIM,
  MODIFIER_INVENTORY_DIM,
  MODIFIER_PHASE_DIM,
  MOVE_BLOCK_DIM,
  POKEMON_BLOCK_DIM,
  TOTAL_POKEMON_SLOTS,
} from "#rl/spaces";
import { buildGameState } from "#rl/state-builder";

// ─── Slot ordering ────────────────────────────────────────────────────
// Mirrors POKEMON_SLOT_KEYS in spaces.ts (~line 1505): actives first
// (player 0-1, enemy 0-1), then player bench, then enemy bench.
export const SLOT_ORDER = [
  "player_0",
  "player_1",
  "enemy_0",
  "enemy_1",
  "player_2",
  "player_3",
  "player_4",
  "player_5",
  "enemy_2",
  "enemy_3",
  "enemy_4",
  "enemy_5",
] as const;

export type SlotKey = (typeof SLOT_ORDER)[number];

// ─── Intra-Pokemon offsets (encodePokemonFromDict, spaces.ts ~684-872) ──
// Width of each segment noted; next offset = previous + width.
export const PKMN = {
  VALID: 0, // 1
  HP_RATIO: 1, // 1
  LEVEL: 2, // 1 (/100)
  BASE_STATS: 3, // 6 (/255)
  STAT_STAGES: 9, // 7 (/6)
  TYPE1: 16, // 19 one-hot
  TYPE2: 35, // 19 one-hot
  STATUS: 54, // 8 one-hot
  NATURE: 62, // 5 (raw multipliers, fallback 1.0; fog: zeroed for enemies)
  ABILITY_FEAT: 67, // 40
  PASSIVE_FEAT: 107, // 40
  IS_TERA: 147, // 1
  TERA_TYPE: 148, // 19 one-hot
  VOLATILE_TAGS: 167, // 69 (CURATED_VOLATILE_TAGS order; v9: 76 → 69)
  OTHER_TAG_COUNT: 236, // 1 (/10)
  IS_BOSS: 237, // 1
  BOSS_SHIELD: 238, // 1
  IS_TRAPPED: 239, // 1
  IS_GROUNDED: 240, // 1
  WEIGHT: 241, // 1 (/1000)
  CATCH_RATE: 242, // 1 (/255)
  IS_FAINTED: 243, // 1
  WAVE_TURN_COUNT: 244, // 1 (/20)
  DAMAGE_TAKEN: 245, // 1 (turn_data.damage_taken / max_hp)
  ACTED: 246, // 1 (turn_data.acted)
  TOXIC_TURNS: 247, // 1 (/16)
  SLEEP_TURNS: 248, // 1 (/4)
  HELD_ITEM_COUNT: 249, // 1 (/10)
  SPECIES_ID: 250, // 1 (/1025)
  GENDER: 251, // 1
  FRIENDSHIP: 252, // 1 (/255)
  MOVE_QUEUE_LEN: 253, // 1 (/2)
  HIT_COUNT: 254, // 1 (battle_data.hit_count / 10)
  ABILITY_SUPPRESSED: 255, // 1
  IS_MEGA: 256, // 1
  IS_MAX: 257, // 1
  MOVE_EFF: 258, // 1 (turn_data.move_effectiveness / 4)
  COMPUTED_STATS: 259, // 5 (stats[1..5] / 500; fog: zeroed for enemies)
  // v9 additions
  AI_TYPE: 264, // 3 one-hot [RANDOM, SMART_RANDOM, SMART]; zeros on player slots
  MOVE_KNOWN: 267, // 4 revealed-indicators (enemy only)
  ABILITY_KNOWN: 271, // 1
  WAS_SEEN: 272, // 1
  MOVES: 273, // 4 × MOVE_BLOCK_DIM (60) = 240 → 513 total
} as const;

// ─── Intra-move offsets (encodeMoveFromDict, spaces.ts ~436-675) ─────
export const MOVE = {
  VALID: 0, // 1
  TYPE: 1, // 19 one-hot
  CATEGORY: 20, // 3 one-hot
  POWER: 23, // 1 (/250)
  ACCURACY: 24, // 1 (/100; <=0 → 1.0)
  PP_RATIO: 25, // 1 (pp_remaining / pp_max)
  PRIORITY: 26, // 1 (/7)
  EFFECT_CHANCE: 27, // 1 (/100)
  DRAIN_RATIO: 28, // 1
  HEAL_RATIO: 29, // 1
  MULTI_HIT_COUNT: 30, // 1 (count/5; 0 = not multi-hit) — v9, replaces IS_MULTI_HIT
  FORCE_SWITCH: 31,
  IS_PROTECT: 32,
  TRAPS_TARGET: 33,
  MAKES_CONTACT: 34,
  IS_USABLE: 35,
  STATUS_EFFECT: 36, // 1 (/7)
  STAT_CHANGE_SELF_SUM: 37, // 1 (/12)
  STAT_CHANGE_TARGET_SUM: 38, // 1 (/12)
  RECOIL_RATIO: 39,
  CRIT_STAGE_BOOST: 40, // 1 (/3)
  TARGET_CLASS: 41, // 3 one-hot [self/ally, single enemy, multi/field]
  // v9 kept effect flags (evidence keep-list)
  IGNORES_PROTECT: 44,
  IS_SOUND_BASED: 45,
  CAN_FLINCH: 46,
  CAN_CONFUSE: 47,
  HAS_VARIABLE_POWER: 48,
  WEATHER_CHANGE: 49, // 1 (WeatherType / 9)
  SETS_ARENA_TAG: 50,
  APPLIES_BATTLER_TAG: 51,
  APPLIES_MOVE_RESTRICTION: 52,
  IS_WIND_MOVE: 53,
  IS_RECKLESS_MOVE: 54,
  IS_REFLECTABLE: 55,
  IS_TRIAGE_MOVE: 56,
  STEALS_ITEM: 57,
  HITS_SEMI_INVULNERABLE: 58,
  HAS_OTHER_EFFECT: 59, // OR of the 77 cut v8 flags — last move dim (MOVE_BLOCK_DIM 60)
} as const;

// ─── Block bases (encodeObservation, spaces.ts ~1518-1573) ───────────
export const FIELD_BASE = TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM; // 6156
export const BATTLE_BASE = FIELD_BASE + FIELD_STATE_DIM; // 6258
export const MODPHASE_BASE = BATTLE_BASE + BATTLE_META_DIM; // 6298
export const MODINV_BASE = MODPHASE_BASE + MODIFIER_PHASE_DIM; // 6661
export const DERIVED_BASE = MODINV_BASE + MODIFIER_INVENTORY_DIM; // 6881
export const LEARN_MOVE_BASE = DERIVED_BASE + DERIVED_FIELDS_DIM; // 6909 (v9)
export const PHASE_BASE = LEARN_MOVE_BASE + LEARN_MOVE_BLOCK_DIM; // 6975

// ─── Field-internal offsets (encodeFieldFromDict, spaces.ts ~876-1001) ──
export const FIELD = {
  WEATHER_OH: 0, // 10 one-hot
  WEATHER_TURNS: 10, // 1 (/8)
  TERRAIN_OH: 11, // 5 one-hot
  TERRAIN_TURNS: 16, // 1 (/8)
  PLAYER_TAGS: 17, // 28 binary (ARENA_TAG_ORDER; BOTH-side tags set both banks)
  PLAYER_SPIKES_LAYERS: 45, // 1 (/3)
  PLAYER_TSPIKES_LAYERS: 46, // 1 (/2)
  ENEMY_TAGS: 47, // 28 binary
  ENEMY_SPIKES_LAYERS: 75, // 1 (/3)
  ENEMY_TSPIKES_LAYERS: 76, // 1 (/2)
  IS_DOUBLE: 77,
  TRICK_ROOM: 78,
  GRAVITY: 79,
  WEATHER_IS_PERMANENT: 80,
  WEATHER_SUPPRESSED: 81,
  TERRAIN_IS_PERMANENT: 82,
  KEY_TAG_TURNS: 83, // 10 = 5 tags × 2 sides, player side first (turn_count / 8)
  PLAYER_TERAS_USED: 93, // 1 (/3)
  // v9: positional tags (Wish / Future Sight per side)
  PLAYER_WISH_ACTIVE: 94,
  PLAYER_WISH_TURNS: 95, // 1 (/8)
  PLAYER_FUTURE_SIGHT_ACTIVE: 96,
  PLAYER_FUTURE_SIGHT_TURNS: 97, // 1 (/8)
  ENEMY_WISH_ACTIVE: 98,
  ENEMY_WISH_TURNS: 99, // 1 (/8)
  ENEMY_FUTURE_SIGHT_ACTIVE: 100,
  ENEMY_FUTURE_SIGHT_TURNS: 101, // 1 (/8) — last field dim (102)
} as const;

/** KEY_ARENA_TAGS order inside the KEY_TAG_TURNS sub-block (spaces.ts ~964-970) */
export const KEY_TAG_TURNS_ORDER: readonly ArenaTagType[] = [
  ArenaTagType.REFLECT,
  ArenaTagType.LIGHT_SCREEN,
  ArenaTagType.AURORA_VEIL,
  ArenaTagType.TAILWIND,
  ArenaTagType.TRICK_ROOM,
];

// ─── Battle-internal offsets (encodeBattleFromDict, spaces.ts ~1005-1106) ──
export const BATTLE = {
  WAVE: 0, // 1 (/200)
  TURN: 1, // 1 (/50)
  BATTLE_TYPE_OH: 2, // 4 one-hot
  MONEY: 6, // 1 (log-normalized)
  SCORE: 7, // 1 (log-normalized)
  POKEBALLS: 8, // 5 (/99)
  PLAYER_ALIVE: 13, // 1 (/6)
  ENEMY_ALIVE: 14, // 1 (/6)
  TERA_AVAILABLE: 15,
  CAN_RUN: 16,
  CAN_CATCH: 17,
  PLAYER_FAINTS: 18, // 1 (/6)
  ENEMY_FAINTS: 19, // 1 (/6)
  COMMAND_FIELD_INDEX: 20, // patched from phase info in encodeObservation
  BIOME: 21, // 1 (/40)
  ESCAPE_ATTEMPTS: 22, // 1 (/10)
  BATTLE_STYLE: 23,
  TIME_OF_DAY: 24,
  LOCK_MODIFIER_TIERS: 25,
  BATTLE_SPEC: 26,
  GAME_MODE: 27,
  TRAINER_SPECIALTY: 28,
  HAS_NO_SHOP: 29,
  SEEN_ENEMY_COUNT: 30,
  IS_CLASSIC: 31,
  IS_ENDLESS: 32,
  IS_DAILY: 33,
  IS_CHALLENGE: 34,
  HAS_MYSTERY_ENCOUNTERS: 35,
  HAS_SHORT_BIOMES: 36,
  HAS_RANDOM_BIOMES: 37,
  HAS_RANDOM_BOSSES: 38,
  INVERSE_BATTLE: 39, // last battle dim
} as const;

// ─── Dim helpers ──────────────────────────────────────────────────────

/** Absolute observation index of a non-move Pokemon feature. */
export function pokemonDim(slot: SlotKey, off: number): number {
  const slotIdx = SLOT_ORDER.indexOf(slot);
  if (slotIdx < 0) {
    throw new Error(`Unknown slot key: ${slot}`);
  }
  return slotIdx * POKEMON_BLOCK_DIM + off;
}

/** Absolute observation index of a move feature inside a Pokemon block. */
export function moveDim(slot: SlotKey, moveIdx: number, off: number): number {
  return pokemonDim(slot, PKMN.MOVES + moveIdx * MOVE_BLOCK_DIM + off);
}

/** Absolute observation index of a field feature. */
export function fieldDim(off: number): number {
  return FIELD_BASE + off;
}

/** Absolute observation index of a battle-meta feature. */
export function battleDim(off: number): number {
  return BATTLE_BASE + off;
}

/** Index of an arena tag inside the 28-dim per-side tag banks. */
export function arenaTagIndex(tag: ArenaTagType): number {
  const idx = ARENA_TAG_ORDER.indexOf(tag);
  if (idx < 0) {
    throw new Error(`Tag ${tag} not in ARENA_TAG_ORDER`);
  }
  return idx;
}

/** Index of a battler tag inside the 48-dim curated volatile tag block. */
export function volatileTagIndex(tag: BattlerTagType): number {
  const idx = CURATED_VOLATILE_TAGS.indexOf(tag);
  if (idx < 0) {
    throw new Error(`Tag ${tag} not in CURATED_VOLATILE_TAGS`);
  }
  return idx;
}

/** Absolute observation index of a key-tag remaining-turns dim. */
export function keyTagTurnsDim(side: "player" | "enemy", tag: ArenaTagType): number {
  const idx = KEY_TAG_TURNS_ORDER.indexOf(tag);
  if (idx < 0) {
    throw new Error(`Tag ${tag} not in KEY_TAG_TURNS_ORDER`);
  }
  return fieldDim(FIELD.KEY_TAG_TURNS + (side === "player" ? 0 : KEY_TAG_TURNS_ORDER.length) + idx);
}

/** Build the live game state from the current globalScene (mid-test). */
// biome-ignore lint/suspicious/noExplicitAny: tests need free-form access to the JSON dict
export function gs(): any {
  // biome-ignore lint/suspicious/noExplicitAny: see above
  return buildGameState(null, 0) as any;
}
