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
  NATURE: 62, // 5 (raw multipliers, fallback 1.0)
  ABILITY_FEAT: 67, // 40
  PASSIVE_FEAT: 107, // 40
  IS_TERA: 147, // 1
  TERA_TYPE: 148, // 19 one-hot
  VOLATILE_TAGS: 167, // 48 (CURATED_VOLATILE_TAGS order)
  OTHER_TAG_COUNT: 215, // 1 (/10)
  IS_BOSS: 216, // 1
  BOSS_SHIELD: 217, // 1
  IS_TRAPPED: 218, // 1
  IS_GROUNDED: 219, // 1
  WEIGHT: 220, // 1 (/1000)
  CATCH_RATE: 221, // 1 (/255)
  IS_FAINTED: 222, // 1
  WAVE_TURN_COUNT: 223, // 1 (/20)
  DAMAGE_TAKEN: 224, // 1 (turn_data.damage_taken / max_hp)
  ACTED: 225, // 1 (turn_data.acted)
  TOXIC_TURNS: 226, // 1 (/16)
  SLEEP_TURNS: 227, // 1 (/4)
  HELD_ITEM_COUNT: 228, // 1 (/10)
  SPECIES_ID: 229, // 1 (/1025)
  GENDER: 230, // 1
  FRIENDSHIP: 231, // 1 (/255)
  MOVE_QUEUE_LEN: 232, // 1 (/2)
  HIT_COUNT: 233, // 1 (battle_data.hit_count / 10)
  ABILITY_SUPPRESSED: 234, // 1
  IS_MEGA: 235, // 1
  IS_MAX: 236, // 1
  MOVE_EFF: 237, // 1 (turn_data.move_effectiveness / 4)
  COMPUTED_STATS: 238, // 5 (stats[1..5] / 500)
  MOVES: 243, // 4 × MOVE_BLOCK_DIM = 528 → 771 total
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
  IS_MULTI_HIT: 30,
  SELF_SWITCH: 31,
  FORCE_SWITCH: 32,
  IS_PROTECT: 33,
  TRAPS_TARGET: 34,
  MAKES_CONTACT: 35,
  IS_USABLE: 36,
  STATUS_EFFECT: 37, // 1 (/7)
  STAT_CHANGE_SELF_SUM: 38, // 1 (/12)
  STAT_CHANGE_TARGET_SUM: 39, // 1 (/12)
  RECOIL_RATIO: 40,
  IS_OHKO: 41,
  IS_CHARGING: 42,
  IS_SACRIFICE: 43,
  CRIT_STAGE_BOOST: 44, // 1 (/3)
  TARGET_CLASS: 45, // 3 one-hot [self/ally, single enemy, multi/field]
  IGNORES_PROTECT: 48,
  IS_SOUND_BASED: 49,
  // ── v6 (spaces.ts ~555-605) ──
  CAN_FLINCH: 50,
  CAN_CONFUSE: 51,
  IS_RECHARGE: 52,
  IS_FRENZY: 53,
  IS_TYPELESS: 54,
  CREATES_SUBSTITUTE: 55,
  SUPPRESSES_ABILITY: 56,
  HAS_VARIABLE_POWER: 57,
  HAS_VARIABLE_TYPE: 58,
  HAS_VARIABLE_CATEGORY: 59,
  BYPASS_BURN_PENALTY: 60,
  IGNORES_STAT_STAGES: 61,
  WEATHER_CHANGE: 62, // 1 (WeatherType / 9)
  TERRAIN_CHANGE: 63, // 1 (TerrainType / 4)
  SETS_ARENA_TAG: 64,
  REMOVES_ARENA_TAGS: 65,
  SETS_HAZARD: 66,
  SETS_SCREEN: 67,
  ARENA_TAG_SELF_SIDE: 68,
  APPLIES_BATTLER_TAG: 69,
  APPLIES_MOVE_RESTRICTION: 70,
  APPLIES_CONTINUOUS_DAMAGE: 71,
  IS_USER_HP_DAMAGE: 72,
  IS_TARGET_HALF_HP: 73,
  IS_COUNTER_DAMAGE: 74,
  IS_LEVEL_DAMAGE: 75,
  IS_DELAYED_ATTACK: 76,
  POST_VICTORY_STAT_BOOST: 77,
  IS_WIND_MOVE: 78,
  IS_RECKLESS_MOVE: 79,
  IS_REFLECTABLE: 80,
  HIDES_USER: 81,
  IS_TRIAGE_MOVE: 82,
  CHECK_ALL_HITS: 83,
  AFFECTED_BY_GRAVITY: 84,
  HIDES_TARGET: 85,
  // ── v7 (spaces.ts ~607-673) ──
  STEALS_ITEM: 86,
  REMOVES_ITEM: 87,
  STEALS_BERRY: 88,
  COPIES_STATS: 89,
  INVERTS_STATS: 90,
  RESETS_STATS: 91,
  SWAPS_STAT_STAGES: 92,
  STEALS_STAT_BOOSTS: 93,
  AVERAGES_STATS: 94,
  SWAPS_SINGLE_STAT: 95,
  SHIFTS_OWN_STAT: 96,
  SPLITS_HP: 97,
  REDUCES_PP: 98,
  REVIVES_ALLY: 99,
  COPIES_LAST_MOVE: 100,
  CALLS_RANDOM_MOVE: 101,
  CALLS_MOVESET_MOVE: 102,
  COPIES_MOVE_TEMP: 103,
  COPIES_MOVE_PERM: 104,
  COPIES_ABILITY: 105,
  SWAPS_ABILITIES: 106,
  CHANGES_ABILITY: 107,
  GIVES_ABILITY: 108,
  SUPPRESSES_IF_ACTED: 109,
  BYPASS_REDIRECT: 110,
  FORCES_TARGET_NEXT: 111,
  FORCES_TARGET_LAST: 112,
  HAS_CONDITIONAL_PRIORITY: 113,
  CURES_PARTY_STATUS: 114,
  TRANSFERS_STATUS: 115,
  HEALS_STATUS: 116,
  REMOVES_BATTLER_TAG: 117,
  REMOVES_SUBSTITUTES: 118,
  TRANSFORMS_INTO_TARGET: 119,
  IS_CURSE: 120,
  IS_WISH: 121,
  IS_DESTINY_BOND: 122,
  SWAPS_ARENA_TAGS: 123,
  CLEARS_WEATHER: 124,
  CLEARS_TERRAIN: 125,
  HAS_VARIABLE_TARGET: 126,
  RESISTS_LAST_TYPE: 127,
  HAS_VARIABLE_ACCURACY: 128,
  USES_ALT_STAT: 129,
  OVERRIDES_TYPE_CHART: 130,
  SCATTERS_MONEY: 131, // last move dim
} as const;

// ─── Block bases (encodeObservation, spaces.ts ~1518-1573) ───────────
export const FIELD_BASE = TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM; // 9252
export const BATTLE_BASE = FIELD_BASE + FIELD_STATE_DIM; // 9346
export const MODPHASE_BASE = BATTLE_BASE + BATTLE_META_DIM; // 9386
export const MODINV_BASE = MODPHASE_BASE + MODIFIER_PHASE_DIM; // 9611
export const DERIVED_BASE = MODINV_BASE + MODIFIER_INVENTORY_DIM; // 9831
export const PHASE_BASE = DERIVED_BASE + DERIVED_FIELDS_DIM; // 9859

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
  PLAYER_TERAS_USED: 93, // 1 (/3) — last field dim
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
