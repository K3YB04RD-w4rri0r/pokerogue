/**
 * Observation and action space encoding utilities for the RL environment.
 *
 * Encodes the GameState JSON dict (from state-builder.ts) into a fixed-size
 * Float32Array observation vector, and extracts action masks from it.
 *
 * Layout v9 (6,991 float32) — see src/rl/docs/OBS_V9_LAYOUT.md:
 *   Pokemon block:            513 dims × 12 slots = 6,156
 *   Field block:              102
 *   Battle block:             40
 *   Modifier phase block:     363
 *   Modifier inventory block: 220
 *   Derived fields block:     28
 *   Learn-move block:         66
 *   Phase block:              16
 */

import { ArenaTagSide } from "#enums/arena-tag-side";
import { ArenaTagType } from "#enums/arena-tag-type";
import { BattlerTagType } from "#enums/battler-tag-type";
import { encodeAbilityFeatures } from "#rl/ability-features";
import { encodeModifierFeatures, MODIFIER_FEATURE_DIM, sortByRLPriority } from "#rl/modifier-features";

// ─── Dimension Constants ──────────────────────────────────────────────

/** Number of valid PokemonType values (NORMAL=0 through STELLAR=18) */
export const NUM_POKEMON_TYPES = 19;

/** Number of StatusEffect values (NONE=0 through FAINT=7) */
export const NUM_STATUS_EFFECTS = 8;

/** Number of WeatherType values (NONE=0 through STRONG_WINDS=9) */
export const NUM_WEATHER_TYPES = 10;

/** Number of TerrainType values (NONE=0 through PSYCHIC=4) */
export const NUM_TERRAIN_TYPES = 5;

/** Number of ArenaTagType values excluding NONE */
export const NUM_ARENA_TAG_TYPES = 28;

/** Number of MoveCategory values */
export const NUM_MOVE_CATEGORIES = 3;

/** Number of BattleType values */
export const NUM_BATTLE_TYPES = 4;

/** Number of ModifierTier values */
export const NUM_MODIFIER_TIERS = 6;

/** Number of Pokeball types tracked (POKEBALL through MASTER_BALL) */
export const NUM_POKEBALL_TYPES = 5;

/** Number of semantic features per ability in the ABILITY_FEATURES lookup table */
export const ABILITY_FEATURE_DIM = 40;

/** Max moves per Pokemon */
export const MAX_MOVES = 4;

/** Max party size */
export const MAX_PARTY_SIZE = 6;

/** Max active Pokemon per side */
export const MAX_ACTIVE_PER_SIDE = 2;

/** Number of bench slots per side */
export const MAX_BENCH_SIZE = 4;

/** Total Pokemon slots in observation */
export const TOTAL_POKEMON_SLOTS = 12;

/** Number of reward modifier options */
export const MAX_REWARD_OPTIONS = 3;

/** Number of shop item slots encoded (all 12, natural/action-id order) */
export const MAX_SHOP_OPTIONS_ENCODED = 12;

/** Max shop options in game state */
export const MAX_SHOP_OPTIONS = 12;

// ─── Move Feature Block ───────────────────────────────────────────────

/**
 * v9 compact move vector (60 dims) — evidence keep-list from the obs audit
 * (docs/OBS_V9_LAYOUT.md §1). Exact order:
 *   [0]     valid
 *   [1-19]  type_onehot(19)
 *   [20-22] category_onehot(3)
 *   [23]    power/250  [24] accuracy/100  [25] pp_ratio  [26] priority/7
 *   [27]    effect_chance/100
 *   [28]    drain_ratio  [29] heal_ratio
 *   [30]    multi_hit_count/5 (NEW; 0=not multi-hit — replaces is_multi_hit)
 *   [31]    force_switch  [32] is_protect  [33] traps_target
 *   [34]    makes_contact  [35] is_usable
 *   [36]    status_effect/7
 *   [37]    stat_change_self_sum/12  [38] stat_change_target_sum/12
 *   [39]    recoil_ratio  [40] crit_stage_boost/3
 *   [41-43] target_class_onehot(3)
 *   [44]    ignores_protect  [45] is_sound_based  [46] can_flinch
 *   [47]    can_confuse  [48] has_variable_power  [49] weather_change/9
 *   [50]    sets_arena_tag  [51] applies_battler_tag
 *   [52]    applies_move_restriction  [53] is_wind_move
 *   [54]    is_reckless_move  [55] is_reflectable  [56] is_triage_move
 *   [57]    steals_item  [58] hits_semi_invulnerable
 *   [59]    has_other_effect (NEW catch-all: OR of the 77 cut v8 flags —
 *           exact list in OBS_V9_LAYOUT.md; a rare-effect move stays
 *           distinguishable from a vanilla one without 77 dims/slot)
 * Cut dims remain serialized in game_state — Python ObservationWrappers
 * can re-add any of them without a protocol change.
 */
export const MOVE_BLOCK_DIM = 60;

// ─── Curated Volatile Tags ────────────────────────────────────────────

/** Strategically important volatile status tags, mapped to fixed indices.
 * v9: 7 TURN_END-transient tags cut (FLINCHED/PROTECTED/ENDURING/
 * HELPING_HAND/MAGIC_COAT/POWDER/CENTER_OF_ATTENTION) — they lapse before
 * every decision boundary, so they were structurally unobservable. */
export const CURATED_VOLATILE_TAGS: BattlerTagType[] = [
  BattlerTagType.CONFUSED,
  BattlerTagType.INFATUATED,
  BattlerTagType.SEEDED,
  BattlerTagType.TRAPPED,
  BattlerTagType.ENCORE,
  BattlerTagType.SUBSTITUTE,
  BattlerTagType.DISABLED,
  BattlerTagType.TAUNT,
  BattlerTagType.TORMENT,
  BattlerTagType.HEAL_BLOCK,
  BattlerTagType.INGRAIN,
  BattlerTagType.AQUA_RING,
  BattlerTagType.FLYING,
  BattlerTagType.UNDERGROUND,
  BattlerTagType.UNDERWATER,
  BattlerTagType.CHARGING,
  BattlerTagType.RECHARGING,
  BattlerTagType.FRENZY,
  BattlerTagType.PERISH_SONG,
  BattlerTagType.DESTINY_BOND,
  BattlerTagType.CURSED,
  BattlerTagType.SALT_CURED,
  BattlerTagType.OCTOLOCK,
  BattlerTagType.DROWSY,
  BattlerTagType.STOCKPILING,
  BattlerTagType.MINIMIZED,
  BattlerTagType.IMPRISON,
  // v2: 7 additional strategically important tags
  BattlerTagType.SLOW_START, // Regigigas halved ATK/SPD for 5 turns
  BattlerTagType.UNBURDEN, // Doubled speed after item loss
  BattlerTagType.RECEIVE_DOUBLE_DAMAGE, // Tar Shot - 2x fire damage
  BattlerTagType.FLOATING, // Magnet Rise/Telekinesis - ground immunity
  BattlerTagType.ALWAYS_CRIT, // Laser Focus - guaranteed crit next turn
  // v3: 9 additional tags from audit
  BattlerTagType.GRUDGE, // If holder faints, attacker's move loses all PP
  BattlerTagType.ICE_FACE, // Eiscue form — absorbs one physical hit
  BattlerTagType.DISGUISE, // Mimikyu form — absorbs one hit
  BattlerTagType.NO_RETREAT, // Can't switch but got +1 all stats
  BattlerTagType.THROAT_CHOPPED, // Can't use sound-based moves for 2 turns
  BattlerTagType.SYRUP_BOMB, // -1 Speed per turn for 3 turns
  BattlerTagType.COMMANDED, // Commander ability — merged into ally
  BattlerTagType.BURNED_UP, // Lost Fire type after Burn Up
  BattlerTagType.DOUBLE_SHOCKED, // Lost Electric type after Double Shock
  // v8: 28 additional tags (partial-trap family, charge/crit/boost states,
  // exposure/ignore states, paradox/overlord boosts, misc disables)
  BattlerTagType.BIND, // Partial-trap: damage + no switch
  BattlerTagType.WRAP, // Partial-trap
  BattlerTagType.CLAMP, // Partial-trap
  BattlerTagType.FIRE_SPIN, // Partial-trap
  BattlerTagType.WHIRLPOOL, // Partial-trap
  BattlerTagType.MAGMA_STORM, // Partial-trap
  BattlerTagType.SAND_TOMB, // Partial-trap
  BattlerTagType.SNAP_TRAP, // Partial-trap
  BattlerTagType.THUNDER_CAGE, // Partial-trap
  BattlerTagType.INFESTATION, // Partial-trap
  BattlerTagType.CHARGED, // Electric move charged (2x next)
  BattlerTagType.CRIT_BOOST, // Focus Energy / Dragon Cheer — +crit stage
  BattlerTagType.DRAGON_CHEER, // Ally crit boost (doubles)
  BattlerTagType.FIRE_BOOST, // Charcoal-like fire boost state
  BattlerTagType.GORILLA_TACTICS, // Locked into one move, +ATK
  BattlerTagType.HIDDEN, // Commander/other hidden-from-field state
  BattlerTagType.IGNORE_ACCURACY, // Lock-On/Mind Reader — next move always hits
  BattlerTagType.IGNORE_DARK, // Miracle Eye — hit Dark with Psychic
  BattlerTagType.IGNORE_FLYING, // Smack Down/Roost — grounded
  BattlerTagType.IGNORE_GHOST, // Foresight/Odor Sleuth — hit Ghost
  BattlerTagType.NIGHTMARE, // 1/4 HP loss per turn while asleep
  BattlerTagType.PROTOSYNTHESIS, // Paradox boost (sun/booster)
  BattlerTagType.QUARK_DRIVE, // Paradox boost (electric terrain/booster)
  BattlerTagType.SUPREME_OVERLORD, // ATK/SPATK boost per fainted ally
  BattlerTagType.TAR_SHOT, // Speed drop + fire weakness
  BattlerTagType.TELEKINESIS, // Floating + always-hit
  BattlerTagType.TRUANT, // Loafs every other turn
  BattlerTagType.ALWAYS_GET_HIT, // Cannot avoid the next hit
];

/** Number of curated volatile tag flags */
export const NUM_CURATED_TAGS = CURATED_VOLATILE_TAGS.length; // 69

// ─── Pokemon Block ────────────────────────────────────────────────────

/**
 * v9 non-move fields (273):
 *   valid(1) + hp_ratio(1) + level(1) + base_stats(6) + stat_stages(7) +
 *   type1_onehot(19) + type2_onehot(19) + status_onehot(8) + nature_mults(5) +
 *   ability_features(40) + passive_ability_features(40) + is_tera(1) + tera_type_onehot(19) +
 *   volatile_tags(69) + other_tag_count(1) + is_boss(1) + boss_shield(1) +
 *   is_trapped(1) + is_grounded(1) + weight(1) + catch_rate(1) + is_fainted(1) +
 *   wave_turn_count(1) + damage_taken(1) + acted(1) + toxic_turn_count(1) +
 *   sleep_turns_remaining(1) + held_item_count(1) +
 *   species_id(1) + gender(1) + friendship(1) + move_queue_len(1) +
 *   battle_data_hit_count(1) + ability_suppressed(1) +
 *   is_mega(1) + is_max(1) + move_effectiveness(1) + computed_stats(5) +
 *   ai_type_onehot(3, NEW — zeros on player slots) +
 *   move_known(4) + ability_known(1) + was_seen(1)  (NEW fog indicators —
 *   constant-truthy for enemies under full observability)
 * = 271 − 7 tags + 3 + 6 = 273
 *
 * Move sub-block: 60 × 4 = 240
 *
 * Total: 273 + 240 = 513
 */
export const POKEMON_BLOCK_DIM = 513;

// ─── Field State Block ────────────────────────────────────────────────

/**
 * weather_onehot(10) + weather_turns(1) + terrain_onehot(5) + terrain_turns(1) +
 * player_arena_tags(28) + player_spikes(1) + player_toxic_spikes(1) +
 * enemy_arena_tags(28) + enemy_spikes(1) + enemy_toxic_spikes(1) +
 * is_double(1) + trick_room(1) + gravity(1) +
 * weather_is_permanent(1) + weather_suppressed(1) + terrain_is_permanent(1) +
 * arena_tag_turns(5_tags × 2_sides = 10) + player_teras_used(1)
 * v9 additions (+8): per side [wish_active(1) + wish_turns(1) +
 * future_sight_active(1) + future_sight_turns(1)] from positional_tags
 * = 102
 */
export const FIELD_STATE_DIM = 102;

// ─── Battle Meta Block ────────────────────────────────────────────────

/**
 * wave(1) + turn(1) + battle_type_onehot(4) + money(1) + score(1) +
 * pokeballs(5) + player_alive(1) + enemy_alive(1) + tera_available(1) +
 * can_run(1) + can_catch(1) + player_faints(1) + enemy_faints(1) +
 * command_field_index(1) + biome(1) + escape_attempts(1) +
 * battle_style(1) + time_of_day(1) + lock_modifier_tiers(1) +
 * battle_spec(1) + game_mode(1) + trainer_specialty_type(1) +
 * has_no_shop(1) + seen_enemy_count(1) +
 * is_classic(1) + is_endless(1) + is_daily(1) + is_challenge(1) +
 * has_mystery_encounters(1) + has_short_biomes(1) + has_random_biomes(1) +
 * has_random_bosses(1) + inverse_battle(1)
 * = 40
 */
export const BATTLE_META_DIM = 40;

// ─── Modifier Phase Block ─────────────────────────────────────────────

/**
 * header(3) + reward_options(3 × 28 = 84) + shop_options(12 × 23 = 276) = 363
 * (v9: shop 6 → 12 encoded slots — actions 40-51 all observable now)
 *
 * Header: modifier_active(1) + can_reroll(1) + reroll_cost_ratio(1)
 * Reward per slot: valid(1) + tier_onehot(6) + is_pokemon(1) + modifier_features(20)
 * Shop per slot: valid(1) + cost_ratio(1) + affordable(1) + modifier_features(20)
 */
export const MODIFIER_PHASE_DIM = 363;

// ─── Modifier Inventory Block ────────────────────────────────────────

/** Max held items encoded per active Pokemon slot */
const MAX_HELD_ITEMS_ENCODED = 2;

/** Dims per held item slot: valid(1) + features(20) + stack_ratio(1) = 22 */
const HELD_ITEM_SLOT_DIM = 1 + MODIFIER_FEATURE_DIM + 1; // 22

/**
 * Held items:    4 × (1 + 2 × 22) = 4 × 45 = 180
 * Party mods:    9 (count + 8 boolean presence flags)
 * Lapsing mods:  23 (count + 1 × (valid + features(20) + battles_remaining))
 * Enemy mods:    8 (count + 7 aggregate stack values)
 * Total: 220
 */
export const MODIFIER_INVENTORY_DIM = 220;

// ─── Derived Fields Block ────────────────────────────────────────────

/**
 * Pre-computed derived fields that close the human-agent knowledge gap:
 *   type_effectiveness: 2 players × 4 moves × 2 enemies = 16
 *   stab_indicators:    2 players × 4 moves             = 8
 *   speed_ordering:     4 active slots                   = 4
 * Total: 28
 */
export const DERIVED_FIELDS_DIM = 28;

// ─── Phase Indicator Block ───────────────────────────────────────────

/**
 * v9 learn-move block: the OFFERED move as one compact move vector (its
 * valid dim doubles as offer-active) + learner party-index one-hot(6).
 * All-zero outside the learn_move phase.
 */
export const LEARN_MOVE_BLOCK_DIM = MOVE_BLOCK_DIM + MAX_PARTY_SIZE; // 66

/** 16-dim one-hot over DecisionPhase enum values */
export const PHASE_INDICATOR_DIM = 16;

/** Mapping from DecisionPhase string values to one-hot indices */
export const PHASE_INDEX_MAP: Record<string, number> = {
  command: 0,
  target: 1,
  modifier: 2,
  modifier_target: 3,
  switch: 4,
  check_switch: 5,
  learn_move: 6,
  evolution: 7,
  starter: 8,
  mystery: 9,
  game_over: 10,
  select_biome: 11,
  revival_blessing: 12,
  form_change: 13,
  title: 14,
  select_gender: 15,
};

/** Total observation vector size */
export const OBSERVATION_DIM =
  TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM
  + FIELD_STATE_DIM
  + BATTLE_META_DIM
  + MODIFIER_PHASE_DIM
  + MODIFIER_INVENTORY_DIM
  + DERIVED_FIELDS_DIM
  + LEARN_MOVE_BLOCK_DIM
  + PHASE_INDICATOR_DIM; // 12*513 + 102 + 40 + 363 + 220 + 28 + 66 + 16 = 6991

// ─── Action Space ─────────────────────────────────────────────────────

/** Total number of discrete actions */
export const ACTION_SPACE_SIZE = 58;

// Battle action index ranges
export const ACTION_FIGHT_ENEMY_START = 0; // 0-3: move 0-3 targeting ENEMY
export const ACTION_FIGHT_ENEMY2_START = 4; // 4-7: move 0-3 targeting ENEMY_2
export const ACTION_FIGHT_ALLY_START = 8; // 8-11: move 0-3 targeting ally
export const ACTION_SWITCH_START = 12; // 12-16: switch to party slot 1-5
export const ACTION_BALL_START = 17; // 17-21: throw ball type 0-4
export const ACTION_RUN = 22;
export const ACTION_TERA_ENEMY_START = 23; // 23-26: tera + move 0-3 targeting ENEMY
export const ACTION_TERA_ENEMY2_START = 27; // 27-30: tera + move 0-3 targeting ENEMY_2
export const ACTION_TERA_ALLY_START = 31; // 31-34: tera + move 0-3 targeting ally

// Modifier action index ranges
export const ACTION_SELECT_REWARD_START = 35; // 35-37: select reward 0-2
export const ACTION_REROLL = 38;
export const ACTION_SKIP = 39;
export const ACTION_BUY_SHOP_START = 40; // 40-51: buy shop item 0-11
export const ACTION_PARTY_TARGET_START = 52; // 52-57: apply modifier to party 0-5

// ─── Arena Tag Type Ordering ──────────────────────────────────────────

/** Fixed ordering of ArenaTagType values (excluding NONE) for binary vector encoding */
export const ARENA_TAG_ORDER: ArenaTagType[] = [
  ArenaTagType.MUD_SPORT,
  ArenaTagType.WATER_SPORT,
  ArenaTagType.SPIKES,
  ArenaTagType.TOXIC_SPIKES,
  ArenaTagType.MIST,
  ArenaTagType.STEALTH_ROCK,
  ArenaTagType.STICKY_WEB,
  ArenaTagType.TRICK_ROOM,
  ArenaTagType.GRAVITY,
  ArenaTagType.REFLECT,
  ArenaTagType.LIGHT_SCREEN,
  ArenaTagType.AURORA_VEIL,
  ArenaTagType.QUICK_GUARD,
  ArenaTagType.WIDE_GUARD,
  ArenaTagType.MAT_BLOCK,
  ArenaTagType.CRAFTY_SHIELD,
  ArenaTagType.TAILWIND,
  ArenaTagType.HAPPY_HOUR,
  ArenaTagType.SAFEGUARD,
  ArenaTagType.NO_CRIT,
  ArenaTagType.IMPRISON,
  ArenaTagType.ION_DELUGE,
  ArenaTagType.FIRE_GRASS_PLEDGE,
  ArenaTagType.WATER_FIRE_PLEDGE,
  ArenaTagType.GRASS_WATER_PLEDGE,
  ArenaTagType.FAIRY_LOCK,
  ArenaTagType.NEUTRALIZING_GAS,
  ArenaTagType.PENDING_HEAL,
];

const ARENA_TAG_INDEX_MAP = new Map<ArenaTagType, number>();
for (let i = 0; i < ARENA_TAG_ORDER.length; i++) {
  ARENA_TAG_INDEX_MAP.set(ARENA_TAG_ORDER[i], i);
}

// ─── Encoding Helpers ─────────────────────────────────────────────────

/** Write a one-hot encoding into the buffer at the given offset */
function writeOneHot(buf: Float32Array, offset: number, size: number, index: number): void {
  if (index >= 0 && index < size) {
    buf[offset + index] = 1.0;
  }
}

/** Clamp a value to [min, max] */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ─── Dict-based encoding helpers ─────────────────────────────────────

/** Safely read a number from a dict, returning fallback if missing/NaN */
function num(dict: Record<string, unknown>, key: string, fallback = 0): number {
  const v = dict[key];
  if (typeof v === "number" && !Number.isNaN(v)) {
    return v;
  }
  return fallback;
}

/** Safely read a boolean from a dict */
function bool(dict: Record<string, unknown>, key: string, fallback = false): boolean {
  const v = dict[key];
  if (typeof v === "boolean") {
    return v;
  }
  return fallback;
}

/** Safely read an array from a dict */
function arr(dict: Record<string, unknown>, key: string): unknown[] {
  const v = dict[key];
  return Array.isArray(v) ? v : [];
}

/** Safely read a sub-dict from a dict */
function sub(dict: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = dict[key];
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

// ─── Move Encoding (from dict) ───────────────────────────────────────

// ─── Move Target Classification ─────────────────────────────────────

/** Self/ally MoveTarget values: USER=0, NEAR_ALLY=10, ALLY=11, USER_OR_NEAR_ALLY=12, USER_AND_ALLIES=13, USER_SIDE=15, PARTY=18 */
const SELF_ALLY_TARGETS = new Set([0, 10, 11, 12, 13, 15, 18]);
/** Single-enemy MoveTarget values: OTHER=1, NEAR_OTHER=3, NEAR_ENEMY=5, ATTACKER=9 */
const SINGLE_ENEMY_TARGETS = new Set([1, 3, 5, 9]);
// Everything else is multi-target/field: ALL_OTHERS=2, ALL_NEAR_OTHERS=4, ALL_NEAR_ENEMIES=6,
// RANDOM_NEAR_ENEMY=7, ALL_ENEMIES=8, ALL=14, ENEMY_SIDE=16, BOTH_SIDES=17, CURSE=19

/**
 * v8 boolean flags folded into the v9 has_other_effect catch-all
 * (76 booleans; terrain_change — a scalar in v8 — is OR'd separately).
 * Exact provenance: OBS_V9_LAYOUT.md §1. Order irrelevant (pure OR).
 */
const OTHER_EFFECT_FLAGS: string[] = [
  // base-section cuts (4)
  "self_switch",
  "is_ohko",
  "is_charging",
  "is_sacrifice",
  // v6 tail cuts (24 booleans; terrain_change handled as scalar)
  "is_recharge",
  "is_frenzy",
  "is_typeless",
  "creates_substitute",
  "suppresses_ability",
  "has_variable_type",
  "has_variable_category",
  "bypass_burn_penalty",
  "ignores_stat_stages",
  "removes_arena_tags",
  "sets_hazard",
  "sets_screen",
  "arena_tag_self_side",
  "applies_continuous_damage",
  "is_user_hp_damage",
  "is_target_half_hp",
  "is_counter_damage",
  "is_level_damage",
  "is_delayed_attack",
  "post_victory_stat_boost",
  "hides_user",
  "hides_target",
  "check_all_hits",
  "affected_by_gravity",
  // v7 tail cuts (45)
  "removes_item",
  "steals_berry",
  "copies_stats",
  "inverts_stats",
  "resets_stats",
  "swaps_stat_stages",
  "steals_stat_boosts",
  "averages_stats",
  "swaps_single_stat",
  "shifts_own_stat",
  "splits_hp",
  "reduces_pp",
  "revives_ally",
  "copies_last_move",
  "calls_random_move",
  "calls_moveset_move",
  "copies_move_temp",
  "copies_move_perm",
  "copies_ability",
  "swaps_abilities",
  "changes_ability",
  "gives_ability",
  "suppresses_if_acted",
  "bypass_redirect",
  "forces_target_next",
  "forces_target_last",
  "has_conditional_priority",
  "cures_party_status",
  "transfers_status",
  "heals_status",
  "removes_battler_tag",
  "removes_substitutes",
  "transforms_into_target",
  "is_curse",
  "is_wish",
  "is_destiny_bond",
  "swaps_arena_tags",
  "clears_weather",
  "clears_terrain",
  "has_variable_target",
  "resists_last_type",
  "has_variable_accuracy",
  "uses_alt_stat",
  "overrides_type_chart",
  "scatters_money",
  // v8 tail cuts (3)
  "survives_at_1hp",
  "matches_user_hp",
  "hp_cost_stat_boost",
];

/**
 * Map the serialized multi_hit_type enum to a hit count.
 * -1 = not multi-hit → 0; TWO(0) → 2; TWO_TO_FIVE(1) → 5; THREE(2) → 3;
 * TEN(3) and BEAT_UP(4) → clamp to 5. Encoded as count/5 ∈ {0,0.4,0.6,1}.
 */
function multiHitCount(multiHitType: number): number {
  switch (multiHitType) {
    case 0:
      return 2;
    case 1:
      return 5;
    case 2:
      return 3;
    case 3:
    case 4:
      return 5;
    default:
      return 0;
  }
}

/**
 * Encode a single move slot from a dict into the buffer (v9 compact
 * vector — order documented at MOVE_BLOCK_DIM and OBS_V9_LAYOUT.md §1).
 * @returns number of floats written (always MOVE_BLOCK_DIM = 60)
 */
function encodeMoveFromDict(buf: Float32Array, offset: number, moveDict: Record<string, unknown>): number {
  const moveId = num(moveDict, "move_id");
  if (moveId <= 0) {
    return MOVE_BLOCK_DIM;
  }

  let pos = offset;

  // [0] valid
  buf[pos++] = 1.0;

  // [1-19] type one-hot(19)
  writeOneHot(buf, pos, NUM_POKEMON_TYPES, num(moveDict, "type", -1));
  pos += NUM_POKEMON_TYPES;

  // [20-22] category one-hot(3)
  writeOneHot(buf, pos, NUM_MOVE_CATEGORIES, num(moveDict, "category", -1));
  pos += NUM_MOVE_CATEGORIES;

  // [23] power /250
  buf[pos++] = clamp(num(moveDict, "power") / 250, 0, 1);

  // [24] accuracy (-1 or 0 means always hits)
  const accuracy = num(moveDict, "accuracy");
  buf[pos++] = accuracy <= 0 ? 1.0 : clamp(accuracy / 100, 0, 1);

  // [25] pp_ratio
  const ppMax = num(moveDict, "pp_max", 1);
  const ppRemaining = num(moveDict, "pp_remaining");
  buf[pos++] = ppMax > 0 ? clamp(ppRemaining / ppMax, 0, 1) : 0;

  // [26] priority /7
  buf[pos++] = clamp(num(moveDict, "priority") / 7, -1, 1);

  // [27] effect_chance /100
  buf[pos++] = clamp(num(moveDict, "effect_chance") / 100, 0, 1);

  // [28] drain_ratio, [29] heal_ratio
  buf[pos++] = clamp(num(moveDict, "drain_ratio"), 0, 1);
  buf[pos++] = clamp(num(moveDict, "heal_ratio"), 0, 1);

  // [30] multi_hit_count /5 (0 = not multi-hit)
  buf[pos++] = multiHitCount(num(moveDict, "multi_hit_type", -1)) / 5;

  // [31-35] force_switch, is_protect, traps_target, makes_contact, is_usable
  buf[pos++] = bool(moveDict, "force_switch") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_protect") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "traps_target") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "makes_contact") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_usable") ? 1.0 : 0.0;

  // [36] status_effect /7 (NONE=0 through FAINT=7)
  buf[pos++] = clamp(num(moveDict, "status_effect") / 7, 0, 1);

  // [37-38] stat_change_self_sum /12 and stat_change_target_sum /12
  const statChanges = arr(moveDict, "stat_changes");
  let selfSum = 0;
  let targetSum = 0;
  for (const sc of statChanges) {
    if (sc && typeof sc === "object") {
      const entry = sc as Record<string, unknown>;
      const stages = num(entry, "stages");
      if (bool(entry, "self_target")) {
        selfSum += stages;
      } else {
        targetSum += stages;
      }
    }
  }
  buf[pos++] = clamp(selfSum / 12, -1, 1);
  buf[pos++] = clamp(targetSum / 12, -1, 1);

  // [39] recoil_ratio
  buf[pos++] = clamp(num(moveDict, "recoil_ratio"), 0, 1);

  // [40] crit_stage_boost /3 (clamped; 99=always_crit maps to 1.0)
  buf[pos++] = clamp(num(moveDict, "crit_stage_boost") / 3, 0, 1);

  // move_target_class (3-dim one-hot: [self_or_ally, single_enemy, multi_target_or_field])
  const moveTarget = num(moveDict, "target", -1);
  if (SELF_ALLY_TARGETS.has(moveTarget)) {
    buf[pos] = 1.0;
  } else if (SINGLE_ENEMY_TARGETS.has(moveTarget)) {
    buf[pos + 1] = 1.0;
  } else if (moveTarget >= 0) {
    buf[pos + 2] = 1.0;
  }
  pos += 3;

  // [44-58] kept effect flags (evidence keep-list)
  buf[pos++] = bool(moveDict, "ignores_protect") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_sound_based") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "can_flinch") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "can_confuse") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "has_variable_power") ? 1.0 : 0.0;
  buf[pos++] = clamp(num(moveDict, "weather_change") / 9, 0, 1);
  buf[pos++] = bool(moveDict, "sets_arena_tag") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "applies_battler_tag") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "applies_move_restriction") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_wind_move") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_reckless_move") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_reflectable") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "is_triage_move") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "steals_item") ? 1.0 : 0.0;
  buf[pos++] = bool(moveDict, "hits_semi_invulnerable") ? 1.0 : 0.0;

  // [59] has_other_effect — OR of the 77 cut v8 flags (list above; the
  // terrain_change scalar counts as "other" when nonzero)
  let hasOther = num(moveDict, "terrain_change") !== 0;
  if (!hasOther) {
    for (const flag of OTHER_EFFECT_FLAGS) {
      if (bool(moveDict, flag)) {
        hasOther = true;
        break;
      }
    }
  }
  buf[pos++] = hasOther ? 1.0 : 0.0;

  return MOVE_BLOCK_DIM;
}

// ─── Pokemon Encoding (from dict) ────────────────────────────────────

/**
 * Encode a single Pokemon from a dict into the observation buffer.
 *
 * @param isEnemy - Whether this slot belongs to the enemy side; drives
 *  ai_type + the revealed-indicator dims and (under fog) masking.
 * @param fogOfWar - When true, enemy private information is masked to
 *  what a human player could know: unseen moves, unrevealed abilities,
 *  IV/nature-derived values, and never-seen bench members (see
 *  docs/OBS_V9_LAYOUT.md §5). Player slots are never masked.
 * @returns number of floats written (always POKEMON_BLOCK_DIM = 513)
 */
function encodePokemonFromDict(
  buf: Float32Array,
  offset: number,
  poke: Record<string, unknown>,
  isEnemy = false,
  fogOfWar = false,
): number {
  if (!bool(poke, "valid")) {
    return POKEMON_BLOCK_DIM;
  }

  const fogged = fogOfWar && isEnemy;
  // Never-seen enemy bench member: the whole block stays zero (including
  // valid and was_seen) — indistinguishable from an empty slot, exactly
  // like for a human player. Slot existence remains inferable from
  // battle.enemy_alive / seen_enemy_count.
  if (fogged && !bool(poke, "was_seen")) {
    return POKEMON_BLOCK_DIM;
  }
  const abilityKnown = !fogged || bool(poke, "ability_revealed");
  const moveKnownRaw = arr(poke, "move_known");

  let pos = offset;

  // valid
  buf[pos++] = 1.0;

  // hp_ratio
  buf[pos++] = clamp(num(poke, "hp_ratio"), 0, 1);

  // level /100
  buf[pos++] = clamp(num(poke, "level") / 100, 0, 1);

  // base_stats(6) /255
  const baseStats = arr(poke, "base_stats");
  for (let i = 0; i < 6; i++) {
    buf[pos++] = clamp((Number(baseStats[i]) || 0) / 255, 0, 1);
  }

  // stat_stages(7) /6
  const statStages = arr(poke, "stat_stages");
  for (let i = 0; i < 7; i++) {
    buf[pos++] = (Number(statStages[i]) || 0) / 6;
  }

  // type1 one-hot(19)
  const types = arr(poke, "types");
  writeOneHot(buf, pos, NUM_POKEMON_TYPES, Number(types[0]) ?? -1);
  pos += NUM_POKEMON_TYPES;

  // type2 one-hot(19)
  writeOneHot(buf, pos, NUM_POKEMON_TYPES, types.length > 1 ? (Number(types[1]) ?? -1) : -1);
  pos += NUM_POKEMON_TYPES;

  // status one-hot(8)
  writeOneHot(buf, pos, NUM_STATUS_EFFECTS, num(poke, "status_effect"));
  pos += NUM_STATUS_EFFECTS;

  // nature_mults(5) — fog: zeroed for enemies (IV/nature-derived)
  if (fogged) {
    pos += 5;
  } else {
    const natureMults = arr(poke, "nature_multipliers");
    for (let i = 0; i < 5; i++) {
      buf[pos++] = Number(natureMults[i]) || 1.0;
    }
  }

  // ability features (40 dims) + passive ability features (40 dims) = 80
  // fog: zeroed until the ability has revealed itself in battle
  const abilityId = num(poke, "ability_id");
  const passiveId = num(poke, "passive_ability_id");
  const suppressed = bool(poke, "ability_suppressed");
  if (suppressed || !abilityKnown) {
    // All zeros for both ability and passive when suppressed or unknown
    pos += ABILITY_FEATURE_DIM * 2;
  } else {
    pos = encodeAbilityFeatures(abilityId, buf, pos);
    pos = encodeAbilityFeatures(passiveId, buf, pos);
  }

  // is_terastallized
  buf[pos++] = bool(poke, "is_terastallized") ? 1.0 : 0.0;

  // tera_type one-hot(19)
  writeOneHot(buf, pos, NUM_POKEMON_TYPES, num(poke, "tera_type", -1));
  pos += NUM_POKEMON_TYPES;

  // volatile_tags(39 curated) + other_tag_count(1)
  const volatileTags = arr(poke, "volatile_tags");
  const activeTagTypes = new Set<string>();
  for (const tag of volatileTags) {
    if (tag && typeof tag === "object") {
      const tagType = (tag as Record<string, unknown>).tag_type;
      if (typeof tagType === "string") {
        activeTagTypes.add(tagType);
      }
    }
  }
  let curatedMatches = 0;
  for (let i = 0; i < CURATED_VOLATILE_TAGS.length; i++) {
    if (activeTagTypes.has(CURATED_VOLATILE_TAGS[i])) {
      buf[pos + i] = 1.0;
      curatedMatches++;
    }
  }
  pos += NUM_CURATED_TAGS;

  // other_tag_count
  const otherCount = volatileTags.length - curatedMatches;
  buf[pos++] = clamp(otherCount / 10, 0, 1);

  // is_boss
  const isBoss = bool(poke, "is_boss");
  buf[pos++] = isBoss ? 1.0 : 0.0;

  // boss_shield_ratio
  if (isBoss) {
    const segments = num(poke, "boss_segments", 1);
    const segmentIndex = num(poke, "boss_segment_index");
    buf[pos++] = segments > 0 ? segmentIndex / segments : 0;
  } else {
    pos++;
  }

  // ── New fields (11 dims) ──

  // is_trapped
  buf[pos++] = bool(poke, "is_trapped") ? 1.0 : 0.0;

  // is_grounded
  buf[pos++] = bool(poke, "is_grounded") ? 1.0 : 0.0;

  // weight /1000
  buf[pos++] = clamp(num(poke, "weight") / 1000, 0, 1);

  // catch_rate /255
  buf[pos++] = clamp(num(poke, "catch_rate") / 255, 0, 1);

  // is_fainted
  buf[pos++] = bool(poke, "is_fainted") ? 1.0 : 0.0;

  // wave_turn_count /20
  buf[pos++] = clamp(num(poke, "wave_turn_count") / 20, 0, 1);

  // damage_taken (from turn_data) normalized by max_hp
  const turnData = sub(poke, "turn_data");
  const maxHp = num(poke, "max_hp", 1);
  buf[pos++] = maxHp > 0 ? clamp(num(turnData, "damage_taken") / maxHp, 0, 1) : 0;

  // acted (from turn_data)
  buf[pos++] = bool(turnData, "acted") ? 1.0 : 0.0;

  // toxic_turn_count /16
  buf[pos++] = clamp(num(poke, "toxic_turn_count") / 16, 0, 1);

  // sleep_turns_remaining /4
  buf[pos++] = clamp(num(poke, "sleep_turns_remaining") / 4, 0, 1);

  // held_item_count /10
  const heldItems = arr(poke, "held_items");
  buf[pos++] = clamp(heldItems.length / 10, 0, 1);

  // ── Additional Pokemon fields (+11 dims) ──

  // species_id /1025
  buf[pos++] = clamp(num(poke, "species_id") / 1025, 0, 1);

  // gender: -1 (genderless) → 0, 0 (male) → 0.5, 1 (female) → 1.0
  const gender = num(poke, "gender", -1);
  buf[pos++] = gender < 0 ? 0.0 : gender === 0 ? 0.5 : 1.0;

  // friendship /255
  buf[pos++] = clamp(num(poke, "friendship") / 255, 0, 1);

  // move_queue length /2
  const moveQueue = arr(poke, "move_queue");
  buf[pos++] = clamp(moveQueue.length / 2, 0, 1);

  // battle_data.hit_count /10 (Rage Fist scaling)
  const battleData = sub(poke, "battle_data");
  buf[pos++] = clamp(num(battleData, "hit_count") / 10, 0, 1);

  // ability_suppressed
  buf[pos++] = bool(poke, "ability_suppressed") ? 1.0 : 0.0;

  // is_mega
  buf[pos++] = bool(poke, "is_mega") ? 1.0 : 0.0;

  // is_max (Gigantamax/Eternamax)
  buf[pos++] = bool(poke, "is_max") ? 1.0 : 0.0;

  // move_effectiveness from turn_data (type effectiveness of last hit received, 0-4 scale /4)
  buf[pos++] = clamp(num(turnData, "move_effectiveness") / 4, 0, 1);

  // computed_stats: ATK/DEF/SPATK/SPDEF/SPD (indices 1-5) /500
  // fog: zeroed for enemies (exact stats are IV/nature-derived)
  if (fogged) {
    pos += 5;
  } else {
    const computedStats = arr(poke, "stats");
    for (let i = 1; i <= 5; i++) {
      buf[pos++] = clamp((Number(computedStats[i]) || 0) / 500, 0, 1);
    }
  }

  const moves = arr(poke, "moves");
  const moveExists = (i: number): boolean => {
    const m = moves[i];
    return !!m && typeof m === "object" && !Array.isArray(m) && num(m as Record<string, unknown>, "move_id") > 0;
  };
  // move j is "known" when full-info (exists = known), or when fog says
  // it has been seen in this battle (move_history-derived, state-builder)
  const moveKnown = (i: number): boolean => moveExists(i) && (!fogged || moveKnownRaw[i] === true);

  // ── v9 additions (9 dims) ──

  // ai_type one-hot(3): RANDOM/SMART_RANDOM/SMART — all-zero on player
  // slots (players have no AI; writeOneHot of -1 writes nothing)
  writeOneHot(buf, pos, 3, isEnemy ? num(poke, "ai_type", -1) : -1);
  pos += 3;

  // move_known(4), ability_known(1), was_seen(1) — revealed-indicators.
  // Constant-truthy for enemies under full observability; all-zero on
  // player slots (not applicable); live values under fog.
  for (let i = 0; i < MAX_MOVES; i++) {
    buf[pos++] = isEnemy && moveKnown(i) ? 1.0 : 0.0;
  }
  buf[pos++] = isEnemy && abilityKnown ? 1.0 : 0.0;
  buf[pos++] = isEnemy ? 1.0 : 0.0; // was_seen (fog never-seen returned early)

  // moves (4 slots × MOVE_BLOCK_DIM) — fog: unseen enemy moves stay zero
  for (let i = 0; i < MAX_MOVES; i++) {
    const moveDict =
      moves[i] && typeof moves[i] === "object" && !Array.isArray(moves[i]) ? (moves[i] as Record<string, unknown>) : {};
    if (!fogged || moveKnown(i)) {
      encodeMoveFromDict(buf, pos, moveDict);
    }
    pos += MOVE_BLOCK_DIM;
  }

  return POKEMON_BLOCK_DIM;
}

// ─── Field State Encoding (from dict) ────────────────────────────────

function encodeFieldFromDict(buf: Float32Array, offset: number, field: Record<string, unknown>): number {
  let pos = offset;

  // weather one-hot(10)
  writeOneHot(buf, pos, NUM_WEATHER_TYPES, num(field, "weather_type"));
  pos += NUM_WEATHER_TYPES;

  // weather_turns_left /8
  buf[pos++] = clamp(num(field, "weather_turns_left") / 8, 0, 1);

  // terrain one-hot(5)
  writeOneHot(buf, pos, NUM_TERRAIN_TYPES, num(field, "terrain_type"));
  pos += NUM_TERRAIN_TYPES;

  // terrain_turns_left /8
  buf[pos++] = clamp(num(field, "terrain_turns_left") / 8, 0, 1);

  // Build arena tag presence maps from the arena_tags array
  const arenaTags = arr(field, "arena_tags");
  const playerTagPresence = new Map<string, boolean>();
  const enemyTagPresence = new Map<string, boolean>();
  const playerSpikeLayers = num(field, "player_spikes_layers");
  const playerToxicSpikeLayers = num(field, "player_toxic_spikes_layers");
  const enemySpikeLayers = num(field, "enemy_spikes_layers");
  const enemyToxicSpikeLayers = num(field, "enemy_toxic_spikes_layers");

  for (const tagObj of arenaTags) {
    if (!tagObj || typeof tagObj !== "object") {
      continue;
    }
    const tag = tagObj as Record<string, unknown>;
    const tagType = tag.tag_type as string;
    const side = num(tag, "side");
    const isPlayerSide = side === ArenaTagSide.PLAYER || side === ArenaTagSide.BOTH;
    const isEnemySide = side === ArenaTagSide.ENEMY || side === ArenaTagSide.BOTH;
    if (isPlayerSide) {
      playerTagPresence.set(tagType, true);
    }
    if (isEnemySide) {
      enemyTagPresence.set(tagType, true);
    }
  }

  // player_arena_tags(28) binary
  const playerTagStart = pos;
  for (let i = 0; i < ARENA_TAG_ORDER.length; i++) {
    if (playerTagPresence.has(ARENA_TAG_ORDER[i])) {
      buf[playerTagStart + i] = 1.0;
    }
  }
  pos += NUM_ARENA_TAG_TYPES;

  // player_spikes /3
  buf[pos++] = playerSpikeLayers / 3;
  // player_toxic_spikes /2
  buf[pos++] = playerToxicSpikeLayers / 2;

  // enemy_arena_tags(28) binary
  const enemyTagStart = pos;
  for (let i = 0; i < ARENA_TAG_ORDER.length; i++) {
    if (enemyTagPresence.has(ARENA_TAG_ORDER[i])) {
      buf[enemyTagStart + i] = 1.0;
    }
  }
  pos += NUM_ARENA_TAG_TYPES;

  // enemy_spikes /3
  buf[pos++] = enemySpikeLayers / 3;
  // enemy_toxic_spikes /2
  buf[pos++] = enemyToxicSpikeLayers / 2;

  // is_double_battle
  buf[pos++] = bool(field, "is_double_battle") ? 1.0 : 0.0;

  // trick_room_active
  buf[pos++] = bool(field, "trick_room_active") ? 1.0 : 0.0;

  // gravity_active
  buf[pos++] = bool(field, "gravity_active") ? 1.0 : 0.0;

  // ── New fields (3 dims) ──

  // weather_is_permanent
  buf[pos++] = bool(field, "weather_is_permanent") ? 1.0 : 0.0;

  // weather_suppressed
  buf[pos++] = bool(field, "weather_suppressed") ? 1.0 : 0.0;

  // terrain_is_permanent
  buf[pos++] = bool(field, "terrain_is_permanent") ? 1.0 : 0.0;

  // ── Arena tag remaining turns (10 dims: 5 tags × 2 sides) ──
  // Key defensive/support tags whose remaining duration matters for strategy.
  // For each tag, look up turn_count from the arena_tags array by tag_type + side.
  const KEY_ARENA_TAGS = [
    ArenaTagType.REFLECT,
    ArenaTagType.LIGHT_SCREEN,
    ArenaTagType.AURORA_VEIL,
    ArenaTagType.TAILWIND,
    ArenaTagType.TRICK_ROOM,
  ];

  // Build a map: "tagType:side" → turn_count for efficient lookup
  const tagTurnMap = new Map<string, number>();
  for (const tagObj of arenaTags) {
    if (!tagObj || typeof tagObj !== "object") {
      continue;
    }
    const tag = tagObj as Record<string, unknown>;
    const tagType = tag.tag_type as string;
    const side = num(tag, "side");
    const turnCount = num(tag, "turn_count");
    // Store for each effective side
    if (side === ArenaTagSide.PLAYER || side === ArenaTagSide.BOTH) {
      tagTurnMap.set(`${tagType}:${ArenaTagSide.PLAYER}`, turnCount);
    }
    if (side === ArenaTagSide.ENEMY || side === ArenaTagSide.BOTH) {
      tagTurnMap.set(`${tagType}:${ArenaTagSide.ENEMY}`, turnCount);
    }
  }

  // Player side (5 dims), then enemy side (5 dims)
  for (const side of [ArenaTagSide.PLAYER, ArenaTagSide.ENEMY]) {
    for (const tagType of KEY_ARENA_TAGS) {
      const turns = tagTurnMap.get(`${tagType}:${side}`) ?? 0;
      buf[pos++] = clamp(turns / 8, 0, 1);
    }
  }

  // player_teras_used /3
  buf[pos++] = clamp(num(field, "player_teras_used") / 3, 0, 1);

  // ── v9: positional tags (+8) — Wish / Future Sight per side ──
  // positional_tags entries carry tag_type, countdown and target_index
  // (BattlerIndex 0-1 = player side, 2-3 = enemy side). Multiple pending
  // on a side: active=1, turns = min countdown.
  const positionalTags = arr(field, "positional_tags");
  // [wishActive, wishTurns, fsActive, fsTurns] × [player, enemy]
  const posAgg = [
    [0, Number.POSITIVE_INFINITY, 0, Number.POSITIVE_INFINITY],
    [0, Number.POSITIVE_INFINITY, 0, Number.POSITIVE_INFINITY],
  ];
  for (const tag of positionalTags) {
    if (!tag || typeof tag !== "object") {
      continue;
    }
    const entry = tag as Record<string, unknown>;
    const tagType = String(entry.tag_type ?? "");
    const sideIdx = num(entry, "target_index", 0) >= 2 ? 1 : 0;
    const countdown = num(entry, "countdown");
    // PositionalTagType enum: WISH covers Wish; DELAYED_ATTACK covers
    // Future Sight / Doom Desire
    const isWish = tagType === "WISH";
    const isFutureSight = tagType === "DELAYED_ATTACK";
    if (isWish) {
      posAgg[sideIdx][0] = 1;
      posAgg[sideIdx][1] = Math.min(posAgg[sideIdx][1], countdown);
    } else if (isFutureSight) {
      posAgg[sideIdx][2] = 1;
      posAgg[sideIdx][3] = Math.min(posAgg[sideIdx][3], countdown);
    }
  }
  for (const agg of posAgg) {
    buf[pos++] = agg[0];
    buf[pos++] = agg[0] ? clamp(agg[1] / 8, 0, 1) : 0;
    buf[pos++] = agg[2];
    buf[pos++] = agg[2] ? clamp(agg[3] / 8, 0, 1) : 0;
  }

  return FIELD_STATE_DIM;
}

// ─── Battle Meta Encoding (from dict) ─────────────────────────────────

function encodeBattleFromDict(buf: Float32Array, offset: number, battle: Record<string, unknown>): number {
  let pos = offset;

  // wave /200
  buf[pos++] = clamp(num(battle, "wave_index") / 200, 0, 1);

  // turn /50
  buf[pos++] = clamp(num(battle, "turn") / 50, 0, 1);

  // battle_type one-hot(4)
  writeOneHot(buf, pos, NUM_BATTLE_TYPES, num(battle, "battle_type"));
  pos += NUM_BATTLE_TYPES;

  // money (log-normalized)
  buf[pos++] = Math.log(1 + Math.max(0, num(battle, "money"))) / Math.log(100001);

  // score (log-normalized)
  buf[pos++] = Math.log(1 + Math.max(0, num(battle, "score"))) / Math.log(100001);

  // pokeball counts (5 types)
  const pokeballCounts = sub(battle, "pokeball_counts");
  const ballKeys = ["pokeball", "great_ball", "ultra_ball", "rogue_ball", "master_ball"];
  for (let i = 0; i < NUM_POKEBALL_TYPES; i++) {
    buf[pos++] = clamp(num(pokeballCounts, ballKeys[i]) / 99, 0, 1);
  }

  // player_alive /6
  buf[pos++] = clamp(num(battle, "player_alive_count") / 6, 0, 1);

  // enemy_alive /6
  buf[pos++] = clamp(num(battle, "enemy_alive_count") / 6, 0, 1);

  // tera_available
  buf[pos++] = bool(battle, "tera_available") ? 1.0 : 0.0;

  // can_run
  buf[pos++] = bool(battle, "can_run") ? 1.0 : 0.0;

  // can_catch
  buf[pos++] = bool(battle, "can_catch") ? 1.0 : 0.0;

  // player_faints /6
  buf[pos++] = clamp(num(battle, "player_faints_battle") / 6, 0, 1);

  // enemy_faints /6
  buf[pos++] = clamp(num(battle, "enemy_faints_battle") / 6, 0, 1);

  // command_field_index: from phase info
  // This is encoded by the caller — we use -1 here (passed externally)
  // We'll read it from the phase section's command_field_index
  // For now encode as 0 (will be overwritten by phase-aware section below)
  pos++; // placeholder — filled in encodeObservation

  // biome /40
  buf[pos++] = clamp(num(battle, "biome_id", num(battle, "biome_type")) / 40, 0, 1);

  // escape_attempts /10
  buf[pos++] = clamp(num(battle, "escape_attempts") / 10, 0, 1);

  // ── New fields (3 dims) ──

  // battle_style /3
  buf[pos++] = clamp(num(battle, "battle_style") / 3, 0, 1);

  // time_of_day /3
  buf[pos++] = clamp(num(battle, "time_of_day") / 3, 0, 1);

  // lock_modifier_tiers
  buf[pos++] = bool(battle, "lock_modifier_tiers") ? 1.0 : 0.0;

  // ── Additional battle fields (+5 dims) ──

  // battle_spec /1 (0=DEFAULT, 1=FINAL_BOSS)
  buf[pos++] = clamp(num(battle, "battle_spec"), 0, 1);

  // game_mode /4 (CLASSIC=0, ENDLESS=1, etc.)
  buf[pos++] = clamp(num(battle, "game_mode") / 4, 0, 1);

  // trainer_specialty_type /19 (from trainer sub-dict, -1 or null → 0)
  const trainer = sub(battle, "trainer");
  const specialtyType = num(trainer, "specialty_type", -1);
  buf[pos++] = specialtyType >= 0 ? clamp(specialtyType / 19, 0, 1) : 0;

  // has_no_shop
  buf[pos++] = bool(battle, "has_no_shop") ? 1.0 : 0.0;

  // seen_enemy_count /6
  buf[pos++] = clamp(num(battle, "seen_enemy_count") / 6, 0, 1);

  // ── GameMode flags + Inverse Battle (9 dims) ──
  buf[pos++] = bool(battle, "is_classic") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "is_endless") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "is_daily") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "is_challenge") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "has_mystery_encounters") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "has_short_biomes") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "has_random_biomes") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "has_random_bosses") ? 1.0 : 0.0;
  buf[pos++] = bool(battle, "inverse_battle") ? 1.0 : 0.0;

  return BATTLE_META_DIM;
}

// ─── Modifier Phase Encoding (from dict) ──────────────────────────────

/** Dims per reward option: valid(1) + tier_onehot(6) + is_pokemon(1) + features(20) = 28 */
const REWARD_OPTION_DIM = 28;

/** Dims per shop option: valid(1) + cost_ratio(1) + affordable(1) + features(20) = 23 */
const SHOP_OPTION_DIM = 23;

function encodeModifierFromDict(
  buf: Float32Array,
  offset: number,
  shop: Record<string, unknown> | null,
  money: number,
): number {
  let pos = offset;

  const isActive = shop !== null;

  // ── Header (3 dims) ──
  buf[pos++] = isActive ? 1.0 : 0.0; // modifier_active
  buf[pos++] = isActive && bool(shop!, "can_reroll") ? 1.0 : 0.0; // can_reroll
  const rerollCost = isActive ? num(shop!, "reroll_cost") : 0;
  buf[pos++] = isActive && money > 0 ? clamp(rerollCost / money, 0, 1) : 0; // reroll_cost_ratio

  if (!isActive) {
    // Skip remaining dims (zeros)
    pos += MAX_REWARD_OPTIONS * REWARD_OPTION_DIM + MAX_SHOP_OPTIONS_ENCODED * SHOP_OPTION_DIM;
    return MODIFIER_PHASE_DIM;
  }

  // ── Reward options (3 × 28 = 84 dims) ──
  const rewardOptions = arr(shop!, "reward_options");
  for (let i = 0; i < MAX_REWARD_OPTIONS; i++) {
    const opt =
      rewardOptions[i] && typeof rewardOptions[i] === "object" ? (rewardOptions[i] as Record<string, unknown>) : null;
    if (opt) {
      buf[pos++] = 1.0; // valid
      // tier one-hot (6 dims)
      writeOneHot(buf, pos, NUM_MODIFIER_TIERS, num(opt, "tier"));
      pos += NUM_MODIFIER_TIERS;
      // is_pokemon_modifier
      buf[pos++] = bool(opt, "is_pokemon_modifier") ? 1.0 : 0.0;
      // 20-dim modifier features
      const modId = (opt.modifier_id as string) ?? "";
      pos = encodeModifierFeatures(modId, opt, buf, pos);
    } else {
      pos += REWARD_OPTION_DIM;
    }
  }

  // ── Shop options (6 × 23 = 138 dims) ──
  // Encode the first 6 shop options in their NATURAL order — the same order
  // the BUY_SHOP actions (40-51) index. They used to be sorted by cost here,
  // which broke the framework's slot contract ("the observation describes
  // that action's slot"): encoded slot k described the k-th cheapest item
  // while action 40+k bought shop_options[k], so the agent could not map
  // shop features to buy actions. Options beyond the first 6 remain buyable
  // but unencoded (valid=0) — a capacity limit, not a misalignment.
  const shopOptions = arr(shop!, "shop_options");
  const validShopOpts: Record<string, unknown>[] = [];
  for (const opt of shopOptions) {
    if (opt && typeof opt === "object") {
      validShopOpts.push(opt as Record<string, unknown>);
    }
  }

  for (let i = 0; i < MAX_SHOP_OPTIONS_ENCODED; i++) {
    const opt = i < validShopOpts.length ? validShopOpts[i] : null;
    if (opt) {
      buf[pos++] = 1.0; // valid
      const cost = num(opt, "cost");
      buf[pos++] = money > 0 ? clamp(cost / money, 0, 1) : 1.0; // cost_ratio
      buf[pos++] = bool(opt, "affordable") ? 1.0 : 0.0; // affordable
      // 20-dim modifier features
      const modId = (opt.modifier_id as string) ?? "";
      pos = encodeModifierFeatures(modId, opt, buf, pos);
    } else {
      pos += SHOP_OPTION_DIM;
    }
  }

  return MODIFIER_PHASE_DIM;
}

// ─── Modifier Inventory Encoding (from dict) ────────────────────────

/** Active Pokemon slot keys (same order as first 4 in POKEMON_SLOT_KEYS) */
const ACTIVE_SLOT_KEYS = ["player_0", "player_1", "enemy_0", "enemy_1"];

/** Well-known party modifier_id strings for boolean presence flags */
const PARTY_FLAG_IDS: readonly string[] = [
  "HEALING_CHARM", // HealingBoosterModifier
  "EXP_SHARE", // ExpShareModifier
  "BERRY_POUCH", // PreserveBerryModifier
  "AMULET_COIN", // MoneyMultiplierModifier (also matches COIN_CASE)
  "LOCK_CAPSULE", // LockModifierTiersModifier
  "GOLDEN_POKEBALL", // ExtraModifierModifier
  "MEGA_BRACELET", // MegaEvolutionAccessModifier
  "TERA_ORB", // TerastallizeAccessModifier
];

/** Known enemy modifier_id strings for aggregate encoding */
const ENEMY_MOD_IDS: readonly string[] = [
  "ENEMY_DAMAGE_BOOSTER",
  "ENEMY_DAMAGE_REDUCTION",
  "ENEMY_HEAL",
  "ENEMY_ATTACK_POISON_CHANCE",
  "ENEMY_ATTACK_PARALYZE_CHANCE",
  "ENEMY_ATTACK_BURN_CHANCE",
  "ENEMY_STATUS_EFFECT_HEAL_CHANCE",
];

/**
 * Encode the modifier inventory into a fixed-size block.
 *
 * Layout (220 dims):
 *   Held items:   4 slots × 45 dims = 180
 *   Party mods:   9 dims (count + 8 boolean flags)
 *   Lapsing mods: 23 dims (count + 1 × (valid + features(20) + battles_remaining))
 *   Enemy mods:   8 dims (count + 7 aggregate stacks)
 *
 * @returns number of floats written (always MODIFIER_INVENTORY_DIM = 220)
 */
function encodeModifierInventory(buf: Float32Array, offset: number, gameState: Record<string, unknown>): number {
  let pos = offset;

  // ── Per active slot held items (4 × 45 = 180 dims) ──
  for (const slotKey of ACTIVE_SLOT_KEYS) {
    const poke = sub(gameState, slotKey);
    const heldItemsRaw = arr(poke, "held_items");

    // Cast to Record<string, unknown>[] for sorting
    const heldItemDicts: Record<string, unknown>[] = [];
    for (const item of heldItemsRaw) {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        heldItemDicts.push(item as Record<string, unknown>);
      }
    }

    // held_item_count / 10
    buf[pos++] = clamp(heldItemDicts.length / 10, 0, 1);

    // Sort by RL priority, encode top 2
    const sorted = sortByRLPriority(heldItemDicts);

    for (let i = 0; i < MAX_HELD_ITEMS_ENCODED; i++) {
      if (i < sorted.length) {
        const item = sorted[i];
        buf[pos++] = 1.0; // valid
        // 20-dim modifier features (static + dynamic)
        const modId = (item.modifier_id as string) ?? "";
        pos = encodeModifierFeatures(modId, item, buf, pos);
        // stack_ratio (redundant with feature[17], kept for compatibility)
        const stack = num(item, "stack_count");
        const maxStack = num(item, "max_stack_count", 1);
        buf[pos++] = maxStack > 0 ? clamp(stack / maxStack, 0, 1) : 0;
      } else {
        // 1 valid + 20 features + 1 stack_ratio = 22 zeros
        pos += HELD_ITEM_SLOT_DIM;
      }
    }
  }

  // ── Party-wide modifiers (9 dims) ──
  const modifiers = sub(gameState, "modifiers");
  const partyMods = arr(modifiers, "party_modifiers");

  // party_mod_count / 20
  buf[pos++] = clamp(partyMods.length / 20, 0, 1);

  // 8 boolean presence flags
  const partyModIdSet = new Set<string>();
  for (const mod of partyMods) {
    if (mod && typeof mod === "object") {
      const modId = (mod as Record<string, unknown>).modifier_id;
      if (typeof modId === "string") {
        partyModIdSet.add(modId);
      }
      // Also check modifier_class for COIN_CASE -> MoneyInterestModifier
      const modClass = (mod as Record<string, unknown>).modifier_class;
      if (modClass === "MoneyInterestModifier") {
        partyModIdSet.add("AMULET_COIN"); // maps to money_boost flag
      }
    }
  }
  for (const flagId of PARTY_FLAG_IDS) {
    buf[pos++] = partyModIdSet.has(flagId) ? 1.0 : 0.0;
  }

  // ── Lapsing modifiers (23 dims) ──
  const lapsingMods = arr(modifiers, "lapsing_modifiers");

  // lapsing_count / 5
  buf[pos++] = clamp(lapsingMods.length / 5, 0, 1);

  // Top 1 lapsing modifier: valid(1) + features(20) + battles_remaining(1) = 22
  let bestLapsing: Record<string, unknown> | null = null;
  let bestBattlesRemaining = 0;
  for (const mod of lapsingMods) {
    if (mod && typeof mod === "object") {
      const m = mod as Record<string, unknown>;
      const remaining = num(m, "battles_remaining");
      if (!bestLapsing || remaining > bestBattlesRemaining) {
        bestLapsing = m;
        bestBattlesRemaining = remaining;
      }
    }
  }

  if (bestLapsing) {
    buf[pos++] = 1.0; // valid
    const modId = (bestLapsing.modifier_id as string) ?? "";
    pos = encodeModifierFeatures(modId, bestLapsing, buf, pos);
    buf[pos++] = clamp(bestBattlesRemaining / 10, 0, 1);
  } else {
    // 1 valid + 20 features + 1 battles_remaining = 22 zeros
    pos += 22;
  }

  // ── Enemy modifiers (8 dims) ──
  const enemyMods = arr(modifiers, "enemy_modifiers");

  // enemy_mod_count / 20
  buf[pos++] = clamp(enemyMods.length / 20, 0, 1);

  // 7 aggregate stack values for known enemy modifier types
  const enemyStackMap = new Map<string, number>();
  for (const mod of enemyMods) {
    if (mod && typeof mod === "object") {
      const m = mod as Record<string, unknown>;
      const modId = (m.modifier_id as string) ?? "";
      const stack = num(m, "stack_count");
      // Sum status attack modifiers under a unified key
      if (
        modId === "ENEMY_ATTACK_POISON_CHANCE"
        || modId === "ENEMY_ATTACK_PARALYZE_CHANCE"
        || modId === "ENEMY_ATTACK_BURN_CHANCE"
      ) {
        enemyStackMap.set(modId, (enemyStackMap.get(modId) ?? 0) + stack);
      } else {
        enemyStackMap.set(modId, (enemyStackMap.get(modId) ?? 0) + stack);
      }
    }
  }

  const enemyNormDivisors: readonly number[] = [50, 50, 20, 20, 20, 20, 20]; // per ENEMY_MOD_IDS order
  for (let i = 0; i < ENEMY_MOD_IDS.length; i++) {
    const stacks = enemyStackMap.get(ENEMY_MOD_IDS[i]) ?? 0;
    buf[pos++] = clamp(stacks / enemyNormDivisors[i], 0, 1);
  }

  return MODIFIER_INVENTORY_DIM;
}

// ─── Derived Fields Encoding (type effectiveness, STAB, speed) ──────

/**
 * Standard Pokemon type effectiveness chart (19×19).
 * Rows = attacking type, Columns = defending type.
 * Types: NORMAL=0, FIGHTING=1, FLYING=2, POISON=3, GROUND=4, ROCK=5,
 *        BUG=6, GHOST=7, STEEL=8, FIRE=9, WATER=10, GRASS=11,
 *        ELECTRIC=12, PSYCHIC=13, ICE=14, DRAGON=15, DARK=16, FAIRY=17,
 *        STELLAR=18
 * STELLAR is neutral (1.0) vs everything.
 */
// prettier-ignore
const TYPE_EFFECTIVENESS: readonly (readonly number[])[] = [
  /*NORMAL  */ [1, 1, 1, 1, 1, 0.5, 1, 0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  /*FIGHTING*/ [2, 1, 0.5, 0.5, 1, 2, 0.5, 0, 2, 1, 1, 1, 1, 0.5, 2, 1, 2, 0.5, 1],
  /*FLYING  */ [1, 2, 1, 1, 1, 0.5, 2, 1, 0.5, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 1],
  /*POISON  */ [1, 1, 1, 0.5, 0.5, 0.5, 1, 0.5, 0, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1],
  /*GROUND  */ [1, 1, 0, 2, 1, 2, 0.5, 1, 2, 2, 1, 0.5, 2, 1, 1, 1, 1, 1, 1],
  /*ROCK    */ [1, 0.5, 2, 1, 0.5, 1, 2, 1, 0.5, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1],
  /*BUG     */ [1, 0.5, 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 1, 2, 1, 2, 1, 1, 2, 0.5, 1],
  /*GHOST   */ [0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0.5, 1, 1],
  /*STEEL   */ [1, 1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 0.5, 1, 0.5, 1, 2, 1, 1, 2, 1],
  /*FIRE    */ [1, 1, 1, 1, 1, 0.5, 2, 1, 2, 0.5, 0.5, 2, 1, 1, 2, 0.5, 1, 1, 1],
  /*WATER   */ [1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 0.5, 0.5, 1, 1, 1, 0.5, 1, 1, 1],
  /*GRASS   */ [1, 1, 0.5, 0.5, 2, 2, 0.5, 1, 0.5, 0.5, 2, 0.5, 1, 1, 1, 0.5, 1, 1, 1],
  /*ELECTRIC*/ [1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 2, 0.5, 0.5, 1, 1, 0.5, 1, 1, 1],
  /*PSYCHIC */ [1, 2, 1, 2, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 0, 1, 1],
  /*ICE     */ [1, 1, 2, 1, 2, 1, 1, 1, 0.5, 0.5, 0.5, 2, 1, 1, 0.5, 2, 1, 1, 1],
  /*DRAGON  */ [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1],
  /*DARK    */ [1, 0.5, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0.5, 0.5, 1],
  /*FAIRY   */ [1, 2, 1, 0.5, 1, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 2, 2, 1, 1],
  /*STELLAR */ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

/** Stat stage multipliers for stages -6..+6 (index = stage+6). Formula: positive=(2+s)/2, negative=2/(2+|s|) */
const STAGE_MULTIPLIERS = [2 / 8, 2 / 7, 2 / 6, 2 / 5, 2 / 4, 2 / 3, 1, 3 / 2, 4 / 2, 5 / 2, 6 / 2, 7 / 2, 8 / 2]; // indices match stage+6

/**
 * Compute type effectiveness of attacking type vs all defending types.
 * Returns the product of multipliers for each defending type.
 */
function computeTypeEffectiveness(atkType: number, defTypes: number[]): number {
  if (atkType < 0 || atkType >= NUM_POKEMON_TYPES) {
    return 1.0;
  }
  let mult = 1.0;
  for (const defType of defTypes) {
    if (defType >= 0 && defType < NUM_POKEMON_TYPES) {
      mult *= TYPE_EFFECTIVENESS[atkType][defType];
    }
  }
  return mult;
}

/**
 * Normalize type effectiveness to [0, 1] range.
 * immune(0) → 0, 0.25 → 0.125, 0.5 → 0.25, 1 → 0.5, 2 → 0.75, 4 → 1.0
 * Uses eff/4 (simple and monotonic).
 */
function normalizeEffectiveness(eff: number): number {
  return clamp(eff / 4.0, 0, 1);
}

/**
 * Encode derived fields: type effectiveness, STAB indicators, speed ordering.
 *
 * @returns number of floats written (always DERIVED_FIELDS_DIM = 28)
 */
function encodeDerivedFields(buf: Float32Array, offset: number, gameState: Record<string, unknown>): number {
  let pos = offset;

  // In SINGLES the v9 slot remap fills player_1/enemy_1 with the first BENCH
  // member (state-builder.ts) — a Pokemon NOT on the field. This block is an
  // ACTIVE-matchup / active-speed summary, so those slots must be excluded in
  // singles: otherwise the benched enemy acts as a phantom second target and
  // the benched mons pollute the speed ranking. Only slot 0 is active per side
  // in singles; in doubles both slots are genuinely on the field.
  const isDouble = bool(sub(gameState, "field"), "is_double_battle");
  const slotOf = (key: string): Record<string, unknown> =>
    !isDouble && (key === "player_1" || key === "enemy_1") ? {} : sub(gameState, key);

  const playerSlots = ["player_0", "player_1"];
  const enemySlots = ["enemy_0", "enemy_1"];

  // Pre-extract enemy types
  const enemyTypes: number[][] = [];
  for (const eKey of enemySlots) {
    const enemy = slotOf(eKey);
    const types = arr(enemy, "types").map(t => Number(t) || 0);
    enemyTypes.push(bool(enemy, "valid") ? types : []);
  }

  // ── Type effectiveness: 2 players × 4 moves × 2 enemies = 16 dims ──
  for (const pKey of playerSlots) {
    const player = slotOf(pKey);
    const moves = arr(player, "moves");
    for (let m = 0; m < MAX_MOVES; m++) {
      const moveDict =
        moves[m] && typeof moves[m] === "object" && !Array.isArray(moves[m])
          ? (moves[m] as Record<string, unknown>)
          : {};
      const moveType = num(moveDict, "type", -1);
      const moveId = num(moveDict, "move_id");
      for (let e = 0; e < 2; e++) {
        if (moveId > 0 && enemyTypes[e].length > 0) {
          const eff = computeTypeEffectiveness(moveType, enemyTypes[e]);
          buf[pos] = normalizeEffectiveness(eff);
        }
        pos++;
      }
    }
  }

  // ── STAB indicators: 2 players × 4 moves = 8 dims ──
  for (const pKey of playerSlots) {
    const player = slotOf(pKey);
    const playerTypes = arr(player, "types").map(t => Number(t));
    const moves = arr(player, "moves");
    for (let m = 0; m < MAX_MOVES; m++) {
      const moveDict =
        moves[m] && typeof moves[m] === "object" && !Array.isArray(moves[m])
          ? (moves[m] as Record<string, unknown>)
          : {};
      const moveType = num(moveDict, "type", -1);
      const moveId = num(moveDict, "move_id");
      if (moveId > 0 && moveType >= 0 && playerTypes.includes(moveType)) {
        buf[pos] = 1.0;
      }
      pos++;
    }
  }

  // ── Speed ordering: 4 active slots = 4 dims ──
  const speedSlots = ["player_0", "player_1", "enemy_0", "enemy_1"];
  const speeds: { index: number; speed: number; valid: boolean }[] = [];
  for (let i = 0; i < speedSlots.length; i++) {
    const poke = slotOf(speedSlots[i]);
    const isValid = bool(poke, "valid") && !bool(poke, "is_fainted");
    let speed = 0;
    if (isValid) {
      // Prefer computed stats (stats[5] = SPD), fall back to base_stats[5]
      const stats = arr(poke, "stats");
      const baseStats = arr(poke, "base_stats");
      speed = (Number(stats[5]) || 0) > 0 ? Number(stats[5]) : Number(baseStats[5]) || 0;
      // Apply speed stat stage (stat_stages[4] = SPD stage)
      const statStages = arr(poke, "stat_stages");
      const spdStage = Number(statStages[4]) || 0;
      const stageIdx = clamp(spdStage + 6, 0, 12);
      speed *= STAGE_MULTIPLIERS[stageIdx];
    }
    speeds.push({ index: i, speed, valid: isValid });
  }

  // Rank by speed (higher = faster). Ties get the same rank.
  const validSpeeds = speeds.filter(s => s.valid);
  const uniqueSpeeds = [...new Set(validSpeeds.map(s => s.speed))].sort((a, b) => b - a);
  const rankValues = [1.0, 0.75, 0.5, 0.25];

  for (let i = 0; i < speedSlots.length; i++) {
    const entry = speeds[i];
    if (entry.valid && uniqueSpeeds.length > 0) {
      const rankIdx = uniqueSpeeds.indexOf(entry.speed);
      buf[pos] = rankValues[Math.min(rankIdx, rankValues.length - 1)];
    }
    pos++;
  }

  return DERIVED_FIELDS_DIM;
}

// ─── Full Observation Encoder ─────────────────────────────────────────

/** Ordered list of Pokemon slot keys in the GameState dict */
const POKEMON_SLOT_KEYS = [
  "player_0",
  "player_1", // active player (2)
  "enemy_0",
  "enemy_1", // active enemy (2)
  "player_2",
  "player_3",
  "player_4",
  "player_5", // player bench (4)
  "enemy_2",
  "enemy_3",
  "enemy_4",
  "enemy_5", // enemy bench (4)
];

/**
 * Encode the v9 learn-move block: the OFFERED move as one compact move
 * vector (its valid dim doubles as offer-active) + learner party-index
 * one-hot. All-zero outside the learn_move phase.
 */
function encodeLearnMoveBlock(buf: Float32Array, offset: number, phase: Record<string, unknown>): number {
  const stats = phase.learn_move_stats;
  if (stats && typeof stats === "object" && !Array.isArray(stats)) {
    encodeMoveFromDict(buf, offset, stats as Record<string, unknown>);
  }
  writeOneHot(buf, offset + MOVE_BLOCK_DIM, MAX_PARTY_SIZE, num(phase, "learn_move_party_index", -1));
  return LEARN_MOVE_BLOCK_DIM;
}

/** Options for {@link encodeObservation}. */
export interface EncodeObservationOptions {
  /**
   * Mask enemy private information to what a human player could know
   * (unseen moves, unrevealed abilities, IV/nature-derived values,
   * never-seen bench members). Default false — full information.
   * See docs/OBS_V9_LAYOUT.md §5 for the exact masking table.
   */
  fogOfWar?: boolean;
}

/**
 * Encode a full observation vector from a GameState dict.
 *
 * @param gameState - The GameState dict from buildGameState()
 * @param opts - Optional encoding options (fog of war)
 * @returns Float32Array of size OBSERVATION_DIM (6991)
 */
export function encodeObservation(gameState: Record<string, unknown>, opts?: EncodeObservationOptions): Float32Array {
  const buf = new Float32Array(OBSERVATION_DIM);
  const fogOfWar = opts?.fogOfWar === true;
  let offset = 0;

  // ── Pokemon blocks (12 × 513 = 6156) ──
  for (const key of POKEMON_SLOT_KEYS) {
    const poke = sub(gameState, key);
    encodePokemonFromDict(buf, offset, poke, key.startsWith("enemy"), fogOfWar);
    offset += POKEMON_BLOCK_DIM;
  }

  // ── Field state (102) ──
  const field = sub(gameState, "field");
  encodeFieldFromDict(buf, offset, field);
  offset += FIELD_STATE_DIM;

  // ── Battle meta (40) ──
  const battle = sub(gameState, "battle");
  const battleStart = offset;
  encodeBattleFromDict(buf, offset, battle);
  offset += BATTLE_META_DIM;

  // Patch command_field_index from phase info
  const phase = sub(gameState, "phase");
  const commandFieldIndex = num(phase, "command_field_index", -1);
  // command_field_index is at battleStart + 2 + 4 + 1 + 1 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = battleStart + 20
  const cmdIdxOffset = battleStart + 20;
  if (commandFieldIndex < 0) {
    buf[cmdIdxOffset] = 0;
  } else {
    buf[cmdIdxOffset] = commandFieldIndex === 0 ? 0.5 : 1.0;
  }

  // ── Modifier phase (363) ──
  const shop = gameState.shop as Record<string, unknown> | null;
  const money = num(battle, "money");
  encodeModifierFromDict(buf, offset, shop ?? null, money);
  offset += MODIFIER_PHASE_DIM;

  // ── Modifier inventory (220) ──
  encodeModifierInventory(buf, offset, gameState);
  offset += MODIFIER_INVENTORY_DIM;

  // ── Derived fields (28) ──
  encodeDerivedFields(buf, offset, gameState);
  offset += DERIVED_FIELDS_DIM;

  // ── Learn-move block (66) ──
  encodeLearnMoveBlock(buf, offset, phase);
  offset += LEARN_MOVE_BLOCK_DIM;

  // ── Phase indicator (16-dim one-hot) ──
  const currentPhase = phase.current_phase as string | undefined;
  if (currentPhase && currentPhase in PHASE_INDEX_MAP) {
    const phaseIdx = PHASE_INDEX_MAP[currentPhase];
    buf[offset + phaseIdx] = 1.0;
  }
  offset += PHASE_INDICATOR_DIM;

  return buf;
}

// ─── Action Mask Extraction ───────────────────────────────────────────

/**
 * Extract the action mask from a GameState dict.
 *
 * The action mask is stored in gameState.phase.action_mask as a boolean[].
 *
 * @param gameState - The GameState dict from buildGameState()
 * @returns boolean array of size ACTION_SPACE_SIZE
 */
export function extractActionMask(gameState: Record<string, unknown>): boolean[] {
  const phase = sub(gameState, "phase");
  const mask = phase.action_mask;

  if (Array.isArray(mask) && mask.length === ACTION_SPACE_SIZE) {
    return mask.map(v => !!v);
  }

  return new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
}
