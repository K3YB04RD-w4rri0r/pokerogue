/**
 * Modifier feature encoding for the RL observation space.
 *
 * Maps modifier_id strings to 20-dimensional semantic feature vectors.
 * Static features (indices 0-7, 12-16) come from the lookup table.
 * Dynamic features (indices 8-11, 17-19) are filled at encoding time.
 *
 * Schema: /tmp/modifier-work/schema.md (frozen v1.0)
 *
 * Feature layout:
 *   0: is_damage_boost     8: target_type (D)      16: is_on_faint
 *   1: is_stat_boost       9: target_stat (D)      17: stack_count_norm (D)
 *   2: is_healing         10: status_effect_type(D) 18: stack_count_raw (D)
 *   3: is_survival        11: berry_type (D)        19: battles_remaining_norm (D)
 *   4: is_speed_priority  12: boost_magnitude
 *   5: is_status_effect   13: proc_chance_base
 *   6: is_economy         14: is_per_turn
 *   7: is_berry           15: is_on_hit
 */

/** Number of features per modifier in the lookup table */
export const MODIFIER_FEATURE_DIM = 20;

/** Default feature vector for unknown modifier_ids (all zeros) */
export const DEFAULT_MODIFIER_FEATURES = new Float32Array(MODIFIER_FEATURE_DIM);

/**
 * Static modifier features lookup table.
 * Maps modifier_id -> 13 static features at indices [0-7, 12-16].
 * Dynamic indices (8-11, 17-19) are always 0.0 in this table.
 *
 * Array layout per entry (20 floats):
 *   [is_damage_boost, is_stat_boost, is_healing, is_survival, is_speed_priority,
 *    is_status_effect, is_economy, is_berry,
 *    0, 0, 0, 0,  -- dynamic target slots (filled at encode time)
 *    boost_magnitude, proc_chance_base, is_per_turn, is_on_hit, is_on_faint,
 *    0, 0, 0]      -- dynamic stack/duration slots
 */
export const MODIFIER_FEATURES: Record<string, Float32Array> = {};

// Helper to build a 20-dim Float32Array from 13 static features
function f(
  d: number, s: number, h: number, sv: number, sp: number, se: number, ec: number, b: number,
  mag: number, proc: number, turn: number, hit: number, faint: number,
): Float32Array {
  //         0   1   2   3   4   5   6   7   8  9  10 11  12    13   14    15   16    17 18 19
  return new Float32Array([d, s, h, sv, sp, se, ec, b, 0, 0, 0, 0, mag, proc, turn, hit, faint, 0, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════════════
// HELD ITEMS — PokemonHeldItemModifier subclasses
// ═══════════════════════════════════════════════════════════════════════

//                                              D  S  H  SV SP SE EC B   mag    proc  turn hit faint
MODIFIER_FEATURES["ATTACK_TYPE_BOOSTER"]    = f(1, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0,   0,  0);
MODIFIER_FEATURES["BASE_STAT_BOOSTER"]      = f(0, 1, 0, 0, 0, 0, 0, 0, 0.10,  0,    0,   0,  0);
MODIFIER_FEATURES["BERRY"]                  = f(0, 0, 0, 0, 0, 0, 0, 1, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["QUICK_CLAW"]             = f(0, 0, 0, 0, 1, 0, 0, 0, 0.10,  0.10, 0,   0,  0);
MODIFIER_FEATURES["GRIP_CLAW"]              = f(0, 0, 0, 0, 0, 0, 0, 0, 0.10,  0.10, 0,   1,  0);
MODIFIER_FEATURES["SCOPE_LENS"]             = f(1, 0, 0, 0, 0, 0, 0, 0, 0.333, 0,    0,   0,  0);
MODIFIER_FEATURES["GOLDEN_PUNCH"]           = f(0, 0, 0, 0, 0, 0, 1, 0, 0.50,  0,    0,   1,  0);
MODIFIER_FEATURES["EVIOLITE"]               = f(0, 1, 0, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["EVOLUTION_TRACKER_GIMMIGHOUL"] = f(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
MODIFIER_FEATURES["MYSTICAL_ROCK"]          = f(0, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0,   0,  0);
MODIFIER_FEATURES["KINGS_ROCK"]             = f(0, 0, 0, 0, 0, 1, 0, 0, 0.10,  0.10, 0,   1,  0);
MODIFIER_FEATURES["SHELL_BELL"]             = f(0, 0, 1, 0, 0, 0, 0, 0, 0.125, 0,    0,   1,  0);
MODIFIER_FEATURES["MINI_BLACK_HOLE"]        = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    1,   0,  0);
MODIFIER_FEATURES["MULTI_LENS"]             = f(1, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   1,  0);
MODIFIER_FEATURES["MYSTERY_ENCOUNTER_MACHO_BRACE"] = f(0, 1, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0);
MODIFIER_FEATURES["MYSTERY_ENCOUNTER_OLD_GATEAU"] = f(0, 1, 0, 0, 0, 0, 0, 0, 0.20, 0, 0, 0, 0);
MODIFIER_FEATURES["MYSTERY_ENCOUNTER_SHUCKLE_JUICE"] = f(0, 1, 0, 0, 0, 0, 0, 0, 0.10, 0, 0, 0, 0);
MODIFIER_FEATURES["FOCUS_BAND"]             = f(0, 0, 0, 1, 0, 0, 0, 0, 0.10,  0.10, 0,   0,  0);
MODIFIER_FEATURES["SPECIES_STAT_BOOSTER"]   = f(0, 1, 0, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["RARE_SPECIES_STAT_BOOSTER"] = f(1, 1, 0, 0, 0, 0, 0, 0, 1.0, 0,   0,   0,  0);
MODIFIER_FEATURES["REVIVER_SEED"]           = f(0, 0, 0, 1, 0, 0, 0, 0, 0.50,  0,    0,   0,  1);
MODIFIER_FEATURES["LEFTOVERS"]              = f(0, 0, 1, 0, 0, 0, 0, 0, 0.0625, 0,   1,   0,  0);
MODIFIER_FEATURES["SOUL_DEW"]               = f(0, 1, 0, 0, 0, 0, 0, 0, 0.10,  0,    0,   0,  0);
MODIFIER_FEATURES["LEEK"]                   = f(1, 0, 0, 0, 0, 0, 0, 0, 0.667, 0,    0,   0,  0);
MODIFIER_FEATURES["TOXIC_ORB"]              = f(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    1,   0,  0);
MODIFIER_FEATURES["FLAME_ORB"]              = f(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    1,   0,  0);
MODIFIER_FEATURES["WHITE_HERB"]             = f(0, 1, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["WIDE_LENS"]              = f(0, 0, 0, 0, 0, 0, 0, 0, 0.05,  0,    0,   0,  0);
MODIFIER_FEATURES["GOLDEN_EGG"]             = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["LUCKY_EGG"]              = f(0, 0, 0, 0, 0, 0, 1, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["SOOTHE_BELL"]            = f(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["BATON"]                  = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["FORM_CHANGE_ITEM"]       = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["RARE_FORM_CHANGE_ITEM"]  = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);

// ═══════════════════════════════════════════════════════════════════════
// PARTY-WIDE PERSISTENT MODIFIERS
// ═══════════════════════════════════════════════════════════════════════

MODIFIER_FEATURES["MAP"]                    = f(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0,   0,  0);
MODIFIER_FEATURES["MEGA_BRACELET"]          = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["DYNAMAX_BAND"]           = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["TERA_ORB"]               = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["CANDY_JAR"]              = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["BERRY_POUCH"]            = f(0, 0, 0, 0, 0, 0, 0, 1, 0.30,  0.30, 0,   0,  0);
MODIFIER_FEATURES["OVAL_CHARM"]             = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["HEALING_CHARM"]          = f(0, 0, 1, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["EXP_CHARM"]              = f(0, 0, 0, 0, 0, 0, 1, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["SUPER_EXP_CHARM"]        = f(0, 0, 0, 0, 0, 0, 1, 0, 0.60,  0,    0,   0,  0);
MODIFIER_FEATURES["GOLDEN_EXP_CHARM"]       = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["EXP_SHARE"]              = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["EXP_BALANCE"]            = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["AMULET_COIN"]            = f(0, 0, 0, 0, 0, 0, 1, 0, 0.20,  0,    0,   0,  0);
MODIFIER_FEATURES["COIN_CASE"]              = f(0, 0, 0, 0, 0, 0, 1, 0, 0.10,  0,    1,   0,  0);
MODIFIER_FEATURES["ABILITY_CHARM"]          = f(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0,   0,  0);
MODIFIER_FEATURES["SHINY_CHARM"]            = f(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0,   0,  0);
MODIFIER_FEATURES["CATCHING_CHARM"]         = f(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["LOCK_CAPSULE"]           = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["MYSTERY_ENCOUNTER_BLACK_SLUDGE"] = f(0, 0, 0, 0, 0, 0, 1, 0, 0.25, 0, 0, 0, 0);
MODIFIER_FEATURES["MYSTERY_ENCOUNTER_GOLDEN_BUG_NET"] = f(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
MODIFIER_FEATURES["IV_SCANNER"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0,   0,  0);
MODIFIER_FEATURES["GOLDEN_POKEBALL"]        = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);

// ═══════════════════════════════════════════════════════════════════════
// LAPSING PERSISTENT MODIFIERS
// ═══════════════════════════════════════════════════════════════════════

MODIFIER_FEATURES["LURE"]                   = f(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["SUPER_LURE"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["MAX_LURE"]               = f(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["TEMP_STAT_STAGE_BOOSTER"] = f(0, 1, 0, 0, 0, 0, 0, 0, 0.20, 0,    0,   0,  0);
MODIFIER_FEATURES["DIRE_HIT"]               = f(1, 0, 0, 0, 0, 0, 0, 0, 0.333, 0,    0,   0,  0);
MODIFIER_FEATURES["SILVER_POKEBALL"]         = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,  0,    0,   0,  0);

// ═══════════════════════════════════════════════════════════════════════
// ENEMY PERSISTENT MODIFIERS
// ═══════════════════════════════════════════════════════════════════════

MODIFIER_FEATURES["ENEMY_DAMAGE_BOOSTER"]   = f(1, 0, 0, 0, 0, 0, 0, 0, 0.05,  0,    0,   0,  0);
MODIFIER_FEATURES["ENEMY_DAMAGE_REDUCTION"] = f(0, 0, 0, 1, 0, 0, 0, 0, 0.025, 0,    0,   0,  0);
MODIFIER_FEATURES["ENEMY_HEAL"]             = f(0, 0, 1, 0, 0, 0, 0, 0, 0.02,  0,    1,   0,  0);
MODIFIER_FEATURES["ENEMY_ATTACK_POISON_CHANCE"] = f(0, 0, 0, 0, 0, 1, 0, 0, 0.05, 0.05, 0, 1, 0);
MODIFIER_FEATURES["ENEMY_ATTACK_PARALYZE_CHANCE"] = f(0, 0, 0, 0, 0, 1, 0, 0, 0.025, 0.025, 0, 1, 0);
MODIFIER_FEATURES["ENEMY_ATTACK_BURN_CHANCE"] = f(0, 0, 0, 0, 0, 1, 0, 0, 0.05, 0.05, 0, 1, 0);
MODIFIER_FEATURES["ENEMY_STATUS_EFFECT_HEAL_CHANCE"] = f(0, 0, 0, 0, 0, 1, 0, 0, 0.025, 0.025, 1, 0, 0);
MODIFIER_FEATURES["ENEMY_ENDURE_CHANCE"]    = f(0, 0, 0, 1, 0, 0, 0, 0, 0.02,  0.02, 0,   0,  0);
MODIFIER_FEATURES["ENEMY_FUSED_CHANCE"]     = f(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0,   0,  0);

// ═══════════════════════════════════════════════════════════════════════
// CONSUMABLE MODIFIERS (reward/shop encoding only)
// ═══════════════════════════════════════════════════════════════════════

MODIFIER_FEATURES["POKEBALL"]               = f(0, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0,   0,  0);
MODIFIER_FEATURES["GREAT_BALL"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 0.40,  0,    0,   0,  0);
MODIFIER_FEATURES["ULTRA_BALL"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 0.60,  0,    0,   0,  0);
MODIFIER_FEATURES["ROGUE_BALL"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 0.80,  0,    0,   0,  0);
MODIFIER_FEATURES["MASTER_BALL"]            = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["VOUCHER"]                = f(0, 0, 0, 0, 0, 0, 1, 0, 0.33,  0,    0,   0,  0);
MODIFIER_FEATURES["VOUCHER_PLUS"]           = f(0, 0, 0, 0, 0, 0, 1, 0, 0.67,  0,    0,   0,  0);
MODIFIER_FEATURES["VOUCHER_PREMIUM"]        = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["NUGGET"]                 = f(0, 0, 0, 0, 0, 0, 1, 0, 0.33,  0,    0,   0,  0);
MODIFIER_FEATURES["BIG_NUGGET"]             = f(0, 0, 0, 0, 0, 0, 1, 0, 0.67,  0,    0,   0,  0);
MODIFIER_FEATURES["RELIC_GOLD"]             = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["POTION"]                 = f(0, 0, 1, 0, 0, 0, 0, 0, 0.05,  0,    0,   0,  0);
MODIFIER_FEATURES["SUPER_POTION"]           = f(0, 0, 1, 0, 0, 0, 0, 0, 0.125, 0,    0,   0,  0);
MODIFIER_FEATURES["HYPER_POTION"]           = f(0, 0, 1, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["MAX_POTION"]             = f(0, 0, 1, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["FULL_RESTORE"]           = f(0, 0, 1, 0, 0, 1, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["REVIVE"]                 = f(0, 0, 1, 1, 0, 0, 0, 0, 0.50,  0,    0,   0,  1);
MODIFIER_FEATURES["MAX_REVIVE"]             = f(0, 0, 1, 1, 0, 0, 0, 0, 1.0,   0,    0,   0,  1);
MODIFIER_FEATURES["SACRED_ASH"]             = f(0, 0, 1, 1, 0, 0, 0, 0, 1.0,   0,    0,   0,  1);
MODIFIER_FEATURES["FULL_HEAL"]              = f(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["RARE_CANDY"]             = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["RARER_CANDY"]            = f(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["ETHER"]                  = f(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["MAX_ETHER"]              = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["ELIXIR"]                 = f(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0,   0,  0);
MODIFIER_FEATURES["MAX_ELIXIR"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["PP_UP"]                  = f(0, 0, 0, 0, 0, 0, 0, 0, 0.33,  0,    0,   0,  0);
MODIFIER_FEATURES["PP_MAX"]                 = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["MINT"]                   = f(0, 1, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["TERA_SHARD"]             = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["EVOLUTION_ITEM"]         = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["RARE_EVOLUTION_ITEM"]    = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["TM_COMMON"]              = f(0, 0, 0, 0, 0, 0, 0, 0, 0.33,  0,    0,   0,  0);
MODIFIER_FEATURES["TM_GREAT"]               = f(0, 0, 0, 0, 0, 0, 0, 0, 0.67,  0,    0,   0,  0);
MODIFIER_FEATURES["TM_ULTRA"]               = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);
MODIFIER_FEATURES["MEMORY_MUSHROOM"]        = f(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0,   0,  0);
MODIFIER_FEATURES["DNA_SPLICERS"]           = f(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0,   0,  0);

// ═══════════════════════════════════════════════════════════════════════
// Encoding Helpers
// ═══════════════════════════════════════════════════════════════════════

/** RL priority ordering for held item sorting (higher priority first) */
const PRIORITY_KEYS: readonly number[] = [0, 3, 2, 1, 4, 5, 7, 6]; // is_damage > survival > healing > stat > speed > status > berry > economy

/**
 * Sort held items by RL importance using category flags.
 * Returns a new sorted array (does not mutate input).
 */
export function sortByRLPriority(items: Record<string, unknown>[]): Record<string, unknown>[] {
  return [...items].sort((a, b) => {
    const aId = (a.modifier_id as string) ?? "";
    const bId = (b.modifier_id as string) ?? "";
    const aFeats = MODIFIER_FEATURES[aId] ?? DEFAULT_MODIFIER_FEATURES;
    const bFeats = MODIFIER_FEATURES[bId] ?? DEFAULT_MODIFIER_FEATURES;

    // Compare by category flags in priority order
    for (const key of PRIORITY_KEYS) {
      if (aFeats[key] !== bFeats[key]) {
        return bFeats[key] - aFeats[key]; // Higher flag value first
      }
    }
    // Tie-break by stack_count descending
    const aStack = (typeof a.stack_count === "number" ? a.stack_count : 0);
    const bStack = (typeof b.stack_count === "number" ? b.stack_count : 0);
    return bStack - aStack;
  });
}

/** Clamp a value to [min, max] */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Safely read a number from a dict */
function num(dict: Record<string, unknown>, key: string, fallback = 0): number {
  const v = dict[key];
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return fallback;
}

/**
 * Encode a full 20-dim modifier feature vector into a buffer.
 * Writes static features from lookup + dynamic features from the item dict.
 *
 * @param modifierId - The modifier_id string
 * @param itemDict - The JSON dict containing dynamic fields (type_id, stat_id, etc.)
 * @param buf - Target Float32Array
 * @param pos - Write offset
 * @returns New position after writing (pos + MODIFIER_FEATURE_DIM)
 */
export function encodeModifierFeatures(
  modifierId: string,
  itemDict: Record<string, unknown>,
  buf: Float32Array,
  pos: number,
): number {
  const features = MODIFIER_FEATURES[modifierId] ?? DEFAULT_MODIFIER_FEATURES;

  // Category flags (0-7) — static
  buf[pos]     = features[0];
  buf[pos + 1] = features[1];
  buf[pos + 2] = features[2];
  buf[pos + 3] = features[3];
  buf[pos + 4] = features[4];
  buf[pos + 5] = features[5];
  buf[pos + 6] = features[6];
  buf[pos + 7] = features[7];

  // Dynamic target parameters (8-11)
  const typeId = num(itemDict, "type_id", -1);
  buf[pos + 8] = typeId >= 0 ? clamp(typeId / 18, 0, 1) : 0;

  const statId = num(itemDict, "stat_id", -1);
  buf[pos + 9] = statId >= 0 ? clamp(statId / 7, 0, 1) : 0;

  const statusEffect = num(itemDict, "status_effect", -1);
  buf[pos + 10] = statusEffect >= 0 ? clamp(statusEffect / 7, 0, 1) : 0;

  const berryType = num(itemDict, "berry_type", -1);
  buf[pos + 11] = berryType >= 0 ? clamp(berryType / 12, 0, 1) : 0;

  // Static effect parameters (12-16)
  buf[pos + 12] = features[12];
  buf[pos + 13] = features[13];
  buf[pos + 14] = features[14];
  buf[pos + 15] = features[15];
  buf[pos + 16] = features[16];

  // Dynamic stack/duration (17-19)
  const stackCount = num(itemDict, "stack_count");
  const maxStackCount = num(itemDict, "max_stack_count", 1);
  buf[pos + 17] = maxStackCount > 0 ? clamp(stackCount / maxStackCount, 0, 1) : 0;
  buf[pos + 18] = clamp(stackCount / 10, 0, 1);

  const battlesRemaining = num(itemDict, "battles_remaining");
  buf[pos + 19] = battlesRemaining > 0 ? clamp(battlesRemaining / 10, 0, 1) : 0;

  return pos + MODIFIER_FEATURE_DIM;
}
