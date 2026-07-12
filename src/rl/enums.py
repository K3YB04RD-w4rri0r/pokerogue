"""
Enum definitions and string-to-int mappings for RL observation encoding.

Every value in the GameState JSON that is a string (tag types, modifier IDs,
class names, phase names) is mapped to a unique integer here. Numeric enums
mirror the TypeScript source exactly.

Usage:
    from rl.enums import PokemonType, lookup_battler_tag, DIMS
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════
# 1. NUMERIC IntEnums (mirror TS enums exactly)
# ═══════════════════════════════════════════════════════════════════════════


class PokemonType(IntEnum):
    """PokemonType: 19 valid types (0-18), UNKNOWN=-1."""
    UNKNOWN = -1
    NORMAL = 0
    FIGHTING = 1
    FLYING = 2
    POISON = 3
    GROUND = 4
    ROCK = 5
    BUG = 6
    GHOST = 7
    STEEL = 8
    FIRE = 9
    WATER = 10
    GRASS = 11
    ELECTRIC = 12
    PSYCHIC = 13
    ICE = 14
    DRAGON = 15
    DARK = 16
    FAIRY = 17
    STELLAR = 18


class StatusEffect(IntEnum):
    """StatusEffect: NONE=0 .. FAINT=7."""
    NONE = 0
    POISON = 1
    TOXIC = 2
    PARALYSIS = 3
    SLEEP = 4
    FREEZE = 5
    BURN = 6
    FAINT = 7


class WeatherType(IntEnum):
    """WeatherType: NONE=0 .. STRONG_WINDS=9."""
    NONE = 0
    SUNNY = 1
    RAIN = 2
    SANDSTORM = 3
    HAIL = 4
    SNOW = 5
    FOG = 6
    HEAVY_RAIN = 7
    HARSH_SUN = 8
    STRONG_WINDS = 9


class TerrainType(IntEnum):
    """TerrainType: NONE=0 .. PSYCHIC=4."""
    NONE = 0
    MISTY = 1
    ELECTRIC = 2
    GRASSY = 3
    PSYCHIC = 4


class MoveCategory(IntEnum):
    """MoveCategory: PHYSICAL=0, SPECIAL=1, STATUS=2."""
    PHYSICAL = 0
    SPECIAL = 1
    STATUS = 2


class Nature(IntEnum):
    """Nature: HARDY=0 .. QUIRKY=24."""
    HARDY = 0
    LONELY = 1
    BRAVE = 2
    ADAMANT = 3
    NAUGHTY = 4
    BOLD = 5
    DOCILE = 6
    RELAXED = 7
    IMPISH = 8
    LAX = 9
    TIMID = 10
    HASTY = 11
    SERIOUS = 12
    JOLLY = 13
    NAIVE = 14
    MODEST = 15
    MILD = 16
    QUIET = 17
    BASHFUL = 18
    RASH = 19
    CALM = 20
    GENTLE = 21
    SASSY = 22
    CAREFUL = 23
    QUIRKY = 24


class Gender(IntEnum):
    """Gender: GENDERLESS=-1, MALE=0, FEMALE=1."""
    GENDERLESS = -1
    MALE = 0
    FEMALE = 1


class BattleType(IntEnum):
    """BattleType: WILD=0 .. MYSTERY_ENCOUNTER=3."""
    WILD = 0
    TRAINER = 1
    CLEAR = 2
    MYSTERY_ENCOUNTER = 3


class ModifierTier(IntEnum):
    """ModifierTier: COMMON=0 .. LUXURY=5."""
    COMMON = 0
    GREAT = 1
    ULTRA = 2
    ROGUE = 3
    MASTER = 4
    LUXURY = 5


class PokeballType(IntEnum):
    """PokeballType: POKEBALL=0 .. LUXURY_BALL=5."""
    POKEBALL = 0
    GREAT_BALL = 1
    ULTRA_BALL = 2
    ROGUE_BALL = 3
    MASTER_BALL = 4
    LUXURY_BALL = 5


class AiType(IntEnum):
    """AiType: RANDOM=0, SMART_RANDOM=1, SMART=2."""
    RANDOM = 0
    SMART_RANDOM = 1
    SMART = 2


class Stat(IntEnum):
    """Stat: HP=0 .. EVA=7."""
    HP = 0
    ATK = 1
    DEF = 2
    SPATK = 3
    SPDEF = 4
    SPD = 5
    ACC = 6
    EVA = 7


class MultiHitType(IntEnum):
    """MultiHitType: TWO=0 .. BEAT_UP=4, N/A=-1."""
    NA = -1
    TWO = 0
    TWO_TO_FIVE = 1
    THREE = 2
    TEN = 3
    BEAT_UP = 4


class ArenaTagSide(IntEnum):
    """ArenaTagSide: BOTH=0, PLAYER=1, ENEMY=2."""
    BOTH = 0
    PLAYER = 1
    ENEMY = 2


class BattlerIndex(IntEnum):
    """BattlerIndex: ATTACKER=-1, PLAYER=0 .. ENEMY_2=3."""
    ATTACKER = -1
    PLAYER = 0
    PLAYER_2 = 1
    ENEMY = 2
    ENEMY_2 = 3


class GameMode(IntEnum):
    """GameModes: CLASSIC=0 .. CHALLENGE=4."""
    CLASSIC = 0
    ENDLESS = 1
    SPLICED_ENDLESS = 2
    DAILY = 3
    CHALLENGE = 4


class TimeOfDay(IntEnum):
    """TimeOfDay: ALL=-1, DAWN=0 .. NIGHT=3."""
    ALL = -1
    DAWN = 0
    DAY = 1
    DUSK = 2
    NIGHT = 3


class BattleStyle(IntEnum):
    """BattleStyle: SWITCH=0, SET=1."""
    SWITCH = 0
    SET = 1


class BerryType(IntEnum):
    """BerryType: SITRUS=0 .. LEPPA=10."""
    SITRUS = 0
    LUM = 1
    ENIGMA = 2
    LIECHI = 3
    GANLON = 4
    PETAYA = 5
    APICOT = 6
    SALAC = 7
    LANSAT = 8
    STARF = 9
    LEPPA = 10


class HitResult(IntEnum):
    """HitResult: starts at 1 (no 0 value)."""
    EFFECTIVE = 1
    SUPER_EFFECTIVE = 2
    NOT_VERY_EFFECTIVE = 3
    ONE_HIT_KO = 4
    NO_EFFECT = 5
    STATUS = 6
    HEAL = 7
    FAIL = 8
    MISS = 9
    INDIRECT = 10
    IMMUNE = 11
    CONFUSION = 12
    INDIRECT_KO = 13


class MoveResult(IntEnum):
    """MoveResult: PENDING=0 .. OTHER=4."""
    PENDING = 0
    SUCCESS = 1
    FAIL = 2
    MISS = 3
    OTHER = 4


class Command(IntEnum):
    """Command: FIGHT=0 .. TERA=4."""
    FIGHT = 0
    BALL = 1
    POKEMON = 2
    RUN = 3
    TERA = 4


# ═══════════════════════════════════════════════════════════════════════════
# 2. STRING ENUM -> INTEGER MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════

# ─── BattlerTagType: 97 members + UNKNOWN=0 ──────────────────────────────

_BATTLER_TAG_VALUES: List[str] = [
    "RECHARGING",
    "FLINCHED",
    "INTERRUPTED",
    "CONFUSED",
    "INFATUATED",
    "SEEDED",
    "NIGHTMARE",
    "FRENZY",
    "CHARGING",
    "ENCORE",
    "HELPING_HAND",
    "INGRAIN",
    "OCTOLOCK",
    "AQUA_RING",
    "DROWSY",
    "TRAPPED",
    "BIND",
    "WRAP",
    "FIRE_SPIN",
    "WHIRLPOOL",
    "CLAMP",
    "SAND_TOMB",
    "MAGMA_STORM",
    "SNAP_TRAP",
    "THUNDER_CAGE",
    "INFESTATION",
    "PROTECTED",
    "SPIKY_SHIELD",
    "KINGS_SHIELD",
    "OBSTRUCT",
    "SILK_TRAP",
    "BANEFUL_BUNKER",
    "BURNING_BULWARK",
    "ENDURING",
    "STURDY",
    "PERISH_SONG",
    "TRUANT",
    "SLOW_START",
    "PROTOSYNTHESIS",
    "QUARK_DRIVE",
    "FLYING",
    "UNDERGROUND",
    "UNDERWATER",
    "HIDDEN",
    "FIRE_BOOST",
    "CRIT_BOOST",
    "ALWAYS_CRIT",
    "IGNORE_ACCURACY",
    "IGNORE_FLYING",
    "SALT_CURED",
    "CURSED",
    "CHARGED",
    "ROOSTED",
    "FLOATING",
    "MINIMIZED",
    "DESTINY_BOND",
    "CENTER_OF_ATTENTION",
    "ICE_FACE",
    "DISGUISE",
    "STOCKPILING",
    "RECEIVE_DOUBLE_DAMAGE",
    "ALWAYS_GET_HIT",
    "DISABLED",
    "SUBSTITUTE",
    "IGNORE_GHOST",
    "IGNORE_DARK",
    "GULP_MISSILE_ARROKUDA",
    "GULP_MISSILE_PIKACHU",
    "BEAK_BLAST_CHARGING",
    "SHELL_TRAP",
    "DRAGON_CHEER",
    "NO_RETREAT",
    "GORILLA_TACTICS",
    "UNBURDEN",
    "THROAT_CHOPPED",
    "TAR_SHOT",
    "BURNED_UP",
    "DOUBLE_SHOCKED",
    "AUTOTOMIZED",
    "MYSTERY_ENCOUNTER_POST_SUMMON",
    "POWER_TRICK",
    "HEAL_BLOCK",
    "TORMENT",
    "TAUNT",
    "IMPRISON",
    "SYRUP_BOMB",
    "ELECTRIFIED",
    "TELEKINESIS",
    "COMMANDED",
    "GRUDGE",
    "PSYCHO_SHIFT",
    "ENDURE_TOKEN",
    "POWDER",
    "MAGIC_COAT",
    "SUPREME_OVERLORD",
    "BYPASS_SPEED",
]

# 0 = UNKNOWN, 1..97 = tag types in declaration order
BATTLER_TAG_STR_TO_ID: Dict[str, int] = {v: i + 1 for i, v in enumerate(_BATTLER_TAG_VALUES)}
BATTLER_TAG_ID_TO_STR: Dict[int, str] = {v: k for k, v in BATTLER_TAG_STR_TO_ID.items()}
BATTLER_TAG_ID_TO_STR[0] = "UNKNOWN"
NUM_BATTLER_TAGS = len(_BATTLER_TAG_VALUES) + 1  # 98 including UNKNOWN

# ─── ArenaTagType: 28 members (excl. NONE) + UNKNOWN=0 + NONE=1 ─────────

_ARENA_TAG_VALUES: List[str] = [
    "NONE",
    "MUD_SPORT",
    "WATER_SPORT",
    "SPIKES",
    "TOXIC_SPIKES",
    "MIST",
    "STEALTH_ROCK",
    "STICKY_WEB",
    "TRICK_ROOM",
    "GRAVITY",
    "REFLECT",
    "LIGHT_SCREEN",
    "AURORA_VEIL",
    "QUICK_GUARD",
    "WIDE_GUARD",
    "MAT_BLOCK",
    "CRAFTY_SHIELD",
    "TAILWIND",
    "HAPPY_HOUR",
    "SAFEGUARD",
    "NO_CRIT",
    "IMPRISON",
    "ION_DELUGE",
    "FIRE_GRASS_PLEDGE",
    "WATER_FIRE_PLEDGE",
    "GRASS_WATER_PLEDGE",
    "FAIRY_LOCK",
    "NEUTRALIZING_GAS",
    "PENDING_HEAL",
]

# 0 = UNKNOWN, 1..29 = tag types in declaration order (NONE=1)
ARENA_TAG_STR_TO_ID: Dict[str, int] = {v: i + 1 for i, v in enumerate(_ARENA_TAG_VALUES)}
ARENA_TAG_ID_TO_STR: Dict[int, str] = {v: k for k, v in ARENA_TAG_STR_TO_ID.items()}
ARENA_TAG_ID_TO_STR[0] = "UNKNOWN"
NUM_ARENA_TAGS = len(_ARENA_TAG_VALUES) + 1  # 30 including UNKNOWN

# ─── ModifierTypeId: 109 modifier type keys (alphabetically sorted) ──────

_MODIFIER_TYPE_VALUES: List[str] = sorted([
    "POKEBALL", "GREAT_BALL", "ULTRA_BALL", "ROGUE_BALL", "MASTER_BALL",
    "RARE_CANDY", "RARER_CANDY", "EVOLUTION_ITEM", "RARE_EVOLUTION_ITEM",
    "FORM_CHANGE_ITEM", "RARE_FORM_CHANGE_ITEM", "EVOLUTION_TRACKER_GIMMIGHOUL",
    "MEGA_BRACELET", "DYNAMAX_BAND", "TERA_ORB", "MAP",
    "POTION", "SUPER_POTION", "HYPER_POTION", "MAX_POTION", "FULL_RESTORE",
    "REVIVE", "MAX_REVIVE", "FULL_HEAL", "SACRED_ASH", "REVIVER_SEED",
    "WHITE_HERB", "ETHER", "MAX_ETHER", "ELIXIR", "MAX_ELIXIR",
    "PP_UP", "PP_MAX", "LURE", "SUPER_LURE", "MAX_LURE",
    "SPECIES_STAT_BOOSTER", "RARE_SPECIES_STAT_BOOSTER",
    "TEMP_STAT_STAGE_BOOSTER", "DIRE_HIT", "BASE_STAT_BOOSTER",
    "ATTACK_TYPE_BOOSTER", "MINT", "MYSTICAL_ROCK", "TERA_SHARD", "BERRY",
    "TM_COMMON", "TM_GREAT", "TM_ULTRA", "MEMORY_MUSHROOM",
    "EXP_SHARE", "EXP_BALANCE", "OVAL_CHARM", "EXP_CHARM",
    "SUPER_EXP_CHARM", "GOLDEN_EXP_CHARM", "LUCKY_EGG", "GOLDEN_EGG",
    "SOOTHE_BELL", "SCOPE_LENS", "LEEK", "EVIOLITE", "SOUL_DEW",
    "NUGGET", "BIG_NUGGET", "RELIC_GOLD", "AMULET_COIN", "GOLDEN_PUNCH",
    "COIN_CASE", "LOCK_CAPSULE", "GRIP_CLAW", "WIDE_LENS", "MULTI_LENS",
    "HEALING_CHARM", "CANDY_JAR", "BERRY_POUCH", "FOCUS_BAND",
    "QUICK_CLAW", "KINGS_ROCK", "LEFTOVERS", "SHELL_BELL",
    "TOXIC_ORB", "FLAME_ORB", "BATON", "SHINY_CHARM", "ABILITY_CHARM",
    "CATCHING_CHARM", "IV_SCANNER", "DNA_SPLICERS", "MINI_BLACK_HOLE",
    "VOUCHER", "VOUCHER_PLUS", "VOUCHER_PREMIUM",
    "GOLDEN_POKEBALL", "SILVER_POKEBALL",
    "ENEMY_DAMAGE_BOOSTER", "ENEMY_DAMAGE_REDUCTION", "ENEMY_HEAL",
    "ENEMY_ATTACK_POISON_CHANCE", "ENEMY_ATTACK_PARALYZE_CHANCE",
    "ENEMY_ATTACK_BURN_CHANCE", "ENEMY_STATUS_EFFECT_HEAL_CHANCE",
    "ENEMY_ENDURE_CHANCE", "ENEMY_FUSED_CHANCE",
    "MYSTERY_ENCOUNTER_SHUCKLE_JUICE", "MYSTERY_ENCOUNTER_OLD_GATEAU",
    "MYSTERY_ENCOUNTER_BLACK_SLUDGE", "MYSTERY_ENCOUNTER_MACHO_BRACE",
    "MYSTERY_ENCOUNTER_GOLDEN_BUG_NET",
])

# 0 = UNKNOWN, 1..109 = modifier types in alphabetical order
MODIFIER_TYPE_STR_TO_ID: Dict[str, int] = {v: i + 1 for i, v in enumerate(_MODIFIER_TYPE_VALUES)}
MODIFIER_TYPE_ID_TO_STR: Dict[int, str] = {v: k for k, v in MODIFIER_TYPE_STR_TO_ID.items()}
MODIFIER_TYPE_ID_TO_STR[0] = "UNKNOWN"
NUM_MODIFIER_TYPES = len(_MODIFIER_TYPE_VALUES) + 1  # 110 including UNKNOWN

# ─── ModifierClassId: 78 concrete modifier class names ───────────────────

_MODIFIER_CLASS_VALUES: List[str] = [
    "AddPokeballModifier",
    "AddVoucherModifier",
    "AttackTypeBoosterModifier",
    "BaseStatModifier",
    "BerryModifier",
    "BoostBugSpawnModifier",
    "BypassSpeedChanceModifier",
    "ContactHeldItemTransferChanceModifier",
    "CritBoosterModifier",
    "CriticalCatchChanceBoosterModifier",
    "DamageMoneyRewardModifier",
    "DoubleBattleChanceBoosterModifier",
    "EnemyAttackStatusEffectChanceModifier",
    "EnemyDamageBoosterModifier",
    "EnemyDamageReducerModifier",
    "EnemyEndureChanceModifier",
    "EnemyFusionChanceModifier",
    "EnemyStatusEffectHealChanceModifier",
    "EnemyTurnHealModifier",
    "EvolutionItemModifier",
    "EvolutionStatBoosterModifier",
    "EvoTrackerModifier",
    "ExpBalanceModifier",
    "ExpBoosterModifier",
    "ExpShareModifier",
    "ExtraModifierModifier",
    "FieldEffectModifier",
    "FlinchChanceModifier",
    "FusePokemonModifier",
    "GigantamaxAccessModifier",
    "HealShopCostModifier",
    "HealingBoosterModifier",
    "HiddenAbilityRateBoosterModifier",
    "HitHealModifier",
    "IvScannerModifier",
    "LevelIncrementBoosterModifier",
    "LockModifierTiersModifier",
    "MapModifier",
    "MegaEvolutionAccessModifier",
    "MoneyInterestModifier",
    "MoneyMultiplierModifier",
    "MoneyRewardModifier",
    "MultipleParticipantExpBonusModifier",
    "PokemonAllMovePpRestoreModifier",
    "PokemonBaseStatFlatModifier",
    "PokemonBaseStatTotalModifier",
    "PokemonExpBoosterModifier",
    "PokemonFormChangeItemModifier",
    "PokemonFriendshipBoosterModifier",
    "PokemonHpRestoreModifier",
    "PokemonIncrementingStatModifier",
    "PokemonInstantReviveModifier",
    "PokemonLevelIncrementModifier",
    "PokemonMoveAccuracyBoosterModifier",
    "PokemonMultiHitModifier",
    "PokemonNatureChangeModifier",
    "PokemonNatureWeightModifier",
    "PokemonPpRestoreModifier",
    "PokemonPpUpModifier",
    "PokemonStatusHealModifier",
    "PreserveBerryModifier",
    "RememberMoveModifier",
    "ResetNegativeStatStageModifier",
    "ShinyRateBoosterModifier",
    "SpeciesCritBoosterModifier",
    "SpeciesStatBoosterModifier",
    "StatBoosterModifier",
    "SurviveDamageModifier",
    "SwitchEffectTransferModifier",
    "TempCritBoosterModifier",
    "TempExtraModifierModifier",
    "TempStatStageBoosterModifier",
    "TerastallizeAccessModifier",
    "TerastallizeModifier",
    "TmModifier",
    "TurnHealModifier",
    "TurnHeldItemTransferModifier",
    "TurnStatusEffectModifier",
]

# 0 = UNKNOWN, 1..78 = modifier classes in alphabetical order
MODIFIER_CLASS_STR_TO_ID: Dict[str, int] = {v: i + 1 for i, v in enumerate(_MODIFIER_CLASS_VALUES)}
MODIFIER_CLASS_ID_TO_STR: Dict[int, str] = {v: k for k, v in MODIFIER_CLASS_STR_TO_ID.items()}
MODIFIER_CLASS_ID_TO_STR[0] = "UNKNOWN"
NUM_MODIFIER_CLASSES = len(_MODIFIER_CLASS_VALUES) + 1  # 79 including UNKNOWN

# ─── PhaseId: 16 phase types matching PHASE_INDEX_MAP in spaces.ts ───────

_PHASE_VALUES: List[str] = [
    "command",          # 0
    "target",           # 1
    "modifier",         # 2
    "modifier_target",  # 3
    "switch",           # 4
    "check_switch",     # 5
    "learn_move",       # 6
    "evolution",        # 7
    "starter",          # 8
    "mystery",          # 9
    "game_over",        # 10
    "select_biome",     # 11
    "revival_blessing",  # 12
    "form_change",      # 13
    "title",            # 14
    "select_gender",    # 15
]

PHASE_STR_TO_ID: Dict[str, int] = {v: i for i, v in enumerate(_PHASE_VALUES)}
# Self-play alias (mirrors spaces.ts PHASE_INDEX_MAP): an enemy_command
# decision is a command decision from the enemy's own perspective.
PHASE_STR_TO_ID["enemy_command"] = PHASE_STR_TO_ID["command"]
PHASE_ID_TO_STR: Dict[int, str] = {v: k for k, v in PHASE_STR_TO_ID.items()}
NUM_PHASES = len(_PHASE_VALUES)  # 16

# ─── TargetKindId: 4 members ─────────────────────────────────────────────

class TargetKindId(IntEnum):
    NONE = 0
    POKEMON = 1
    MOVE = 2
    POKEMON_PAIR = 3


TARGET_KIND_STR_TO_ID: Dict[str, int] = {
    "none": 0, "pokemon": 1, "move": 2, "pokemon_pair": 3,
}

# ─── PositionalTagId: 3 members ──────────────────────────────────────────

class PositionalTagId(IntEnum):
    UNKNOWN = 0
    DELAYED_ATTACK = 1
    WISH = 2


POSITIONAL_TAG_STR_TO_ID: Dict[str, int] = {
    "DELAYED_ATTACK": 1, "WISH": 2,
}

# ═══════════════════════════════════════════════════════════════════════════
# 3. CURATED ORDERINGS (match spaces.ts exactly)
# ═══════════════════════════════════════════════════════════════════════════

# Strategically important volatile tags, in spaces.ts order.
# v9: 7 TURN_END-transient tags cut (FLINCHED/PROTECTED/ENDURING/
# HELPING_HAND/MAGIC_COAT/POWDER/CENTER_OF_ATTENTION) — they lapse before
# every decision boundary. Must match spaces.ts CURATED_VOLATILE_TAGS.
CURATED_VOLATILE_TAGS: List[str] = [
    "CONFUSED", "INFATUATED", "SEEDED", "TRAPPED",
    "ENCORE", "SUBSTITUTE", "DISABLED",
    "TAUNT", "TORMENT", "HEAL_BLOCK", "INGRAIN", "AQUA_RING",
    "FLYING", "UNDERGROUND", "UNDERWATER", "CHARGING", "RECHARGING",
    "FRENZY", "PERISH_SONG", "DESTINY_BOND", "CURSED", "SALT_CURED",
    "OCTOLOCK", "DROWSY", "STOCKPILING", "MINIMIZED", "IMPRISON",
    # v2 additions (v9-kept)
    "SLOW_START",           # Regigigas halved ATK/SPD for 5 turns
    "UNBURDEN",             # Doubled speed after item loss
    "RECEIVE_DOUBLE_DAMAGE",  # Tar Shot — 2x fire damage
    "FLOATING",             # Magnet Rise/Telekinesis — ground immunity
    "ALWAYS_CRIT",          # Laser Focus — guaranteed crit next turn
    # v3: 9 additional tags from audit
    "GRUDGE",               # If holder faints, attacker's move loses all PP
    "ICE_FACE",             # Eiscue form — absorbs one physical hit
    "DISGUISE",             # Mimikyu form — absorbs one hit
    "NO_RETREAT",           # Can't switch but got +1 all stats
    "THROAT_CHOPPED",       # Can't use sound-based moves for 2 turns
    "SYRUP_BOMB",           # -1 Speed per turn for 3 turns
    "COMMANDED",            # Commander ability — merged into ally
    "BURNED_UP",            # Lost Fire type after Burn Up
    "DOUBLE_SHOCKED",       # Lost Electric type after Double Shock
    # v8: 28 additional tags (must match spaces.ts CURATED_VOLATILE_TAGS order)
    "BIND", "WRAP", "CLAMP", "FIRE_SPIN", "WHIRLPOOL", "MAGMA_STORM",
    "SAND_TOMB", "SNAP_TRAP", "THUNDER_CAGE", "INFESTATION",  # partial-trap family
    "CHARGED", "CRIT_BOOST", "DRAGON_CHEER", "FIRE_BOOST", "GORILLA_TACTICS",
    "HIDDEN", "IGNORE_ACCURACY", "IGNORE_DARK", "IGNORE_FLYING", "IGNORE_GHOST",
    "NIGHTMARE", "PROTOSYNTHESIS", "QUARK_DRIVE", "SUPREME_OVERLORD",
    "TAR_SHOT", "TELEKINESIS", "TRUANT", "ALWAYS_GET_HIT",
]

# 28 arena tag types in spaces.ts order (lines 232-261)
ARENA_TAG_ORDER: List[str] = [
    "MUD_SPORT", "WATER_SPORT", "SPIKES", "TOXIC_SPIKES", "MIST",
    "STEALTH_ROCK", "STICKY_WEB", "TRICK_ROOM", "GRAVITY", "REFLECT",
    "LIGHT_SCREEN", "AURORA_VEIL", "QUICK_GUARD", "WIDE_GUARD", "MAT_BLOCK",
    "CRAFTY_SHIELD", "TAILWIND", "HAPPY_HOUR", "SAFEGUARD", "NO_CRIT",
    "IMPRISON", "ION_DELUGE", "FIRE_GRASS_PLEDGE", "WATER_FIRE_PLEDGE",
    "GRASS_WATER_PLEDGE", "FAIRY_LOCK", "NEUTRALIZING_GAS", "PENDING_HEAL",
]

# ═══════════════════════════════════════════════════════════════════════════
# 4. SAFE LOOKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def lookup_battler_tag(tag_str: str) -> int:
    """Map a BattlerTagType string to its integer ID. Returns 0 (UNKNOWN) for unrecognized."""
    return BATTLER_TAG_STR_TO_ID.get(tag_str, 0)


def lookup_arena_tag(tag_str: str) -> int:
    """Map an ArenaTagType string to its integer ID. Returns 0 (UNKNOWN) for unrecognized."""
    return ARENA_TAG_STR_TO_ID.get(tag_str, 0)


def lookup_modifier_type(type_str: str) -> int:
    """Map a modifier type ID string to its integer ID. Returns 0 (UNKNOWN) for unrecognized."""
    return MODIFIER_TYPE_STR_TO_ID.get(type_str, 0)


def lookup_modifier_class(class_str: str) -> int:
    """Map a modifier class name to its integer ID. Returns 0 (UNKNOWN) for unrecognized."""
    return MODIFIER_CLASS_STR_TO_ID.get(class_str, 0)


def lookup_phase(phase_str: str) -> int:
    """Map a phase name to its integer ID. Returns -1 for unrecognized."""
    return PHASE_STR_TO_ID.get(phase_str, -1)


def lookup_target_kind(kind_str: str) -> int:
    """Map a target kind string to its integer ID. Returns 0 (NONE) for unrecognized."""
    return TARGET_KIND_STR_TO_ID.get(kind_str, 0)


def lookup_positional_tag(tag_str: str) -> int:
    """Map a positional tag type string to its integer ID. Returns 0 (UNKNOWN) for unrecognized."""
    return POSITIONAL_TAG_STR_TO_ID.get(tag_str, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. DISPLAY NAME DICTIONARIES (for play.py TUI)
# ═══════════════════════════════════════════════════════════════════════════

TYPE_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in PokemonType}
TYPE_NAMES[-1] = "\u2014"

STATUS_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in StatusEffect}
STATUS_NAMES[0] = "\u2014"

WEATHER_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in WeatherType}
WEATHER_NAMES[0] = "None"

TERRAIN_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in TerrainType}
TERRAIN_NAMES[0] = "None"

CATEGORY_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in MoveCategory}

BATTLE_TYPE_NAMES: Dict[int, str] = {m.value: m.name.replace("_", " ").title() for m in BattleType}

TYPE_ABBREV: Dict[int, str] = {
    0: "Nor", 1: "Fig", 2: "Fly", 3: "Psn", 4: "Gnd", 5: "Rck", 6: "Bug",
    7: "Gho", 8: "Stl", 9: "Fir", 10: "Wat", 11: "Grs", 12: "Elc", 13: "Psy",
    14: "Ice", 15: "Drg", 16: "Drk", 17: "Fai", 18: "Str", -1: "\u2014",
}

CATEGORY_ABBREV: Dict[int, str] = {0: "Phy", 1: "Spe", 2: "Sta"}

# ═══════════════════════════════════════════════════════════════════════════
# 6. DIMENSION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

DIMS: Dict[str, int] = {
    "pokemon_type": 19,
    "status_effect": 8,
    "weather_type": 10,
    "terrain_type": 5,
    "move_category": 3,
    "nature": 25,
    "battle_type": 4,
    "modifier_tier": 6,
    "pokeball_type": 6,
    "battler_tag": NUM_BATTLER_TAGS,
    "arena_tag": NUM_ARENA_TAGS,
    "modifier_type": NUM_MODIFIER_TYPES,
    "modifier_class": NUM_MODIFIER_CLASSES,
    "phase": NUM_PHASES,
    "target_kind": 4,
    "positional_tag": 3,
    "curated_volatile": len(CURATED_VOLATILE_TAGS),
    "arena_tag_order": len(ARENA_TAG_ORDER),
}

# Pokemon slot keys in observation order (matches spaces.ts POKEMON_SLOT_KEYS)
POKEMON_SLOT_KEYS: List[str] = [
    "player_0", "player_1",  # active player (2)
    "enemy_0", "enemy_1",    # active enemy (2)
    "player_2", "player_3", "player_4", "player_5",  # player bench (4)
    "enemy_2", "enemy_3", "enemy_4", "enemy_5",      # enemy bench (4)
]
