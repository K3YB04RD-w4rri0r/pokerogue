"""
Feature name table for the 9,875-dim RL observation vector.

Provides a human-readable name for every dimension written by
``encode_observation()`` in observation.py. The name list mirrors the EXACT
write order of the encoders:

  Pokemon blocks:     12 x 771 = 9,252   (_encode_pokemon / _encode_move)
  Field state:        94                 (_encode_field)
  Battle meta:        40                 (_encode_battle)
  Modifier phase:     225                (_encode_modifier)
  Modifier inventory: 220                (_encode_modifier_inventory)
  Derived fields:     28                 (_encode_derived_fields)
  Phase indicator:    16                 (_encode_phase)
  Total:              9,875

Usage:
    from rl.feature_names import FEATURE_NAMES, ONE_HOT_GROUPS
    from rl.feature_names import dim_to_name, name_to_dim, block_of

    dim_to_name(0)                      # 'player_0/valid'
    name_to_dim('field/is_double_battle')
    block_of(9252)                      # ('field', 0)

ONE_HOT_GROUPS is a list of (start_index, size, label) for every genuine
one-hot range (type/status/tera one-hots per slot, move type/category/target
one-hots per move, weather/terrain, battle type, reward tier, phase
indicator). Multi-hot banks (volatile tags, arena tags) are NOT included.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .enums import (
    ARENA_TAG_ORDER,
    CURATED_VOLATILE_TAGS,
    PHASE_ID_TO_STR,
    POKEMON_SLOT_KEYS,
)
from .observation import (
    ABILITY_FEATURE_DIM,
    BATTLE_META_DIM,
    DERIVED_FIELDS_DIM,
    FIELD_STATE_DIM,
    HELD_ITEM_SLOT_DIM,
    MAX_HELD_ITEMS_ENCODED,
    MAX_MOVES,
    MAX_REWARD_OPTIONS,
    MAX_SHOP_OPTIONS_ENCODED,
    MODIFIER_FEATURE_DIM,
    MODIFIER_INVENTORY_DIM,
    MODIFIER_PHASE_DIM,
    MOVE_BLOCK_DIM,
    NUM_ARENA_TAG_TYPES,
    NUM_BATTLE_TYPES,
    NUM_CURATED_TAGS,
    NUM_MODIFIER_TIERS,
    NUM_MOVE_CATEGORIES,
    NUM_POKEMON_TYPES,
    NUM_STATUS_EFFECTS,
    NUM_TERRAIN_TYPES,
    NUM_WEATHER_TYPES,
    OBSERVATION_DIM,
    PHASE_INDICATOR_DIM,
    POKEMON_BLOCK_DIM,
    REWARD_OPTION_DIM,
    SHOP_OPTION_DIM,
    TOTAL_POKEMON_SLOTS,
    _ENEMY_MOD_IDS,
    _KEY_ARENA_TAGS,
    _PARTY_FLAG_IDS,
)

# ─── Label vocabularies (mirror state_schema.py orderings) ─────────────────

# base_stats / stats lists are ordered [HP, ATK, DEF, SPATK, SPDEF, SPD]
_STAT6_LABELS = ["hp", "atk", "def", "spatk", "spdef", "spd"]

# stat_stages is ordered [ATK, DEF, SPATK, SPDEF, SPD, ACC, EVA]
_STAT7_LABELS = ["atk", "def", "spatk", "spdef", "spd", "acc", "eva"]

# nature_multipliers / computed stats[1:6] cover the 5 effective stats
_STAT5_LABELS = ["atk", "def", "spatk", "spdef", "spd"]

# Pokeball counts written in _encode_battle order
_POKEBALL_LABELS = ["pokeball", "great_ball", "ultra_ball", "rogue_ball", "master_ball"]

# 20-dim modifier feature vector (see _encode_modifier_features_vec):
#   0-7  static category flags, 8-11 dynamic target params,
#   12-16 static effect params, 17-19 dynamic stack/duration
MODIFIER_FEATURE_LABELS = [
    "is_damage_boost",      # 0
    "is_stat_boost",        # 1
    "is_healing",           # 2
    "is_survival",          # 3
    "is_speed_priority",    # 4
    "is_status_effect",     # 5
    "is_economy",           # 6
    "is_berry",             # 7
    "type_id",              # 8  dynamic: type_id / 18
    "stat_id",              # 9  dynamic: stat_id / 7
    "status_effect",        # 10 dynamic: status_effect / 7
    "berry_type",           # 11 dynamic: berry_type / 12
    "boost_magnitude",      # 12
    "proc_chance_base",     # 13
    "is_per_turn",          # 14
    "is_on_hit",            # 15
    "is_on_faint",          # 16
    "stack_ratio",          # 17 dynamic: stack_count / max_stack_count
    "stack_count",          # 18 dynamic: stack_count / 10
    "battles_remaining",    # 19 dynamic: battles_remaining / 10
]

# 40-dim ability feature vector semantics (schema v1.0, see observation.py).
# Names use indexed form `ability_feat[i]`; this list documents index meaning.
ABILITY_FEATURE_LABELS = [
    "immune_ground", "immune_electric", "immune_water", "immune_fire",
    "immune_grass", "immunity_is_absorb", "non_se_immunity", "sturdy_endure",
    "mold_breaker", "ignorable", "blocks_priority", "priority_mod",
    "sets_weather", "sets_terrain", "atk_multiplier", "spatk_multiplier",
    "spd_multiplier", "phys_damage_reduction", "spec_damage_reduction",
    "se_damage_reduction", "type_boost_type", "type_boost_value",
    "intimidate", "trapping", "suppress_weather", "adaptability",
    "magic_bounce", "type_change_on_move", "normal_to_type", "contact_damage",
    "contact_status_type", "contact_status_chance", "blocks_crits",
    "stat_stage_multiplier", "unaware", "magic_guard", "regenerator",
    "tinted_lens", "parental_bond", "good_as_gold",
]

# Scalar move fields written after the category one-hot, in _encode_move order
_MOVE_SCALAR_FIELDS = [
    "power", "accuracy", "pp_ratio", "priority", "effect_chance",
    "drain_ratio", "heal_ratio", "is_multi_hit", "self_switch",
    "force_switch", "is_protect", "traps_target", "makes_contact",
    "is_usable",
    # secondary effect fields
    "status_effect", "stat_change_self_sum", "stat_change_target_sum",
    "recoil_ratio", "is_ohko", "is_charging", "is_sacrifice",
    "crit_stage_boost",
]

# Scalar move fields written after the target-class one-hot
_MOVE_POST_TARGET_FIELDS = ["ignores_protect", "is_sound_based"]

# v6 move semantic flags (36), in _encode_move write order
_MOVE_V6_FIELDS = [
    # Group 1: boolean attr flags (12)
    "can_flinch", "can_confuse", "is_recharge", "is_frenzy", "is_typeless",
    "creates_substitute", "suppresses_ability", "has_variable_power",
    "has_variable_type", "has_variable_category", "bypass_burn_penalty",
    "ignores_stat_stages",
    # Group 2: field control (4)
    "weather_change", "terrain_change", "sets_arena_tag", "removes_arena_tags",
    # Group 3: arena tag semantics (3)
    "sets_hazard", "sets_screen", "arena_tag_self_side",
    # Group 4: battler tag semantics (3)
    "applies_battler_tag", "applies_move_restriction",
    "applies_continuous_damage",
    # Group 5: fixed damage discrimination (4)
    "is_user_hp_damage", "is_target_half_hp", "is_counter_damage",
    "is_level_damage",
    # Group 6: additional strategic flags (2)
    "is_delayed_attack", "post_victory_stat_boost",
    # Group 7: missing MoveFlags (8)
    "is_wind_move", "is_reckless_move", "is_reflectable", "hides_user",
    "is_triage_move", "check_all_hits", "affected_by_gravity", "hides_target",
]

# v7 MoveAttr boolean flags (46), in _encode_move write order
_MOVE_V7_FIELDS = [
    # Group 8: item manipulation (3)
    "steals_item", "removes_item", "steals_berry",
    # Group 9: stat manipulation (8)
    "copies_stats", "inverts_stats", "resets_stats", "swaps_stat_stages",
    "steals_stat_boosts", "averages_stats", "swaps_single_stat",
    "shifts_own_stat",
    # Group 10: HP / PP / revival (3)
    "splits_hp", "reduces_pp", "revives_ally",
    # Group 11: move-calling (5)
    "copies_last_move", "calls_random_move", "calls_moveset_move",
    "copies_move_temp", "copies_move_perm",
    # Group 12: ability manipulation (5)
    "copies_ability", "swaps_abilities", "changes_ability", "gives_ability",
    "suppresses_if_acted",
    # Group 13: targeting & priority (4)
    "bypass_redirect", "forces_target_next", "forces_target_last",
    "has_conditional_priority",
    # Group 14: status & tag manipulation (5)
    "cures_party_status", "transfers_status", "heals_status",
    "removes_battler_tag", "removes_substitutes",
    # Group 15: transform & special moves (4)
    "transforms_into_target", "is_curse", "is_wish", "is_destiny_bond",
    # Group 16: field control (3)
    "swaps_arena_tags", "clears_weather", "clears_terrain",
    # Group 17: damage calc & misc (6)
    "has_variable_target", "resists_last_type", "has_variable_accuracy",
    "uses_alt_stat", "overrides_type_chart", "scatters_money",
]

# Non-move scalar pokemon fields between tera one-hot bank and the move blocks
_POKEMON_TAIL_FIELDS = [
    "other_tag_count",
    "is_boss", "boss_shield",
    "is_trapped", "is_grounded", "weight", "catch_rate", "is_fainted",
    "wave_turn_count", "damage_taken", "acted", "toxic_turn_count",
    "sleep_turns_remaining", "held_item_count",
    "species_id", "gender", "friendship", "move_queue_len", "hit_count",
    "ability_suppressed", "is_mega", "is_max", "move_effectiveness",
]

# Battle scalar fields after the pokeball counts, in _encode_battle order
_BATTLE_TAIL_FIELDS = [
    "player_alive_count", "enemy_alive_count", "tera_available", "can_run",
    "can_catch", "player_faints_battle", "enemy_faints_battle",
    "command_field_index", "biome_id", "escape_attempts", "battle_style",
    "time_of_day", "lock_modifier_tiers", "battle_spec", "game_mode",
    "trainer_specialty_type", "has_no_shop", "seen_enemy_count",
    "is_classic", "is_endless", "is_daily", "is_challenge",
    "has_mystery_encounters", "has_short_biomes", "has_random_biomes",
    "has_random_bosses", "inverse_battle",
]

# Active-field slots used by _encode_modifier_inventory / _encode_derived_fields
_ACTIVE_SLOT_KEYS = POKEMON_SLOT_KEYS[:4]   # player_0, player_1, enemy_0, enemy_1
_PLAYER_ACTIVE_KEYS = POKEMON_SLOT_KEYS[:2]  # player_0, player_1
_ENEMY_ACTIVE_KEYS = POKEMON_SLOT_KEYS[2:4]  # enemy_0, enemy_1

# Phase indicator labels in PhaseId order (16)
_PHASE_LABELS = [PHASE_ID_TO_STR[i] for i in range(PHASE_INDICATOR_DIM)]


def build_feature_names() -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Build (names, one_hot_groups) for the 9,875-dim observation.

    names[i] is the feature name of observation dimension i.
    one_hot_groups is a list of (start_index, size, label) tuples for every
    genuine one-hot range written by the encoders.
    """
    names: List[str] = []
    one_hot_groups: List[Tuple[int, int, str]] = []

    def add(name: str) -> None:
        names.append(name)

    def add_one_hot(label: str, size: int, item_labels=None) -> None:
        one_hot_groups.append((len(names), size, label))
        for i in range(size):
            item = item_labels[i] if item_labels is not None else i
            names.append(f"{label}[{item}]")

    # ── Move block (132 dims), mirrors _encode_move ──────────────────────
    def add_move(prefix: str) -> None:
        start = len(names)
        add(f"{prefix}/valid")
        add_one_hot(f"{prefix}/type_onehot", NUM_POKEMON_TYPES)
        add_one_hot(f"{prefix}/category_onehot", NUM_MOVE_CATEGORIES)
        for f in _MOVE_SCALAR_FIELDS:
            add(f"{prefix}/{f}")
        # 3-dim one-hot: [self_or_ally, single_enemy, multi_target_or_field]
        add_one_hot(f"{prefix}/target_class_onehot", 3)
        for f in _MOVE_POST_TARGET_FIELDS:
            add(f"{prefix}/{f}")
        for f in _MOVE_V6_FIELDS:
            add(f"{prefix}/{f}")
        for f in _MOVE_V7_FIELDS:
            add(f"{prefix}/{f}")
        assert len(names) - start == MOVE_BLOCK_DIM, (
            f"move block for {prefix} is {len(names) - start}, expected {MOVE_BLOCK_DIM}"
        )

    # ── Pokemon block (771 dims), mirrors _encode_pokemon ────────────────
    def add_pokemon(slot: str) -> None:
        start = len(names)
        add(f"{slot}/valid")
        add(f"{slot}/hp_ratio")
        add(f"{slot}/level")
        for s in _STAT6_LABELS:
            add(f"{slot}/base_stats[{s}]")
        for s in _STAT7_LABELS:
            add(f"{slot}/stat_stages[{s}]")
        add_one_hot(f"{slot}/type1_onehot", NUM_POKEMON_TYPES)
        add_one_hot(f"{slot}/type2_onehot", NUM_POKEMON_TYPES)
        add_one_hot(f"{slot}/status_onehot", NUM_STATUS_EFFECTS)
        for s in _STAT5_LABELS:
            add(f"{slot}/nature_mults[{s}]")
        for i in range(ABILITY_FEATURE_DIM):
            add(f"{slot}/ability_feat[{i}]")
        for i in range(ABILITY_FEATURE_DIM):
            add(f"{slot}/passive_feat[{i}]")
        add(f"{slot}/is_terastallized")
        add_one_hot(f"{slot}/tera_type_onehot", NUM_POKEMON_TYPES)
        for tag in CURATED_VOLATILE_TAGS:
            add(f"{slot}/volatile_tags[{tag}]")
        for f in _POKEMON_TAIL_FIELDS:
            add(f"{slot}/{f}")
        # computed stats: indices 1-5 of the [HP,ATK,DEF,SPATK,SPDEF,SPD] list
        for s in _STAT5_LABELS:
            add(f"{slot}/stats[{s}]")
        for mi in range(MAX_MOVES):
            add_move(f"{slot}/moves[{mi}]")
        assert len(names) - start == POKEMON_BLOCK_DIM, (
            f"pokemon block for {slot} is {len(names) - start}, expected {POKEMON_BLOCK_DIM}"
        )

    # ── Field block (94 dims), mirrors _encode_field ─────────────────────
    def add_field() -> None:
        start = len(names)
        add_one_hot("field/weather_onehot", NUM_WEATHER_TYPES)
        add("field/weather_turns_left")
        add_one_hot("field/terrain_onehot", NUM_TERRAIN_TYPES)
        add("field/terrain_turns_left")
        for tag in ARENA_TAG_ORDER:
            add(f"field/player_arena_tags[{tag}]")
        add("field/player_spikes_layers")
        add("field/player_toxic_spikes_layers")
        for tag in ARENA_TAG_ORDER:
            add(f"field/enemy_arena_tags[{tag}]")
        add("field/enemy_spikes_layers")
        add("field/enemy_toxic_spikes_layers")
        add("field/is_double_battle")
        add("field/trick_room_active")
        add("field/gravity_active")
        add("field/weather_is_permanent")
        add("field/weather_suppressed")
        add("field/terrain_is_permanent")
        # key arena tag remaining turns: sides [PLAYER, ENEMY] x 5 key tags
        for side in ["player", "enemy"]:
            for tag in _KEY_ARENA_TAGS:
                add(f"field/tag_turns/{side}[{tag}]")
        add("field/player_teras_used")
        assert len(names) - start == FIELD_STATE_DIM, (
            f"field block is {len(names) - start}, expected {FIELD_STATE_DIM}"
        )

    # ── Battle block (40 dims), mirrors _encode_battle ───────────────────
    def add_battle() -> None:
        start = len(names)
        add("battle/wave_index")
        add("battle/turn")
        add_one_hot("battle/battle_type_onehot", NUM_BATTLE_TYPES)
        add("battle/money_log")
        add("battle/score_log")
        for ball in _POKEBALL_LABELS:
            add(f"battle/pokeball_counts[{ball}]")
        for f in _BATTLE_TAIL_FIELDS:
            add(f"battle/{f}")
        assert len(names) - start == BATTLE_META_DIM, (
            f"battle block is {len(names) - start}, expected {BATTLE_META_DIM}"
        )

    # ── Modifier phase block (225 dims), mirrors _encode_modifier ────────
    def add_modifier_features(prefix: str) -> None:
        for f in MODIFIER_FEATURE_LABELS:
            add(f"{prefix}/feat[{f}]")

    def add_modifier_phase() -> None:
        start = len(names)
        add("shop/header/modifier_active")
        add("shop/header/can_reroll")
        add("shop/header/reroll_cost_ratio")
        for i in range(MAX_REWARD_OPTIONS):
            opt_start = len(names)
            add(f"shop/reward[{i}]/valid")
            add_one_hot(f"shop/reward[{i}]/tier_onehot", NUM_MODIFIER_TIERS)
            add(f"shop/reward[{i}]/is_pokemon_modifier")
            add_modifier_features(f"shop/reward[{i}]")
            assert len(names) - opt_start == REWARD_OPTION_DIM
        for i in range(MAX_SHOP_OPTIONS_ENCODED):
            opt_start = len(names)
            add(f"shop/shop[{i}]/valid")
            add(f"shop/shop[{i}]/cost_ratio")
            add(f"shop/shop[{i}]/affordable")
            add_modifier_features(f"shop/shop[{i}]")
            assert len(names) - opt_start == SHOP_OPTION_DIM
        assert len(names) - start == MODIFIER_PHASE_DIM, (
            f"modifier phase block is {len(names) - start}, expected {MODIFIER_PHASE_DIM}"
        )

    # ── Modifier inventory block (220), mirrors _encode_modifier_inventory ─
    def add_modifier_inventory() -> None:
        start = len(names)
        # Held items: 4 active slots x (count + 2 x 22) = 4 x 45 = 180
        for slot in _ACTIVE_SLOT_KEYS:
            add(f"inventory/held/{slot}/item_count")
            for i in range(MAX_HELD_ITEMS_ENCODED):
                item_start = len(names)
                add(f"inventory/held/{slot}/item[{i}]/valid")
                add_modifier_features(f"inventory/held/{slot}/item[{i}]")
                add(f"inventory/held/{slot}/item[{i}]/stack_ratio")
                assert len(names) - item_start == HELD_ITEM_SLOT_DIM
        # Party-wide modifiers (9): count + 8 boolean presence flags
        add("inventory/party/mod_count")
        for flag in _PARTY_FLAG_IDS:
            add(f"inventory/party/flag[{flag}]")
        # Lapsing modifiers (23): count + top-1 (valid + 20 + battles_remaining)
        add("inventory/lapsing/count")
        add("inventory/lapsing/top/valid")
        add_modifier_features("inventory/lapsing/top")
        add("inventory/lapsing/top/battles_remaining")
        # Enemy modifiers (8): count + 7 aggregate stacks
        add("inventory/enemy/mod_count")
        for eid in _ENEMY_MOD_IDS:
            add(f"inventory/enemy/stacks[{eid}]")
        assert len(names) - start == MODIFIER_INVENTORY_DIM, (
            f"modifier inventory block is {len(names) - start}, expected {MODIFIER_INVENTORY_DIM}"
        )

    # ── Derived fields block (28), mirrors _encode_derived_fields ────────
    def add_derived() -> None:
        start = len(names)
        # Type effectiveness: 2 players x 4 moves x 2 enemies = 16
        for p in _PLAYER_ACTIVE_KEYS:
            for mi in range(MAX_MOVES):
                for e in _ENEMY_ACTIVE_KEYS:
                    add(f"derived/type_eff/{p}_move{mi}_vs_{e}")
        # STAB indicators: 2 players x 4 moves = 8
        for p in _PLAYER_ACTIVE_KEYS:
            for mi in range(MAX_MOVES):
                add(f"derived/stab/{p}_move{mi}")
        # Speed ranks: 4 active slots
        for slot in _ACTIVE_SLOT_KEYS:
            add(f"derived/speed_rank/{slot}")
        assert len(names) - start == DERIVED_FIELDS_DIM, (
            f"derived block is {len(names) - start}, expected {DERIVED_FIELDS_DIM}"
        )

    # ═══ Walk the exact encode_observation() order ═══════════════════════
    for si, slot in enumerate(POKEMON_SLOT_KEYS):
        add_pokemon(slot)
        if si == 0:
            assert len(names) == POKEMON_BLOCK_DIM
    assert len(names) == TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM

    add_field()
    assert len(names) == TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM + FIELD_STATE_DIM

    add_battle()
    assert len(names) == (
        TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM + FIELD_STATE_DIM + BATTLE_META_DIM
    )

    add_modifier_phase()
    assert len(names) == (
        TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM + FIELD_STATE_DIM + BATTLE_META_DIM
        + MODIFIER_PHASE_DIM
    )

    add_modifier_inventory()
    assert len(names) == (
        TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM + FIELD_STATE_DIM + BATTLE_META_DIM
        + MODIFIER_PHASE_DIM + MODIFIER_INVENTORY_DIM
    )

    add_derived()
    assert len(names) == (
        TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM + FIELD_STATE_DIM + BATTLE_META_DIM
        + MODIFIER_PHASE_DIM + MODIFIER_INVENTORY_DIM + DERIVED_FIELDS_DIM
    )

    # Phase indicator (16), mirrors _encode_phase
    add_one_hot("phase_onehot", PHASE_INDICATOR_DIM, _PHASE_LABELS)
    assert len(names) == OBSERVATION_DIM, (
        f"total names {len(names)}, expected {OBSERVATION_DIM}"
    )

    return names, one_hot_groups


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL TABLES AND HARD CHECKS
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES, ONE_HOT_GROUPS = build_feature_names()

assert len(FEATURE_NAMES) == OBSERVATION_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected {OBSERVATION_DIM}"
)
assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES), "duplicate feature names"
assert NUM_CURATED_TAGS == len(CURATED_VOLATILE_TAGS)
assert NUM_ARENA_TAG_TYPES == len(ARENA_TAG_ORDER)
assert MODIFIER_FEATURE_DIM == len(MODIFIER_FEATURE_LABELS)
assert ABILITY_FEATURE_DIM == len(ABILITY_FEATURE_LABELS)
assert PHASE_INDICATOR_DIM == len(_PHASE_LABELS)
assert len(_MOVE_V6_FIELDS) == 36 and len(_MOVE_V7_FIELDS) == 46

# Block ranges: (label, start, size) in observation order
_POKEMON_TOTAL = TOTAL_POKEMON_SLOTS * POKEMON_BLOCK_DIM
BLOCK_RANGES: List[Tuple[str, int, int]] = [
    (slot, i * POKEMON_BLOCK_DIM, POKEMON_BLOCK_DIM)
    for i, slot in enumerate(POKEMON_SLOT_KEYS)
] + [
    ("field", _POKEMON_TOTAL, FIELD_STATE_DIM),
    ("battle", _POKEMON_TOTAL + FIELD_STATE_DIM, BATTLE_META_DIM),
    ("modifier_phase",
     _POKEMON_TOTAL + FIELD_STATE_DIM + BATTLE_META_DIM,
     MODIFIER_PHASE_DIM),
    ("modifier_inventory",
     _POKEMON_TOTAL + FIELD_STATE_DIM + BATTLE_META_DIM + MODIFIER_PHASE_DIM,
     MODIFIER_INVENTORY_DIM),
    ("derived",
     _POKEMON_TOTAL + FIELD_STATE_DIM + BATTLE_META_DIM + MODIFIER_PHASE_DIM
     + MODIFIER_INVENTORY_DIM,
     DERIVED_FIELDS_DIM),
    ("phase_indicator",
     _POKEMON_TOTAL + FIELD_STATE_DIM + BATTLE_META_DIM + MODIFIER_PHASE_DIM
     + MODIFIER_INVENTORY_DIM + DERIVED_FIELDS_DIM,
     PHASE_INDICATOR_DIM),
]
assert BLOCK_RANGES[-1][1] + BLOCK_RANGES[-1][2] == OBSERVATION_DIM

_NAME_TO_DIM: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}


def dim_to_name(i: int) -> str:
    """Return the feature name of observation dimension i."""
    if not 0 <= i < OBSERVATION_DIM:
        raise IndexError(f"dimension {i} out of range [0, {OBSERVATION_DIM})")
    return FEATURE_NAMES[i]


def name_to_dim(name: str) -> int:
    """Return the observation dimension index of a feature name."""
    return _NAME_TO_DIM[name]


def block_of(i: int) -> Tuple[str, int]:
    """Return (block_label, offset_within_block) for observation dimension i."""
    if not 0 <= i < OBSERVATION_DIM:
        raise IndexError(f"dimension {i} out of range [0, {OBSERVATION_DIM})")
    for label, start, size in BLOCK_RANGES:
        if start <= i < start + size:
            return label, i - start
    raise IndexError(f"dimension {i} not covered by any block")  # unreachable
