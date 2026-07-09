"""
Clean observation state: dataclasses, parser, and encoder for RL.

Converts the raw GameState JSON dict (364 fields, 35 TypedDicts) into typed
dataclasses with all strings mapped to unique integers, then encodes into a
fixed-size float32 observation vector compatible with spaces.ts (6,991 dims).

Layout (6,991 float32):
  Pokemon block:            771 dims x 12 slots = 9,252
  Field block:              94
  Battle block:             40
  Modifier phase block:     225
  Modifier inventory block: 220
  Derived fields block:     28
  Phase block:              16

Usage:
    from observation import parse_game_state, encode_observation, extract_action_mask

    state = parse_game_state(raw_json)  # dict -> CleanGameState
    obs = encode_observation(state)      # -> np.ndarray(6991, float32)
    mask = extract_action_mask(state)    # -> np.ndarray(58, bool)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

import numpy as np

from .enums import (
    ARENA_TAG_ORDER,
    CURATED_VOLATILE_TAGS,
    MODIFIER_TYPE_ID_TO_STR,
    PHASE_STR_TO_ID,
    POKEMON_SLOT_KEYS,
    lookup_arena_tag,
    lookup_battler_tag,
    lookup_modifier_class,
    lookup_modifier_type,
    lookup_positional_tag,
    lookup_target_kind,
)

# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION CONSTANTS (mirror spaces.ts)
# ═══════════════════════════════════════════════════════════════════════════

NUM_POKEMON_TYPES = 19
NUM_STATUS_EFFECTS = 8
NUM_WEATHER_TYPES = 10
NUM_TERRAIN_TYPES = 5
NUM_MOVE_CATEGORIES = 3
NUM_BATTLE_TYPES = 4
NUM_MODIFIER_TIERS = 6
NUM_POKEBALL_TYPES = 5  # Tracked in pokeball_counts (0-4)
NUM_CURATED_TAGS = 69   # v9: 7 TURN_END-transient tags cut (see OBS_V9_LAYOUT.md)
NUM_ARENA_TAG_TYPES = 28
MAX_MOVES = 4
MAX_PARTY_SIZE = 6
MAX_REWARD_OPTIONS = 3
MAX_SHOP_OPTIONS = 12
MAX_SHOP_OPTIONS_ENCODED = 12  # v9: all 12 shop options, natural (action-id) order
MAX_HELD_ITEMS_ENCODED = 2   # Top-N held items encoded per active slot

ABILITY_FEATURE_DIM = 40     # v3: semantic features per ability (replaces ability_id/310)
MODIFIER_FEATURE_DIM = 20    # v4: semantic features per modifier
MOVE_BLOCK_DIM = 60           # v9: compact evidence keep-list (was 136)
POKEMON_BLOCK_DIM = 513      # v9: 273 non-move (tags 69, +ai_type 3, +indicators 6) + 4*60 moves
FIELD_STATE_DIM = 102        # v9: +8 positional tags (Wish/Future Sight per side)
BATTLE_META_DIM = 40         # +9: game mode flags, inverse_battle
MODIFIER_PHASE_DIM = 363     # v9: header(3) + reward(3*28) + shop(12*23)
MODIFIER_INVENTORY_DIM = 220 # v4: held(4*45) + party(9) + lapsing(23) + enemy(8)
DERIVED_FIELDS_DIM = 28      # type effectiveness, STAB, speed ordering
LEARN_MOVE_BLOCK_DIM = 66    # v9: offered move (60) + learner party-index one-hot (6)
PHASE_INDICATOR_DIM = 16
TOTAL_POKEMON_SLOTS = 12
OBSERVATION_DIM = 6991       # 12*513 + 102 + 40 + 363 + 220 + 28 + 66 + 16
ACTION_SPACE_SIZE = 58

# Held item slot dims: valid(1) + features(20) + stack_ratio(1) = 22
HELD_ITEM_SLOT_DIM = 22
# Reward option dims: valid(1) + tier_onehot(6) + is_pokemon(1) + features(20) = 28
REWARD_OPTION_DIM = 28
# Shop option dims: valid(1) + cost_ratio(1) + affordable(1) + features(20) = 23
SHOP_OPTION_DIM = 23

# ═══════════════════════════════════════════════════════════════════════════
# PADDING LIMITS
# ═══════════════════════════════════════════════════════════════════════════

MAX_VOLATILE_TAGS = 16
MAX_HELD_ITEMS = 12
MAX_ARENA_TAGS = 16
MAX_POSITIONAL_TAGS = 4
MAX_ATTACKS_RECEIVED = 6
MAX_MOVE_HISTORY = 8
MAX_STAT_CHANGES = 4
MAX_PARTY_MODIFIERS = 20
MAX_LAPSING_MODIFIERS = 8
MAX_ENEMY_MODIFIERS = 10
MAX_CHALLENGES = 4


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ObsStatChange:
    stat_id: int = 0
    stages: int = 0
    self_target: bool = False
    chance: int = 0


@dataclass(slots=True)
class ObsMove:
    move_id: int = 0
    type: int = -1
    category: int = -1
    power: int = 0
    accuracy: int = 0
    priority: int = 0
    pp_max: int = 0
    pp_used: int = 0
    pp_remaining: int = 0
    pp_up: int = 0
    target: int = 0
    is_usable: bool = False
    makes_contact: bool = False
    is_sound_based: bool = False
    is_powder: bool = False
    is_punching: bool = False
    is_slicing: bool = False
    is_biting: bool = False
    is_ballistic: bool = False
    effect_chance: int = 0
    status_effect: int = 0
    stat_changes: List[ObsStatChange] = dc_field(default_factory=list)
    drain_ratio: float = 0.0
    recoil_ratio: float = 0.0
    heal_ratio: float = 0.0
    is_multi_hit: bool = False
    multi_hit_type: int = -1
    crit_stage_boost: int = 0
    is_charging: bool = False
    self_switch: bool = False
    force_switch: bool = False
    traps_target: bool = False
    is_protect: bool = False
    is_sacrifice: bool = False
    is_ohko: bool = False
    ignores_protect: bool = False
    ignores_abilities: bool = False
    ignores_substitute: bool = False
    fixed_damage: int = 0
    is_pulse: bool = False
    is_dance: bool = False
    # ── v6: Move semantic encoding (+36 fields) ──
    # Group 1: Boolean attr flags (12)
    can_flinch: bool = False
    can_confuse: bool = False
    is_recharge: bool = False
    is_frenzy: bool = False
    is_typeless: bool = False
    creates_substitute: bool = False
    suppresses_ability: bool = False
    has_variable_power: bool = False
    has_variable_type: bool = False
    has_variable_category: bool = False
    bypass_burn_penalty: bool = False
    ignores_stat_stages: bool = False
    # Group 2: Field control (4)
    weather_change: int = 0
    terrain_change: int = 0
    sets_arena_tag: bool = False
    removes_arena_tags: bool = False
    # Group 3: Arena tag semantics (3)
    sets_hazard: bool = False
    sets_screen: bool = False
    arena_tag_self_side: bool = False
    # Group 4: Battler tag semantics (3)
    applies_battler_tag: bool = False
    applies_move_restriction: bool = False
    applies_continuous_damage: bool = False
    # Group 5: Fixed damage discrimination (4)
    is_user_hp_damage: bool = False
    is_target_half_hp: bool = False
    is_counter_damage: bool = False
    is_level_damage: bool = False
    # Group 6: Additional strategic flags (2)
    is_delayed_attack: bool = False
    post_victory_stat_boost: bool = False
    # Group 7: Missing MoveFlags (8)
    is_wind_move: bool = False
    is_reckless_move: bool = False
    is_reflectable: bool = False
    hides_user: bool = False
    is_triage_move: bool = False
    check_all_hits: bool = False
    affected_by_gravity: bool = False
    hides_target: bool = False
    # ── v7: MoveAttr boolean flags (+46 fields) ──
    # Group 8: Item Manipulation (3)
    steals_item: bool = False
    removes_item: bool = False
    steals_berry: bool = False
    # Group 9: Stat Manipulation (8)
    copies_stats: bool = False
    inverts_stats: bool = False
    resets_stats: bool = False
    swaps_stat_stages: bool = False
    steals_stat_boosts: bool = False
    averages_stats: bool = False
    swaps_single_stat: bool = False
    shifts_own_stat: bool = False
    # Group 10: HP / PP / Revival (3)
    splits_hp: bool = False
    reduces_pp: bool = False
    revives_ally: bool = False
    # Group 11: Move-Calling (5)
    copies_last_move: bool = False
    calls_random_move: bool = False
    calls_moveset_move: bool = False
    copies_move_temp: bool = False
    copies_move_perm: bool = False
    # Group 12: Ability Manipulation (5)
    copies_ability: bool = False
    swaps_abilities: bool = False
    changes_ability: bool = False
    gives_ability: bool = False
    suppresses_if_acted: bool = False
    # Group 13: Targeting & Priority (4)
    bypass_redirect: bool = False
    forces_target_next: bool = False
    forces_target_last: bool = False
    has_conditional_priority: bool = False
    # Group 14: Status & Tag Manipulation (5)
    cures_party_status: bool = False
    transfers_status: bool = False
    heals_status: bool = False
    removes_battler_tag: bool = False
    removes_substitutes: bool = False
    # Group 15: Transform & Special Moves (4)
    transforms_into_target: bool = False
    is_curse: bool = False
    is_wish: bool = False
    is_destiny_bond: bool = False
    # Group 16: Field Control (3)
    swaps_arena_tags: bool = False
    clears_weather: bool = False
    clears_terrain: bool = False
    # Group 17: Damage Calc & Misc (6)
    has_variable_target: bool = False
    resists_last_type: bool = False
    has_variable_accuracy: bool = False
    uses_alt_stat: bool = False
    overrides_type_chart: bool = False
    scatters_money: bool = False
    # Group 18: v8 survival / HP-relative semantics (4)
    survives_at_1hp: bool = False
    matches_user_hp: bool = False
    hp_cost_stat_boost: bool = False
    hits_semi_invulnerable: bool = False


@dataclass(slots=True)
class ObsVolatileTag:
    tag_type_id: int = 0       # from lookup_battler_tag
    tag_type_str: str = ""     # original string for curated matching
    turn_count: int = 0
    source_id: int = -1
    source_move: int = -1
    substitute_hp: int = -1
    stockpile_count: int = -1
    encore_move_id: int = -1
    disabled_move_id: int = -1
    type_boost_type: int = -1
    type_boost_value: float = 0.0
    crit_boost_stages: int = -1
    gorilla_tactics_move_id: int = -1
    highest_stat_boost_stat: int = -1
    highest_stat_boost_multiplier: float = 0.0
    supreme_overlord_faint_count: int = -1
    autotomize_count: int = -1


@dataclass(slots=True)
class ObsQueuedMove:
    move_id: int = 0
    targets: List[int] = dc_field(default_factory=list)
    use_mode: int = 0
    result: int = -1


@dataclass(slots=True)
class ObsAttackReceived:
    source_battler_index: int = -1
    source_id: int = -1
    move_id: int = 0
    damage: int = 0
    critical: bool = False
    result: int = 0


@dataclass(slots=True)
class ObsHeldItem:
    modifier_class_id: int = 0
    modifier_type_id: int = 0
    stack_count: int = 0
    max_stack_count: int = 0
    is_transferable: bool = False
    type_id: int = -1
    stat_id: int = -1
    status_effect: int = -1
    berry_type: int = -1
    consumed: bool = False
    stat_modifier: int = 0
    form_change_item: int = -1
    form_change_active: bool = False


@dataclass(slots=True)
class ObsTurnData:
    damage_taken: int = 0
    total_damage_dealt: int = 0
    attacks_received: List[ObsAttackReceived] = dc_field(default_factory=list)
    order: int = 0
    hit_count: int = 0
    acted: bool = False
    switched_in_this_turn: bool = False
    stat_stages_increased: bool = False
    stat_stages_decreased: bool = False
    berries_eaten: List[int] = dc_field(default_factory=list)
    move_effectiveness: float = 0.0
    hits_left: int = 0
    single_hit_damage_dealt: int = 0


@dataclass(slots=True)
class ObsBattleData:
    hit_count: int = 0
    has_eaten_berry: bool = False
    berries_eaten: List[int] = dc_field(default_factory=list)
    abilities_applied: List[int] = dc_field(default_factory=list)


@dataclass(slots=True)
class ObsPokemon:
    valid: bool = False
    species_id: int = 0
    form_index: int = 0
    level: int = 0
    gender: int = -1
    friendship: int = 0
    shiny: bool = False
    variant: int = 0
    hp: int = 0
    max_hp: int = 0
    hp_ratio: float = 0.0
    base_stats: List[int] = dc_field(default_factory=lambda: [0] * 6)
    ivs: List[int] = dc_field(default_factory=lambda: [0] * 6)
    stats: List[int] = dc_field(default_factory=lambda: [0] * 6)
    stat_stages: List[int] = dc_field(default_factory=lambda: [0] * 7)
    status_effect: int = 0
    toxic_turn_count: int = 0
    sleep_turns_remaining: int = 0
    types: List[int] = dc_field(default_factory=list)
    tera_type: int = -1
    is_terastallized: bool = False
    added_type: int = -1
    ability_id: int = 0
    passive_ability_id: int = 0
    has_passive: bool = False
    ability_suppressed: bool = False
    ability_revealed: bool = False
    # v9 fog-of-war inputs
    was_seen: bool = True
    move_known: List[bool] = dc_field(default_factory=list)
    nature: int = 0
    nature_multipliers: List[float] = dc_field(default_factory=lambda: [1.0] * 5)
    moves: List[ObsMove] = dc_field(default_factory=list)
    move_history: List[ObsQueuedMove] = dc_field(default_factory=list)
    pokeball: int = 0
    volatile_tags: List[ObsVolatileTag] = dc_field(default_factory=list)
    is_boss: bool = False
    boss_segments: int = 0
    boss_segment_index: int = 0
    ai_type: int = -1  # -1 = unknown/absent (all-zero one-hot); RANDOM=0
    is_fusion: bool = False
    fusion_species_id: int = -1
    is_on_field: bool = False
    is_player: bool = False
    battler_index: int = -1
    field_index: int = -1
    held_items: List[ObsHeldItem] = dc_field(default_factory=list)
    move_queue: List[ObsQueuedMove] = dc_field(default_factory=list)
    wave_turn_count: int = 0
    is_fainted: bool = False
    is_active: bool = False
    is_trapped: bool = False
    is_grounded: bool = True
    transform_species_id: int = -1
    illusion_species_id: int = -1
    attacks_received: List[ObsAttackReceived] = dc_field(default_factory=list)
    turn_data: ObsTurnData = dc_field(default_factory=ObsTurnData)
    battle_data: ObsBattleData = dc_field(default_factory=ObsBattleData)
    weight: float = 0.0
    catch_rate: int = 0
    base_total: int = 0
    stellar_types_boosted: List[int] = dc_field(default_factory=list)
    berries_eaten_last: List[int] = dc_field(default_factory=list)
    exp_to_next_level: int = 0
    luck: int = 0
    endured_this_wave: bool = False
    is_mega: bool = False
    is_max: bool = False


@dataclass(slots=True)
class ObsArenaTag:
    tag_type_id: int = 0
    tag_type_str: str = ""
    side: int = 0
    turn_count: int = 0
    layers: int = 0
    source_id: int = -1


@dataclass(slots=True)
class ObsPositionalTag:
    tag_type_id: int = 0
    countdown: int = 0
    target_index: int = -1
    source_id: int = -1
    move_id: int = -1
    heal_hp: int = -1


@dataclass(slots=True)
class ObsField:
    biome_id: int = 0
    weather_type: int = 0
    weather_turns_left: int = 0
    weather_is_permanent: bool = False
    weather_suppressed: bool = False
    terrain_type: int = 0
    terrain_turns_left: int = 0
    terrain_is_permanent: bool = False
    player_teras_used: int = 0
    arena_tags: List[ObsArenaTag] = dc_field(default_factory=list)
    positional_tags: List[ObsPositionalTag] = dc_field(default_factory=list)
    is_double_battle: bool = False
    trick_room_active: bool = False
    gravity_active: bool = False
    ignore_abilities: bool = False
    player_spikes_layers: int = 0
    player_toxic_spikes_layers: int = 0
    player_stealth_rock: bool = False
    player_sticky_web: bool = False
    enemy_spikes_layers: int = 0
    enemy_toxic_spikes_layers: int = 0
    enemy_stealth_rock: bool = False
    enemy_sticky_web: bool = False


@dataclass(slots=True)
class ObsPokeballCounts:
    pokeball: int = 0
    great_ball: int = 0
    ultra_ball: int = 0
    rogue_ball: int = 0
    master_ball: int = 0


@dataclass(slots=True)
class ObsTrainerInfo:
    trainer_type: int = 0
    is_double: bool = False
    is_boss: bool = False
    party_template_size: int = 0
    specialty_type: int = -1
    tera_mode: int = -1


@dataclass(slots=True)
class ObsChallengeInfo:
    challenge_type: int = 0
    value: int = 0
    severity: int = 0


@dataclass(slots=True)
class ObsPartyModifier:
    modifier_class_id: int = 0
    modifier_type_id: int = 0
    stack_count: int = 0
    max_stack_count: int = 0
    type_id: int = -1
    stat_id: int = -1
    status_effect: int = -1


@dataclass(slots=True)
class ObsLapsingModifier:
    modifier_class_id: int = 0
    modifier_type_id: int = 0
    stack_count: int = 0
    battles_remaining: int = 0
    stat_id: int = -1
    boost: float = 0.0


@dataclass(slots=True)
class ObsModifiers:
    held_items: Dict[int, List[ObsHeldItem]] = dc_field(default_factory=dict)
    party_modifiers: List[ObsPartyModifier] = dc_field(default_factory=list)
    lapsing_modifiers: List[ObsLapsingModifier] = dc_field(default_factory=list)
    enemy_modifiers: List[ObsPartyModifier] = dc_field(default_factory=list)


@dataclass(slots=True)
class ObsRewardOption:
    index: int = 0
    tier: int = 0
    upgrade_count: int = 0
    modifier_type_id: int = 0
    modifier_class_id: int = 0
    target_kind_id: int = 0
    is_pokemon_modifier: bool = False
    type_id: int = -1
    stat_id: int = -1
    status_effect: int = -1
    berry_type: int = -1


@dataclass(slots=True)
class ObsShopOption:
    index: int = 0
    cost: int = 0
    tier: int = 0
    modifier_type_id: int = 0
    modifier_class_id: int = 0
    target_kind_id: int = 0
    affordable: bool = False
    type_id: int = -1
    stat_id: int = -1
    status_effect: int = -1
    berry_type: int = -1


@dataclass(slots=True)
class ObsShop:
    reward_options: List[ObsRewardOption] = dc_field(default_factory=list)
    shop_options: List[ObsShopOption] = dc_field(default_factory=list)
    can_reroll: bool = False
    reroll_cost: int = 0
    money: int = 0


@dataclass(slots=True)
class ObsBattle:
    biome_id: int = 0
    wave_index: int = 0
    turn: int = 0
    battle_type: int = 0
    battle_spec: int = 0
    is_double: bool = False
    escape_attempts: int = 0
    player_alive_count: int = 0
    enemy_alive_count: int = 0
    player_faints_battle: int = 0
    enemy_faints_battle: int = 0
    last_move_id: int = -1
    money: int = 0
    score: int = 0
    pokeball_counts: ObsPokeballCounts = dc_field(default_factory=ObsPokeballCounts)
    can_run: bool = False
    can_catch: bool = False
    tera_available: bool = False
    game_mode: int = 0
    trainer: Optional[ObsTrainerInfo] = None
    challenges: List[ObsChallengeInfo] = dc_field(default_factory=list)
    battle_style: int = 0
    time_of_day: int = 0
    player_faints_biome: int = 0
    money_scattered: int = 0
    lock_modifier_tiers: bool = False
    reroll_count: int = 0
    failed_run_away: bool = False
    has_no_shop: bool = False
    has_trainers: bool = False
    is_spliced_only: bool = False
    seen_enemy_count: int = 0
    enemy_switch_counter: int = 0
    offset_gym: bool = False
    is_classic: bool = False
    is_endless: bool = False
    is_daily: bool = False
    is_challenge: bool = False
    has_mystery_encounters: bool = False
    has_short_biomes: bool = False
    has_random_biomes: bool = False
    has_random_bosses: bool = False
    inverse_battle: bool = False


@dataclass(slots=True)
class ObsPhase:
    current_phase: str = "unknown"
    current_phase_id: int = -1
    command_field_index: int = -1
    action_mask: List[bool] = dc_field(default_factory=lambda: [False] * ACTION_SPACE_SIZE)
    valid_actions: List[int] = dc_field(default_factory=list)
    learn_move_id: int = -1
    # v9: full feature payload for the offered move + who is learning
    learn_move_stats: Optional[ObsMove] = None
    learn_move_party_index: int = -1
    # Biome names offered by the select_biome phase (not encoded; decision metadata)
    biome_options: List[str] = dc_field(default_factory=list)
    mystery_option_count: int = -1
    is_game_over: bool = False
    is_victory: bool = False


@dataclass(slots=True)
class CleanGameState:
    pokemon: List[ObsPokemon] = dc_field(default_factory=list)
    field: ObsField = dc_field(default_factory=ObsField)
    battle: ObsBattle = dc_field(default_factory=ObsBattle)
    modifiers: ObsModifiers = dc_field(default_factory=ObsModifiers)
    phase: ObsPhase = dc_field(default_factory=ObsPhase)
    shop: Optional[ObsShop] = None
    step: int = 0
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PARSING: dict -> dataclasses
# ═══════════════════════════════════════════════════════════════════════════

def _g(d: dict, k: str, default=0):
    """Safe get with default."""
    v = d.get(k)
    return v if v is not None else default


def _gb(d: dict, k: str, default: bool = False) -> bool:
    v = d.get(k)
    return bool(v) if v is not None else default


def _gl(d: dict, k: str) -> list:
    v = d.get(k)
    return v if isinstance(v, list) else []


def _gd(d: dict, k: str) -> dict:
    v = d.get(k)
    return v if isinstance(v, dict) else {}


def _parse_stat_change(d: dict) -> ObsStatChange:
    return ObsStatChange(
        stat_id=_g(d, "stat_id"),
        stages=_g(d, "stages"),
        self_target=_gb(d, "self_target"),
        chance=_g(d, "chance", 100),
    )


def _parse_move(d: dict) -> ObsMove:
    if not d:
        return ObsMove()
    stat_changes = [_parse_stat_change(sc) for sc in _gl(d, "stat_changes")]
    return ObsMove(
        move_id=_g(d, "move_id"),
        type=_g(d, "type", -1),
        category=_g(d, "category", -1),
        power=_g(d, "power"),
        accuracy=_g(d, "accuracy"),
        priority=_g(d, "priority"),
        pp_max=_g(d, "pp_max"),
        pp_used=_g(d, "pp_used"),
        pp_remaining=_g(d, "pp_remaining"),
        pp_up=_g(d, "pp_up"),
        target=_g(d, "target"),
        is_usable=_gb(d, "is_usable"),
        makes_contact=_gb(d, "makes_contact"),
        is_sound_based=_gb(d, "is_sound_based"),
        is_powder=_gb(d, "is_powder"),
        is_punching=_gb(d, "is_punching"),
        is_slicing=_gb(d, "is_slicing"),
        is_biting=_gb(d, "is_biting"),
        is_ballistic=_gb(d, "is_ballistic"),
        effect_chance=_g(d, "effect_chance"),
        status_effect=_g(d, "status_effect"),
        stat_changes=stat_changes,
        drain_ratio=float(_g(d, "drain_ratio", 0.0)),
        recoil_ratio=float(_g(d, "recoil_ratio", 0.0)),
        heal_ratio=float(_g(d, "heal_ratio", 0.0)),
        is_multi_hit=_gb(d, "is_multi_hit"),
        multi_hit_type=_g(d, "multi_hit_type", -1),
        crit_stage_boost=_g(d, "crit_stage_boost"),
        is_charging=_gb(d, "is_charging"),
        self_switch=_gb(d, "self_switch"),
        force_switch=_gb(d, "force_switch"),
        traps_target=_gb(d, "traps_target"),
        is_protect=_gb(d, "is_protect"),
        is_sacrifice=_gb(d, "is_sacrifice"),
        is_ohko=_gb(d, "is_ohko"),
        ignores_protect=_gb(d, "ignores_protect"),
        ignores_abilities=_gb(d, "ignores_abilities"),
        ignores_substitute=_gb(d, "ignores_substitute"),
        fixed_damage=_g(d, "fixed_damage"),
        is_pulse=_gb(d, "is_pulse"),
        is_dance=_gb(d, "is_dance"),
        # ── v6: Move semantic encoding (+36 fields) ──
        can_flinch=_gb(d, "can_flinch"),
        can_confuse=_gb(d, "can_confuse"),
        is_recharge=_gb(d, "is_recharge"),
        is_frenzy=_gb(d, "is_frenzy"),
        is_typeless=_gb(d, "is_typeless"),
        creates_substitute=_gb(d, "creates_substitute"),
        suppresses_ability=_gb(d, "suppresses_ability"),
        has_variable_power=_gb(d, "has_variable_power"),
        has_variable_type=_gb(d, "has_variable_type"),
        has_variable_category=_gb(d, "has_variable_category"),
        bypass_burn_penalty=_gb(d, "bypass_burn_penalty"),
        ignores_stat_stages=_gb(d, "ignores_stat_stages"),
        weather_change=_g(d, "weather_change"),
        terrain_change=_g(d, "terrain_change"),
        sets_arena_tag=_gb(d, "sets_arena_tag"),
        removes_arena_tags=_gb(d, "removes_arena_tags"),
        sets_hazard=_gb(d, "sets_hazard"),
        sets_screen=_gb(d, "sets_screen"),
        arena_tag_self_side=_gb(d, "arena_tag_self_side"),
        applies_battler_tag=_gb(d, "applies_battler_tag"),
        applies_move_restriction=_gb(d, "applies_move_restriction"),
        applies_continuous_damage=_gb(d, "applies_continuous_damage"),
        is_user_hp_damage=_gb(d, "is_user_hp_damage"),
        is_target_half_hp=_gb(d, "is_target_half_hp"),
        is_counter_damage=_gb(d, "is_counter_damage"),
        is_level_damage=_gb(d, "is_level_damage"),
        is_delayed_attack=_gb(d, "is_delayed_attack"),
        post_victory_stat_boost=_gb(d, "post_victory_stat_boost"),
        is_wind_move=_gb(d, "is_wind_move"),
        is_reckless_move=_gb(d, "is_reckless_move"),
        is_reflectable=_gb(d, "is_reflectable"),
        hides_user=_gb(d, "hides_user"),
        is_triage_move=_gb(d, "is_triage_move"),
        check_all_hits=_gb(d, "check_all_hits"),
        affected_by_gravity=_gb(d, "affected_by_gravity"),
        hides_target=_gb(d, "hides_target"),
        # ── v7: MoveAttr boolean flags (+46 fields) ──
        steals_item=_gb(d, "steals_item"),
        removes_item=_gb(d, "removes_item"),
        steals_berry=_gb(d, "steals_berry"),
        copies_stats=_gb(d, "copies_stats"),
        inverts_stats=_gb(d, "inverts_stats"),
        resets_stats=_gb(d, "resets_stats"),
        swaps_stat_stages=_gb(d, "swaps_stat_stages"),
        steals_stat_boosts=_gb(d, "steals_stat_boosts"),
        averages_stats=_gb(d, "averages_stats"),
        swaps_single_stat=_gb(d, "swaps_single_stat"),
        shifts_own_stat=_gb(d, "shifts_own_stat"),
        splits_hp=_gb(d, "splits_hp"),
        reduces_pp=_gb(d, "reduces_pp"),
        revives_ally=_gb(d, "revives_ally"),
        copies_last_move=_gb(d, "copies_last_move"),
        calls_random_move=_gb(d, "calls_random_move"),
        calls_moveset_move=_gb(d, "calls_moveset_move"),
        copies_move_temp=_gb(d, "copies_move_temp"),
        copies_move_perm=_gb(d, "copies_move_perm"),
        copies_ability=_gb(d, "copies_ability"),
        swaps_abilities=_gb(d, "swaps_abilities"),
        changes_ability=_gb(d, "changes_ability"),
        gives_ability=_gb(d, "gives_ability"),
        suppresses_if_acted=_gb(d, "suppresses_if_acted"),
        bypass_redirect=_gb(d, "bypass_redirect"),
        forces_target_next=_gb(d, "forces_target_next"),
        forces_target_last=_gb(d, "forces_target_last"),
        has_conditional_priority=_gb(d, "has_conditional_priority"),
        cures_party_status=_gb(d, "cures_party_status"),
        transfers_status=_gb(d, "transfers_status"),
        heals_status=_gb(d, "heals_status"),
        removes_battler_tag=_gb(d, "removes_battler_tag"),
        removes_substitutes=_gb(d, "removes_substitutes"),
        transforms_into_target=_gb(d, "transforms_into_target"),
        is_curse=_gb(d, "is_curse"),
        is_wish=_gb(d, "is_wish"),
        is_destiny_bond=_gb(d, "is_destiny_bond"),
        swaps_arena_tags=_gb(d, "swaps_arena_tags"),
        clears_weather=_gb(d, "clears_weather"),
        clears_terrain=_gb(d, "clears_terrain"),
        has_variable_target=_gb(d, "has_variable_target"),
        resists_last_type=_gb(d, "resists_last_type"),
        has_variable_accuracy=_gb(d, "has_variable_accuracy"),
        uses_alt_stat=_gb(d, "uses_alt_stat"),
        overrides_type_chart=_gb(d, "overrides_type_chart"),
        scatters_money=_gb(d, "scatters_money"),
        survives_at_1hp=_gb(d, "survives_at_1hp"),
        matches_user_hp=_gb(d, "matches_user_hp"),
        hp_cost_stat_boost=_gb(d, "hp_cost_stat_boost"),
        hits_semi_invulnerable=_gb(d, "hits_semi_invulnerable"),
    )


def _parse_volatile_tag(d: dict) -> ObsVolatileTag:
    tag_str = d.get("tag_type", "")
    return ObsVolatileTag(
        tag_type_id=lookup_battler_tag(tag_str),
        tag_type_str=tag_str,
        turn_count=_g(d, "turn_count"),
        source_id=_g(d, "source_id", -1),
        source_move=_g(d, "source_move", -1),
        substitute_hp=_g(d, "substitute_hp", -1),
        stockpile_count=_g(d, "stockpile_count", -1),
        encore_move_id=_g(d, "encore_move_id", -1),
        disabled_move_id=_g(d, "disabled_move_id", -1),
        type_boost_type=_g(d, "type_boost_type", -1),
        type_boost_value=float(_g(d, "type_boost_value", 0.0)),
        crit_boost_stages=_g(d, "crit_boost_stages", -1),
        gorilla_tactics_move_id=_g(d, "gorilla_tactics_move_id", -1),
        highest_stat_boost_stat=_g(d, "highest_stat_boost_stat", -1),
        highest_stat_boost_multiplier=float(_g(d, "highest_stat_boost_multiplier", 0.0)),
        supreme_overlord_faint_count=_g(d, "supreme_overlord_faint_count", -1),
        autotomize_count=_g(d, "autotomize_count", -1),
    )


def _parse_queued_move(d: dict) -> ObsQueuedMove:
    return ObsQueuedMove(
        move_id=_g(d, "move_id"),
        targets=_gl(d, "targets"),
        use_mode=_g(d, "use_mode"),
        result=_g(d, "result", -1),
    )


def _parse_attack_received(d: dict) -> ObsAttackReceived:
    return ObsAttackReceived(
        source_battler_index=_g(d, "source_battler_index", -1),
        source_id=_g(d, "source_id", -1),
        move_id=_g(d, "move_id"),
        damage=_g(d, "damage"),
        critical=_gb(d, "critical"),
        result=_g(d, "result"),
    )


def _parse_held_item(d: dict) -> ObsHeldItem:
    return ObsHeldItem(
        modifier_class_id=lookup_modifier_class(d.get("modifier_class", "")),
        modifier_type_id=lookup_modifier_type(d.get("modifier_id", "")),
        stack_count=_g(d, "stack_count"),
        # Default 1 mirrors TS num(dict, "max_stack_count", 1); explicit 0 stays 0
        max_stack_count=_g(d, "max_stack_count", 1),
        is_transferable=_gb(d, "is_transferable"),
        type_id=_g(d, "type_id", -1),
        stat_id=_g(d, "stat_id", -1),
        status_effect=_g(d, "status_effect", -1),
        berry_type=_g(d, "berry_type", -1),
        consumed=_gb(d, "consumed"),
        stat_modifier=_g(d, "stat_modifier"),
        form_change_item=_g(d, "form_change_item", -1),
        form_change_active=_gb(d, "form_change_active"),
    )


def _parse_turn_data(d: dict) -> ObsTurnData:
    return ObsTurnData(
        damage_taken=_g(d, "damage_taken"),
        total_damage_dealt=_g(d, "total_damage_dealt"),
        attacks_received=[_parse_attack_received(a) for a in _gl(d, "attacks_received")],
        order=_g(d, "order"),
        hit_count=_g(d, "hit_count"),
        acted=_gb(d, "acted"),
        switched_in_this_turn=_gb(d, "switched_in_this_turn"),
        stat_stages_increased=_gb(d, "stat_stages_increased"),
        stat_stages_decreased=_gb(d, "stat_stages_decreased"),
        berries_eaten=_gl(d, "berries_eaten"),
        move_effectiveness=float(_g(d, "move_effectiveness", 0.0)),
        hits_left=_g(d, "hits_left"),
        single_hit_damage_dealt=_g(d, "single_hit_damage_dealt"),
    )


def _parse_battle_data(d: dict) -> ObsBattleData:
    return ObsBattleData(
        hit_count=_g(d, "hit_count"),
        has_eaten_berry=_gb(d, "has_eaten_berry"),
        berries_eaten=_gl(d, "berries_eaten"),
        abilities_applied=_gl(d, "abilities_applied"),
    )


def _parse_pokemon(d: dict) -> ObsPokemon:
    if not d or not d.get("valid"):
        return ObsPokemon()

    return ObsPokemon(
        valid=True,
        species_id=_g(d, "species_id"),
        form_index=_g(d, "form_index"),
        level=_g(d, "level"),
        gender=_g(d, "gender", -1),
        friendship=_g(d, "friendship"),
        shiny=_gb(d, "shiny"),
        variant=_g(d, "variant"),
        hp=_g(d, "hp"),
        max_hp=_g(d, "max_hp"),
        hp_ratio=float(_g(d, "hp_ratio", 0.0)),
        base_stats=_gl(d, "base_stats") or [0] * 6,
        ivs=_gl(d, "ivs") or [0] * 6,
        stats=_gl(d, "stats") or [0] * 6,
        stat_stages=_gl(d, "stat_stages") or [0] * 7,
        status_effect=_g(d, "status_effect"),
        toxic_turn_count=_g(d, "toxic_turn_count"),
        sleep_turns_remaining=_g(d, "sleep_turns_remaining"),
        types=_gl(d, "types"),
        tera_type=_g(d, "tera_type", -1),
        is_terastallized=_gb(d, "is_terastallized"),
        added_type=_g(d, "added_type", -1),
        ability_id=_g(d, "ability_id"),
        passive_ability_id=_g(d, "passive_ability_id"),
        has_passive=_gb(d, "has_passive"),
        ability_suppressed=_gb(d, "ability_suppressed"),
        ability_revealed=_gb(d, "ability_revealed"),
        # Default False to match the TS encoder (spaces.ts `bool()` defaults
        # false, then zeroes the whole enemy block under fog). Defaulting True
        # here made a fog run LEAK a valid enemy's block whenever its serialized
        # dict lacked a boolean was_seen, where TS would have hidden it.
        was_seen=_gb(d, "was_seen", False),
        move_known=[bool(x) for x in (d.get("move_known") or [])],
        nature=_g(d, "nature"),
        nature_multipliers=_gl(d, "nature_multipliers") or [1.0] * 5,
        moves=[_parse_move(m) for m in _gl(d, "moves")],
        move_history=[_parse_queued_move(m) for m in _gl(d, "move_history")],
        pokeball=_g(d, "pokeball"),
        volatile_tags=[_parse_volatile_tag(t) for t in _gl(d, "volatile_tags")],
        is_boss=_gb(d, "is_boss"),
        boss_segments=_g(d, "boss_segments"),
        boss_segment_index=_g(d, "boss_segment_index"),
        ai_type=_g(d, "ai_type", -1),
        is_fusion=_gb(d, "is_fusion"),
        fusion_species_id=_g(d, "fusion_species_id", -1),
        is_on_field=_gb(d, "is_on_field"),
        is_player=_gb(d, "is_player"),
        battler_index=_g(d, "battler_index", -1),
        field_index=_g(d, "field_index", -1),
        held_items=[_parse_held_item(i) for i in _gl(d, "held_items")],
        move_queue=[_parse_queued_move(m) for m in _gl(d, "move_queue")],
        wave_turn_count=_g(d, "wave_turn_count"),
        is_fainted=_gb(d, "is_fainted"),
        is_active=_gb(d, "is_active"),
        is_trapped=_gb(d, "is_trapped"),
        # Default False mirrors the TS encoder's bool() helper (missing -> 0);
        # state-builder always emits the key for real states
        is_grounded=_gb(d, "is_grounded", False),
        transform_species_id=_g(d, "transform_species_id", -1),
        illusion_species_id=_g(d, "illusion_species_id", -1),
        attacks_received=[_parse_attack_received(a) for a in _gl(d, "attacks_received")],
        turn_data=_parse_turn_data(_gd(d, "turn_data")),
        battle_data=_parse_battle_data(_gd(d, "battle_data")),
        weight=float(_g(d, "weight", 0.0)),
        catch_rate=_g(d, "catch_rate"),
        base_total=_g(d, "base_total"),
        stellar_types_boosted=_gl(d, "stellar_types_boosted"),
        berries_eaten_last=_gl(d, "berries_eaten_last"),
        exp_to_next_level=_g(d, "exp_to_next_level"),
        luck=_g(d, "luck"),
        endured_this_wave=_gb(d, "endured_this_wave"),
        is_mega=_gb(d, "is_mega"),
        is_max=_gb(d, "is_max"),
    )


def _parse_arena_tag(d: dict) -> ObsArenaTag:
    tag_str = d.get("tag_type", "")
    return ObsArenaTag(
        tag_type_id=lookup_arena_tag(tag_str),
        tag_type_str=tag_str,
        side=_g(d, "side"),
        turn_count=_g(d, "turn_count"),
        layers=_g(d, "layers", 1),
        source_id=_g(d, "source_id", -1),
    )


def _parse_positional_tag(d: dict) -> ObsPositionalTag:
    return ObsPositionalTag(
        tag_type_id=lookup_positional_tag(d.get("tag_type", "")),
        countdown=_g(d, "countdown"),
        target_index=_g(d, "target_index", -1),
        source_id=_g(d, "source_id", -1),
        move_id=_g(d, "move_id", -1),
        heal_hp=_g(d, "heal_hp", -1),
    )


def _parse_field(d: dict) -> ObsField:
    return ObsField(
        biome_id=_g(d, "biome_id"),
        weather_type=_g(d, "weather_type"),
        weather_turns_left=_g(d, "weather_turns_left"),
        weather_is_permanent=_gb(d, "weather_is_permanent"),
        weather_suppressed=_gb(d, "weather_suppressed"),
        terrain_type=_g(d, "terrain_type"),
        terrain_turns_left=_g(d, "terrain_turns_left"),
        terrain_is_permanent=_gb(d, "terrain_is_permanent"),
        player_teras_used=_g(d, "player_teras_used"),
        arena_tags=[_parse_arena_tag(t) for t in _gl(d, "arena_tags")],
        positional_tags=[_parse_positional_tag(t) for t in _gl(d, "positional_tags")],
        is_double_battle=_gb(d, "is_double_battle"),
        trick_room_active=_gb(d, "trick_room_active"),
        gravity_active=_gb(d, "gravity_active"),
        ignore_abilities=_gb(d, "ignore_abilities"),
        player_spikes_layers=_g(d, "player_spikes_layers"),
        player_toxic_spikes_layers=_g(d, "player_toxic_spikes_layers"),
        player_stealth_rock=_gb(d, "player_stealth_rock"),
        player_sticky_web=_gb(d, "player_sticky_web"),
        enemy_spikes_layers=_g(d, "enemy_spikes_layers"),
        enemy_toxic_spikes_layers=_g(d, "enemy_toxic_spikes_layers"),
        enemy_stealth_rock=_gb(d, "enemy_stealth_rock"),
        enemy_sticky_web=_gb(d, "enemy_sticky_web"),
    )


def _parse_pokeball_counts(d: dict) -> ObsPokeballCounts:
    return ObsPokeballCounts(
        pokeball=_g(d, "pokeball"),
        great_ball=_g(d, "great_ball"),
        ultra_ball=_g(d, "ultra_ball"),
        rogue_ball=_g(d, "rogue_ball"),
        master_ball=_g(d, "master_ball"),
    )


def _parse_trainer(d: dict) -> ObsTrainerInfo:
    return ObsTrainerInfo(
        trainer_type=_g(d, "trainer_type"),
        is_double=_gb(d, "is_double"),
        is_boss=_gb(d, "is_boss"),
        party_template_size=_g(d, "party_template_size"),
        specialty_type=_g(d, "specialty_type", -1),
        tera_mode=_g(d, "tera_mode", -1),
    )


def _parse_challenge(d: dict) -> ObsChallengeInfo:
    return ObsChallengeInfo(
        challenge_type=_g(d, "challenge_type"),
        value=_g(d, "value"),
        severity=_g(d, "severity"),
    )


def _parse_party_modifier(d: dict) -> ObsPartyModifier:
    return ObsPartyModifier(
        modifier_class_id=lookup_modifier_class(d.get("modifier_class", "")),
        modifier_type_id=lookup_modifier_type(d.get("modifier_id", "")),
        stack_count=_g(d, "stack_count"),
        # Default 1 mirrors TS num(dict, "max_stack_count", 1); explicit 0 stays 0
        max_stack_count=_g(d, "max_stack_count", 1),
        type_id=_g(d, "type_id", -1),
        stat_id=_g(d, "stat_id", -1),
        status_effect=_g(d, "status_effect", -1),
    )


def _parse_lapsing_modifier(d: dict) -> ObsLapsingModifier:
    return ObsLapsingModifier(
        modifier_class_id=lookup_modifier_class(d.get("modifier_class", "")),
        modifier_type_id=lookup_modifier_type(d.get("modifier_id", "")),
        stack_count=_g(d, "stack_count"),
        battles_remaining=_g(d, "battles_remaining"),
        stat_id=_g(d, "stat_id", -1),
        boost=float(_g(d, "boost", 0.0)),
    )


def _parse_battle(d: dict) -> ObsBattle:
    trainer_raw = d.get("trainer")
    trainer = _parse_trainer(trainer_raw) if isinstance(trainer_raw, dict) else None
    return ObsBattle(
        # Mirrors TS num(battle, "biome_id", num(battle, "biome_type")):
        # older states carried only biome_type
        biome_id=_g(d, "biome_id", _g(d, "biome_type", 0)),
        wave_index=_g(d, "wave_index"),
        turn=_g(d, "turn"),
        battle_type=_g(d, "battle_type"),
        battle_spec=_g(d, "battle_spec"),
        is_double=_gb(d, "is_double"),
        escape_attempts=_g(d, "escape_attempts"),
        player_alive_count=_g(d, "player_alive_count"),
        enemy_alive_count=_g(d, "enemy_alive_count"),
        player_faints_battle=_g(d, "player_faints_battle"),
        enemy_faints_battle=_g(d, "enemy_faints_battle"),
        last_move_id=_g(d, "last_move_id", -1),
        money=_g(d, "money"),
        score=_g(d, "score"),
        pokeball_counts=_parse_pokeball_counts(_gd(d, "pokeball_counts")),
        can_run=_gb(d, "can_run"),
        can_catch=_gb(d, "can_catch"),
        tera_available=_gb(d, "tera_available"),
        game_mode=_g(d, "game_mode"),
        trainer=trainer,
        challenges=[_parse_challenge(c) for c in _gl(d, "challenges")],
        battle_style=_g(d, "battle_style"),
        time_of_day=_g(d, "time_of_day"),
        player_faints_biome=_g(d, "player_faints_biome"),
        money_scattered=_g(d, "money_scattered"),
        lock_modifier_tiers=_gb(d, "lock_modifier_tiers"),
        reroll_count=_g(d, "reroll_count"),
        failed_run_away=_gb(d, "failed_run_away"),
        has_no_shop=_gb(d, "has_no_shop"),
        has_trainers=_gb(d, "has_trainers"),
        is_spliced_only=_gb(d, "is_spliced_only"),
        seen_enemy_count=_g(d, "seen_enemy_count"),
        enemy_switch_counter=_g(d, "enemy_switch_counter"),
        offset_gym=_gb(d, "offset_gym"),
        is_classic=_gb(d, "is_classic"),
        is_endless=_gb(d, "is_endless"),
        is_daily=_gb(d, "is_daily"),
        is_challenge=_gb(d, "is_challenge"),
        has_mystery_encounters=_gb(d, "has_mystery_encounters"),
        has_short_biomes=_gb(d, "has_short_biomes"),
        has_random_biomes=_gb(d, "has_random_biomes"),
        has_random_bosses=_gb(d, "has_random_bosses"),
        inverse_battle=_gb(d, "inverse_battle"),
    )


def _parse_modifiers(d: dict) -> ObsModifiers:
    held_raw = d.get("held_items", {})
    held: Dict[int, List[ObsHeldItem]] = {}
    if isinstance(held_raw, dict):
        for slot_str, items in held_raw.items():
            try:
                slot = int(slot_str)
            except (ValueError, TypeError):
                continue
            if isinstance(items, list):
                held[slot] = [_parse_held_item(i) for i in items if isinstance(i, dict)]
    return ObsModifiers(
        held_items=held,
        party_modifiers=[_parse_party_modifier(m) for m in _gl(d, "party_modifiers")],
        lapsing_modifiers=[_parse_lapsing_modifier(m) for m in _gl(d, "lapsing_modifiers")],
        enemy_modifiers=[_parse_party_modifier(m) for m in _gl(d, "enemy_modifiers")],
    )


def _parse_reward_option(d: dict) -> ObsRewardOption:
    return ObsRewardOption(
        index=_g(d, "index"),
        tier=_g(d, "tier"),
        upgrade_count=_g(d, "upgrade_count"),
        modifier_type_id=lookup_modifier_type(d.get("modifier_id", "")),
        modifier_class_id=lookup_modifier_class(d.get("modifier_class", "")),
        target_kind_id=lookup_target_kind(d.get("target_kind", "none")),
        is_pokemon_modifier=_gb(d, "is_pokemon_modifier"),
        type_id=_g(d, "type_id", -1),
        stat_id=_g(d, "stat_id", -1),
        status_effect=_g(d, "status_effect", -1),
        berry_type=_g(d, "berry_type", -1),
    )


def _parse_shop_option(d: dict) -> ObsShopOption:
    return ObsShopOption(
        index=_g(d, "index"),
        cost=_g(d, "cost"),
        tier=_g(d, "tier"),
        modifier_type_id=lookup_modifier_type(d.get("modifier_id", "")),
        modifier_class_id=lookup_modifier_class(d.get("modifier_class", "")),
        target_kind_id=lookup_target_kind(d.get("target_kind", "none")),
        affordable=_gb(d, "affordable"),
        type_id=_g(d, "type_id", -1),
        stat_id=_g(d, "stat_id", -1),
        status_effect=_g(d, "status_effect", -1),
        berry_type=_g(d, "berry_type", -1),
    )


def _parse_shop(d: dict) -> ObsShop:
    return ObsShop(
        reward_options=[_parse_reward_option(r) for r in _gl(d, "reward_options")],
        shop_options=[_parse_shop_option(s) for s in _gl(d, "shop_options")],
        can_reroll=_gb(d, "can_reroll"),
        reroll_cost=_g(d, "reroll_cost"),
        money=_g(d, "money"),
    )


def _parse_phase(d: dict) -> ObsPhase:
    phase_str = d.get("current_phase", "unknown")
    mask_raw = d.get("action_mask", [])
    mask = [bool(v) for v in mask_raw] if isinstance(mask_raw, list) else [False] * ACTION_SPACE_SIZE
    if len(mask) < ACTION_SPACE_SIZE:
        mask.extend([False] * (ACTION_SPACE_SIZE - len(mask)))
    return ObsPhase(
        current_phase=phase_str,
        current_phase_id=PHASE_STR_TO_ID.get(phase_str, -1),
        command_field_index=_g(d, "command_field_index", -1),
        action_mask=mask[:ACTION_SPACE_SIZE],
        valid_actions=_gl(d, "valid_actions"),
        learn_move_id=_g(d, "learn_move_id", -1),
        learn_move_stats=_parse_move(d["learn_move_stats"]) if isinstance(d.get("learn_move_stats"), dict) else None,
        learn_move_party_index=_g(d, "learn_move_party_index", -1),
        biome_options=[str(b) for b in _gl(d, "biome_options")],
        mystery_option_count=_g(d, "mystery_option_count", -1),
        is_game_over=_gb(d, "is_game_over"),
        is_victory=_gb(d, "is_victory"),
    )


def parse_game_state(raw: dict) -> CleanGameState:
    """Parse a raw GameState JSON dict into typed dataclasses.

    All string fields are mapped to integer IDs via enums.py lookup functions.
    Missing/null fields use safe defaults. Variable-length lists are kept as-is
    (padding happens during encoding).
    """
    pokemon = [_parse_pokemon(raw.get(key, {})) for key in POKEMON_SLOT_KEYS]
    field_state = _parse_field(raw.get("field", {}))
    battle = _parse_battle(raw.get("battle", {}))
    modifiers = _parse_modifiers(raw.get("modifiers", {}))
    phase = _parse_phase(raw.get("phase", {}))
    shop_raw = raw.get("shop")
    shop = _parse_shop(shop_raw) if isinstance(shop_raw, dict) else None
    return CleanGameState(
        pokemon=pokemon,
        field=field_state,
        battle=battle,
        modifiers=modifiers,
        phase=phase,
        shop=shop,
        step=_g(raw, "step"),
        timestamp=float(_g(raw, "timestamp", 0.0)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENCODING: CleanGameState -> float32 vector (6,991 dims)
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _write_one_hot(buf: np.ndarray, offset: int, size: int, index: int) -> None:
    if 0 <= index < size:
        buf[offset + index] = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# ABILITY FEATURES LOOKUP TABLE (311 entries x 40 features)
# Mirrors ABILITY_FEATURES in ability-features.ts exactly.
#
# Schema v1.0 (FROZEN) — 40 features per ability:
#  [0]  immune_ground        [1]  immune_electric     [2]  immune_water
#  [3]  immune_fire          [4]  immune_grass        [5]  immunity_is_absorb
#  [6]  non_se_immunity      [7]  sturdy_endure       [8]  mold_breaker
#  [9]  ignorable            [10] blocks_priority     [11] priority_mod
#  [12] sets_weather         [13] sets_terrain        [14] atk_multiplier
#  [15] spatk_multiplier     [16] spd_multiplier      [17] phys_damage_reduction
#  [18] spec_damage_reduction [19] se_damage_reduction [20] type_boost_type
#  [21] type_boost_value     [22] intimidate          [23] trapping
#  [24] suppress_weather     [25] adaptability        [26] magic_bounce
#  [27] type_change_on_move  [28] normal_to_type      [29] contact_damage
#  [30] contact_status_type  [31] contact_status_chance [32] blocks_crits
#  [33] stat_stage_multiplier [34] unaware            [35] magic_guard
#  [36] regenerator          [37] tinted_lens         [38] parental_bond
#  [39] good_as_gold
#
# Defaults: [14-16]=0.333 (neutral 1x/3), [17-19]=1.0 (no reduction), rest=0.0
# ═══════════════════════════════════════════════════════════════════════════

# Default feature vector: stat mults=0.333, damage reduction=1.0, rest=0
_ABILITY_DEFAULT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# fmt: off
_ABILITY_FEATURES: list = [
#   0 NONE
_ABILITY_DEFAULT,
#   1 STENCH
_ABILITY_DEFAULT,
#   2 DRIZZLE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   3 SPEED_BOOST
_ABILITY_DEFAULT,
#   4 BATTLE_ARMOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
#   5 STURDY
[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   6 DAMP
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   7 LIMBER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   8 SAND_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   9 STATIC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.429, 0.3, 0, 0, 0, 0, 0, 0, 0, 0],
#  10 VOLT_ABSORB
[0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  11 WATER_ABSORB
[0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  12 OBLIVIOUS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  13 CLOUD_NINE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  14 COMPOUND_EYES
_ABILITY_DEFAULT,
#  15 INSOMNIA
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  16 COLOR_CHANGE
_ABILITY_DEFAULT,
#  17 IMMUNITY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  18 FLASH_FIRE
[0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  19 SHIELD_DUST
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  20 OWN_TEMPO
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  21 SUCTION_CUPS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  22 INTIMIDATE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  23 SHADOW_TAG
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  24 ROUGH_SKIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  25 WONDER_GUARD
[0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  26 LEVITATE
[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  27 EFFECT_SPORE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143, 0.3, 0, 0, 0, 0, 0, 0, 0, 0],
#  28 SYNCHRONIZE
_ABILITY_DEFAULT,
#  29 CLEAR_BODY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  30 NATURAL_CURE
_ABILITY_DEFAULT,
#  31 LIGHTNING_ROD
[0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  32 SERENE_GRACE
_ABILITY_DEFAULT,
#  33 SWIFT_SWIM
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  34 CHLOROPHYLL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  35 ILLUMINATE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  36 TRACE
_ABILITY_DEFAULT,
#  37 HUGE_POWER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.667, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  38 POISON_POINT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143, 0.3, 0, 0, 0, 0, 0, 0, 0, 0],
#  39 INNER_FOCUS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  40 MAGMA_ARMOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  41 WATER_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  42 MAGNET_PULL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  43 SOUNDPROOF
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  44 RAIN_DISH
_ABILITY_DEFAULT,
#  45 SAND_STREAM
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  46 PRESSURE
_ABILITY_DEFAULT,
#  47 THICK_FAT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  48 EARLY_BIRD
_ABILITY_DEFAULT,
#  49 FLAME_BODY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.857, 0.3, 0, 0, 0, 0, 0, 0, 0, 0],
#  50 RUN_AWAY
_ABILITY_DEFAULT,
#  51 KEEN_EYE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  52 HYPER_CUTTER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  53 PICKUP
_ABILITY_DEFAULT,
#  54 TRUANT
_ABILITY_DEFAULT,
#  55 HUSTLE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  56 CUTE_CHARM
_ABILITY_DEFAULT,
#  57 PLUS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.5, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  58 MINUS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.5, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  59 FORECAST
_ABILITY_DEFAULT,
#  60 STICKY_HOLD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  61 SHED_SKIN
_ABILITY_DEFAULT,
#  62 GUTS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  63 MARVEL_SCALE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  64 LIQUID_OOZE
_ABILITY_DEFAULT,
#  65 OVERGROW
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.579, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  66 BLAZE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.474, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  67 TORRENT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.526, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  68 SWARM
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.316, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  69 ROCK_HEAD
_ABILITY_DEFAULT,
#  70 DROUGHT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  71 ARENA_TRAP
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  72 VITAL_SPIRIT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  73 WHITE_SMOKE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  74 PURE_POWER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.667, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  75 SHELL_ARMOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
#  76 AIR_LOCK
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  77 TANGLED_FEET
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  78 MOTOR_DRIVE
[0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  79 RIVALRY
_ABILITY_DEFAULT,
#  80 STEADFAST
_ABILITY_DEFAULT,
#  81 SNOW_CLOAK
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  82 GLUTTONY
_ABILITY_DEFAULT,
#  83 ANGER_POINT
_ABILITY_DEFAULT,
#  84 UNBURDEN
_ABILITY_DEFAULT,
#  85 HEATPROOF
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  86 SIMPLE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
#  87 DRY_SKIN
[0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  88 DOWNLOAD
_ABILITY_DEFAULT,
#  89 IRON_FIST
_ABILITY_DEFAULT,
#  90 POISON_HEAL
_ABILITY_DEFAULT,
#  91 ADAPTABILITY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  92 SKILL_LINK
_ABILITY_DEFAULT,
#  93 HYDRATION
_ABILITY_DEFAULT,
#  94 SOLAR_POWER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.5, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  95 QUICK_FEET
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  96 NORMALIZE
_ABILITY_DEFAULT,
#  97 SNIPER
_ABILITY_DEFAULT,
#  98 MAGIC_GUARD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
#  99 NO_GUARD
_ABILITY_DEFAULT,
# 100 STALL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 101 TECHNICIAN
_ABILITY_DEFAULT,
# 102 LEAF_GUARD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 103 KLUTZ
_ABILITY_DEFAULT,
# 104 MOLD_BREAKER
[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 105 SUPER_LUCK
_ABILITY_DEFAULT,
# 106 AFTERMATH
_ABILITY_DEFAULT,
# 107 ANTICIPATION
_ABILITY_DEFAULT,
# 108 FOREWARN
_ABILITY_DEFAULT,
# 109 UNAWARE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
# 110 TINTED_LENS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
# 111 FILTER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 112 SLOW_START
_ABILITY_DEFAULT,
# 113 SCRAPPY
_ABILITY_DEFAULT,
# 114 STORM_DRAIN
[0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 115 ICE_BODY
_ABILITY_DEFAULT,
# 116 SOLID_ROCK
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 117 SNOW_WARNING
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 118 HONEY_GATHER
_ABILITY_DEFAULT,
# 119 FRISK
_ABILITY_DEFAULT,
# 120 RECKLESS
_ABILITY_DEFAULT,
# 121 MULTITYPE
_ABILITY_DEFAULT,
# 122 FLOWER_GIFT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.5, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 123 BAD_DREAMS
_ABILITY_DEFAULT,
# 124 PICKPOCKET
_ABILITY_DEFAULT,
# 125 SHEER_FORCE
_ABILITY_DEFAULT,
# 126 CONTRARY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0],
# 127 UNNERVE
_ABILITY_DEFAULT,
# 128 DEFIANT
_ABILITY_DEFAULT,
# 129 DEFEATIST
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.167, 0.167, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 130 CURSED_BODY
_ABILITY_DEFAULT,
# 131 HEALER
_ABILITY_DEFAULT,
# 132 FRIEND_GUARD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 133 WEAK_ARMOR
_ABILITY_DEFAULT,
# 134 HEAVY_METAL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 135 LIGHT_METAL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 136 MULTISCALE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 0.5, 0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 137 TOXIC_BOOST
_ABILITY_DEFAULT,
# 138 FLARE_BOOST
_ABILITY_DEFAULT,
# 139 HARVEST
_ABILITY_DEFAULT,
# 140 TELEPATHY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 141 MOODY
_ABILITY_DEFAULT,
# 142 OVERCOAT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 143 POISON_TOUCH
_ABILITY_DEFAULT,
# 144 REGENERATOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
# 145 BIG_PECKS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 146 SAND_RUSH
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 147 WONDER_SKIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 148 ANALYTIC
_ABILITY_DEFAULT,
# 149 ILLUSION
_ABILITY_DEFAULT,
# 150 IMPOSTER
_ABILITY_DEFAULT,
# 151 INFILTRATOR
_ABILITY_DEFAULT,
# 152 MUMMY
_ABILITY_DEFAULT,
# 153 MOXIE
_ABILITY_DEFAULT,
# 154 JUSTIFIED
_ABILITY_DEFAULT,
# 155 RATTLED
_ABILITY_DEFAULT,
# 156 MAGIC_BOUNCE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 157 SAP_SIPPER
[0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 158 PRANKSTER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 159 SAND_FORCE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.263, 0.52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 160 IRON_BARBS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 161 ZEN_MODE
_ABILITY_DEFAULT,
# 162 VICTORY_STAR
_ABILITY_DEFAULT,
# 163 TURBOBLAZE
[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 164 TERAVOLT
[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 165 AROMA_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 166 FLOWER_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 167 CHEEK_POUCH
_ABILITY_DEFAULT,
# 168 PROTEAN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 169 FUR_COAT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 0.5, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 170 MAGICIAN
_ABILITY_DEFAULT,
# 171 BULLETPROOF
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 172 COMPETITIVE
_ABILITY_DEFAULT,
# 173 STRONG_JAW
_ABILITY_DEFAULT,
# 174 REFRIGERATE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 175 SWEET_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 176 STANCE_CHANGE
_ABILITY_DEFAULT,
# 177 GALE_WINGS
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.143, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 178 MEGA_LAUNCHER
_ABILITY_DEFAULT,
# 179 GRASS_PELT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 180 SYMBIOSIS
_ABILITY_DEFAULT,
# 181 TOUGH_CLAWS
_ABILITY_DEFAULT,
# 182 PIXILATE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.895, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 183 GOOEY
_ABILITY_DEFAULT,
# 184 AERILATE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 185 PARENTAL_BOND
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
# 186 DARK_AURA
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.842, 0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 187 FAIRY_AURA
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.895, 0.533, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 188 AURA_BREAK
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.842, 0.225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 189 PRIMORDIAL_SEA
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 190 DESOLATE_LAND
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 191 DELTA_STREAM
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 192 STAMINA
_ABILITY_DEFAULT,
# 193 WIMP_OUT
_ABILITY_DEFAULT,
# 194 EMERGENCY_EXIT
_ABILITY_DEFAULT,
# 195 WATER_COMPACTION
_ABILITY_DEFAULT,
# 196 MERCILESS
_ABILITY_DEFAULT,
# 197 SHIELDS_DOWN
_ABILITY_DEFAULT,
# 198 STAKEOUT
_ABILITY_DEFAULT,
# 199 WATER_BUBBLE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.526, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 200 STEELWORKER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.421, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 201 BERSERK
_ABILITY_DEFAULT,
# 202 SLUSH_RUSH
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 203 LONG_REACH
_ABILITY_DEFAULT,
# 204 LIQUID_VOICE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.526, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 205 TRIAGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.429, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 206 GALVANIZE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 207 SURGE_SURFER
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.667, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 208 SCHOOLING
_ABILITY_DEFAULT,
# 209 DISGUISE
[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 210 BATTLE_BOND
_ABILITY_DEFAULT,
# 211 POWER_CONSTRUCT
_ABILITY_DEFAULT,
# 212 CORROSION
_ABILITY_DEFAULT,
# 213 COMATOSE
_ABILITY_DEFAULT,
# 214 QUEENLY_MAJESTY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 215 INNARDS_OUT
_ABILITY_DEFAULT,
# 216 DANCER
_ABILITY_DEFAULT,
# 217 BATTERY
_ABILITY_DEFAULT,
# 218 FLUFFY
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 0.5, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 219 DAZZLING
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 220 SOUL_HEART
_ABILITY_DEFAULT,
# 221 TANGLING_HAIR
_ABILITY_DEFAULT,
# 222 RECEIVER
_ABILITY_DEFAULT,
# 223 POWER_OF_ALCHEMY
_ABILITY_DEFAULT,
# 224 BEAST_BOOST
_ABILITY_DEFAULT,
# 225 RKS_SYSTEM
_ABILITY_DEFAULT,
# 226 ELECTRIC_SURGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 227 PSYCHIC_SURGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 228 MISTY_SURGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 229 GRASSY_SURGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 230 FULL_METAL_BODY
_ABILITY_DEFAULT,
# 231 SHADOW_SHIELD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 0.5, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 232 PRISM_ARMOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 233 NEUROFORCE
_ABILITY_DEFAULT,
# 234 INTREPID_SWORD
_ABILITY_DEFAULT,
# 235 DAUNTLESS_SHIELD
_ABILITY_DEFAULT,
# 236 LIBERO
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 237 BALL_FETCH
_ABILITY_DEFAULT,
# 238 COTTON_DOWN
_ABILITY_DEFAULT,
# 239 PROPELLER_TAIL
_ABILITY_DEFAULT,
# 240 MIRROR_ARMOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 241 GULP_MISSILE
_ABILITY_DEFAULT,
# 242 STALWART
_ABILITY_DEFAULT,
# 243 STEAM_ENGINE
_ABILITY_DEFAULT,
# 244 PUNK_ROCK
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 245 SAND_SPIT
_ABILITY_DEFAULT,
# 246 ICE_SCALES
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 247 RIPEN
_ABILITY_DEFAULT,
# 248 ICE_FACE
[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 249 POWER_SPOT
_ABILITY_DEFAULT,
# 250 MIMICRY
_ABILITY_DEFAULT,
# 251 SCREEN_CLEANER
_ABILITY_DEFAULT,
# 252 STEELY_SPIRIT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.421, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 253 PERISH_BODY
_ABILITY_DEFAULT,
# 254 WANDERING_SPIRIT
_ABILITY_DEFAULT,
# 255 GORILLA_TACTICS
_ABILITY_DEFAULT,
# 256 NEUTRALIZING_GAS
_ABILITY_DEFAULT,
# 257 PASTEL_VEIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 258 HUNGER_SWITCH
_ABILITY_DEFAULT,
# 259 QUICK_DRAW
_ABILITY_DEFAULT,
# 260 UNSEEN_FIST
_ABILITY_DEFAULT,
# 261 CURIOUS_MEDICINE
_ABILITY_DEFAULT,
# 262 TRANSISTOR
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.632, 0.52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 263 DRAGONS_MAW
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.789, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 264 CHILLING_NEIGH
_ABILITY_DEFAULT,
# 265 GRIM_NEIGH
_ABILITY_DEFAULT,
# 266 AS_ONE_GLASTRIER
_ABILITY_DEFAULT,
# 267 AS_ONE_SPECTRIER
_ABILITY_DEFAULT,
# 268 LINGERING_AROMA
_ABILITY_DEFAULT,
# 269 SEED_SOWER
_ABILITY_DEFAULT,
# 270 THERMAL_EXCHANGE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 271 ANGER_SHELL
_ABILITY_DEFAULT,
# 272 PURIFYING_SALT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 273 WELL_BAKED_BODY
[0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 274 WIND_RIDER
[0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 275 GUARD_DOG
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 276 ROCKY_PAYLOAD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0.263, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 277 WIND_POWER
_ABILITY_DEFAULT,
# 278 ZERO_TO_HERO
_ABILITY_DEFAULT,
# 279 COMMANDER
_ABILITY_DEFAULT,
# 280 ELECTROMORPHOSIS
_ABILITY_DEFAULT,
# 281 PROTOSYNTHESIS
_ABILITY_DEFAULT,
# 282 QUARK_DRIVE
_ABILITY_DEFAULT,
# 283 GOOD_AS_GOLD
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
# 284 VESSEL_OF_RUIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 285 SWORD_OF_RUIN
_ABILITY_DEFAULT,
# 286 TABLETS_OF_RUIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 287 BEADS_OF_RUIN
_ABILITY_DEFAULT,
# 288 ORICHALCUM_PULSE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.444, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 289 HADRON_ENGINE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.333, 0.444, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 290 OPPORTUNIST
_ABILITY_DEFAULT,
# 291 CUD_CHEW
_ABILITY_DEFAULT,
# 292 SHARPNESS
_ABILITY_DEFAULT,
# 293 SUPREME_OVERLORD
_ABILITY_DEFAULT,
# 294 COSTAR
_ABILITY_DEFAULT,
# 295 TOXIC_DEBRIS
_ABILITY_DEFAULT,
# 296 ARMOR_TAIL
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 297 EARTH_EATER
[1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 298 MYCELIUM_MIGHT
[0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, -1.0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 299 MINDS_EYE
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 300 SUPERSWEET_SYRUP
_ABILITY_DEFAULT,
# 301 HOSPITALITY
_ABILITY_DEFAULT,
# 302 TOXIC_CHAIN
_ABILITY_DEFAULT,
# 303 EMBODY_ASPECT_T
_ABILITY_DEFAULT,
# 304 EMBODY_ASPECT_W
_ABILITY_DEFAULT,
# 305 EMBODY_ASPECT_H
_ABILITY_DEFAULT,
# 306 EMBODY_ASPECT_C
_ABILITY_DEFAULT,
# 307 TERA_SHIFT
_ABILITY_DEFAULT,
# 308 TERA_SHELL
[0, 0, 0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 309 TERAFORM_ZERO
_ABILITY_DEFAULT,
# 310 POISON_PUPPETEER
_ABILITY_DEFAULT,
]
# fmt: on

_NUM_ABILITIES = len(_ABILITY_FEATURES)  # 311


def _encode_ability_features(ability_id: int, buf: np.ndarray, pos: int) -> int:
    """Encode 40 ability feature floats into the buffer. Returns new position."""
    if 0 <= ability_id < _NUM_ABILITIES:
        features = _ABILITY_FEATURES[ability_id]
    else:
        features = _ABILITY_FEATURES[0]  # NONE defaults
    for i in range(ABILITY_FEATURE_DIM):
        buf[pos + i] = features[i]
    return pos + ABILITY_FEATURE_DIM


_SELF_ALLY_TARGETS = {0, 10, 11, 12, 13, 15, 18}
_SINGLE_ENEMY_TARGETS = {1, 3, 5, 9}



# v8 boolean flags folded into the v9 has_other_effect catch-all (76
# booleans; terrain_change is OR'd separately). Mirrors spaces.ts
# OTHER_EFFECT_FLAGS exactly — see docs/OBS_V9_LAYOUT.md §1.
_OTHER_EFFECT_FLAGS = (
    "self_switch", "is_ohko", "is_charging", "is_sacrifice",
    "is_recharge", "is_frenzy", "is_typeless", "creates_substitute",
    "suppresses_ability", "has_variable_type", "has_variable_category",
    "bypass_burn_penalty", "ignores_stat_stages", "removes_arena_tags",
    "sets_hazard", "sets_screen", "arena_tag_self_side",
    "applies_continuous_damage", "is_user_hp_damage", "is_target_half_hp",
    "is_counter_damage", "is_level_damage", "is_delayed_attack",
    "post_victory_stat_boost", "hides_user", "hides_target",
    "check_all_hits", "affected_by_gravity",
    "removes_item", "steals_berry", "copies_stats", "inverts_stats",
    "resets_stats", "swaps_stat_stages", "steals_stat_boosts",
    "averages_stats", "swaps_single_stat", "shifts_own_stat", "splits_hp",
    "reduces_pp", "revives_ally", "copies_last_move", "calls_random_move",
    "calls_moveset_move", "copies_move_temp", "copies_move_perm",
    "copies_ability", "swaps_abilities", "changes_ability", "gives_ability",
    "suppresses_if_acted", "bypass_redirect", "forces_target_next",
    "forces_target_last", "has_conditional_priority", "cures_party_status",
    "transfers_status", "heals_status", "removes_battler_tag",
    "removes_substitutes", "transforms_into_target", "is_curse", "is_wish",
    "is_destiny_bond", "swaps_arena_tags", "clears_weather",
    "clears_terrain", "has_variable_target", "resists_last_type",
    "has_variable_accuracy", "uses_alt_stat", "overrides_type_chart",
    "scatters_money", "survives_at_1hp", "matches_user_hp",
    "hp_cost_stat_boost",
)


def _multi_hit_count(multi_hit_type: int) -> int:
    """-1 -> 0 (not multi-hit); TWO(0) -> 2; TWO_TO_FIVE(1) -> 5;
    THREE(2) -> 3; TEN(3)/BEAT_UP(4) -> clamp 5. Mirrors spaces.ts."""
    if multi_hit_type == 0:
        return 2
    if multi_hit_type == 1:
        return 5
    if multi_hit_type == 2:
        return 3
    if multi_hit_type in (3, 4):
        return 5
    return 0


def _encode_move(buf: np.ndarray, offset: int, m: ObsMove) -> None:
    """Encode one move slot (60 dims, v9). Matches spaces.ts encodeMoveFromDict()."""
    if m.move_id <= 0:
        return
    pos = offset
    buf[pos] = 1.0; pos += 1                                      # valid
    _write_one_hot(buf, pos, NUM_POKEMON_TYPES, m.type); pos += NUM_POKEMON_TYPES  # type(19)
    _write_one_hot(buf, pos, NUM_MOVE_CATEGORIES, m.category); pos += NUM_MOVE_CATEGORIES  # cat(3)
    buf[pos] = _clamp(m.power / 250, 0, 1); pos += 1              # power
    acc = m.accuracy
    buf[pos] = 1.0 if acc <= 0 else _clamp(acc / 100, 0, 1); pos += 1  # accuracy
    pp_max = max(m.pp_max, 1)
    buf[pos] = _clamp(m.pp_remaining / pp_max, 0, 1); pos += 1    # pp_ratio
    buf[pos] = _clamp(m.priority / 7, -1, 1); pos += 1            # priority
    buf[pos] = _clamp(m.effect_chance / 100, 0, 1); pos += 1      # effect_chance
    buf[pos] = _clamp(m.drain_ratio, 0, 1); pos += 1              # drain_ratio
    buf[pos] = _clamp(m.heal_ratio, 0, 1); pos += 1               # heal_ratio
    buf[pos] = _multi_hit_count(m.multi_hit_type) / 5; pos += 1  # multi_hit_count (v9)
    buf[pos] = 1.0 if m.force_switch else 0.0; pos += 1           # force_switch
    buf[pos] = 1.0 if m.is_protect else 0.0; pos += 1             # is_protect
    buf[pos] = 1.0 if m.traps_target else 0.0; pos += 1           # traps_target
    buf[pos] = 1.0 if m.makes_contact else 0.0; pos += 1          # makes_contact
    buf[pos] = 1.0 if m.is_usable else 0.0; pos += 1              # is_usable

    # status_effect /7
    buf[pos] = _clamp(m.status_effect / 7, 0, 1); pos += 1

    # stat_change_self_sum /12 and stat_change_target_sum /12
    self_sum = 0
    target_sum = 0
    for sc in m.stat_changes:
        if sc.self_target:
            self_sum += sc.stages
        else:
            target_sum += sc.stages
    buf[pos] = _clamp(self_sum / 12, -1, 1); pos += 1
    buf[pos] = _clamp(target_sum / 12, -1, 1); pos += 1

    # recoil_ratio
    buf[pos] = _clamp(m.recoil_ratio, 0, 1); pos += 1

    # crit_stage_boost /3 (99=always_crit maps to 1.0)
    buf[pos] = _clamp(m.crit_stage_boost / 3, 0, 1); pos += 1

    # move_target_class (3-dim one-hot: [self_or_ally, single_enemy, multi_target_or_field])
    if m.target in _SELF_ALLY_TARGETS:
        buf[pos] = 1.0
    elif m.target in _SINGLE_ENEMY_TARGETS:
        buf[pos + 1] = 1.0
    elif m.target >= 0:
        buf[pos + 2] = 1.0
    pos += 3

    # [44-58] kept effect flags (evidence keep-list)
    buf[pos] = 1.0 if m.ignores_protect else 0.0; pos += 1
    buf[pos] = 1.0 if m.is_sound_based else 0.0; pos += 1
    buf[pos] = 1.0 if m.can_flinch else 0.0; pos += 1
    buf[pos] = 1.0 if m.can_confuse else 0.0; pos += 1
    buf[pos] = 1.0 if m.has_variable_power else 0.0; pos += 1
    buf[pos] = _clamp(m.weather_change / 9, 0, 1); pos += 1
    buf[pos] = 1.0 if m.sets_arena_tag else 0.0; pos += 1
    buf[pos] = 1.0 if m.applies_battler_tag else 0.0; pos += 1
    buf[pos] = 1.0 if m.applies_move_restriction else 0.0; pos += 1
    buf[pos] = 1.0 if m.is_wind_move else 0.0; pos += 1
    buf[pos] = 1.0 if m.is_reckless_move else 0.0; pos += 1
    buf[pos] = 1.0 if m.is_reflectable else 0.0; pos += 1
    buf[pos] = 1.0 if m.is_triage_move else 0.0; pos += 1
    buf[pos] = 1.0 if m.steals_item else 0.0; pos += 1
    buf[pos] = 1.0 if m.hits_semi_invulnerable else 0.0; pos += 1

    # [59] has_other_effect — OR of the 77 cut v8 flags (OTHER_EFFECT_FLAGS
    # mirrors spaces.ts; terrain_change scalar counts as "other" if nonzero)
    has_other = m.terrain_change != 0
    if not has_other:
        for flag in _OTHER_EFFECT_FLAGS:
            if getattr(m, flag, False):
                has_other = True
                break
    buf[pos] = 1.0 if has_other else 0.0


def _encode_pokemon(
    buf: np.ndarray, offset: int, poke: ObsPokemon, is_enemy: bool = False, fog_of_war: bool = False
) -> None:
    """Encode one Pokemon slot (513 dims, v9). Matches spaces.ts encodePokemonFromDict()."""
    if not poke.valid:
        return
    fogged = fog_of_war and is_enemy
    # Never-seen enemy bench member: whole block stays zero (like an empty
    # slot) — mirrors spaces.ts.
    if fogged and not poke.was_seen:
        return
    ability_known = (not fogged) or poke.ability_revealed
    pos = offset
    buf[pos] = 1.0; pos += 1                                      # valid
    buf[pos] = _clamp(poke.hp_ratio, 0, 1); pos += 1              # hp_ratio
    buf[pos] = _clamp(poke.level / 100, 0, 1); pos += 1           # level

    # base_stats(6) /255
    for i in range(6):
        v = poke.base_stats[i] if i < len(poke.base_stats) else 0
        buf[pos] = _clamp(v / 255, 0, 1); pos += 1

    # stat_stages(7) /6
    for i in range(7):
        v = poke.stat_stages[i] if i < len(poke.stat_stages) else 0
        buf[pos] = v / 6; pos += 1

    # type1 one-hot(19)
    t1 = poke.types[0] if len(poke.types) > 0 else -1
    _write_one_hot(buf, pos, NUM_POKEMON_TYPES, t1); pos += NUM_POKEMON_TYPES

    # type2 one-hot(19)
    t2 = poke.types[1] if len(poke.types) > 1 else -1
    _write_one_hot(buf, pos, NUM_POKEMON_TYPES, t2); pos += NUM_POKEMON_TYPES

    # status one-hot(8)
    _write_one_hot(buf, pos, NUM_STATUS_EFFECTS, poke.status_effect); pos += NUM_STATUS_EFFECTS

    # nature_mults(5) — fog: zeroed for enemies (IV/nature-derived)
    if fogged:
        pos += 5
    else:
        for i in range(5):
            v = poke.nature_multipliers[i] if i < len(poke.nature_multipliers) else 1.0
            buf[pos] = v; pos += 1

    # ability features (40 dims) + passive ability features (40 dims) = 80
    # fog: zeroed until the ability has revealed itself in battle
    if poke.ability_suppressed or not ability_known:
        # All zeros for both ability and passive when suppressed or unknown
        pos += ABILITY_FEATURE_DIM * 2
    else:
        pos = _encode_ability_features(poke.ability_id, buf, pos)
        pos = _encode_ability_features(poke.passive_ability_id, buf, pos)

    # is_terastallized
    buf[pos] = 1.0 if poke.is_terastallized else 0.0; pos += 1

    # tera_type one-hot(19)
    _write_one_hot(buf, pos, NUM_POKEMON_TYPES, poke.tera_type); pos += NUM_POKEMON_TYPES

    # volatile_tags(39 curated) + other_tag_count(1)
    active_tags = {t.tag_type_str for t in poke.volatile_tags}
    curated_matches = 0
    for i, tag_name in enumerate(CURATED_VOLATILE_TAGS):
        if tag_name in active_tags:
            buf[pos + i] = 1.0
            curated_matches += 1
    pos += NUM_CURATED_TAGS
    other_count = len(poke.volatile_tags) - curated_matches
    buf[pos] = _clamp(other_count / 10, 0, 1); pos += 1

    # is_boss + boss_shield
    buf[pos] = 1.0 if poke.is_boss else 0.0; pos += 1
    if poke.is_boss and poke.boss_segments > 0:
        buf[pos] = poke.boss_segment_index / poke.boss_segments
    pos += 1

    # is_trapped, is_grounded, weight, catch_rate, is_fainted
    buf[pos] = 1.0 if poke.is_trapped else 0.0; pos += 1
    buf[pos] = 1.0 if poke.is_grounded else 0.0; pos += 1
    buf[pos] = _clamp(poke.weight / 1000, 0, 1); pos += 1
    buf[pos] = _clamp(poke.catch_rate / 255, 0, 1); pos += 1
    buf[pos] = 1.0 if poke.is_fainted else 0.0; pos += 1

    # wave_turn_count, damage_taken, acted, toxic_turn_count, sleep_turns_remaining, held_item_count
    buf[pos] = _clamp(poke.wave_turn_count / 20, 0, 1); pos += 1
    max_hp = max(poke.max_hp, 1)
    buf[pos] = _clamp(poke.turn_data.damage_taken / max_hp, 0, 1); pos += 1
    buf[pos] = 1.0 if poke.turn_data.acted else 0.0; pos += 1
    buf[pos] = _clamp(poke.toxic_turn_count / 16, 0, 1); pos += 1
    buf[pos] = _clamp(poke.sleep_turns_remaining / 4, 0, 1); pos += 1
    buf[pos] = _clamp(len(poke.held_items) / 10, 0, 1); pos += 1

    # ── Additional Pokemon fields (+11 dims) ──

    # species_id /1025
    buf[pos] = _clamp(poke.species_id / 1025, 0, 1); pos += 1

    # gender: -1 (genderless) -> 0, 0 (male) -> 0.5, 1 (female) -> 1.0
    if poke.gender < 0:
        buf[pos] = 0.0
    elif poke.gender == 0:
        buf[pos] = 0.5
    else:
        buf[pos] = 1.0
    pos += 1

    # friendship /255
    buf[pos] = _clamp(poke.friendship / 255, 0, 1); pos += 1

    # move_queue length /2
    buf[pos] = _clamp(len(poke.move_queue) / 2, 0, 1); pos += 1

    # battle_data.hit_count /10 (Rage Fist scaling)
    buf[pos] = _clamp(poke.battle_data.hit_count / 10, 0, 1); pos += 1

    # ability_suppressed
    buf[pos] = 1.0 if poke.ability_suppressed else 0.0; pos += 1

    # is_mega
    buf[pos] = 1.0 if getattr(poke, "is_mega", False) else 0.0; pos += 1

    # is_max (Gigantamax/Eternamax)
    buf[pos] = 1.0 if getattr(poke, "is_max", False) else 0.0; pos += 1

    # move_effectiveness from turn_data (type effectiveness of last hit received, 0-4 /4)
    buf[pos] = _clamp(getattr(poke.turn_data, "move_effectiveness", 0) / 4, 0, 1); pos += 1

    # computed_stats: ATK/DEF/SPATK/SPDEF/SPD (indices 1-5) /500
    # fog: zeroed for enemies (exact stats are IV/nature-derived)
    if fogged:
        pos += 5
    else:
        for i in range(1, 6):
            v = poke.stats[i] if i < len(poke.stats) else 0
            buf[pos] = _clamp(v / 500, 0, 1); pos += 1

    def _move_exists(i: int) -> bool:
        return i < len(poke.moves) and poke.moves[i].move_id > 0

    def _move_known(i: int) -> bool:
        if not _move_exists(i):
            return False
        if not fogged:
            return True
        return i < len(poke.move_known) and poke.move_known[i] is True

    # ── v9 additions (9 dims) ──

    # ai_type one-hot(3): RANDOM/SMART_RANDOM/SMART — all-zero on players
    _write_one_hot(buf, pos, 3, poke.ai_type if is_enemy else -1); pos += 3

    # move_known(4), ability_known(1), was_seen(1) — revealed-indicators;
    # constant-truthy for enemies under full obs, all-zero on player slots
    for i in range(MAX_MOVES):
        buf[pos] = 1.0 if (is_enemy and _move_known(i)) else 0.0; pos += 1
    buf[pos] = 1.0 if (is_enemy and ability_known) else 0.0; pos += 1
    buf[pos] = 1.0 if is_enemy else 0.0; pos += 1  # was_seen (fog never-seen returned early)

    # moves (4 slots x 60 dims) — fog: unseen enemy moves stay zero
    for i in range(MAX_MOVES):
        m = poke.moves[i] if i < len(poke.moves) else ObsMove()
        if (not fogged) or _move_known(i):
            _encode_move(buf, pos, m)
        pos += MOVE_BLOCK_DIM


_KEY_ARENA_TAGS = ["REFLECT", "LIGHT_SCREEN", "AURORA_VEIL", "TAILWIND", "TRICK_ROOM"]


def _encode_field(buf: np.ndarray, offset: int, fld: ObsField) -> None:
    """Encode field state (102 dims, v9). Matches spaces.ts encodeFieldFromDict()."""
    pos = offset

    # weather one-hot(10) + turns
    _write_one_hot(buf, pos, NUM_WEATHER_TYPES, fld.weather_type); pos += NUM_WEATHER_TYPES
    buf[pos] = _clamp(fld.weather_turns_left / 8, 0, 1); pos += 1

    # terrain one-hot(5) + turns
    _write_one_hot(buf, pos, NUM_TERRAIN_TYPES, fld.terrain_type); pos += NUM_TERRAIN_TYPES
    buf[pos] = _clamp(fld.terrain_turns_left / 8, 0, 1); pos += 1

    # Build arena tag presence maps and turn count map
    player_tags: set = set()
    enemy_tags: set = set()
    # "tagType:side" -> turn_count
    tag_turn_map: dict = {}
    for tag in fld.arena_tags:
        s = tag.tag_type_str
        side = tag.side
        if side == 0 or side == 1:  # BOTH or PLAYER
            player_tags.add(s)
            tag_turn_map[f"{s}:1"] = tag.turn_count
        if side == 0 or side == 2:  # BOTH or ENEMY
            enemy_tags.add(s)
            tag_turn_map[f"{s}:2"] = tag.turn_count

    # player_arena_tags(28)
    for i, tag_name in enumerate(ARENA_TAG_ORDER):
        if tag_name in player_tags:
            buf[pos + i] = 1.0
    pos += NUM_ARENA_TAG_TYPES

    # player spikes/toxic_spikes
    buf[pos] = fld.player_spikes_layers / 3; pos += 1
    buf[pos] = fld.player_toxic_spikes_layers / 2; pos += 1

    # enemy_arena_tags(28)
    for i, tag_name in enumerate(ARENA_TAG_ORDER):
        if tag_name in enemy_tags:
            buf[pos + i] = 1.0
    pos += NUM_ARENA_TAG_TYPES

    # enemy spikes/toxic_spikes
    buf[pos] = fld.enemy_spikes_layers / 3; pos += 1
    buf[pos] = fld.enemy_toxic_spikes_layers / 2; pos += 1

    # is_double, trick_room, gravity
    buf[pos] = 1.0 if fld.is_double_battle else 0.0; pos += 1
    buf[pos] = 1.0 if fld.trick_room_active else 0.0; pos += 1
    buf[pos] = 1.0 if fld.gravity_active else 0.0; pos += 1

    # ── New fields (3 dims) ──
    buf[pos] = 1.0 if fld.weather_is_permanent else 0.0; pos += 1
    buf[pos] = 1.0 if fld.weather_suppressed else 0.0; pos += 1
    buf[pos] = 1.0 if fld.terrain_is_permanent else 0.0; pos += 1

    # ── Arena tag remaining turns (10 dims: 5 tags x 2 sides) ──
    for side in [1, 2]:  # PLAYER=1, ENEMY=2
        for tag_name in _KEY_ARENA_TAGS:
            turns = tag_turn_map.get(f"{tag_name}:{side}", 0)
            buf[pos] = _clamp(turns / 8, 0, 1); pos += 1

    # player_teras_used /3
    buf[pos] = _clamp(fld.player_teras_used / 3, 0, 1); pos += 1

    # ── v9: positional tags (+8) — Wish / Future Sight per side ──
    # target_index (BattlerIndex): 0-1 = player side, 2-3 = enemy side.
    # Multiple pending on a side: active=1, turns = min countdown.
    # tag_type_id: DELAYED_ATTACK=1 (Future Sight/Doom Desire), WISH=2.
    pos_agg = [[0, float("inf"), 0, float("inf")], [0, float("inf"), 0, float("inf")]]
    for tag in fld.positional_tags:
        side_idx = 1 if tag.target_index >= 2 else 0
        if tag.tag_type_id == 2:  # WISH
            pos_agg[side_idx][0] = 1
            pos_agg[side_idx][1] = min(pos_agg[side_idx][1], tag.countdown)
        elif tag.tag_type_id == 1:  # DELAYED_ATTACK
            pos_agg[side_idx][2] = 1
            pos_agg[side_idx][3] = min(pos_agg[side_idx][3], tag.countdown)
    for agg in pos_agg:
        buf[pos] = agg[0]; pos += 1
        buf[pos] = _clamp(agg[1] / 8, 0, 1) if agg[0] else 0.0; pos += 1
        buf[pos] = agg[2]; pos += 1
        buf[pos] = _clamp(agg[3] / 8, 0, 1) if agg[2] else 0.0; pos += 1


def _encode_battle(buf: np.ndarray, offset: int, battle: ObsBattle, phase: ObsPhase) -> None:
    """Encode battle meta (40 dims). Matches spaces.ts encodeBattleFromDict()."""
    pos = offset

    buf[pos] = _clamp(battle.wave_index / 200, 0, 1); pos += 1     # wave
    buf[pos] = _clamp(battle.turn / 50, 0, 1); pos += 1            # turn

    # battle_type one-hot(4)
    _write_one_hot(buf, pos, NUM_BATTLE_TYPES, battle.battle_type); pos += NUM_BATTLE_TYPES

    # money, score (log-normalized)
    buf[pos] = math.log(1 + max(0, battle.money)) / math.log(100001); pos += 1
    buf[pos] = math.log(1 + max(0, battle.score)) / math.log(100001); pos += 1

    # pokeball counts (5)
    bc = battle.pokeball_counts
    for v in [bc.pokeball, bc.great_ball, bc.ultra_ball, bc.rogue_ball, bc.master_ball]:
        buf[pos] = _clamp(v / 99, 0, 1); pos += 1

    buf[pos] = _clamp(battle.player_alive_count / 6, 0, 1); pos += 1
    buf[pos] = _clamp(battle.enemy_alive_count / 6, 0, 1); pos += 1
    buf[pos] = 1.0 if battle.tera_available else 0.0; pos += 1
    buf[pos] = 1.0 if battle.can_run else 0.0; pos += 1
    buf[pos] = 1.0 if battle.can_catch else 0.0; pos += 1
    buf[pos] = _clamp(battle.player_faints_battle / 6, 0, 1); pos += 1
    buf[pos] = _clamp(battle.enemy_faints_battle / 6, 0, 1); pos += 1

    # command_field_index (from phase) — placeholder, patched in encode_observation
    cfi = phase.command_field_index
    if cfi < 0:
        buf[pos] = 0
    else:
        buf[pos] = 0.5 if cfi == 0 else 1.0
    pos += 1

    # biome /40
    buf[pos] = _clamp(battle.biome_id / 40, 0, 1); pos += 1

    # escape_attempts /10
    buf[pos] = _clamp(battle.escape_attempts / 10, 0, 1); pos += 1

    # ── New fields (3 dims) ──
    buf[pos] = _clamp(battle.battle_style / 3, 0, 1); pos += 1
    buf[pos] = _clamp(battle.time_of_day / 3, 0, 1); pos += 1
    buf[pos] = 1.0 if battle.lock_modifier_tiers else 0.0; pos += 1

    # ── Additional battle fields (+5 dims) ──
    buf[pos] = _clamp(battle.battle_spec, 0, 1); pos += 1          # battle_spec
    buf[pos] = _clamp(battle.game_mode / 4, 0, 1); pos += 1        # game_mode
    # trainer_specialty_type /19
    trainer = battle.trainer
    specialty = trainer.specialty_type if trainer is not None else -1
    buf[pos] = _clamp(specialty / 19, 0, 1) if specialty >= 0 else 0; pos += 1
    buf[pos] = 1.0 if battle.has_no_shop else 0.0; pos += 1        # has_no_shop
    buf[pos] = _clamp(battle.seen_enemy_count / 6, 0, 1); pos += 1  # seen_enemy_count

    # ── GameMode flags + Inverse Battle (9 dims) ──
    buf[pos] = 1.0 if getattr(battle, "is_classic", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "is_endless", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "is_daily", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "is_challenge", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "has_mystery_encounters", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "has_short_biomes", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "has_random_biomes", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "has_random_bosses", False) else 0.0; pos += 1
    buf[pos] = 1.0 if getattr(battle, "inverse_battle", False) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MODIFIER FEATURES LOOKUP TABLE (mirrors modifier-features.ts)
# ═══════════════════════════════════════════════════════════════════════════
#
# 20-dim feature vector per modifier_id. Static features at indices 0-7, 12-16.
# Dynamic slots (8-11, 17-19) are always 0.0 in this table; filled at encode time.
#
# Layout: [is_damage_boost, is_stat_boost, is_healing, is_survival, is_speed_priority,
#          is_status_effect, is_economy, is_berry,
#          0, 0, 0, 0,  # dynamic target slots
#          boost_magnitude, proc_chance_base, is_per_turn, is_on_hit, is_on_faint,
#          0, 0, 0]      # dynamic stack/duration

def _mf(d, s, h, sv, sp, se, ec, b, mag, proc, turn, hit, faint):
    return [d, s, h, sv, sp, se, ec, b, 0, 0, 0, 0, mag, proc, turn, hit, faint, 0, 0, 0]

_MODIFIER_FEATURES: Dict[str, list] = {
    # Held items (A-L)
    "ATTACK_TYPE_BOOSTER":    _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0, 0, 0),
    "BASE_STAT_BOOSTER":      _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.10,  0,    0, 0, 0),
    "BERRY":                  _mf(0, 0, 0, 0, 0, 0, 0, 1, 0.50,  0,    0, 0, 0),
    "QUICK_CLAW":             _mf(0, 0, 0, 0, 1, 0, 0, 0, 0.10,  0.10, 0, 0, 0),
    "GRIP_CLAW":              _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.10,  0.10, 0, 1, 0),
    "SCOPE_LENS":             _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.333, 0,    0, 0, 0),
    "GOLDEN_PUNCH":           _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.50,  0,    0, 1, 0),
    "EVIOLITE":               _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "EVOLUTION_TRACKER_GIMMIGHOUL": _mf(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "MYSTICAL_ROCK":          _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0, 0, 0),
    "KINGS_ROCK":             _mf(0, 0, 0, 0, 0, 1, 0, 0, 0.10,  0.10, 0, 1, 0),
    "SHELL_BELL":             _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.125, 0,    0, 1, 0),
    # Held items (M-W)
    "MINI_BLACK_HOLE":        _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    1, 0, 0),
    "MULTI_LENS":             _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 1, 0),
    "MYSTERY_ENCOUNTER_MACHO_BRACE": _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.02, 0, 0, 0, 0),
    "MYSTERY_ENCOUNTER_OLD_GATEAU": _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.20, 0, 0, 0, 0),
    "MYSTERY_ENCOUNTER_SHUCKLE_JUICE": _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.10, 0, 0, 0, 0),
    "FOCUS_BAND":             _mf(0, 0, 0, 1, 0, 0, 0, 0, 0.10,  0.10, 0, 0, 0),
    "SPECIES_STAT_BOOSTER":   _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "RARE_SPECIES_STAT_BOOSTER": _mf(1, 1, 0, 0, 0, 0, 0, 0, 1.0, 0,   0, 0, 0),
    "REVIVER_SEED":           _mf(0, 0, 0, 1, 0, 0, 0, 0, 0.50,  0,    0, 0, 1),
    "LEFTOVERS":              _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.0625, 0,   1, 0, 0),
    "SOUL_DEW":               _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.10,  0,    0, 0, 0),
    "LEEK":                   _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.667, 0,    0, 0, 0),
    "TOXIC_ORB":              _mf(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    1, 0, 0),
    "FLAME_ORB":              _mf(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    1, 0, 0),
    "WHITE_HERB":             _mf(0, 1, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "WIDE_LENS":              _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.05,  0,    0, 0, 0),
    "GOLDEN_EGG":             _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "LUCKY_EGG":              _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.50,  0,    0, 0, 0),
    "SOOTHE_BELL":            _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "BATON":                  _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "FORM_CHANGE_ITEM":       _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "RARE_FORM_CHANGE_ITEM":  _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    # Party-wide modifiers
    "MAP":                    _mf(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0, 0, 0),
    "MEGA_BRACELET":          _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "DYNAMAX_BAND":           _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "TERA_ORB":               _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "CANDY_JAR":              _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "BERRY_POUCH":            _mf(0, 0, 0, 0, 0, 0, 0, 1, 0.30,  0.30, 0, 0, 0),
    "OVAL_CHARM":             _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "HEALING_CHARM":          _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "EXP_CHARM":              _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.25,  0,    0, 0, 0),
    "SUPER_EXP_CHARM":        _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.60,  0,    0, 0, 0),
    "GOLDEN_EXP_CHARM":       _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "EXP_SHARE":              _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "EXP_BALANCE":            _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "AMULET_COIN":            _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.20,  0,    0, 0, 0),
    "COIN_CASE":              _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.10,  0,    1, 0, 0),
    "ABILITY_CHARM":          _mf(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0, 0, 0),
    "SHINY_CHARM":            _mf(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0, 0, 0),
    "CATCHING_CHARM":         _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "LOCK_CAPSULE":           _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "MYSTERY_ENCOUNTER_BLACK_SLUDGE": _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.25, 0, 0, 0, 0),
    "MYSTERY_ENCOUNTER_GOLDEN_BUG_NET": _mf(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "IV_SCANNER":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0, 0, 0),
    "GOLDEN_POKEBALL":        _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    # Lapsing modifiers
    "LURE":                   _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 0, 0),
    "SUPER_LURE":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 0, 0),
    "MAX_LURE":               _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 0, 0),
    "TEMP_STAT_STAGE_BOOSTER": _mf(0, 1, 0, 0, 0, 0, 0, 0, 0.20, 0,   0, 0, 0),
    "DIRE_HIT":               _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.333, 0,   0, 0, 0),
    "SILVER_POKEBALL":        _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,   0, 0, 0),
    # Enemy modifiers
    "ENEMY_DAMAGE_BOOSTER":   _mf(1, 0, 0, 0, 0, 0, 0, 0, 0.05,  0,    0, 0, 0),
    "ENEMY_DAMAGE_REDUCTION": _mf(0, 0, 0, 1, 0, 0, 0, 0, 0.025, 0,    0, 0, 0),
    "ENEMY_HEAL":             _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.02,  0,    1, 0, 0),
    "ENEMY_ATTACK_POISON_CHANCE": _mf(0, 0, 0, 0, 0, 1, 0, 0, 0.05, 0.05, 0, 1, 0),
    "ENEMY_ATTACK_PARALYZE_CHANCE": _mf(0, 0, 0, 0, 0, 1, 0, 0, 0.025, 0.025, 0, 1, 0),
    "ENEMY_ATTACK_BURN_CHANCE": _mf(0, 0, 0, 0, 0, 1, 0, 0, 0.05, 0.05, 0, 1, 0),
    "ENEMY_STATUS_EFFECT_HEAL_CHANCE": _mf(0, 0, 0, 0, 0, 1, 0, 0, 0.025, 0.025, 1, 0, 0),
    "ENEMY_ENDURE_CHANCE":    _mf(0, 0, 0, 1, 0, 0, 0, 0, 0.02,  0.02, 0, 0, 0),
    "ENEMY_FUSED_CHANCE":     _mf(0, 0, 0, 0, 0, 0, 0, 0, 0,     0,    0, 0, 0),
    # Consumables (reward/shop encoding only)
    "POKEBALL":               _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.20,  0,    0, 0, 0),
    "GREAT_BALL":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.40,  0,    0, 0, 0),
    "ULTRA_BALL":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.60,  0,    0, 0, 0),
    "ROGUE_BALL":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.80,  0,    0, 0, 0),
    "MASTER_BALL":            _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "VOUCHER":                _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.33,  0,    0, 0, 0),
    "VOUCHER_PLUS":           _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.67,  0,    0, 0, 0),
    "VOUCHER_PREMIUM":        _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "NUGGET":                 _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.33,  0,    0, 0, 0),
    "BIG_NUGGET":             _mf(0, 0, 0, 0, 0, 0, 1, 0, 0.67,  0,    0, 0, 0),
    "RELIC_GOLD":             _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "POTION":                 _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.05,  0,    0, 0, 0),
    "SUPER_POTION":           _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.125, 0,    0, 0, 0),
    "HYPER_POTION":           _mf(0, 0, 1, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "MAX_POTION":             _mf(0, 0, 1, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "FULL_RESTORE":           _mf(0, 0, 1, 0, 0, 1, 0, 0, 1.0,   0,    0, 0, 0),
    "REVIVE":                 _mf(0, 0, 1, 1, 0, 0, 0, 0, 0.50,  0,    0, 0, 1),
    "MAX_REVIVE":             _mf(0, 0, 1, 1, 0, 0, 0, 0, 1.0,   0,    0, 0, 1),
    "SACRED_ASH":             _mf(0, 0, 1, 1, 0, 0, 0, 0, 1.0,   0,    0, 0, 1),
    "FULL_HEAL":              _mf(0, 0, 0, 0, 0, 1, 0, 0, 1.0,   0,    0, 0, 0),
    "RARE_CANDY":             _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "RARER_CANDY":            _mf(0, 0, 0, 0, 0, 0, 1, 0, 1.0,   0,    0, 0, 0),
    "ETHER":                  _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 0, 0),
    "MAX_ETHER":              _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "ELIXIR":                 _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.25,  0,    0, 0, 0),
    "MAX_ELIXIR":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "PP_UP":                  _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.33,  0,    0, 0, 0),
    "PP_MAX":                 _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "MINT":                   _mf(0, 1, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "TERA_SHARD":             _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "EVOLUTION_ITEM":         _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "RARE_EVOLUTION_ITEM":    _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "TM_COMMON":              _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.33,  0,    0, 0, 0),
    "TM_GREAT":               _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.67,  0,    0, 0, 0),
    "TM_ULTRA":               _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
    "MEMORY_MUSHROOM":        _mf(0, 0, 0, 0, 0, 0, 0, 0, 0.50,  0,    0, 0, 0),
    "DNA_SPLICERS":           _mf(0, 0, 0, 0, 0, 0, 0, 0, 1.0,   0,    0, 0, 0),
}

_DEFAULT_MODIFIER_FEATURES = [0.0] * MODIFIER_FEATURE_DIM

# RL priority ordering for held item sorting (higher = more important)
_PRIORITY_KEYS = [0, 3, 2, 1, 4, 5, 7, 6]  # damage > survival > healing > stat > speed > status > berry > economy

# Party modifier_id flags (8 boolean presence flags)
_PARTY_FLAG_IDS = [
    "HEALING_CHARM",
    "EXP_SHARE",
    "BERRY_POUCH",
    "AMULET_COIN",  # also matches COIN_CASE via modifier_class check
    "LOCK_CAPSULE",
    "GOLDEN_POKEBALL",
    "MEGA_BRACELET",
    "TERA_ORB",
]

# Known enemy modifier_id strings for aggregate encoding
_ENEMY_MOD_IDS = [
    "ENEMY_DAMAGE_BOOSTER",
    "ENEMY_DAMAGE_REDUCTION",
    "ENEMY_HEAL",
    "ENEMY_ATTACK_POISON_CHANCE",
    "ENEMY_ATTACK_PARALYZE_CHANCE",
    "ENEMY_ATTACK_BURN_CHANCE",
    "ENEMY_STATUS_EFFECT_HEAL_CHANCE",
]
_ENEMY_NORM_DIVISORS = [50, 50, 20, 20, 20, 20, 20]


def _get_modifier_id_str(modifier_type_id: int) -> str:
    """Reverse-lookup modifier_id string from integer ID."""
    return MODIFIER_TYPE_ID_TO_STR.get(modifier_type_id, "")


def _get_modifier_features(modifier_id: str) -> list:
    """Get the 20-dim static feature vector for a modifier_id."""
    return _MODIFIER_FEATURES.get(modifier_id, _DEFAULT_MODIFIER_FEATURES)


def _encode_modifier_features_vec(
    buf: np.ndarray, pos: int, modifier_id: str,
    type_id: int, stat_id: int, status_effect: int, berry_type: int,
    stack_count: int, max_stack_count: int, battles_remaining: int,
) -> int:
    """Encode a full 20-dim modifier feature vector into the buffer.
    Returns new position after writing."""
    feats = _get_modifier_features(modifier_id)

    # Category flags (0-7) -- static
    for i in range(8):
        buf[pos + i] = feats[i]

    # Dynamic target parameters (8-11)
    buf[pos + 8] = _clamp(type_id / 18, 0, 1) if type_id >= 0 else 0.0
    buf[pos + 9] = _clamp(stat_id / 7, 0, 1) if stat_id >= 0 else 0.0
    buf[pos + 10] = _clamp(status_effect / 7, 0, 1) if status_effect >= 0 else 0.0
    buf[pos + 11] = _clamp(berry_type / 12, 0, 1) if berry_type >= 0 else 0.0

    # Static effect parameters (12-16)
    for i in range(12, 17):
        buf[pos + i] = feats[i]

    # Dynamic stack/duration (17-19)
    # Mirrors modifier-features.ts: max_stack_count <= 0 encodes as 0, NOT
    # stack/1 — items can legitimately report max_stack_count 0.
    buf[pos + 17] = _clamp(stack_count / max_stack_count, 0, 1) if max_stack_count > 0 else 0.0
    buf[pos + 18] = _clamp(stack_count / 10, 0, 1)
    buf[pos + 19] = _clamp(battles_remaining / 10, 0, 1) if battles_remaining > 0 else 0.0

    return pos + MODIFIER_FEATURE_DIM


def _sort_held_items_by_priority(items: list) -> list:
    """Sort held items by RL importance. Returns a new sorted list."""
    def priority_key(item):
        mod_id = _get_modifier_id_str(item.modifier_type_id)
        feats = _get_modifier_features(mod_id)
        # Negate for descending sort: higher flag value = more important
        return tuple(-feats[k] for k in _PRIORITY_KEYS) + (-item.stack_count,)
    return sorted(items, key=priority_key)


# ── Modifier Phase Encoding ───────────────────────────────────────────

def _encode_modifier(buf: np.ndarray, offset: int, shop: Optional[ObsShop], money: int) -> None:
    """Encode modifier phase (225 dims)."""
    pos = offset

    is_active = shop is not None

    # Header (3 dims)
    buf[pos] = 1.0 if is_active else 0.0; pos += 1  # modifier_active
    buf[pos] = 1.0 if (is_active and shop.can_reroll) else 0.0; pos += 1  # can_reroll
    reroll_cost = shop.reroll_cost if is_active else 0
    buf[pos] = _clamp(reroll_cost / money, 0, 1) if (is_active and money > 0) else 0.0; pos += 1

    if not is_active:
        return  # rest stays zero

    # Reward options (3 x 28 = 84 dims)
    for i in range(MAX_REWARD_OPTIONS):
        if i < len(shop.reward_options):
            opt = shop.reward_options[i]
            buf[pos] = 1.0; pos += 1  # valid
            _write_one_hot(buf, pos, NUM_MODIFIER_TIERS, opt.tier); pos += NUM_MODIFIER_TIERS
            buf[pos] = 1.0 if opt.is_pokemon_modifier else 0.0; pos += 1
            # 20-dim modifier features
            mod_id = _get_modifier_id_str(opt.modifier_type_id)
            pos = _encode_modifier_features_vec(
                buf, pos, mod_id,
                opt.type_id, opt.stat_id, opt.status_effect, opt.berry_type,
                0, 0, 0,
            )
        else:
            pos += REWARD_OPTION_DIM

    # Shop options (first 6 in NATURAL order, 6 x 23 = 138 dims).
    # Natural order is the order the BUY_SHOP actions (40-51) index — encoded
    # slot k must describe action 40+k. (These used to be sorted by cost,
    # which decoupled shop features from buy actions; mirrors spaces.ts.)
    valid_shop = list(shop.shop_options)

    for i in range(MAX_SHOP_OPTIONS_ENCODED):
        if i < len(valid_shop):
            opt = valid_shop[i]
            buf[pos] = 1.0; pos += 1  # valid
            cost_ratio = _clamp(opt.cost / money, 0, 1) if money > 0 else 1.0
            buf[pos] = cost_ratio; pos += 1
            buf[pos] = 1.0 if opt.affordable else 0.0; pos += 1
            # 20-dim modifier features
            mod_id = _get_modifier_id_str(opt.modifier_type_id)
            pos = _encode_modifier_features_vec(
                buf, pos, mod_id,
                opt.type_id, opt.stat_id, opt.status_effect, opt.berry_type,
                0, 0, 0,
            )
        else:
            pos += SHOP_OPTION_DIM


def _encode_phase(buf: np.ndarray, offset: int, phase: ObsPhase) -> None:
    """Encode phase indicator (16-dim one-hot)."""
    pid = phase.current_phase_id
    if 0 <= pid < PHASE_INDICATOR_DIM:
        buf[offset + pid] = 1.0


# ── Type effectiveness chart (19x19) ────────────────────────────────────
# Rows = attacking type, Columns = defending type.
# Same as spaces.ts TYPE_EFFECTIVENESS.

_TYPE_EFFECTIVENESS = [
    #NOR FIG FLY PSN GND RCK BUG GHO STL FIR WAT GRS ELC PSY ICE DRG DRK FAI STR
    [1,  1,  1,  1,  1, .5,  1,  0, .5,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # NORMAL
    [2,  1, .5, .5,  1,  2, .5,  0,  2,  1,  1,  1,  1, .5,  2,  1,  2, .5,  1],  # FIGHTING
    [1,  2,  1,  1,  1, .5,  2,  1, .5,  1,  1,  2, .5,  1,  1,  1,  1,  1,  1],  # FLYING
    [1,  1,  1, .5, .5, .5,  1, .5,  0,  1,  1,  2,  1,  1,  1,  1,  1,  2,  1],  # POISON
    [1,  1,  0,  2,  1,  2, .5,  1,  2,  2,  1, .5,  2,  1,  1,  1,  1,  1,  1],  # GROUND
    [1, .5,  2,  1, .5,  1,  2,  1, .5,  2,  1,  1,  1,  1,  2,  1,  1,  1,  1],  # ROCK
    [1, .5, .5, .5,  1,  1,  1, .5, .5, .5,  1,  2,  1,  2,  1,  1,  2, .5,  1],  # BUG
    [0,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1,  1,  1,  2,  1,  1, .5,  1,  1],  # GHOST
    [1,  1,  1,  1,  1,  2,  1,  1, .5, .5, .5,  1, .5,  1,  2,  1,  1,  2,  1],  # STEEL
    [1,  1,  1,  1,  1, .5,  2,  1,  2, .5, .5,  2,  1,  1,  2, .5,  1,  1,  1],  # FIRE
    [1,  1,  1,  1,  2,  2,  1,  1,  1,  2, .5, .5,  1,  1,  1, .5,  1,  1,  1],  # WATER
    [1,  1, .5, .5,  2,  2, .5,  1, .5, .5,  2, .5,  1,  1,  1, .5,  1,  1,  1],  # GRASS
    [1,  1,  2,  1,  0,  1,  1,  1,  1,  1,  2, .5, .5,  1,  1, .5,  1,  1,  1],  # ELECTRIC
    [1,  2,  1,  2,  1,  1,  1,  1, .5,  1,  1,  1,  1, .5,  1,  1,  0,  1,  1],  # PSYCHIC
    [1,  1,  2,  1,  2,  1,  1,  1, .5, .5, .5,  2,  1,  1, .5,  2,  1,  1,  1],  # ICE
    [1,  1,  1,  1,  1,  1,  1,  1, .5,  1,  1,  1,  1,  1,  1,  2,  1,  0,  1],  # DRAGON
    [1, .5,  1,  1,  1,  1,  1,  2,  1,  1,  1,  1,  1,  2,  1,  1, .5, .5,  1],  # DARK
    [1,  2,  1, .5,  1,  1,  1,  1, .5, .5,  1,  1,  1,  1,  1,  2,  2,  1,  1],  # FAIRY
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # STELLAR
]

# Stage multipliers for stages -6..+6 (index = stage+6)
_STAGE_MULTIPLIERS = [2/8, 2/7, 2/6, 2/5, 2/4, 2/3, 1, 3/2, 4/2, 5/2, 6/2, 7/2, 8/2]


def _compute_type_effectiveness(atk_type: int, def_types: list) -> float:
    """Type effectiveness of attacking type vs all defending types."""
    if atk_type < 0 or atk_type >= NUM_POKEMON_TYPES:
        return 1.0
    mult = 1.0
    for def_type in def_types:
        if 0 <= def_type < NUM_POKEMON_TYPES:
            mult *= _TYPE_EFFECTIVENESS[atk_type][def_type]
    return mult


def _encode_modifier_inventory(buf: np.ndarray, offset: int, state: CleanGameState) -> None:
    """Encode modifier inventory (220 dims). Matches spaces.ts encodeModifierInventory().

    Layout:
      Held items:    4 slots x 45 = 180
      Party mods:    9 (count + 8 boolean flags)
      Lapsing mods:  23 (count + 1 x (valid + features(20) + battles_remaining))
      Enemy mods:    8 (count + 7 aggregate stacks)
    """
    pos = offset

    # Active Pokemon slot keys in same order as spaces.ts
    active_keys = [0, 1, 2, 3]  # player_0, player_1, enemy_0, enemy_1

    # ── Per active slot held items (4 x 45 = 180 dims) ──
    for slot_idx in active_keys:
        poke = state.pokemon[slot_idx] if slot_idx < len(state.pokemon) else ObsPokemon()
        items = poke.held_items

        # held_item_count / 10
        buf[pos] = _clamp(len(items) / 10, 0, 1); pos += 1

        # Sort by RL priority, encode top 2
        sorted_items = _sort_held_items_by_priority(items)

        for i in range(MAX_HELD_ITEMS_ENCODED):
            if i < len(sorted_items):
                item = sorted_items[i]
                buf[pos] = 1.0; pos += 1  # valid
                # 20-dim modifier features
                mod_id = _get_modifier_id_str(item.modifier_type_id)
                pos = _encode_modifier_features_vec(
                    buf, pos, mod_id,
                    item.type_id, item.stat_id, item.status_effect, item.berry_type,
                    item.stack_count, item.max_stack_count, 0,
                )
                # stack_ratio (redundant with feature[17], kept for compatibility)
                # Mirrors spaces.ts: max_stack_count <= 0 encodes as 0
                if item.max_stack_count > 0:
                    buf[pos] = _clamp(item.stack_count / item.max_stack_count, 0, 1)
                pos += 1
            else:
                pos += HELD_ITEM_SLOT_DIM  # 22 zeros

    # ── Party-wide modifiers (9 dims) ──
    party_mods = state.modifiers.party_modifiers

    # party_mod_count / 20
    buf[pos] = _clamp(len(party_mods) / 20, 0, 1); pos += 1

    # 8 boolean presence flags
    from .enums import MODIFIER_CLASS_ID_TO_STR
    party_mod_id_set = set()
    for mod in party_mods:
        mod_id = _get_modifier_id_str(mod.modifier_type_id)
        if mod_id:
            party_mod_id_set.add(mod_id)
        # Also check modifier_class for COIN_CASE -> MoneyInterestModifier
        cls_name = MODIFIER_CLASS_ID_TO_STR.get(mod.modifier_class_id, "")
        if cls_name == "MoneyInterestModifier":
            party_mod_id_set.add("AMULET_COIN")  # maps to money_boost flag

    for flag_id in _PARTY_FLAG_IDS:
        buf[pos] = 1.0 if flag_id in party_mod_id_set else 0.0; pos += 1

    # ── Lapsing modifiers (23 dims) ──
    lapsing = state.modifiers.lapsing_modifiers

    # lapsing_count / 5
    buf[pos] = _clamp(len(lapsing) / 5, 0, 1); pos += 1

    # Top 1 lapsing modifier: valid(1) + features(20) + battles_remaining(1) = 22
    best_lapsing = None
    best_remaining = 0
    for mod in lapsing:
        if mod.battles_remaining > best_remaining:
            best_lapsing = mod
            best_remaining = mod.battles_remaining

    if best_lapsing is not None:
        buf[pos] = 1.0; pos += 1  # valid
        mod_id = _get_modifier_id_str(best_lapsing.modifier_type_id)
        # max_stack_count=1: lapsing dicts carry no max_stack_count key, and the
        # TS encoder's num(dict, "max_stack_count", 1) defaults missing keys to 1
        pos = _encode_modifier_features_vec(
            buf, pos, mod_id,
            -1, best_lapsing.stat_id, -1, -1,
            best_lapsing.stack_count, 1, best_lapsing.battles_remaining,
        )
        buf[pos] = _clamp(best_remaining / 10, 0, 1); pos += 1
    else:
        pos += 22  # 1 valid + 20 features + 1 battles_remaining

    # ── Enemy modifiers (8 dims) ──
    enemy_mods = state.modifiers.enemy_modifiers

    # enemy_mod_count / 20
    buf[pos] = _clamp(len(enemy_mods) / 20, 0, 1); pos += 1

    # 7 aggregate stack values for known enemy modifier types
    enemy_stack_map: Dict[str, int] = {}
    for mod in enemy_mods:
        mod_id = _get_modifier_id_str(mod.modifier_type_id)
        if mod_id:
            enemy_stack_map[mod_id] = enemy_stack_map.get(mod_id, 0) + mod.stack_count

    for i, eid in enumerate(_ENEMY_MOD_IDS):
        stacks = enemy_stack_map.get(eid, 0)
        buf[pos] = _clamp(stacks / _ENEMY_NORM_DIVISORS[i], 0, 1); pos += 1


def _encode_derived_fields(buf: np.ndarray, offset: int, state: CleanGameState) -> None:
    """Encode derived fields (28 dims). Matches spaces.ts encodeDerivedFields()."""
    pos = offset

    # In SINGLES the v9 slot remap fills slot indices 1 (player_1) and 3
    # (enemy_1) with the first BENCH member — not an active combatant. This
    # block is an active-matchup / active-speed summary, so those slots are
    # excluded in singles (mirrors spaces.ts): only slot 0 is active per side.
    is_double = state.field.is_double_battle

    def _slot(idx: int) -> ObsPokemon:
        if not is_double and idx in (1, 3):
            return ObsPokemon()
        return state.pokemon[idx] if idx < len(state.pokemon) else ObsPokemon()

    player_slots = [0, 1]  # player_0, player_1
    enemy_slots = [2, 3]   # enemy_0, enemy_1

    # Pre-extract enemy types
    enemy_types_list = []
    for e_idx in enemy_slots:
        poke = _slot(e_idx)
        if poke.valid:
            enemy_types_list.append(poke.types)
        else:
            enemy_types_list.append([])

    # ── Type effectiveness: 2 players x 4 moves x 2 enemies = 16 dims ──
    for p_idx in player_slots:
        poke = _slot(p_idx)
        for m_i in range(MAX_MOVES):
            m = poke.moves[m_i] if m_i < len(poke.moves) else ObsMove()
            for e_i in range(2):
                if m.move_id > 0 and len(enemy_types_list[e_i]) > 0:
                    eff = _compute_type_effectiveness(m.type, enemy_types_list[e_i])
                    buf[pos] = _clamp(eff / 4.0, 0, 1)
                pos += 1

    # ── STAB indicators: 2 players x 4 moves = 8 dims ──
    for p_idx in player_slots:
        poke = _slot(p_idx)
        ptypes = set(poke.types)
        for m_i in range(MAX_MOVES):
            m = poke.moves[m_i] if m_i < len(poke.moves) else ObsMove()
            if m.move_id > 0 and m.type >= 0 and m.type in ptypes:
                buf[pos] = 1.0
            pos += 1

    # ── Speed ordering: 4 active slots = 4 dims ──
    speed_slot_indices = [0, 1, 2, 3]  # player_0, player_1, enemy_0, enemy_1
    speeds = []
    for s_idx in speed_slot_indices:
        poke = _slot(s_idx)
        is_valid = poke.valid and not poke.is_fainted
        speed = 0.0
        if is_valid:
            # Prefer computed stats[5] = SPD
            spd_stat = poke.stats[5] if 5 < len(poke.stats) else 0
            if spd_stat <= 0:
                spd_stat = poke.base_stats[5] if 5 < len(poke.base_stats) else 0
            speed = float(spd_stat)
            # Apply speed stat stage (stat_stages[4] = SPD stage)
            spd_stage = poke.stat_stages[4] if 4 < len(poke.stat_stages) else 0
            stage_idx = max(0, min(12, spd_stage + 6))
            speed *= _STAGE_MULTIPLIERS[stage_idx]
        speeds.append((is_valid, speed))

    valid_speeds = [s for v, s in speeds if v]
    unique_speeds = sorted(set(valid_speeds), reverse=True)
    rank_values = [1.0, 0.75, 0.5, 0.25]

    for is_valid, speed in speeds:
        if is_valid and unique_speeds:
            rank_idx = unique_speeds.index(speed)
            buf[pos] = rank_values[min(rank_idx, len(rank_values) - 1)]
        pos += 1


def _encode_learn_move(buf: np.ndarray, offset: int, phase: ObsPhase) -> None:
    """v9 learn-move block (66): offered move (60) + learner party one-hot (6).
    All-zero outside the learn_move phase. Matches spaces.ts."""
    if phase.learn_move_stats is not None:
        _encode_move(buf, offset, phase.learn_move_stats)
    _write_one_hot(buf, offset + MOVE_BLOCK_DIM, MAX_PARTY_SIZE, phase.learn_move_party_index)


def encode_observation(state: CleanGameState, fog_of_war: bool = False) -> np.ndarray:
    """Encode a CleanGameState into a 6,991-dim float32 observation vector.

    Compatible with the TypeScript encodeObservation() in spaces.ts.
    Same normalization formulas, same ordering.

    Layout (v9 — docs/OBS_V9_LAYOUT.md):
      Pokemon blocks:     12 x 513 = 6,156
      Field state:        102
      Battle meta:        40
      Modifier phase:     363
      Modifier inventory: 220
      Derived fields:     28
      Learn-move block:   66
      Phase indicator:    16
      Total:              6,991

    fog_of_war: mask enemy private info to what a human could know
    (unseen moves, unrevealed abilities, IV/nature-derived values,
    never-seen bench members). Default False = full information.
    """
    buf = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    offset = 0

    # Pokemon blocks (12 x 513 = 6,156); slot order mirrors POKEMON_SLOT_KEYS
    for slot_key, poke in zip(POKEMON_SLOT_KEYS, state.pokemon):
        _encode_pokemon(buf, offset, poke, is_enemy=slot_key.startswith("enemy"), fog_of_war=fog_of_war)
        offset += POKEMON_BLOCK_DIM

    # Field state (94)
    _encode_field(buf, offset, state.field)
    offset += FIELD_STATE_DIM

    # Battle meta (31)
    _encode_battle(buf, offset, state.battle, state.phase)
    offset += BATTLE_META_DIM

    # Modifier phase (225)
    _encode_modifier(buf, offset, state.shop, state.battle.money)
    offset += MODIFIER_PHASE_DIM

    # Modifier inventory (220)
    _encode_modifier_inventory(buf, offset, state)
    offset += MODIFIER_INVENTORY_DIM

    # Derived fields (28)
    _encode_derived_fields(buf, offset, state)
    offset += DERIVED_FIELDS_DIM

    # Learn-move block (66)
    _encode_learn_move(buf, offset, state.phase)
    offset += LEARN_MOVE_BLOCK_DIM

    # Phase indicator (16)
    _encode_phase(buf, offset, state.phase)

    return buf


def extract_action_mask(state: CleanGameState) -> np.ndarray:
    """Extract the 58-dim boolean action mask from a parsed CleanGameState."""
    return np.array(state.phase.action_mask[:ACTION_SPACE_SIZE], dtype=np.bool_)
