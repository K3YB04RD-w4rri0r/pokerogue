"""
Complete Python dictionary schema for the PokéRogue RL game state.

This module defines TypedDict classes that describe every field the RL agent
can observe. The top-level ``GameState`` dict is serialized as JSON-lines over
the wire between the Node.js headless runner and the Python agent.

Design goals:
  - Human-readable: every field has a clear name and comment.
  - JSON-serializable: only int, float, bool, str, None, list, and dict.
  - Complete: captures everything an optimal agent could need, including
    fields the current Float32 vector encoding (spaces.ts, 9,875 dims) omits.
  - Typed: uses typing.TypedDict so agents get IDE autocomplete and
    static-analysis support out of the box.

Slot layout (12 Pokemon total):
  player_0  - player active slot 0  (PLAYER   / BattlerIndex 0)
  player_1  - player active slot 1  (PLAYER_2 / BattlerIndex 1, doubles only)
  player_2  - player bench slot 0
  player_3  - player bench slot 1
  player_4  - player bench slot 2
  player_5  - player bench slot 3
  enemy_0   - enemy active slot 0   (ENEMY    / BattlerIndex 2)
  enemy_1   - enemy active slot 1   (ENEMY_2  / BattlerIndex 3, doubles only)
  enemy_2   - enemy bench slot 0
  enemy_3   - enemy bench slot 1
  enemy_4   - enemy bench slot 2
  enemy_5   - enemy bench slot 3
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# Constants — mirror the TypeScript enum counts in spaces.ts / enums/
# ---------------------------------------------------------------------------

NUM_POKEMON_TYPES = 19       # PokemonType: NORMAL=0 .. STELLAR=18, UNKNOWN=-1
NUM_STATUS_EFFECTS = 8       # StatusEffect: NONE=0 .. FAINT=7
NUM_STATS = 6                # HP, ATK, DEF, SPATK, SPDEF, SPD
NUM_BATTLE_STATS = 7         # ATK, DEF, SPATK, SPDEF, SPD, ACC, EVA
NUM_EFFECTIVE_STATS = 5      # ATK, DEF, SPATK, SPDEF, SPD (nature-affected)
NUM_WEATHER_TYPES = 10       # WeatherType: NONE=0 .. STRONG_WINDS=9
NUM_TERRAIN_TYPES = 5        # TerrainType: NONE=0 .. PSYCHIC=4
NUM_ARENA_TAG_TYPES = 28     # ArenaTagType values excluding NONE
NUM_CURATED_TAGS = 48        # Strategically important volatile tags encoded as binary flags
NUM_MOVE_CATEGORIES = 3      # MoveCategory: PHYSICAL=0, SPECIAL=1, STATUS=2
NUM_BATTLE_TYPES = 4         # BattleType: WILD=0, TRAINER=1, CLEAR=2, MYSTERY_ENCOUNTER=3
NUM_MODIFIER_TIERS = 6       # ModifierTier: COMMON=0 .. LUXURY=5
NUM_POKEBALL_TYPES = 6       # PokeballType enum has 6 values: POKEBALL=0 .. LUXURY_BALL=5
                              # But pokeballCounts only tracks 0-4 (LUXURY_BALL excluded by game)
NUM_NATURES = 25             # Nature: HARDY=0 .. QUIRKY=24
NUM_GENDERS = 3              # Gender: GENDERLESS=-1, MALE=0, FEMALE=1
MAX_MOVES = 4                # Max moves per Pokemon
MAX_PARTY_SIZE = 6           # Max party size per side
MAX_ACTIVE_PER_SIDE = 2      # Max active Pokemon per side (doubles)
MAX_REWARD_OPTIONS = 3       # Free reward slots after a battle
MAX_SHOP_OPTIONS = 12        # Purchasable shop slots
ACTION_SPACE_SIZE = 58       # Total discrete actions

# ─── Observation Vector Layout (mirrors spaces.ts) ───
ABILITY_FEATURE_DIM = 40     # Semantic features per ability (v3: replaces ability_id/310)
MODIFIER_FEATURE_DIM = 20   # Semantic features per modifier (v4: 20-dim feature vector)
MOVE_BLOCK_DIM = 132         # Dims per move slot (v7: +46 MoveAttr boolean flags)
POKEMON_BLOCK_DIM = 771      # 243 non-move + 4*132 moves (v7: MoveAttr boolean flags)
FIELD_STATE_DIM = 94         # Weather/terrain/arena tags/turns
BATTLE_META_DIM = 40         # Wave/turn/money/score/pokeballs/capabilities/game_mode_flags
MODIFIER_PHASE_DIM = 225     # v4: header(3) + reward(3*28) + shop(6*23)
MODIFIER_INVENTORY_DIM = 220 # v4: held(4*45) + party(9) + lapsing(23) + enemy(8)
DERIVED_FIELDS_DIM = 28      # Type effectiveness, STAB, speed ordering
PHASE_INDICATOR_DIM = 16     # One-hot over DecisionPhase
TOTAL_POKEMON_SLOTS = 12     # 2 active + 4 bench per side
OBSERVATION_DIM = 9875       # 12*771 + 94 + 40 + 225 + 220 + 28 + 16

# Move target enum values (MoveTarget) for reference
MOVE_TARGET_USER = 0
MOVE_TARGET_OTHER = 1
MOVE_TARGET_ALL_OTHERS = 2
MOVE_TARGET_NEAR_OTHER = 3
MOVE_TARGET_ALL_NEAR_OTHERS = 4
MOVE_TARGET_NEAR_ENEMY = 5
MOVE_TARGET_ALL_NEAR_ENEMIES = 6
MOVE_TARGET_RANDOM_NEAR_ENEMY = 7
MOVE_TARGET_ALL_ENEMIES = 8
MOVE_TARGET_ATTACKER = 9
MOVE_TARGET_NEAR_ALLY = 10
MOVE_TARGET_ALLY = 11
MOVE_TARGET_USER_OR_NEAR_ALLY = 12
MOVE_TARGET_USER_AND_ALLIES = 13
MOVE_TARGET_ALL = 14
MOVE_TARGET_USER_SIDE = 15
MOVE_TARGET_ENEMY_SIDE = 16
MOVE_TARGET_BOTH_SIDES = 17
MOVE_TARGET_PARTY = 18
MOVE_TARGET_CURSE = 19

# ArenaTagSide values
ARENA_TAG_SIDE_BOTH = 0
ARENA_TAG_SIDE_PLAYER = 1
ARENA_TAG_SIDE_ENEMY = 2

# BattlerIndex values
BATTLER_INDEX_ATTACKER = -1
BATTLER_INDEX_PLAYER = 0
BATTLER_INDEX_PLAYER_2 = 1
BATTLER_INDEX_ENEMY = 2
BATTLER_INDEX_ENEMY_2 = 3


# ═══════════════════════════════════════════════════════════════════════════
# 1a. STAT CHANGE (used by MoveSlot.stat_changes)
# ═══════════════════════════════════════════════════════════════════════════

class StatChange(TypedDict):
    """A stat stage change caused by a move (e.g. Swords Dance +2 ATK)."""

    stat_id: int                # Stat enum: ATK=1..EVA=7
    stages: int                 # Number of stages changed (-3 to +3 typically)
    self_target: bool           # True if the change applies to the user, False for target
    chance: int                 # Probability (100 = guaranteed, e.g. Close Combat -1 DEF)


# ═══════════════════════════════════════════════════════════════════════════
# 1b. MOVE SLOT
# ═══════════════════════════════════════════════════════════════════════════

class MoveSlot(TypedDict):
    """One of up to 4 move slots on a Pokemon."""

    # --- Identity ---
    move_id: int                # Unique numeric ID of the move (Moves enum)
    name: str                   # Localized display name (e.g. "Thunderbolt")

    # --- Typing ---
    type: int                   # PokemonType enum (0-18)
    category: int               # MoveCategory: PHYSICAL=0, SPECIAL=1, STATUS=2

    # --- Stats ---
    power: int                  # Base power (-1 for status moves, >0 for attacks)
    accuracy: int               # Base accuracy (100 = 100%, -1 = always hits)
    priority: int               # Move priority (-7 to +5 typically)

    # --- PP ---
    pp_max: int                 # Max PP for this move (after PP Ups)
    pp_used: int                # PP already consumed
    pp_remaining: int           # pp_max - pp_used
    pp_up: int                  # Number of PP Up applications (0-3)

    # --- Targeting ---
    target: int                 # MoveTarget enum value (see constants above)

    # --- Flags (basic) ---
    is_usable: bool             # Whether the move can currently be selected
    makes_contact: bool         # Whether the move makes contact
    is_sound_based: bool        # Whether the move is sound-based
    is_powder: bool             # Whether the move is powder-based
    is_punching: bool           # Whether the move is a punching move
    is_slicing: bool            # Whether the move is a slicing move
    is_biting: bool             # Whether the move is a biting move
    is_ballistic: bool          # Whether the move is ballistic (Bulletproof)

    # --- Secondary Effects ---
    effect_chance: int          # % chance of secondary effect (-1 if none, 100 if guaranteed)
    status_effect: int          # StatusEffect inflicted (0=NONE, 1=POISON, ..., 7=FAINT)
    stat_changes: List[StatChange]  # Stat stage changes (e.g. [{stat:1, stages:2, self:True, chance:100}])
    drain_ratio: float          # Fraction of damage dealt healed back (Drain Punch=0.5, 0.0 if none)
    recoil_ratio: float         # Fraction of damage dealt as recoil (Brave Bird=0.33, 0.0 if none)
                                # NOTE: some moves use fraction of maxHP instead (Struggle=0.25, Chloroblast=0.5)
    heal_ratio: float           # Fraction of max HP healed for recovery moves (Recover=0.5, 0.0 if none)

    # --- Multi-hit ---
    is_multi_hit: bool          # Whether the move hits multiple times (Bullet Seed, Double Kick)
    multi_hit_type: int         # MultiHitType: -1=N/A, TWO=0, TWO_TO_FIVE=1, THREE=2, TEN=3, BEAT_UP=4
    crit_stage_boost: int       # Extra crit stages (0=normal, 1=Slash/Stone Edge, 99=always crit)

    # --- Strategic Flags ---
    is_charging: bool           # Two-turn move (Solar Beam, Fly, Dig)
    self_switch: bool           # User switches out after (U-turn, Volt Switch, Flip Turn)
    force_switch: bool          # Forces target to switch out (Whirlwind, Dragon Tail)
    traps_target: bool          # Traps target for 4-5 turns (Fire Spin, Wrap, Infestation)
    is_protect: bool            # Protection move (Protect, Detect, Baneful Bunker)
    is_sacrifice: bool          # User faints or loses HP (Explosion, Memento, Healing Wish)
    is_ohko: bool               # One-hit KO move (Sheer Cold, Fissure, Guillotine)
    ignores_protect: bool       # Bypasses Protect (Feint, Shadow Force, Phantom Force)
    ignores_abilities: bool     # Bypasses target ability (Sunsteel Strike, Moongeist Beam)
    ignores_substitute: bool    # Can hit through Substitute
    fixed_damage: int           # Fixed damage amount (Dragon Rage=40, Sonic Boom=20, 0 if none)

    # --- Ability-Interaction Flags ---
    is_pulse: bool              # Boosted by Mega Launcher (+50%)
    is_dance: bool              # Copied by Dancer ability

    # --- v6: Move Semantic Encoding (+36 fields) ---

    # Boolean attr flags (12)
    can_flinch: bool            # Has FlinchAttr (Iron Head, Rock Slide)
    can_confuse: bool           # Has ConfuseAttr (Hurricane, Confuse Ray)
    is_recharge: bool           # Has RechargeAttr — user loses next turn (Hyper Beam)
    is_frenzy: bool             # Has FrenzyAttr — locked for 2-3 turns, confused after (Outrage)
    is_typeless: bool           # Has TypelessAttr — no type for effectiveness (Struggle)
    creates_substitute: bool    # Has AddSubstituteAttr (Substitute)
    suppresses_ability: bool    # Has SuppressAbilitiesAttr (Gastro Acid)
    has_variable_power: bool    # Has VariablePowerAttr — power depends on context (Eruption, Gyro Ball)
    has_variable_type: bool     # Has VariableMoveTypeAttr — type changes (Weather Ball, Tera Blast)
    has_variable_category: bool # Has VariableMoveCategoryAttr (Photon Geyser, Shell Side Arm)
    bypass_burn_penalty: bool   # Has BypassBurnDamageReductionAttr (Facade)
    ignores_stat_stages: bool   # Has IgnoreOpponentStatStagesAttr (Sacred Sword, Chip Away)

    # Field control (4)
    weather_change: int         # WeatherType set by move (0=none, 1-9=specific)
    terrain_change: int         # TerrainType set by move (0=none, 1-4=specific)
    sets_arena_tag: bool        # Has AddArenaTagAttr
    removes_arena_tags: bool    # Has RemoveArenaTagsAttr (Defog, Rapid Spin)

    # Arena tag semantics (3)
    sets_hazard: bool           # Arena tag is entry hazard (Stealth Rock, Spikes, etc.)
    sets_screen: bool           # Arena tag is screen (Reflect, Light Screen, Aurora Veil)
    arena_tag_self_side: bool   # Arena tag targets own side

    # Battler tag semantics (3)
    applies_battler_tag: bool   # Has AddBattlerTagAttr (excl. flinch/confuse/recharge)
    applies_move_restriction: bool  # Tag restricts moves (Taunt, Encore, Disable, etc.)
    applies_continuous_damage: bool # Tag applies ongoing damage (Leech Seed, Salt Cure, etc.)

    # Fixed damage discrimination (4)
    is_user_hp_damage: bool     # UserHpDamageAttr (Endeavor)
    is_target_half_hp: bool     # TargetHalfHpDamageAttr (Super Fang)
    is_counter_damage: bool     # CounterDamageAttr (Counter, Mirror Coat, Metal Burst)
    is_level_damage: bool       # LevelDamageAttr (Seismic Toss, Night Shade)

    # Additional strategic flags (2)
    is_delayed_attack: bool     # DelayedAttackAttr (Future Sight, Doom Desire)
    post_victory_stat_boost: bool  # PostVictoryStatStageChangeAttr

    # Missing MoveFlags (8)
    is_wind_move: bool          # WIND_MOVE — Wind Rider/Wind Power interaction
    is_reckless_move: bool      # RECKLESS_MOVE — Reckless ability boost
    is_reflectable: bool        # REFLECTABLE — Magic Bounce interaction
    hides_user: bool            # HIDE_USER — semi-invulnerable (Fly, Dig)
    is_triage_move: bool        # TRIAGE_MOVE — Triage ability priority boost
    check_all_hits: bool        # CHECK_ALL_HITS — multi-hit interaction
    affected_by_gravity: bool   # GRAVITY — disabled under Gravity
    hides_target: bool          # HIDE_TARGET — Phantom Force etc.

    # ── v7: MoveAttr boolean flags (+46 fields) ──

    # Group 8: Item Manipulation (3)
    steals_item: bool           # StealHeldItemChanceAttr (Thief, Covet)
    removes_item: bool          # RemoveHeldItemAttr (Knock Off, Incinerate)
    steals_berry: bool          # StealEatBerryAttr (Pluck, Bug Bite)

    # Group 9: Stat Manipulation (8)
    copies_stats: bool          # CopyStatsAttr (Psych Up)
    inverts_stats: bool         # InvertStatsAttr (Topsy-Turvy)
    resets_stats: bool          # ResetStatsAttr (Clear Smog, Haze, Freezy Frost)
    swaps_stat_stages: bool     # SwapStatStagesAttr (Heart Swap, Power/Guard Swap)
    steals_stat_boosts: bool    # SpectralThiefAttr (Spectral Thief)
    averages_stats: bool        # AverageStatsAttr (Power/Guard Split)
    swaps_single_stat: bool     # SwapStatAttr (Speed Swap)
    shifts_own_stat: bool       # ShiftStatAttr (Power Shift)

    # Group 10: HP / PP / Revival (3)
    splits_hp: bool             # HpSplitAttr (Pain Split)
    reduces_pp: bool            # ReducePpMoveAttr (Spite, Eerie Spell)
    revives_ally: bool          # RevivalBlessingAttr (Revival Blessing)

    # Group 11: Move-Calling (5)
    copies_last_move: bool      # CopyMoveAttr (Mirror Move)
    calls_random_move: bool     # RandomMoveAttr (Metronome)
    calls_moveset_move: bool    # RandomMovesetMoveAttr (Sleep Talk, Assist)
    copies_move_temp: bool      # MovesetCopyMoveAttr (Mimic)
    copies_move_perm: bool      # SketchAttr (Sketch)

    # Group 12: Ability Manipulation (5)
    copies_ability: bool        # AbilityCopyAttr (Role Play, Doodle)
    swaps_abilities: bool       # SwitchAbilitiesAttr (Skill Swap)
    changes_ability: bool       # AbilityChangeAttr (Worry Seed, Simple Beam)
    gives_ability: bool         # AbilityGiveAttr (Entrainment)
    suppresses_if_acted: bool   # SuppressAbilitiesIfActedAttr (Core Enforcer)

    # Group 13: Targeting & Priority (4)
    bypass_redirect: bool       # BypassRedirectAttr (Snipe Shot)
    forces_target_next: bool    # AfterYouAttr (After You)
    forces_target_last: bool    # ForceLastAttr (Quash)
    has_conditional_priority: bool  # IncrementMovePriorityAttr (Grassy Glide)

    # Group 14: Status & Tag Manipulation (5)
    cures_party_status: bool    # PartyStatusCureAttr (Aromatherapy, Heal Bell)
    transfers_status: bool      # PsychoShiftEffectAttr (Psycho Shift)
    heals_status: bool          # HealStatusEffectAttr (~10 self-cure moves)
    removes_battler_tag: bool   # RemoveBattlerTagAttr (Rapid Spin)
    removes_substitutes: bool   # RemoveAllSubstitutesAttr (Tidy Up)

    # Group 15: Transform & Special Moves (4)
    transforms_into_target: bool  # TransformAttr (Transform)
    is_curse: bool              # CurseAttr (Curse)
    is_wish: bool               # WishAttr (Wish)
    is_destiny_bond: bool       # DestinyBondAttr (Destiny Bond)

    # Group 16: Field Control (3)
    swaps_arena_tags: bool      # SwapArenaTagsAttr (Court Change)
    clears_weather: bool        # ClearWeatherAttr (weather-clearing moves)
    clears_terrain: bool        # ClearTerrainAttr (terrain-clearing moves)

    # Group 17: Damage Calc & Misc (6)
    has_variable_target: bool   # VariableTargetAttr (Expanding Force)
    resists_last_type: bool     # ResistLastMoveTypeAttr (Conversion 2)
    has_variable_accuracy: bool # VariableAccuracyAttr (Thunder in rain)
    uses_alt_stat: bool         # VariableAtkAttr || VariableDefAttr (Psyshock, Body Press)
    overrides_type_chart: bool  # MoveTypeChartOverrideAttr (Freeze-Dry)
    scatters_money: bool        # MoneyAttr (Pay Day, Happy Hour)
    # Group 18: v8 survival / HP-relative semantics
    survives_at_1hp: bool       # SurviveDamageAttr (False Swipe, Hold Back)
    matches_user_hp: bool       # MatchHpAttr (Endeavor)
    hp_cost_stat_boost: bool    # CutHpStatStageBoostAttr (Belly Drum)
    hits_semi_invulnerable: bool  # HitsTagAttr / HitsTagForDoubleDamageAttr (Stomp; Earthquake vs Dig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VOLATILE TAG (BattlerTag)
# ═══════════════════════════════════════════════════════════════════════════

class VolatileTag(TypedDict):
    """An active volatile status condition on a Pokemon (BattlerTag).

    Many BattlerTag subclasses carry extra state beyond tagType/turnCount.
    The optional fields below capture the most RL-relevant extra state.
    """

    tag_type: str               # BattlerTagType string value (e.g. "CONFUSED")
    turn_count: int             # Number of turns remaining, or <=0 if indefinite
    source_id: Optional[int]    # Pokemon PID (Pokemon.id) of the setter, or None
    source_move: Optional[int]  # MoveId that created this tag, or None

    # --- Extra state for specific tag subclasses ---
    substitute_hp: Optional[int]       # SubstituteTag: remaining HP of the substitute
    stockpile_count: Optional[int]     # StockpilingTag: Stockpile stacks (1-3)
    encore_move_id: Optional[int]      # EncoreTag: which move is locked
    disabled_move_id: Optional[int]    # DisabledTag: which move is disabled
    type_boost_type: Optional[int]     # TypeBoostTag: which PokemonType is boosted
    type_boost_value: Optional[float]  # TypeBoostTag: boost multiplier
    crit_boost_stages: Optional[int]   # CritBoostTag: crit stage bonus (Focus Energy, etc.)

    # --- v5 additions (completeness audit) ---
    gorilla_tactics_move_id: Optional[int]       # GorillaTacticsTag: move locked into (Choice-like)
    highest_stat_boost_stat: Optional[int]       # HighestStatBoostTag: which EffectiveStat (Protosynthesis/Quark Drive)
    highest_stat_boost_multiplier: Optional[float]  # HighestStatBoostTag: 1.3 or 1.5 (1.5 for Speed)
    supreme_overlord_faint_count: Optional[int]  # SupremeOverlordTag: prior faints (0-5, +10% dmg each)
    autotomize_count: Optional[int]    # AutotomizedTag: stacks reducing weight by 100kg each


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUEUED MOVE
# ═══════════════════════════════════════════════════════════════════════════

class QueuedMove(TypedDict):
    """A move queued for execution or in move history.

    In move_history entries, ``result`` contains the outcome of the move.
    In move_queue entries, ``result`` is None (not yet executed).
    """

    move_id: int                # Moves enum ID
    targets: List[int]          # BattlerIndex targets
    use_mode: int               # MoveUseMode enum value
    result: Optional[int]       # MoveResult enum (SUCCESS, MISS, FAIL, etc.), None if pending


# ═══════════════════════════════════════════════════════════════════════════
# 4. ATTACK RECEIVED
# ═══════════════════════════════════════════════════════════════════════════

class AttackReceived(TypedDict):
    """Record of an attack this Pokemon received during the current turn/wave.

    Mirrors the game's ``AttackMoveResult`` interface.
    """

    source_battler_index: int   # BattlerIndex of the attacker
    source_id: int              # Unique ID of the attacking Pokemon
    move_id: int                # The move that dealt damage (MoveId enum)
    damage: int                 # Amount of HP damage dealt
    critical: bool              # Whether the hit was a critical hit
    result: int                 # HitResult enum value (DamageResult subset)


# ═══════════════════════════════════════════════════════════════════════════
# 5. HELD ITEM
# ═══════════════════════════════════════════════════════════════════════════

class HeldItem(TypedDict):
    """A held item (PokemonHeldItemModifier) on a specific Pokemon."""

    modifier_class: str         # Class name (e.g. "AttackTypeBoosterModifier")
    modifier_id: str            # Modifier type ID string
    name: str                   # Display name (e.g. "Silk Scarf")
    stack_count: int            # Number of stacks (most items stack 1-5)
    max_stack_count: int        # Maximum stacks allowed
    is_transferable: bool       # Whether the item can be transferred (Baton Pass, etc.)

    # Type-specific fields (present only for certain modifier classes)
    type_id: Optional[int]      # PokemonType for type-boosting items (None if N/A)
    stat_id: Optional[int]      # Stat ID for single-stat items like BaseStatModifier (None if N/A)
    status_effect: Optional[int]  # StatusEffect for TurnStatusEffectModifier (Toxic/Flame Orb)

    # Berry-specific fields (BerryModifier)
    berry_type: Optional[int]   # BerryType enum value (None if not a berry)
    consumed: Optional[bool]    # Whether the berry has been consumed (None if not a berry)

    # v5 additions (completeness audit)
    stat_modifier: Optional[int]       # PokemonBaseStatTotalModifier: +10 (Shuckle Juice good) or -15 (bad)
    form_change_item: Optional[int]    # PokemonFormChangeItemModifier: FormChangeItem enum value
    form_change_active: Optional[bool] # PokemonFormChangeItemModifier: whether currently active


# ═══════════════════════════════════════════════════════════════════════════
# 6. TURN DATA
# ═══════════════════════════════════════════════════════════════════════════

class TurnData(TypedDict):
    """Per-turn transient data for a Pokemon (reset each turn).

    Mirrors the game's ``PokemonTurnData`` class.
    """

    damage_taken: int           # Total damage taken this turn (damageTaken)
    total_damage_dealt: int     # Total damage dealt this turn (totalDamageDealt)
    attacks_received: List[AttackReceived]  # All attacks received this turn
    order: int                  # Turn order index (lower = faster)
    hit_count: int              # Number of hits for current multi-hit move
    acted: bool                 # Whether the Pokemon has acted this turn
    switched_in_this_turn: bool # Whether switched in this turn (not initial summon)
    stat_stages_increased: bool # Whether any stat stages were raised
    stat_stages_decreased: bool # Whether any stat stages were lowered
    berries_eaten: List[int]    # BerryType values of berries eaten this turn


# ═══════════════════════════════════════════════════════════════════════════
# 7. BATTLE DATA (cumulative within current battle/wave)
# ═══════════════════════════════════════════════════════════════════════════

class BattleData(TypedDict):
    """Cumulative data for a Pokemon within the current battle/wave.

    Mirrors the game's ``PokemonBattleData`` class.
    """

    hit_count: int              # Direct hits received this battle (PokemonBattleData.hitCount, for Rage Fist)
    has_eaten_berry: bool       # Whether a berry was eaten this battle (for Belch)
    berries_eaten: List[int]    # BerryType values of berries eaten (for Harvest)
    abilities_applied: List[int]  # Ability IDs that triggered (from PokemonWaveData)


# ═══════════════════════════════════════════════════════════════════════════
# 8. POKEMON STATE (per slot)
# ═══════════════════════════════════════════════════════════════════════════

class PokemonState(TypedDict):
    """Complete state of a single Pokemon slot.

    The ``valid`` field is False for empty slots (e.g. bench slots when
    the party has fewer than 6 Pokemon).
    """

    # --- Presence ---
    valid: bool                 # True if this slot is occupied

    # --- Identity ---
    species_id: int             # National Dex number (SpeciesId enum)
    species_name: str           # Display name (e.g. "Pikachu")
    form_index: int             # Form variant index (0 = default)
    level: int                  # 1-100
    gender: int                 # Gender enum: GENDERLESS=-1, MALE=0, FEMALE=1
    friendship: int             # 0-255 (affects Return/Frustration)
    shiny: bool                 # Whether this Pokemon is shiny
    variant: int                # Shiny variant (0-2)

    # --- HP ---
    hp: int                     # Current hit points
    max_hp: int                 # Maximum hit points (from stats)
    hp_ratio: float             # hp / max_hp (0.0-1.0)

    # --- Stats ---
    # Each list is ordered [HP, ATK, DEF, SPATK, SPDEF, SPD]
    base_stats: List[int]       # Species base stats (length 6)
    ivs: List[int]              # Individual values 0-31 (length 6)
    stats: List[int]            # Computed battle stats (length 6)

    # --- Stat Stages ---
    # Ordered [ATK, DEF, SPATK, SPDEF, SPD, ACC, EVA] — range -6 to +6
    # Only meaningful for on-field Pokemon with summonData.
    stat_stages: List[int]      # Length 7

    # --- Status ---
    status_effect: int          # StatusEffect enum (0=NONE, 1=POISON, ..., 7=FAINT)
    toxic_turn_count: int       # Turns of toxic damage accumulated (0 if not toxic)
    sleep_turns_remaining: int  # Turns of sleep left (0 if not asleep)

    # --- Typing ---
    types: List[int]            # Current types as PokemonType ints (length 1-2)
    tera_type: int              # Tera type (PokemonType), or -1 if none
    is_terastallized: bool      # Whether currently Terastallized
    added_type: int             # Type added by Forest's Curse / Trick-or-Treat (-1 if none)

    # --- Abilities ---
    ability_id: int             # Current (effective) ability ID (AbilityId enum)
    ability_name: str           # Display name of current ability
    passive_ability_id: int     # Passive ability ID
    passive_ability_name: str   # Display name of passive ability
    has_passive: bool           # Whether passive ability is unlocked
    ability_suppressed: bool    # Whether ability is currently suppressed (Gastro Acid)
    ability_revealed: bool      # Whether ability has been revealed (summonData flag)

    # --- Nature ---
    nature: int                 # Nature enum (0-24)
    nature_multipliers: List[float]  # Stat multipliers for ATK/DEF/SPATK/SPDEF/SPD (length 5)
                                # Each is 0.9, 1.0, or 1.1

    # --- Moves ---
    moves: List[MoveSlot]       # Up to 4 move slots (may be fewer)
    move_history: List[QueuedMove]  # Recent move history (from summonData.moveHistory)

    # --- Catch Ball ---
    pokeball: int               # PokeballType used to catch this Pokemon (0-5)

    # --- Volatile Tags ---
    volatile_tags: List[VolatileTag]  # All active BattlerTag volatile conditions

    # --- Boss ---
    is_boss: bool               # Whether this is a boss Pokemon
    boss_segments: int          # Total boss shield segments (0 if not boss)
    boss_segment_index: int     # Current shield segment index (0 if not boss)

    # --- AI (enemies only) ---
    ai_type: int                # AiType enum: RANDOM=0, SMART_RANDOM=1, SMART=2

    # --- Fusion ---
    is_fusion: bool             # Whether this Pokemon is a fusion
    fusion_species_id: Optional[int]  # Species ID of the fusion partner, or None

    # --- Positioning ---
    is_on_field: bool           # Whether currently on the battlefield
    is_player: bool             # True for player Pokemon, False for enemy
    battler_index: int          # BattlerIndex: PLAYER=0, PLAYER_2=1, ENEMY=2, ENEMY_2=3
    field_index: int            # 0 or 1 (slot on that side)

    # --- Held Items ---
    held_items: List[HeldItem]  # All held items on this Pokemon

    # === FIELDS MISSING FROM CURRENT FLOAT32 ENCODING (spaces.ts) ===

    # --- Queued Moves ---
    move_queue: List[QueuedMove]  # Queued moves (Encore, two-turn, Outrage, etc.)

    # --- Battle Participation ---
    wave_turn_count: int        # Turns this Pokemon has been active this wave
    is_fainted: bool            # Whether HP <= 0 (isFainted())
    is_active: bool             # isActive() — on field AND not fainted

    # --- Trapping & Grounding ---
    is_trapped: bool            # Cannot switch out (Mean Look, Arena Trap, etc.)
    is_grounded: bool           # Affected by ground-based effects (isGrounded())

    # --- Transform / Illusion ---
    transform_species_id: Optional[int]  # Species ID if Transform is active, else None
    transform_moves: Optional[List[MoveSlot]]  # Moveset from Transform, or None
    illusion_species_id: Optional[int]    # Species ID of Illusion disguise, or None

    # --- Combat History ---
    attacks_received: List[AttackReceived]  # Convenience copy of turnData.attacks_received (resets each turn)
    turn_data: TurnData         # Per-turn transient data
    battle_data: BattleData     # Cumulative battle data

    # --- Weight & Catch ---
    weight: float               # Species weight in kg (Low Kick, Heavy Slam)
    catch_rate: int             # Species base catch rate (0-255)
    base_total: int             # Sum of base stats

    # --- Stellar ---
    stellar_types_boosted: List[int]  # PokemonType values already Stellar-boosted

    # --- v5 additions (completeness audit) ---
    berries_eaten_last: List[int]     # BerryType values eaten last turn (Cud Chew re-eats these)
    exp_to_next_level: int            # EXP needed to reach next level (for Rare Candy / EXP Share value)
    luck: int                         # Luck value (affects modifier tier generation in shop)
    endured_this_wave: bool           # Whether Focus Band / Endure Token already triggered this wave


# ═══════════════════════════════════════════════════════════════════════════
# 9. ARENA TAG
# ═══════════════════════════════════════════════════════════════════════════

class ArenaTagState(TypedDict):
    """An active arena tag (hazard, screen, field condition)."""

    tag_type: str               # ArenaTagType string value (e.g. "STEALTH_ROCK")
    side: int                   # ArenaTagSide: BOTH=0, PLAYER=1, ENEMY=2
    turn_count: int             # Turns remaining; <=0 if indefinite (hazards use 0)
    layers: int                 # Layer count for stackable tags (Spikes: 1-3,
                                # Toxic Spikes: 1-2); 1 for non-stackable
    source_id: Optional[int]    # Pokemon PID (Pokemon.id) of the setter, or None


# ═══════════════════════════════════════════════════════════════════════════
# 10. POSITIONAL TAG (Future Sight, Wish, etc.)
# ═══════════════════════════════════════════════════════════════════════════

class PositionalTag(TypedDict):
    """A delayed positional effect targeting a specific field slot."""

    tag_type: str               # PositionalTagType: "DELAYED_ATTACK" or "WISH"
    countdown: int              # Turns until activation
    target_index: int           # BattlerIndex of the target slot
    source_id: Optional[int]    # Pokemon PID of the user (DelayedAttackTag only), or None
    move_id: Optional[int]      # Move ID (Future Sight / Doom Desire), or None
    heal_hp: Optional[int]      # WishTag: HP amount to heal, or None


# ═══════════════════════════════════════════════════════════════════════════
# 11. FIELD STATE (weather, terrain, arena tags)
# ═══════════════════════════════════════════════════════════════════════════

class FieldState(TypedDict):
    """Complete arena / field / environment state."""

    # --- Biome ---
    biome_id: int               # BiomeId enum (0-40+)
    biome_name: str             # Display name of the biome

    # --- Weather ---
    weather_type: int           # WeatherType enum (0-9), 0 = NONE
    weather_turns_left: int     # Turns remaining; 0 if permanent (ability-set), 0 if none
    weather_is_permanent: bool  # Whether weather was set by an ability (turnsLeft=0)
    weather_suppressed: bool    # Whether Cloud Nine / Air Lock is suppressing weather

    # --- Terrain ---
    terrain_type: int           # TerrainType enum (0-4), 0 = NONE
    terrain_turns_left: int     # Turns remaining; 0 if permanent or none
    terrain_is_permanent: bool  # Whether terrain was set by an ability (permanent)

    # --- Tera Usage ---
    player_teras_used: int      # Number of teras used by the player this battle

    # --- Arena Tags ---
    arena_tags: List[ArenaTagState]  # All active arena tags (screens, hazards, rooms)

    # --- Positional Tags ---
    positional_tags: List[PositionalTag]  # Delayed effects (Future Sight, Wish, etc.)

    # --- Battle Configuration ---
    is_double_battle: bool      # Whether the current battle is a double battle

    # --- Strategic Flags (redundant but important for RL) ---
    trick_room_active: bool     # Whether Trick Room is active
    gravity_active: bool        # Whether Gravity is active
    ignore_abilities: bool      # Whether Mold Breaker / Teravolt / Turboblaze is active

    # --- Hazard Layers (convenience, also in arena_tags) ---
    player_spikes_layers: int   # 0-3
    player_toxic_spikes_layers: int  # 0-2
    player_stealth_rock: bool   # True if Stealth Rock is on player side
    player_sticky_web: bool     # True if Sticky Web is on player side
    enemy_spikes_layers: int    # 0-3
    enemy_toxic_spikes_layers: int   # 0-2
    enemy_stealth_rock: bool    # True if Stealth Rock is on enemy side
    enemy_sticky_web: bool      # True if Sticky Web is on enemy side


# ═══════════════════════════════════════════════════════════════════════════
# 12. POKEBALL COUNTS
# ═══════════════════════════════════════════════════════════════════════════

class PokeballCounts(TypedDict):
    """Count of each Pokeball type in the player's inventory."""

    pokeball: int               # PokeballType.POKEBALL (0)
    great_ball: int             # PokeballType.GREAT_BALL (1)
    ultra_ball: int             # PokeballType.ULTRA_BALL (2)
    rogue_ball: int             # PokeballType.ROGUE_BALL (3)
    master_ball: int            # PokeballType.MASTER_BALL (4)


# ═══════════════════════════════════════════════════════════════════════════
# 13. TRAINER INFO
# ═══════════════════════════════════════════════════════════════════════════

class TrainerInfo(TypedDict):
    """Info about the current trainer, if applicable."""

    trainer_type: int           # TrainerType enum value
    trainer_name: str           # Display name (e.g. "Youngster Joey")
    is_double: bool             # Whether this trainer forces a double battle
    is_boss: bool               # Whether this is a gym leader / elite four / champion
    party_template_size: int    # Expected number of Pokemon in trainer's party
    specialty_type: Optional[int]  # PokemonType the trainer specializes in (None if N/A)
    tera_mode: Optional[int]    # TeraAIMode enum: NO_TERA=0, etc. (None if unknown)


# ═══════════════════════════════════════════════════════════════════════════
# 14. MYSTERY ENCOUNTER
# ═══════════════════════════════════════════════════════════════════════════

class MysteryEncounterOption(TypedDict):
    """A selectable option in a mystery encounter."""

    index: int                  # Option index (0-based)
    label: str                  # Display text
    has_requirements: bool      # Whether the option has prerequisites
    is_available: bool          # Whether the option can currently be selected


class MysteryEncounterState(TypedDict):
    """State of the current mystery encounter, if any."""

    encounter_type: int         # MysteryEncounterType enum value
    encounter_name: str         # Display name
    options: List[MysteryEncounterOption]  # Available choices


# ═══════════════════════════════════════════════════════════════════════════
# 15. CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════

class ChallengeInfo(TypedDict):
    """An active challenge modifier for the run."""

    challenge_type: int         # ChallengeType enum value
    challenge_name: str         # Display name (e.g. "Single Type", "Nuzlocke")
    value: int                  # Challenge parameter value
    severity: int               # Difficulty severity level


# ═══════════════════════════════════════════════════════════════════════════
# 16. BATTLE / RUN STATE
# ═══════════════════════════════════════════════════════════════════════════

class BattleState(TypedDict):
    """Metadata about the current battle and overall run."""

    # --- Biome (B3 fix: also available at battle level for convenience) ---
    biome_id: int               # BiomeId enum (0-40+), same as field.biome_id

    # --- Current Battle ---
    wave_index: int             # Current wave number (1-200 in Classic)
    turn: int                   # Current turn within this battle (starts at 0, incremented by incrementTurn())
    battle_type: int            # BattleType: WILD=0, TRAINER=1, CLEAR=2, MYSTERY_ENCOUNTER=3
    battle_spec: int            # 0=DEFAULT, 1=FINAL_BOSS
    is_double: bool             # Whether current battle is doubles
    escape_attempts: int        # Number of failed escape attempts this battle

    # --- Combatant Counts ---
    player_alive_count: int     # Number of non-fainted player Pokemon
    enemy_alive_count: int      # Number of non-fainted enemy Pokemon
    player_faints_battle: int   # Player faints in current battle (derived: Battle.playerFaintsHistory.length)
    enemy_faints_battle: int    # Enemy faints in current battle (Battle.enemyFaints)

    # --- Last Move ---
    last_move_id: Optional[int] # Move ID of the last move used, or None

    # --- Economy ---
    money: int                  # Current money
    score: int                  # Current score
    pokeball_counts: PokeballCounts  # Pokeball inventory

    # --- Capabilities ---
    can_run: bool               # Whether running is allowed
    can_catch: bool             # Whether catching is allowed (wild battle, singles)
    tera_available: bool        # Whether Tera is available (not yet used this battle)

    # --- Run Metadata ---
    game_mode: int              # GameModes enum: CLASSIC=0, ENDLESS=1, etc.
    seed: str                   # Current RNG seed string

    # --- Trainer ---
    trainer: Optional[TrainerInfo]  # Trainer info if battle_type == TRAINER, else None

    # --- Mystery Encounter ---
    mystery_encounter: Optional[MysteryEncounterState]  # If in mystery encounter

    # === FIELDS MISSING FROM CURRENT FLOAT32 ENCODING ===

    # --- Settings ---
    battle_style: int           # 0 = SWITCH (can swap on KO), 1 = SET (no free swap)
    time_of_day: int            # TimeOfDay enum: DAWN=0, DAY=1, DUSK=2, NIGHT=3

    # --- Run Totals ---
    player_faints_biome: int    # Player faints in the current biome (arena.playerFaints)
    money_scattered: int        # Money scattered from moves like Pay Day/Make It Rain this battle

    # --- Challenges ---
    challenges: List[ChallengeInfo]  # Active challenge modifiers

    # --- Modifier Phase Control ---
    lock_modifier_tiers: bool   # Whether modifier tiers are locked (reroll preserves tiers)
    reroll_count: int           # Number of rerolls done this wave

    # --- Escape ---
    failed_run_away: bool       # Whether the player tried to run and failed this turn

    # --- v5 additions (completeness audit) ---
    has_no_shop: bool           # Whether the shop is disabled (e.g. Daily runs)
    has_trainers: bool          # Whether trainer battles can occur (false in some Endless modes)
    is_spliced_only: bool       # Whether all enemies are fusions (Spliced Endless mode)
    seen_enemy_count: int       # Number of distinct enemy Pokemon revealed this battle (trainer)
    enemy_switch_counter: int   # Times the enemy has switched this battle
    offset_gym: bool            # Whether gyms are offset (wave%30==0 instead of default wave%30==20)


# ═══════════════════════════════════════════════════════════════════════════
# 17. PARTY MODIFIER (non-Pokemon-specific modifiers)
# ═══════════════════════════════════════════════════════════════════════════

class PartyModifier(TypedDict):
    """A party-wide modifier (not attached to a specific Pokemon)."""

    modifier_class: str         # Class name (e.g. "ExpShareModifier")
    modifier_id: str            # Modifier type ID string
    name: str                   # Display name (e.g. "EXP. Share")
    stack_count: int            # Number of stacks
    max_stack_count: int        # Maximum allowed stacks

    # Subtype-specific fields
    type_id: Optional[int]      # For type-specific modifiers
    stat_id: Optional[int]      # For stat-specific modifiers
    status_effect: Optional[int]  # StatusEffect for EnemyAttackStatusEffectChanceModifier


# ═══════════════════════════════════════════════════════════════════════════
# 18. LAPSING MODIFIER (battle-count limited)
# ═══════════════════════════════════════════════════════════════════════════

class LapsingModifier(TypedDict):
    """A modifier that expires after a set number of battles.

    Includes TempStatStageBoosterModifier (X Attack, etc.) and
    DoubleBattleChanceBoosterModifier (Max Lure, etc.).
    """

    modifier_class: str         # Class name (e.g. "TempStatStageBoosterModifier")
    modifier_id: str            # Modifier type ID string
    name: str                   # Display name (e.g. "X Attack")
    stack_count: int            # Number of stacks
    battles_remaining: int      # Battles until this modifier expires

    # TempStatStageBoosterModifier-specific fields
    stat_id: Optional[int]      # TempBattleStat enum (which stat is boosted, None if N/A)
    boost: Optional[float]      # Boost amount (stat stage multiplier increase, None if N/A)


# ═══════════════════════════════════════════════════════════════════════════
# 19. MODIFIER INVENTORY (all player modifiers)
# ═══════════════════════════════════════════════════════════════════════════

class ModifierInventory(TypedDict):
    """Complete player modifier inventory.

    ``held_items`` is keyed by party slot index (0-5) as a string because
    JSON object keys must be strings.
    """

    # Per-pokemon held items, keyed by party slot index ("0" - "5")
    held_items: Dict[str, List[HeldItem]]

    # Party-wide modifiers (EXP Share, Lucky Egg, Amulet Coin, etc.)
    party_modifiers: List[PartyModifier]

    # Modifiers that expire after N battles
    lapsing_modifiers: List[LapsingModifier]

    # Enemy-side hidden modifiers (enemy stat boosts, etc.)
    enemy_modifiers: List[PartyModifier]


# ═══════════════════════════════════════════════════════════════════════════
# 20. REWARD OPTION (SelectModifierPhase)
# ═══════════════════════════════════════════════════════════════════════════

class RewardOption(TypedDict):
    """A free reward option shown at end of wave."""

    index: int                  # Position (0-2)
    tier: int                   # ModifierTier: COMMON=0 .. LUXURY=5
    upgrade_count: int          # Number of tier upgrades applied
    name: str                   # Display name
    modifier_id: str            # Modifier type ID string
    modifier_class: str         # Modifier class name
    target_kind: str            # "none", "pokemon", "move", or "pokemon_pair"
    is_pokemon_modifier: bool   # Whether this targets a specific Pokemon

    # Item effect fields (for RL evaluation of rewards)
    type_id: Optional[int]      # PokemonType for type-boosting items (e.g. Silk Scarf → NORMAL)
    stat_id: Optional[int]      # Stat for stat-boosting items (e.g. Protein → ATK)
    description: str            # Brief effect description for RL understanding


# ═══════════════════════════════════════════════════════════════════════════
# 21. SHOP OPTION (SelectModifierPhase)
# ═══════════════════════════════════════════════════════════════════════════

class ShopOption(TypedDict):
    """A purchasable shop item shown at end of wave."""

    index: int                  # Position (0-11)
    cost: int                   # Price in money
    tier: int                   # ModifierTier: COMMON=0 .. LUXURY=5
    name: str                   # Display name
    modifier_id: str            # Modifier type ID string
    modifier_class: str         # Modifier class name
    target_kind: str            # "none", "pokemon", "move", or "pokemon_pair"
    affordable: bool            # Whether the player has enough money

    # Item effect fields (for RL evaluation of shop items)
    type_id: Optional[int]      # PokemonType for type-boosting items
    stat_id: Optional[int]      # Stat for stat-boosting items
    description: str            # Brief effect description for RL understanding


# ═══════════════════════════════════════════════════════════════════════════
# 22. SHOP STATE (only during SelectModifierPhase)
# ═══════════════════════════════════════════════════════════════════════════

class ShopState(TypedDict):
    """State of the reward/shop selection (only present during SelectModifierPhase)."""

    reward_options: List[RewardOption]  # Free reward choices (up to 3)
    shop_options: List[ShopOption]      # Purchasable items (up to 12)
    can_reroll: bool                    # Whether rerolling is available
    reroll_cost: int                    # Money cost to reroll
    money: int                          # Current money (for affordability checks)


# ═══════════════════════════════════════════════════════════════════════════
# 23. PHASE / DECISION STATE
# ═══════════════════════════════════════════════════════════════════════════

class PhaseInfo(TypedDict):
    """Current game phase and decision point information."""

    # --- Phase Identity ---
    current_phase: str          # Phase name string. One of:
                                #   "command"          - choosing move/switch/ball/run
                                #   "modifier"         - selecting rewards or shop items
                                #   "modifier_target"  - choosing which Pokemon gets the item
                                #   "switch"           - forced switch (faint or move effect)
                                #   "check_switch"     - optional switch offer (SET/SWITCH style)
                                #   "learn_move"       - choosing to learn or skip a new move
                                #   "evolution"        - evolution happening (auto-handled)
                                #   "select_biome"     - choosing next biome
                                #   "revival_blessing"  - choosing which fainted mon to revive
                                #   "mystery"          - mystery encounter option selection
                                #   "game_over"        - run ended
                                #   "title"            - pre-game title screen
                                #   "target"           - selecting a move target (doubles)
                                #   "starter"          - starter selection (auto-handled by RL)
                                #   "form_change"      - form change (auto-handled)
                                #   "select_gender"    - gender selection (auto-handled)
                                #   "unknown"          - unrecognized phase

    # --- Command Phase Context ---
    command_field_index: int     # Which Pokemon is choosing (-1 if not command phase)
    command_pokemon_species: Optional[str]  # Species name of the choosing Pokemon

    # --- Action Space ---
    action_mask: List[bool]     # 58 booleans for valid actions
    valid_actions: List[int]    # Indices of valid actions (convenience)

    # --- Phase-Specific Metadata ---
    # These are present only for specific phases:
    learn_move_id: Optional[int]         # Move ID of the new move (learn_move phase)
    learn_move_name: Optional[str]       # Name of the move to learn (learn_move phase)
    learn_move_stats: Optional[MoveSlot] # Full stats of the new move (learn_move phase)
    learn_move_current: Optional[List[str]]  # Current moveset names (learn_move phase)
    biome_options: Optional[List[str]]   # Available biome names (select_biome phase)
    mystery_option_count: Optional[int]  # Number of ME options (mystery phase)
    is_game_over: Optional[bool]         # True if this is a game_over phase
    is_victory: Optional[bool]           # True if game_over was a victory


# ═══════════════════════════════════════════════════════════════════════════
# 24. ACTION INFO (for human-readable action labels)
# ═══════════════════════════════════════════════════════════════════════════

class ActionInfo(TypedDict):
    """Human-readable label for a valid action."""

    index: int                  # Action index (0-57)
    label: str                  # Human-readable label (e.g. "Fight: Thunderbolt -> Rattata")


# ═══════════════════════════════════════════════════════════════════════════
# 25. TOP-LEVEL GAME STATE
# ═══════════════════════════════════════════════════════════════════════════

class GameState(TypedDict):
    """Complete game state dictionary — the target schema for RL agent observation.

    ``state-builder.ts`` serializes the full live game state into this schema.
    Wire protocol messages include ``gameState`` matching this TypedDict, plus
    extra metadata fields (``step``, ``timestamp``, ``action_labels``).

    Slot naming convention:
      ``player_0`` through ``player_5``: player party (0-1 active, 2-5 bench)
      ``enemy_0``  through ``enemy_5``:  enemy party (0-1 active, 2-5 bench)
    """

    # --- Pokemon Slots (12 total) ---
    player_0: PokemonState      # Player active slot 0
    player_1: PokemonState      # Player active slot 1 (doubles), or bench if singles
    player_2: PokemonState      # Player bench slot 0
    player_3: PokemonState      # Player bench slot 1
    player_4: PokemonState      # Player bench slot 2
    player_5: PokemonState      # Player bench slot 3
    enemy_0: PokemonState       # Enemy active slot 0
    enemy_1: PokemonState       # Enemy active slot 1 (doubles), or bench if singles
    enemy_2: PokemonState       # Enemy bench slot 0
    enemy_3: PokemonState       # Enemy bench slot 1
    enemy_4: PokemonState       # Enemy bench slot 2
    enemy_5: PokemonState       # Enemy bench slot 3

    # --- Field State ---
    field: FieldState           # Weather, terrain, arena tags, hazards

    # --- Battle / Run State ---
    battle: BattleState         # Wave, turn, money, score, battle type, etc.

    # --- Modifier Inventory ---
    modifiers: ModifierInventory  # All player (and enemy) modifiers

    # --- Phase / Decision ---
    phase: PhaseInfo            # Current phase, action mask, valid actions

    # --- Shop (only during SelectModifierPhase) ---
    shop: Optional[ShopState]   # Reward/shop state, or None if not in shop phase

    # --- Action Labels (human-readable, optional) ---
    action_labels: Optional[List[ActionInfo]]  # Human-readable action descriptions

    # --- Protocol Metadata ---
    step: int                   # Decision step counter (0-indexed)
    timestamp: float            # Unix timestamp of state capture


# ═══════════════════════════════════════════════════════════════════════════
# 26. WIRE PROTOCOL MESSAGES
# ═══════════════════════════════════════════════════════════════════════════

class StateMessage(TypedDict):
    """JSON-line message sent from runner to agent: game state at a decision point.

    ``gameState`` now contains the full ``GameState`` dict (364 fields) built
    by ``state-builder.ts``.
    """

    type: str                   # Always "state"
    step: int                   # Decision step counter
    phase: str                  # Decision phase name
    gameState: GameState        # Full game state matching GameState schema
    actions: List[ActionInfo]   # Valid actions with labels
    metadata: Optional[dict]    # Phase-specific metadata (command info, learn move, etc.)


class ActionMessage(TypedDict):
    """JSON-line message sent from agent to runner: chosen action."""

    action: int                 # Action index (0-57)


class GameOverMessage(TypedDict):
    """JSON-line message sent when the game ends."""

    type: str                   # Always "game_over"
    step: int                   # Final decision step
    victory: bool               # Whether the run was a victory
    gameState: dict             # Final game state (currently simplified; target: GameState)


class ReadyMessage(TypedDict):
    """JSON-line message sent on boot when the runner is ready."""

    type: str                   # Always "ready"
    seed: str                   # RNG seed string
    maxWaves: int               # Maximum waves for this run
    bootTime: int               # Boot time in milliseconds


class WarningMessage(TypedDict):
    """JSON-line message for invalid action fallback."""

    type: str                   # Always "warning"
    message: str                # Warning description


class DoneMessage(TypedDict):
    """JSON-line message sent when the interactive session ends."""

    type: str                   # Always "done"
    steps: int                  # Total decision steps taken


class InfoMessage(TypedDict):
    """JSON-line message for non-decision info (e.g. IV Scanner results)."""

    type: str                   # Always "info"
    message: str                # Human-readable info string


class ErrorMessage(TypedDict):
    """JSON-line message for errors."""

    type: str                   # Always "error"
    message: str                # Error description


# ═══════════════════════════════════════════════════════════════════════════
# 27. ACTION SPACE CONSTANTS (mirror spaces.ts)
# ═══════════════════════════════════════════════════════════════════════════

# Battle action indices
ACTION_FIGHT_ENEMY_START = 0      # 0-3:   move 0-3 targeting ENEMY slot 0
ACTION_FIGHT_ENEMY2_START = 4     # 4-7:   move 0-3 targeting ENEMY slot 1
ACTION_FIGHT_ALLY_START = 8       # 8-11:  move 0-3 targeting ally
ACTION_SWITCH_START = 12          # 12-16: switch to party slot 1-5
ACTION_BALL_START = 17            # 17-21: throw Pokeball type 0-4
ACTION_RUN = 22                   # 22:    run away
ACTION_TERA_ENEMY_START = 23      # 23-26: Tera + move 0-3 targeting ENEMY slot 0
ACTION_TERA_ENEMY2_START = 27     # 27-30: Tera + move 0-3 targeting ENEMY slot 1
ACTION_TERA_ALLY_START = 31       # 31-34: Tera + move 0-3 targeting ally

# Modifier action indices
ACTION_SELECT_REWARD_START = 35   # 35-37: select free reward 0-2
ACTION_REROLL = 38                # 38:    reroll modifiers
ACTION_SKIP = 39                  # 39:    skip / decline
ACTION_BUY_SHOP_START = 40        # 40-51: buy shop item 0-11
ACTION_PARTY_TARGET_START = 52    # 52-57: apply modifier to party slot 0-5


# ═══════════════════════════════════════════════════════════════════════════
# 28. HELPER: EMPTY STATE FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def empty_move_slot() -> MoveSlot:
    """Return a zeroed-out MoveSlot for empty move positions."""
    return MoveSlot(
        move_id=0,
        name="",
        type=0,
        category=0,
        power=0,
        accuracy=0,
        priority=0,
        pp_max=0,
        pp_used=0,
        pp_remaining=0,
        pp_up=0,
        target=0,
        is_usable=False,
        makes_contact=False,
        is_sound_based=False,
        is_powder=False,
        is_punching=False,
        is_slicing=False,
        is_biting=False,
        is_ballistic=False,
        # Secondary effects
        effect_chance=0,
        status_effect=0,
        stat_changes=[],
        drain_ratio=0.0,
        recoil_ratio=0.0,
        heal_ratio=0.0,
        # Multi-hit
        is_multi_hit=False,
        multi_hit_type=-1,
        crit_stage_boost=0,
        # Strategic flags
        is_charging=False,
        self_switch=False,
        force_switch=False,
        traps_target=False,
        is_protect=False,
        is_sacrifice=False,
        is_ohko=False,
        ignores_protect=False,
        ignores_abilities=False,
        ignores_substitute=False,
        fixed_damage=0,
        # Ability-interaction flags
        is_pulse=False,
        is_dance=False,
        # ── v6: Move semantic encoding (+36 fields) ──
        # Group 1: Boolean attr flags (12)
        can_flinch=False,
        can_confuse=False,
        is_recharge=False,
        is_frenzy=False,
        is_typeless=False,
        creates_substitute=False,
        suppresses_ability=False,
        has_variable_power=False,
        has_variable_type=False,
        has_variable_category=False,
        bypass_burn_penalty=False,
        ignores_stat_stages=False,
        # Group 2: Field control (4)
        weather_change=0,
        terrain_change=0,
        sets_arena_tag=False,
        removes_arena_tags=False,
        # Group 3: Arena tag semantics (3)
        sets_hazard=False,
        sets_screen=False,
        arena_tag_self_side=False,
        # Group 4: Battler tag semantics (3)
        applies_battler_tag=False,
        applies_move_restriction=False,
        applies_continuous_damage=False,
        # Group 5: Fixed damage discrimination (4)
        is_user_hp_damage=False,
        is_target_half_hp=False,
        is_counter_damage=False,
        is_level_damage=False,
        # Group 6: Additional strategic flags (2)
        is_delayed_attack=False,
        post_victory_stat_boost=False,
        # Group 7: Missing MoveFlags (8)
        is_wind_move=False,
        is_reckless_move=False,
        is_reflectable=False,
        hides_user=False,
        is_triage_move=False,
        check_all_hits=False,
        affected_by_gravity=False,
        hides_target=False,
        # ── v7: MoveAttr boolean flags (+46 fields) ──
        # Group 8: Item Manipulation (3)
        steals_item=False,
        removes_item=False,
        steals_berry=False,
        # Group 9: Stat Manipulation (8)
        copies_stats=False,
        inverts_stats=False,
        resets_stats=False,
        swaps_stat_stages=False,
        steals_stat_boosts=False,
        averages_stats=False,
        swaps_single_stat=False,
        shifts_own_stat=False,
        # Group 10: HP / PP / Revival (3)
        splits_hp=False,
        reduces_pp=False,
        revives_ally=False,
        # Group 11: Move-Calling (5)
        copies_last_move=False,
        calls_random_move=False,
        calls_moveset_move=False,
        copies_move_temp=False,
        copies_move_perm=False,
        # Group 12: Ability Manipulation (5)
        copies_ability=False,
        swaps_abilities=False,
        changes_ability=False,
        gives_ability=False,
        suppresses_if_acted=False,
        # Group 13: Targeting & Priority (4)
        bypass_redirect=False,
        forces_target_next=False,
        forces_target_last=False,
        has_conditional_priority=False,
        # Group 14: Status & Tag Manipulation (5)
        cures_party_status=False,
        transfers_status=False,
        heals_status=False,
        removes_battler_tag=False,
        removes_substitutes=False,
        # Group 15: Transform & Special Moves (4)
        transforms_into_target=False,
        is_curse=False,
        is_wish=False,
        is_destiny_bond=False,
        # Group 16: Field Control (3)
        swaps_arena_tags=False,
        clears_weather=False,
        clears_terrain=False,
        # Group 17: Damage Calc & Misc (6)
        has_variable_target=False,
        resists_last_type=False,
        has_variable_accuracy=False,
        uses_alt_stat=False,
        overrides_type_chart=False,
        scatters_money=False,
        survives_at_1hp=False,
        matches_user_hp=False,
        hp_cost_stat_boost=False,
        hits_semi_invulnerable=False,
    )


def empty_turn_data() -> TurnData:
    """Return a zeroed-out TurnData."""
    return TurnData(
        damage_taken=0,
        total_damage_dealt=0,
        attacks_received=[],
        order=0,
        hit_count=0,
        acted=False,
        switched_in_this_turn=False,
        stat_stages_increased=False,
        stat_stages_decreased=False,
        berries_eaten=[],
    )


def empty_battle_data() -> BattleData:
    """Return a zeroed-out BattleData."""
    return BattleData(
        hit_count=0,
        has_eaten_berry=False,
        berries_eaten=[],
        abilities_applied=[],
    )


def empty_pokemon_state() -> PokemonState:
    """Return a PokemonState representing an empty (invalid) slot."""
    return PokemonState(
        valid=False,
        species_id=0,
        species_name="",
        form_index=0,
        level=0,
        gender=-1,
        friendship=0,
        shiny=False,
        variant=0,
        hp=0,
        max_hp=0,
        hp_ratio=0.0,
        base_stats=[0] * NUM_STATS,
        ivs=[0] * NUM_STATS,
        stats=[0] * NUM_STATS,
        stat_stages=[0] * NUM_BATTLE_STATS,
        status_effect=0,
        toxic_turn_count=0,
        sleep_turns_remaining=0,
        types=[],
        tera_type=-1,
        is_terastallized=False,
        added_type=-1,
        ability_id=0,
        ability_name="",
        passive_ability_id=0,
        passive_ability_name="",
        has_passive=False,
        ability_suppressed=False,
        ability_revealed=False,
        nature=0,
        nature_multipliers=[1.0] * NUM_EFFECTIVE_STATS,
        moves=[],
        move_history=[],
        pokeball=0,
        volatile_tags=[],
        is_boss=False,
        boss_segments=0,
        boss_segment_index=0,
        ai_type=0,
        is_fusion=False,
        fusion_species_id=None,
        is_on_field=False,
        is_player=False,
        battler_index=-1,
        field_index=-1,
        held_items=[],
        move_queue=[],
        wave_turn_count=0,
        is_fainted=False,
        is_active=False,
        is_trapped=False,
        is_grounded=True,
        transform_species_id=None,
        transform_moves=None,
        illusion_species_id=None,
        attacks_received=[],
        turn_data=empty_turn_data(),
        battle_data=empty_battle_data(),
        weight=0.0,
        catch_rate=0,
        base_total=0,
        stellar_types_boosted=[],
        # v5 additions
        berries_eaten_last=[],
        exp_to_next_level=0,
        luck=0,
        endured_this_wave=False,
    )


def empty_field_state() -> FieldState:
    """Return a FieldState with no active conditions."""
    return FieldState(
        biome_id=0,
        biome_name="",
        weather_type=0,
        weather_turns_left=0,
        weather_is_permanent=False,
        weather_suppressed=False,
        terrain_type=0,
        terrain_turns_left=0,
        terrain_is_permanent=False,
        player_teras_used=0,
        arena_tags=[],
        positional_tags=[],
        is_double_battle=False,
        trick_room_active=False,
        gravity_active=False,
        ignore_abilities=False,
        player_spikes_layers=0,
        player_toxic_spikes_layers=0,
        player_stealth_rock=False,
        player_sticky_web=False,
        enemy_spikes_layers=0,
        enemy_toxic_spikes_layers=0,
        enemy_stealth_rock=False,
        enemy_sticky_web=False,
    )


def empty_pokeball_counts() -> PokeballCounts:
    """Return a PokeballCounts with zero of each type."""
    return PokeballCounts(
        pokeball=0,
        great_ball=0,
        ultra_ball=0,
        rogue_ball=0,
        master_ball=0,
    )
