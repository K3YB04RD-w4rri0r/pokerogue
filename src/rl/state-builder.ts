/**
 * Comprehensive game state serializer for the RL framework.
 *
 * Serializes the full live game state into a JSON dict matching the
 * GameState TypedDict defined in state_schema.py (35 TypedDicts, 364 fields).
 *
 * Replaces the minimal 6-field buildGameState() previously in cli.ts.
 */

import type { AttackMoveResult } from "#app/@types/attack-move-result";
import type { TurnMove } from "#app/@types/turn-move";
import { MAX_TERAS_PER_ARENA } from "#app/constants";
import { globalScene } from "#app/global-scene";
import type { ArenaTag } from "#data/arena-tag";
import type { BattlerTag } from "#data/battler-tags";
// BattlerTag subclass imports for instanceof checks
import {
  AutotomizedTag,
  CritBoostTag,
  DisabledTag,
  EncoreTag,
  GorillaTacticsTag,
  HighestStatBoostTag,
  StockpilingTag,
  SubstituteTag,
  SupremeOverlordTag,
  TypeBoostTag,
} from "#data/battler-tags";
import { getLevelTotalExp } from "#data/exp";
import { getNatureStatMultiplier } from "#data/nature";
import type { PokemonBattleData, PokemonTurnData, PokemonWaveData } from "#data/pokemon/pokemon-data";
import { DelayedAttackTag, WishTag } from "#data/positional-tags/positional-tag";
import { ArenaTagSide } from "#enums/arena-tag-side";
import { ArenaTagType } from "#enums/arena-tag-type";
import { BattleType } from "#enums/battle-type";
import { BiomeId } from "#enums/biome-id";
import { Challenges } from "#enums/challenges";
import { MoveFlags } from "#enums/move-flags";
import { MoveTarget } from "#enums/move-target";
import { EFFECTIVE_STATS } from "#enums/stat";
import { StatusEffect } from "#enums/status-effect";
import type { Pokemon } from "#field/pokemon";
import type { PersistentModifier } from "#modifiers/modifier";
// Modifier subclass imports for instanceof checks
import {
  AttackTypeBoosterModifier,
  BaseStatModifier,
  BerryModifier,
  CritBoosterModifier,
  EnemyAttackStatusEffectChanceModifier,
  LapsingPersistentModifier,
  PokemonBaseStatTotalModifier,
  PokemonFormChangeItemModifier,
  PokemonHeldItemModifier,
  StatBoosterModifier,
  TempStatStageBoosterModifier,
  TurnStatusEffectModifier,
} from "#modifiers/modifier";
import type { PokemonMove } from "#moves/pokemon-move";
import { getAvailableModifiers } from "#rl/modifier-api";
import type { PhaseState } from "#rl/phase-router";
import { getLegalBallTypes } from "#rl/phase-router";

// ─── Empty State Factories ───────────────────────────────────────────

function emptyTurnData(): Record<string, unknown> {
  return {
    damage_taken: 0,
    total_damage_dealt: 0,
    attacks_received: [],
    order: 0,
    hit_count: 0,
    acted: false,
    switched_in_this_turn: false,
    stat_stages_increased: false,
    stat_stages_decreased: false,
    berries_eaten: [],
    move_effectiveness: 0,
    hits_left: 0,
    single_hit_damage_dealt: 0,
  };
}

function emptyBattleData(): Record<string, unknown> {
  return {
    hit_count: 0,
    has_eaten_berry: false,
    berries_eaten: [],
    abilities_applied: [],
  };
}

function emptyPokemonState(): Record<string, unknown> {
  return {
    valid: false,
    species_id: 0,
    species_name: "",
    form_index: 0,
    level: 0,
    gender: -1,
    friendship: 0,
    shiny: false,
    variant: 0,
    hp: 0,
    max_hp: 0,
    hp_ratio: 0.0,
    base_stats: [0, 0, 0, 0, 0, 0],
    ivs: [0, 0, 0, 0, 0, 0],
    stats: [0, 0, 0, 0, 0, 0],
    stat_stages: [0, 0, 0, 0, 0, 0, 0],
    status_effect: 0,
    toxic_turn_count: 0,
    sleep_turns_remaining: 0,
    types: [],
    tera_type: -1,
    is_terastallized: false,
    added_type: -1,
    ability_id: 0,
    ability_name: "",
    passive_ability_id: 0,
    passive_ability_name: "",
    has_passive: false,
    ability_suppressed: false,
    ability_revealed: false,
    nature: 0,
    nature_multipliers: [1.0, 1.0, 1.0, 1.0, 1.0],
    moves: [],
    move_history: [],
    pokeball: 0,
    volatile_tags: [],
    is_boss: false,
    boss_segments: 0,
    boss_segment_index: 0,
    ai_type: 0,
    is_fusion: false,
    fusion_species_id: null,
    is_on_field: false,
    is_player: false,
    battler_index: -1,
    field_index: -1,
    held_items: [],
    move_queue: [],
    wave_turn_count: 0,
    is_fainted: false,
    is_active: false,
    is_trapped: false,
    is_grounded: true,
    transform_species_id: null,
    transform_moves: null,
    illusion_species_id: null,
    attacks_received: [],
    turn_data: emptyTurnData(),
    battle_data: emptyBattleData(),
    weight: 0.0,
    catch_rate: 0,
    base_total: 0,
    stellar_types_boosted: [],
    berries_eaten_last: [],
    exp_to_next_level: 0,
    luck: 0,
    endured_this_wave: false,
    is_mega: false,
    is_max: false,
  };
}

function emptyMoveSlot(): Record<string, unknown> {
  return {
    move_id: 0,
    name: "",
    type: 0,
    category: 0,
    power: 0,
    accuracy: 0,
    priority: 0,
    pp_max: 0,
    pp_used: 0,
    pp_remaining: 0,
    pp_up: 0,
    target: 0,
    is_usable: false,
    makes_contact: false,
    is_sound_based: false,
    is_powder: false,
    is_punching: false,
    is_slicing: false,
    is_biting: false,
    is_ballistic: false,
    effect_chance: 0,
    status_effect: 0,
    stat_changes: [],
    drain_ratio: 0.0,
    recoil_ratio: 0.0,
    heal_ratio: 0.0,
    is_multi_hit: false,
    multi_hit_type: -1,
    crit_stage_boost: 0,
    is_charging: false,
    self_switch: false,
    force_switch: false,
    traps_target: false,
    is_protect: false,
    is_sacrifice: false,
    is_ohko: false,
    ignores_protect: false,
    ignores_abilities: false,
    ignores_substitute: false,
    fixed_damage: 0,
    is_pulse: false,
    is_dance: false,
    // ── v6: Move semantic encoding (+36 fields) ──
    // Group 1: Boolean attr flags (12)
    can_flinch: false,
    can_confuse: false,
    is_recharge: false,
    is_frenzy: false,
    is_typeless: false,
    creates_substitute: false,
    suppresses_ability: false,
    has_variable_power: false,
    has_variable_type: false,
    has_variable_category: false,
    bypass_burn_penalty: false,
    ignores_stat_stages: false,
    // Group 2: Field control (4)
    weather_change: 0,
    terrain_change: 0,
    sets_arena_tag: false,
    removes_arena_tags: false,
    // Group 3: Arena tag semantics (3)
    sets_hazard: false,
    sets_screen: false,
    arena_tag_self_side: false,
    // Group 4: Battler tag semantics (3)
    applies_battler_tag: false,
    applies_move_restriction: false,
    applies_continuous_damage: false,
    // Group 5: Fixed damage discrimination (4)
    is_user_hp_damage: false,
    is_target_half_hp: false,
    is_counter_damage: false,
    is_level_damage: false,
    // Group 6: Additional strategic flags (2)
    is_delayed_attack: false,
    post_victory_stat_boost: false,
    // Group 7: Missing MoveFlags (8)
    is_wind_move: false,
    is_reckless_move: false,
    is_reflectable: false,
    hides_user: false,
    is_triage_move: false,
    check_all_hits: false,
    affected_by_gravity: false,
    hides_target: false,
    // ── v7: MoveAttr boolean flags (+46 fields) ──
    // Group 8: Item Manipulation (3)
    steals_item: false,
    removes_item: false,
    steals_berry: false,
    // Group 9: Stat Manipulation (8)
    copies_stats: false,
    inverts_stats: false,
    resets_stats: false,
    swaps_stat_stages: false,
    steals_stat_boosts: false,
    averages_stats: false,
    swaps_single_stat: false,
    shifts_own_stat: false,
    // Group 10: HP / PP / Revival (3)
    splits_hp: false,
    reduces_pp: false,
    revives_ally: false,
    // Group 11: Move-Calling (5)
    copies_last_move: false,
    calls_random_move: false,
    calls_moveset_move: false,
    copies_move_temp: false,
    copies_move_perm: false,
    // Group 12: Ability Manipulation (5)
    copies_ability: false,
    swaps_abilities: false,
    changes_ability: false,
    gives_ability: false,
    suppresses_if_acted: false,
    // Group 13: Targeting & Priority (4)
    bypass_redirect: false,
    forces_target_next: false,
    forces_target_last: false,
    has_conditional_priority: false,
    // Group 14: Status & Tag Manipulation (5)
    cures_party_status: false,
    transfers_status: false,
    heals_status: false,
    removes_battler_tag: false,
    removes_substitutes: false,
    // Group 15: Transform & Special Moves (4)
    transforms_into_target: false,
    is_curse: false,
    is_wish: false,
    is_destiny_bond: false,
    // Group 16: Field Control (3)
    swaps_arena_tags: false,
    clears_weather: false,
    clears_terrain: false,
    // Group 17: Damage Calc & Misc (6)
    has_variable_target: false,
    resists_last_type: false,
    has_variable_accuracy: false,
    uses_alt_stat: false,
    overrides_type_chart: false,
    scatters_money: false,
    // Group 18: v8 survival / HP-relative semantics (4)
    survives_at_1hp: false,
    matches_user_hp: false,
    hp_cost_stat_boost: false,
    hits_semi_invulnerable: false,
  };
}

// ─── Tiny Helpers ────────────────────────────────────────────────────

function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch {
    return fallback;
  }
}

function getHazardLayers(tagType: ArenaTagType, side: ArenaTagSide): number {
  const arena = globalScene.arena;
  if (!arena?.tags) {
    return 0;
  }
  for (const tag of arena.tags) {
    if (tag.tagType === tagType && (tag.side === side || tag.side === ArenaTagSide.BOTH)) {
      return (tag as any).layers ?? 1;
    }
  }
  return 0;
}

function hasArenaTag(tagType: ArenaTagType, side: ArenaTagSide): boolean {
  const arena = globalScene.arena;
  if (!arena?.tags) {
    return false;
  }
  return arena.tags.some(t => t.tagType === tagType && (t.side === side || t.side === ArenaTagSide.BOTH));
}

// ─── QueuedMove / AttackReceived ─────────────────────────────────────

function buildQueuedMove(turnMove: TurnMove): Record<string, unknown> {
  return {
    move_id: turnMove.move ?? 0,
    targets: turnMove.targets ?? [],
    use_mode: turnMove.useMode ?? 0,
    result: turnMove.result ?? null,
  };
}

function buildAttackReceived(atk: AttackMoveResult): Record<string, unknown> {
  return {
    source_battler_index: atk.sourceBattlerIndex ?? 0,
    source_id: atk.sourceId ?? 0,
    move_id: atk.move ?? 0,
    damage: atk.damage ?? 0,
    critical: atk.critical ?? false,
    result: atk.result ?? 0,
  };
}

// ─── TurnData / BattleData ───────────────────────────────────────────

function buildTurnData(td: PokemonTurnData | null | undefined): Record<string, unknown> {
  if (!td) {
    return emptyTurnData();
  }
  return {
    damage_taken: td.damageTaken ?? 0,
    total_damage_dealt: td.totalDamageDealt ?? 0,
    attacks_received: (td.attacksReceived ?? []).map(buildAttackReceived),
    order: td.order ?? 0,
    hit_count: td.hitCount ?? 0,
    acted: td.acted ?? false,
    switched_in_this_turn: td.switchedInThisTurn ?? false,
    stat_stages_increased: td.statStagesIncreased ?? false,
    stat_stages_decreased: td.statStagesDecreased ?? false,
    berries_eaten: (td as any).berriesEaten ?? [],
    move_effectiveness: td.moveEffectiveness ?? 0,
    hits_left: td.hitsLeft ?? 0,
    single_hit_damage_dealt: td.singleHitDamageDealt ?? 0,
  };
}

function buildBattleDataDict(
  bd: PokemonBattleData | null | undefined,
  wd: PokemonWaveData | null | undefined,
): Record<string, unknown> {
  return {
    hit_count: bd?.hitCount ?? 0,
    has_eaten_berry: bd?.hasEatenBerry ?? false,
    berries_eaten: bd?.berriesEaten ?? [],
    abilities_applied: wd?.abilitiesApplied ? Array.from(wd.abilitiesApplied) : [],
  };
}

// ─── Volatile Tag ────────────────────────────────────────────────────

function buildVolatileTag(tag: BattlerTag): Record<string, unknown> {
  const result: Record<string, unknown> = {
    tag_type: tag.tagType,
    turn_count: tag.turnCount ?? 0,
    source_id: tag.sourceId ?? null,
    source_move: tag.sourceMove ?? null,
    // Optional fields default to null
    substitute_hp: null,
    stockpile_count: null,
    encore_move_id: null,
    disabled_move_id: null,
    type_boost_type: null,
    type_boost_value: null,
    crit_boost_stages: null,
    gorilla_tactics_move_id: null,
    highest_stat_boost_stat: null,
    highest_stat_boost_multiplier: null,
    supreme_overlord_faint_count: null,
    autotomize_count: null,
  };

  try {
    if (tag instanceof SubstituteTag) {
      result.substitute_hp = tag.hp ?? null;
    }
    if (tag instanceof StockpilingTag) {
      result.stockpile_count = tag.stockpiledCount ?? null;
    }
    if (tag instanceof EncoreTag) {
      result.encore_move_id = tag.moveId ?? null;
    }
    if (tag instanceof DisabledTag) {
      result.disabled_move_id = tag.moveId ?? null;
    }
    if (tag instanceof TypeBoostTag) {
      result.type_boost_type = tag.boostedType ?? null;
      result.type_boost_value = tag.boostValue ?? null;
    }
    if (tag instanceof CritBoostTag) {
      result.crit_boost_stages = tag.critStages ?? null;
    }
    if (tag instanceof GorillaTacticsTag) {
      result.gorilla_tactics_move_id = tag.moveId ?? null;
    }
    if (tag instanceof HighestStatBoostTag) {
      result.highest_stat_boost_stat = tag.stat ?? null;
      result.highest_stat_boost_multiplier = tag.multiplier ?? null;
    }
    if (tag instanceof SupremeOverlordTag) {
      result.supreme_overlord_faint_count = tag.faintCount ?? null;
    }
    if (tag instanceof AutotomizedTag) {
      result.autotomize_count = tag.autotomizeCount ?? null;
    }
  } catch (err) {
    console.error("[state-builder] Error extracting volatile tag extra state:", err);
  }

  return result;
}

// ─── Held Item ───────────────────────────────────────────────────────

function buildHeldItem(modifier: PokemonHeldItemModifier): Record<string, unknown> {
  const result: Record<string, unknown> = {
    modifier_class: modifier.constructor.name,
    modifier_id: modifier.type?.id ?? "",
    name: modifier.type?.name ?? "",
    stack_count: modifier.stackCount ?? 0,
    max_stack_count: safe(() => modifier.getMaxStackCount(), 0),
    is_transferable: modifier.isTransferable ?? true,
    // Optional type-specific fields
    type_id: null,
    stat_id: null,
    stat_ids: null,
    status_effect: null,
    berry_type: null,
    consumed: null,
    stat_modifier: null,
    stat_boost_multiplier: null,
    crit_stage_increment: null,
    form_change_item: null,
    form_change_active: null,
  };

  try {
    if (modifier instanceof AttackTypeBoosterModifier) {
      result.type_id = modifier.moveType;
    }
    if (modifier instanceof StatBoosterModifier) {
      result.stat_ids = (modifier as any).stats ? Array.from((modifier as any).stats) : null;
      result.stat_boost_multiplier = (modifier as any).multiplier ?? null;
    }
    if (modifier instanceof BaseStatModifier) {
      result.stat_id = (modifier as any).stat ?? null;
    }
    if (modifier instanceof CritBoosterModifier) {
      result.crit_stage_increment = (modifier as any).stageIncrement ?? null;
    }
    if (modifier instanceof BerryModifier) {
      result.berry_type = modifier.berryType;
      result.consumed = modifier.consumed;
    }
    if (modifier instanceof TurnStatusEffectModifier) {
      result.status_effect = (modifier as any).effect ?? null;
    }
    if (modifier instanceof PokemonBaseStatTotalModifier) {
      result.stat_modifier = modifier.statModifier;
    }
    if (modifier instanceof PokemonFormChangeItemModifier) {
      result.form_change_item = modifier.formChangeItem;
      result.form_change_active = modifier.active;
    }
  } catch (err) {
    console.error("[state-builder] Error extracting held item subclass fields:", err);
  }

  return result;
}

// ─── Move Slot ───────────────────────────────────────────────────────

function buildMoveSlot(pokemonMove: PokemonMove | null | undefined, pokemon: Pokemon | null): Record<string, unknown> {
  if (!pokemonMove) {
    return emptyMoveSlot();
  }

  try {
    const move = pokemonMove.getMove();
    if (!move) {
      return emptyMoveSlot();
    }

    const ppMax = safe(() => pokemonMove.getMovePp(), 0);
    const ppUsed = pokemonMove.ppUsed ?? 0;

    // --- is_usable: needs pokemon ref ---
    let isUsable = ppMax - ppUsed > 0;
    if (pokemon) {
      try {
        isUsable = pokemonMove.isUsable(pokemon)[0];
      } catch {
        // fallback to PP check
      }
    }

    // --- Secondary effect extraction via getAttrs ---
    // Status effect
    let statusEffect = StatusEffect.NONE;
    try {
      const statusAttrs = move.getAttrs("StatusEffectAttr");
      if (statusAttrs.length > 0) {
        statusEffect = (statusAttrs[0] as any).effect ?? StatusEffect.NONE;
      }
    } catch {
      /* ignore */
    }

    // Stat changes
    const statChanges: Record<string, unknown>[] = [];
    try {
      const statChangeAttrs = move.getAttrs("StatStageChangeAttr");
      for (const attr of statChangeAttrs) {
        const chance = (attr as any).options?.effectChanceOverride ?? (move.chance > 0 ? move.chance : 100);
        for (const statId of (attr as any).stats ?? []) {
          statChanges.push({
            stat_id: statId,
            stages: (attr as any).stages ?? 0,
            self_target: !!(attr as any).selfTarget,
            chance,
          });
        }
      }
    } catch {
      /* ignore */
    }

    // Drain ratio (HitHealAttr)
    let drainRatio = 0.0;
    try {
      const hitHealAttrs = move.getAttrs("HitHealAttr");
      if (hitHealAttrs.length > 0) {
        drainRatio = (hitHealAttrs[0] as any).healRatio ?? 0.5;
      }
    } catch {
      /* ignore */
    }

    // Recoil ratio (RecoilAttr)
    let recoilRatio = 0.0;
    try {
      const recoilAttrs = move.getAttrs("RecoilAttr");
      if (recoilAttrs.length > 0) {
        recoilRatio = (recoilAttrs[0] as any).damageRatio ?? 0.25;
      }
    } catch {
      /* ignore */
    }

    // Heal ratio (HealAttr)
    let healRatio = 0.0;
    try {
      const healAttrs = move.getAttrs("HealAttr");
      if (healAttrs.length > 0) {
        healRatio = (healAttrs[0] as any).healRatio ?? 0.5;
      }
    } catch {
      /* ignore */
    }

    // Multi-hit
    let isMultiHit = false;
    let multiHitType = -1;
    try {
      const multiHitAttrs = move.getAttrs("MultiHitAttr");
      if (multiHitAttrs.length > 0) {
        isMultiHit = true;
        multiHitType = (multiHitAttrs[0] as any).intrinsicMultiHitType ?? (multiHitAttrs[0] as any).multiHitType ?? -1;
      }
    } catch {
      /* ignore */
    }

    // Crit stage boost
    let critStageBoost = 0;
    try {
      if (move.hasAttr("CritOnlyAttr")) {
        critStageBoost = 99;
      } else if (move.hasAttr("HighCritAttr")) {
        critStageBoost = 1;
      }
    } catch {
      /* ignore */
    }

    // Charging move
    const isCharging = safe(() => move.isChargingMove(), false);

    // Force switch / self switch
    let selfSwitch = false;
    let forceSwitch = false;
    try {
      const forceSwitchAttrs = move.getAttrs("ForceSwitchOutAttr");
      for (const attr of forceSwitchAttrs) {
        if ((attr as any).selfSwitch) {
          selfSwitch = true;
        } else {
          forceSwitch = true;
        }
      }
    } catch {
      /* ignore */
    }

    // Traps target
    const trapsTarget = safe(() => move.hasAttr("TrapAttr"), false);

    // Protect
    const isProtect = safe(() => move.hasAttr("ProtectAttr"), false);

    // Sacrifice
    const isSacrifice = safe(
      () =>
        move.hasAttr("SacrificialAttr") || move.hasAttr("SacrificialAttrOnHit") || move.hasAttr("HalfSacrificialAttr"),
      false,
    );

    // OHKO
    const isOhko = safe(() => move.hasAttr("OneHitKOAttr"), false);

    // Fixed damage
    let fixedDamage = 0;
    try {
      const fixedDmgAttrs = move.getAttrs("FixedDamageAttr");
      if (fixedDmgAttrs.length > 0) {
        fixedDamage = (fixedDmgAttrs[0] as any).damage ?? 0;
      }
    } catch {
      /* ignore */
    }

    // ── v6: Move semantic encoding (+36 fields) ──

    // Group 1: Boolean attr flags
    const canFlinch = safe(() => move.hasAttr("FlinchAttr"), false);
    const canConfuse = safe(() => move.hasAttr("ConfuseAttr"), false);
    const isRecharge = safe(() => move.hasAttr("RechargeAttr"), false);
    const isFrenzy = safe(() => move.hasAttr("FrenzyAttr"), false);
    const isTypeless = safe(() => move.hasAttr("TypelessAttr"), false);
    const createsSubstitute = safe(() => move.hasAttr("AddSubstituteAttr"), false);
    const suppressesAbility = safe(() => move.hasAttr("SuppressAbilitiesAttr"), false);
    const hasVariablePower = safe(() => move.hasAttr("VariablePowerAttr"), false);
    const hasVariableType = safe(() => move.hasAttr("VariableMoveTypeAttr"), false);
    const hasVariableCategory = safe(() => move.hasAttr("VariableMoveCategoryAttr"), false);
    const bypassBurnPenalty = safe(() => move.hasAttr("BypassBurnDamageReductionAttr"), false);
    const ignoresStatStages = safe(() => move.hasAttr("IgnoreOpponentStatStagesAttr"), false);

    // Group 2: Field control
    let weatherChange = 0;
    try {
      const weatherAttrs = move.getAttrs("WeatherChangeAttr");
      if (weatherAttrs.length > 0) {
        weatherChange = (weatherAttrs[0] as any).weatherType ?? 0;
      }
    } catch {
      /* ignore */
    }

    let terrainChange = 0;
    try {
      const terrainAttrs = move.getAttrs("TerrainChangeAttr");
      if (terrainAttrs.length > 0) {
        terrainChange = (terrainAttrs[0] as any).terrainType ?? 0;
      }
    } catch {
      /* ignore */
    }

    const setsArenaTag = safe(() => move.hasAttr("AddArenaTagAttr"), false);
    const removesArenaTags = safe(() => move.hasAttr("RemoveArenaTagsAttr"), false);

    // Group 3: Arena tag semantics
    let setsHazard = false;
    let setsScreen = false;
    let arenaTagSelfSide = false;
    try {
      const arenaTagAttrs = move.getAttrs("AddArenaTagAttr");
      if (arenaTagAttrs.length > 0) {
        const tagType = (arenaTagAttrs[0] as any).tagType as string;
        // The side a tag lands on is determined by the move's TARGET
        // (screens: USER_SIDE, hazards: ENEMY_SIDE, Trick Room: BOTH_SIDES).
        // The attr's `selfSideTarget` constructor param is NOT that semantic —
        // e.g. Reflect is `AddArenaTagAttr(REFLECT, 5, true)` where `true` is
        // failOnOverlap and selfSideTarget stays default false.
        arenaTagSelfSide = move.moveTarget === MoveTarget.USER_SIDE || move.moveTarget === MoveTarget.BOTH_SIDES;
        // Hazards: enemy-side entry hazards
        const HAZARD_TAGS = new Set([
          ArenaTagType.STEALTH_ROCK,
          ArenaTagType.SPIKES,
          ArenaTagType.TOXIC_SPIKES,
          ArenaTagType.STICKY_WEB,
        ]);
        // Screens: self-side damage reduction
        const SCREEN_TAGS = new Set([ArenaTagType.REFLECT, ArenaTagType.LIGHT_SCREEN, ArenaTagType.AURORA_VEIL]);
        setsHazard = HAZARD_TAGS.has(tagType as ArenaTagType);
        setsScreen = SCREEN_TAGS.has(tagType as ArenaTagType);
      }
    } catch {
      /* ignore */
    }

    // Group 4: Battler tag semantics (excluding Flinch/Confuse/Recharge subclasses)
    let appliesBattlerTag = false;
    let appliesMoveRestriction = false;
    let appliesContinuousDamage = false;
    try {
      const battlerTagAttrs = move.getAttrs("AddBattlerTagAttr");
      // Filter out FlinchAttr/ConfuseAttr/RechargeAttr which are subclasses
      const isFlinchOrConfuseOrRecharge = (a: unknown): boolean => {
        try {
          return (
            (move.hasAttr("FlinchAttr") && (a as any).tagType === "FLINCHED")
            || (move.hasAttr("ConfuseAttr") && (a as any).tagType === "CONFUSED")
            || (move.hasAttr("RechargeAttr") && (a as any).tagType === "RECHARGING")
          );
        } catch {
          return false;
        }
      };
      const MOVE_RESTRICTION_TAGS = new Set([
        "TAUNT",
        "ENCORE",
        "DISABLED",
        "TORMENT",
        "IMPRISON",
        "HEAL_BLOCK",
        "THROAT_CHOPPED",
      ]);
      const CONTINUOUS_DAMAGE_TAGS = new Set(["SEEDED", "SALT_CURED", "CURSED", "NIGHTMARE", "PERISH_SONG"]);
      for (const attr of battlerTagAttrs) {
        const tagType = String((attr as any).tagType ?? "");
        if (isFlinchOrConfuseOrRecharge(attr)) {
          continue;
        }
        appliesBattlerTag = true;
        if (MOVE_RESTRICTION_TAGS.has(tagType)) {
          appliesMoveRestriction = true;
        }
        if (CONTINUOUS_DAMAGE_TAGS.has(tagType)) {
          appliesContinuousDamage = true;
        }
      }
    } catch {
      /* ignore */
    }

    // Group 5: Fixed damage discrimination
    const isUserHpDamage = safe(() => move.hasAttr("UserHpDamageAttr"), false);
    const isTargetHalfHp = safe(() => move.hasAttr("TargetHalfHpDamageAttr"), false);
    const isCounterDamage = safe(() => move.hasAttr("CounterDamageAttr"), false);
    const isLevelDamage = safe(() => move.hasAttr("LevelDamageAttr"), false);

    // Group 6: Additional strategic flags
    const isDelayedAttack = safe(() => move.hasAttr("DelayedAttackAttr"), false);
    const postVictoryStatBoost = safe(() => move.hasAttr("PostVictoryStatStageChangeAttr"), false);

    // ── v7: MoveAttr boolean flags (+46 fields) ──

    // Group 8: Item Manipulation (3)
    const stealsItem = safe(() => move.hasAttr("StealHeldItemChanceAttr"), false);
    const removesItem = safe(() => move.hasAttr("RemoveHeldItemAttr"), false);
    const stealsBerry = safe(() => move.hasAttr("StealEatBerryAttr"), false);

    // Group 9: Stat Manipulation (8)
    const copiesStats = safe(() => move.hasAttr("CopyStatsAttr"), false);
    const invertsStats = safe(() => move.hasAttr("InvertStatsAttr"), false);
    const resetsStats = safe(() => move.hasAttr("ResetStatsAttr"), false);
    const swapsStatStages = safe(() => move.hasAttr("SwapStatStagesAttr"), false);
    const stealsStatBoosts = safe(() => move.hasAttr("SpectralThiefAttr"), false);
    const averagesStats = safe(() => move.hasAttr("AverageStatsAttr"), false);
    const swapsSingleStat = safe(() => move.hasAttr("SwapStatAttr"), false);
    const shiftsOwnStat = safe(() => move.hasAttr("ShiftStatAttr"), false);

    // Group 10: HP / PP / Revival (3)
    const splitsHp = safe(() => move.hasAttr("HpSplitAttr"), false);
    const reducesPp = safe(() => move.hasAttr("ReducePpMoveAttr"), false);
    const revivesAlly = safe(() => move.hasAttr("RevivalBlessingAttr"), false);

    // Group 11: Move-Calling (5)
    const copiesLastMove = safe(() => move.hasAttr("CopyMoveAttr"), false);
    const callsRandomMove = safe(() => move.hasAttr("RandomMoveAttr"), false);
    const callsMovesetMove = safe(() => move.hasAttr("RandomMovesetMoveAttr"), false);
    const copiesMoveTemp = safe(() => move.hasAttr("MovesetCopyMoveAttr"), false);
    const copiesMovePerm = safe(() => move.hasAttr("SketchAttr"), false);

    // Group 12: Ability Manipulation (5)
    const copiesAbility = safe(() => move.hasAttr("AbilityCopyAttr"), false);
    const swapsAbilities = safe(() => move.hasAttr("SwitchAbilitiesAttr"), false);
    const changesAbility = safe(() => move.hasAttr("AbilityChangeAttr"), false);
    const givesAbility = safe(() => move.hasAttr("AbilityGiveAttr"), false);
    const suppressesIfActed = safe(() => move.hasAttr("SuppressAbilitiesIfActedAttr"), false);

    // Group 13: Targeting & Priority (4)
    const bypassRedirect = safe(() => move.hasAttr("BypassRedirectAttr"), false);
    const forcesTargetNext = safe(() => move.hasAttr("AfterYouAttr"), false);
    const forcesTargetLast = safe(() => move.hasAttr("ForceLastAttr"), false);
    const hasConditionalPriority = safe(() => move.hasAttr("IncrementMovePriorityAttr"), false);

    // Group 14: Status & Tag Manipulation (5)
    const curesPartyStatus = safe(() => move.hasAttr("PartyStatusCureAttr"), false);
    const transfersStatus = safe(() => move.hasAttr("PsychoShiftEffectAttr"), false);
    const healsStatus = safe(() => move.hasAttr("HealStatusEffectAttr"), false);
    const removesBattlerTag = safe(() => move.hasAttr("RemoveBattlerTagAttr"), false);
    const removesSubstitutes = safe(() => move.hasAttr("RemoveAllSubstitutesAttr"), false);

    // Group 15: Transform & Special Moves (4)
    const transformsIntoTarget = safe(() => move.hasAttr("TransformAttr"), false);
    const isCurse = safe(() => move.hasAttr("CurseAttr"), false);
    const isWish = safe(() => move.hasAttr("WishAttr"), false);
    const isDestinyBond = safe(() => move.hasAttr("DestinyBondAttr"), false);

    // Group 16: Field Control (3)
    const swapsArenaTags = safe(() => move.hasAttr("SwapArenaTagsAttr"), false);
    const clearsWeather = safe(() => move.hasAttr("ClearWeatherAttr"), false);
    const clearsTerrain = safe(() => move.hasAttr("ClearTerrainAttr"), false);

    // Group 17: Damage Calc & Misc (6)
    const hasVariableTarget = safe(() => move.hasAttr("VariableTargetAttr"), false);
    const resistsLastType = safe(() => move.hasAttr("ResistLastMoveTypeAttr"), false);
    const hasVariableAccuracy = safe(() => move.hasAttr("VariableAccuracyAttr"), false);
    const usesAltStat = safe(() => move.hasAttr("VariableAtkAttr") || move.hasAttr("VariableDefAttr"), false);
    const overridesTypeChart = safe(() => move.hasAttr("MoveTypeChartOverrideAttr"), false);
    const scattersMoney = safe(() => move.hasAttr("MoneyAttr"), false);

    // ── v8: survival / HP-relative semantics (4) ──
    // hasAttr uses instanceof, so HitsTagForDoubleDamageAttr (extends HitsTagAttr)
    // is caught by the HitsTagAttr check.
    const survivesAt1hp = safe(() => move.hasAttr("SurviveDamageAttr"), false);
    const matchesUserHp = safe(() => move.hasAttr("MatchHpAttr"), false);
    const hpCostStatBoost = safe(() => move.hasAttr("CutHpStatStageBoostAttr"), false);
    const hitsSemiInvulnerable = safe(() => move.hasAttr("HitsTagAttr"), false);

    return {
      move_id: pokemonMove.moveId ?? 0,
      name: move.name ?? "",
      type: move.type ?? 0,
      category: move.category ?? 0,
      power: move.power ?? 0,
      accuracy: move.accuracy ?? 0,
      priority: move.priority ?? 0,
      pp_max: ppMax,
      pp_used: ppUsed,
      pp_remaining: Math.max(0, ppMax - ppUsed),
      pp_up: pokemonMove.ppUp ?? 0,
      target: move.moveTarget ?? 0,
      is_usable: isUsable,
      makes_contact: move.hasFlag(MoveFlags.MAKES_CONTACT),
      is_sound_based: move.hasFlag(MoveFlags.SOUND_BASED),
      is_powder: move.hasFlag(MoveFlags.POWDER_MOVE),
      is_punching: move.hasFlag(MoveFlags.PUNCHING_MOVE),
      is_slicing: move.hasFlag(MoveFlags.SLICING_MOVE),
      is_biting: move.hasFlag(MoveFlags.BITING_MOVE),
      is_ballistic: move.hasFlag(MoveFlags.BALLBOMB_MOVE),
      effect_chance: move.chance ?? 0,
      status_effect: statusEffect,
      stat_changes: statChanges,
      drain_ratio: drainRatio,
      recoil_ratio: recoilRatio,
      heal_ratio: healRatio,
      is_multi_hit: isMultiHit,
      multi_hit_type: multiHitType,
      crit_stage_boost: critStageBoost,
      is_charging: isCharging,
      self_switch: selfSwitch,
      force_switch: forceSwitch,
      traps_target: trapsTarget,
      is_protect: isProtect,
      is_sacrifice: isSacrifice,
      is_ohko: isOhko,
      ignores_protect: move.hasFlag(MoveFlags.IGNORE_PROTECT),
      ignores_abilities: move.hasFlag(MoveFlags.IGNORE_ABILITIES),
      ignores_substitute: move.hasFlag(MoveFlags.IGNORE_SUBSTITUTE),
      fixed_damage: fixedDamage,
      is_pulse: move.hasFlag(MoveFlags.PULSE_MOVE),
      is_dance: move.hasFlag(MoveFlags.DANCE_MOVE),
      // ── v6: Move semantic encoding (+36 fields) ──
      // Group 1: Boolean attr flags (12)
      can_flinch: canFlinch,
      can_confuse: canConfuse,
      is_recharge: isRecharge,
      is_frenzy: isFrenzy,
      is_typeless: isTypeless,
      creates_substitute: createsSubstitute,
      suppresses_ability: suppressesAbility,
      has_variable_power: hasVariablePower,
      has_variable_type: hasVariableType,
      has_variable_category: hasVariableCategory,
      bypass_burn_penalty: bypassBurnPenalty,
      ignores_stat_stages: ignoresStatStages,
      // Group 2: Field control (4)
      weather_change: weatherChange,
      terrain_change: terrainChange,
      sets_arena_tag: setsArenaTag,
      removes_arena_tags: removesArenaTags,
      // Group 3: Arena tag semantics (3)
      sets_hazard: setsHazard,
      sets_screen: setsScreen,
      arena_tag_self_side: arenaTagSelfSide,
      // Group 4: Battler tag semantics (3)
      applies_battler_tag: appliesBattlerTag,
      applies_move_restriction: appliesMoveRestriction,
      applies_continuous_damage: appliesContinuousDamage,
      // Group 5: Fixed damage discrimination (4)
      is_user_hp_damage: isUserHpDamage,
      is_target_half_hp: isTargetHalfHp,
      is_counter_damage: isCounterDamage,
      is_level_damage: isLevelDamage,
      // Group 6: Additional strategic flags (2)
      is_delayed_attack: isDelayedAttack,
      post_victory_stat_boost: postVictoryStatBoost,
      // Group 7: Missing MoveFlags (8)
      is_wind_move: move.hasFlag(MoveFlags.WIND_MOVE),
      is_reckless_move: move.hasFlag(MoveFlags.RECKLESS_MOVE),
      is_reflectable: move.hasFlag(MoveFlags.REFLECTABLE),
      hides_user: move.hasFlag(MoveFlags.HIDE_USER),
      is_triage_move: move.hasFlag(MoveFlags.TRIAGE_MOVE),
      check_all_hits: move.hasFlag(MoveFlags.CHECK_ALL_HITS),
      affected_by_gravity: move.hasFlag(MoveFlags.GRAVITY),
      hides_target: move.hasFlag(MoveFlags.HIDE_TARGET),
      // ── v7: MoveAttr boolean flags (+46 fields) ──
      // Group 8: Item Manipulation (3)
      steals_item: stealsItem,
      removes_item: removesItem,
      steals_berry: stealsBerry,
      // Group 9: Stat Manipulation (8)
      copies_stats: copiesStats,
      inverts_stats: invertsStats,
      resets_stats: resetsStats,
      swaps_stat_stages: swapsStatStages,
      steals_stat_boosts: stealsStatBoosts,
      averages_stats: averagesStats,
      swaps_single_stat: swapsSingleStat,
      shifts_own_stat: shiftsOwnStat,
      // Group 10: HP / PP / Revival (3)
      splits_hp: splitsHp,
      reduces_pp: reducesPp,
      revives_ally: revivesAlly,
      // Group 11: Move-Calling (5)
      copies_last_move: copiesLastMove,
      calls_random_move: callsRandomMove,
      calls_moveset_move: callsMovesetMove,
      copies_move_temp: copiesMoveTemp,
      copies_move_perm: copiesMovePerm,
      // Group 12: Ability Manipulation (5)
      copies_ability: copiesAbility,
      swaps_abilities: swapsAbilities,
      changes_ability: changesAbility,
      gives_ability: givesAbility,
      suppresses_if_acted: suppressesIfActed,
      // Group 13: Targeting & Priority (4)
      bypass_redirect: bypassRedirect,
      forces_target_next: forcesTargetNext,
      forces_target_last: forcesTargetLast,
      has_conditional_priority: hasConditionalPriority,
      // Group 14: Status & Tag Manipulation (5)
      cures_party_status: curesPartyStatus,
      transfers_status: transfersStatus,
      heals_status: healsStatus,
      removes_battler_tag: removesBattlerTag,
      removes_substitutes: removesSubstitutes,
      // Group 15: Transform & Special Moves (4)
      transforms_into_target: transformsIntoTarget,
      is_curse: isCurse,
      is_wish: isWish,
      is_destiny_bond: isDestinyBond,
      // Group 16: Field Control (3)
      swaps_arena_tags: swapsArenaTags,
      clears_weather: clearsWeather,
      clears_terrain: clearsTerrain,
      // Group 17: Damage Calc & Misc (6)
      has_variable_target: hasVariableTarget,
      resists_last_type: resistsLastType,
      has_variable_accuracy: hasVariableAccuracy,
      uses_alt_stat: usesAltStat,
      overrides_type_chart: overridesTypeChart,
      scatters_money: scattersMoney,
      // Group 18: v8 survival / HP-relative semantics (4)
      survives_at_1hp: survivesAt1hp,
      matches_user_hp: matchesUserHp,
      hp_cost_stat_boost: hpCostStatBoost,
      hits_semi_invulnerable: hitsSemiInvulnerable,
    };
  } catch (err) {
    console.error("[state-builder] Error building move slot:", err);
    return emptyMoveSlot();
  }
}

// ─── Pokemon State ───────────────────────────────────────────────────

function buildPokemonState(
  pokemon: Pokemon | null | undefined,
  isPlayer: boolean,
  _slotIndex: number,
): Record<string, unknown> {
  if (!pokemon) {
    return emptyPokemonState();
  }

  try {
    const maxHp = safe(() => pokemon.getMaxHp(), 0);
    const hasSummonData = !!pokemon.summonData;

    // Base stats from species form
    const baseStats = safe(() => pokemon.getSpeciesForm(true).baseStats.slice(), [0, 0, 0, 0, 0, 0]);

    // Computed stats (from IVs + EVs + nature + level).
    // getStats(true) returns the LIVE `this.stats` array by reference. The dumped
    // game state must be a frozen snapshot: without copying, an in-place mutation
    // of this.stats (e.g. an evolution recalculating stats between the moment the
    // observation is encoded and the moment the dict is serialized to JSON) makes
    // the serialized stats diverge from the already-encoded observation — a
    // TS<->Python parity mismatch seen only at evolution boundaries (the dict
    // ends up internally inconsistent: pre-evolution species + post-evolution
    // stats). Snapshot it, exactly as baseStats does one line above.
    const computedStats = safe(() => pokemon.getStats(true).slice(), [0, 0, 0, 0, 0, 0]);

    // IVs
    const ivs = safe(() => Array.from(pokemon.ivs), [0, 0, 0, 0, 0, 0]);

    // Stat stages (only meaningful for on-field Pokemon with summonData)
    const statStages = hasSummonData
      ? safe(() => Array.from(pokemon.getStatStages()), [0, 0, 0, 0, 0, 0, 0])
      : [0, 0, 0, 0, 0, 0, 0];

    // Nature multipliers for the 5 effective stats
    const natureMults = safe(
      // getNature() (not the raw .nature) so an applied Mint is reflected —
      // the game computes stats from getNature(), and the scalar `nature`
      // field below already uses it; the raw nature would contradict the
      // computed_stats after a Mint.
      () => EFFECTIVE_STATS.map(s => getNatureStatMultiplier(pokemon.getNature(), s)),
      [1.0, 1.0, 1.0, 1.0, 1.0],
    );

    // Moves
    const moveset = safe(() => pokemon.getMoveset(false) ?? [], []);
    const moves = moveset.map(m => buildMoveSlot(m, pokemon));

    // Move history (from summonData)
    const moveHistory = hasSummonData
      ? safe(() => (pokemon.summonData.moveHistory ?? []).map(buildQueuedMove), [])
      : [];

    // Volatile tags
    const volatileTags = hasSummonData ? safe(() => (pokemon.summonData.tags ?? []).map(buildVolatileTag), []) : [];

    // Move queue
    const moveQueue = hasSummonData ? safe(() => (pokemon.summonData.moveQueue ?? []).map(buildQueuedMove), []) : [];

    // Boss detection
    const isBoss = "bossSegments" in pokemon && (pokemon as any).bossSegments > 0;

    // Held items
    const heldItems = safe(() => pokemon.getHeldItems().map(buildHeldItem), []);

    // Turn data and battle data
    const turnData = buildTurnData(pokemon.turnData);
    const battleData = buildBattleDataDict(pokemon.battleData, pokemon.waveData);

    // Attacks received (convenience copy from turnData)
    const attacksReceived = safe(() => (pokemon.turnData?.attacksReceived ?? []).map(buildAttackReceived), []);

    // Exp to next level
    let expToNextLevel = 0;
    try {
      const nextLevelExp = getLevelTotalExp(pokemon.level + 1, pokemon.species.growthRate);
      expToNextLevel = Math.max(0, nextLevelExp - pokemon.exp);
    } catch {
      /* ignore */
    }

    // Types
    const types = safe(() => pokemon.getTypes(false, false, false), []);

    // Added type (Forest's Curse / Trick-or-Treat)
    const addedType = hasSummonData ? (pokemon.summonData.addedType ?? -1) : -1;

    return {
      valid: true,
      species_id: pokemon.species?.speciesId ?? 0,
      species_name: safe(() => pokemon.species?.getName() ?? "", ""),
      form_index: pokemon.formIndex ?? 0,
      level: pokemon.level ?? 0,
      gender: pokemon.gender ?? -1,
      friendship: pokemon.friendship ?? 0,
      shiny: pokemon.shiny ?? false,
      variant: pokemon.variant ?? 0,
      hp: pokemon.hp ?? 0,
      max_hp: maxHp,
      hp_ratio: maxHp > 0 ? pokemon.hp / maxHp : 0.0,
      base_stats: baseStats,
      ivs,
      stats: computedStats,
      stat_stages: statStages,
      status_effect: pokemon.status?.effect ?? StatusEffect.NONE,
      toxic_turn_count: pokemon.status?.toxicTurnCount ?? 0,
      sleep_turns_remaining: pokemon.status?.sleepTurnsRemaining ?? 0,
      types,
      tera_type: safe(() => pokemon.getTeraType(), pokemon.teraType ?? -1),
      is_terastallized: !!pokemon.isTerastallized,
      added_type: addedType,
      ability_id: safe(() => pokemon.getAbility()?.id ?? 0, 0),
      ability_name: safe(() => pokemon.getAbility()?.name ?? "", ""),
      passive_ability_id: safe(() => pokemon.getPassiveAbility()?.id ?? 0, 0),
      passive_ability_name: safe(() => pokemon.getPassiveAbility()?.name ?? "", ""),
      has_passive: safe(() => pokemon.hasPassive(), false),
      ability_suppressed: hasSummonData ? (pokemon.summonData.abilitySuppressed ?? false) : false,
      ability_revealed: pokemon.waveData?.abilityRevealed ?? false,
      // v9 fog-of-war inputs: whether a human player could know this mon /
      // its moves. Players are always fully known. move_known derives from
      // the CURRENT summon's move history (conservative: knowledge resets
      // when the enemy re-summons — refine when fog training starts).
      was_seen: isPlayer
        ? true
        : safe(() => globalScene.currentBattle?.seenEnemyPartyMemberIds?.has(pokemon.id) ?? false, false),
      move_known: isPlayer
        ? moveset.map(() => true)
        : safe(
            () => {
              const seenIds = new Set((pokemon.summonData?.moveHistory ?? []).map(mh => mh.move));
              return moveset.map(m => (m?.moveId != null ? seenIds.has(m.moveId) : false));
            },
            moveset.map(() => false),
          ),
      nature: safe(() => pokemon.getNature(), pokemon.nature ?? 0),
      nature_multipliers: natureMults,
      moves,
      move_history: moveHistory,
      pokeball: pokemon.pokeball ?? 0,
      volatile_tags: volatileTags,
      is_boss: isBoss,
      boss_segments: isBoss ? ((pokemon as any).bossSegments ?? 0) : 0,
      boss_segment_index: isBoss ? ((pokemon as any).bossSegmentIndex ?? 0) : 0,
      ai_type: (pokemon as any).aiType ?? 0,
      is_fusion: !!pokemon.fusionSpecies,
      fusion_species_id: pokemon.fusionSpecies?.speciesId ?? null,
      is_on_field: safe(() => pokemon.isOnField(), false),
      is_player: isPlayer,
      battler_index: safe(() => pokemon.getBattlerIndex(), -1),
      field_index: safe(() => pokemon.getFieldIndex(), -1),
      held_items: heldItems,
      move_queue: moveQueue,
      wave_turn_count: pokemon.tempSummonData?.waveTurnCount ?? 0,
      is_fainted: safe(() => pokemon.isFainted(), false),
      is_active: safe(() => pokemon.isActive(), false),
      is_trapped: safe(() => pokemon.isTrapped([], true), false),
      is_grounded: safe(() => pokemon.isGrounded(), true),
      transform_species_id: hasSummonData ? (pokemon.summonData.speciesForm?.speciesId ?? null) : null,
      transform_moves: null, // Transform moveset is already reflected in getMoveset()
      illusion_species_id: hasSummonData ? (pokemon.summonData.illusion?.species ?? null) : null,
      attacks_received: attacksReceived,
      turn_data: turnData,
      battle_data: battleData,
      weight: safe(() => pokemon.getWeight(), 0.0),
      catch_rate: pokemon.species?.catchRate ?? 0,
      base_total: baseStats.reduce((s: number, v: number) => s + v, 0),
      stellar_types_boosted: pokemon.stellarTypesBoosted ? Array.from(pokemon.stellarTypesBoosted) : [],
      berries_eaten_last: hasSummonData ? (pokemon.summonData.berriesEatenLast ?? []) : [],
      exp_to_next_level: expToNextLevel,
      luck: pokemon.luck ?? 0,
      endured_this_wave: pokemon.waveData?.endured ?? false,
      is_mega: safe(() => pokemon.isMega(), false),
      is_max: safe(() => pokemon.isMax(), false),
    };
  } catch (err) {
    console.error("[state-builder] Error building pokemon state:", err);
    return emptyPokemonState();
  }
}

// ─── Arena Tag ───────────────────────────────────────────────────────

function buildArenaTagDict(tag: ArenaTag): Record<string, unknown> {
  return {
    tag_type: tag.tagType,
    side: tag.side,
    turn_count: tag.turnCount ?? 0,
    layers: (tag as any).layers ?? 1,
    source_id: tag.sourceId ?? null,
  };
}

// ─── Field State ─────────────────────────────────────────────────────

function buildFieldState(): Record<string, unknown> {
  try {
    const arena = globalScene.arena;
    const battle = globalScene.currentBattle;

    if (!arena) {
      return {
        biome_id: 0,
        biome_name: "",
        weather_type: 0,
        weather_turns_left: 0,
        weather_is_permanent: false,
        weather_suppressed: false,
        terrain_type: 0,
        terrain_turns_left: 0,
        terrain_is_permanent: false,
        player_teras_used: 0,
        arena_tags: [],
        positional_tags: [],
        is_double_battle: false,
        trick_room_active: false,
        gravity_active: false,
        ignore_abilities: false,
        player_spikes_layers: 0,
        player_toxic_spikes_layers: 0,
        player_stealth_rock: false,
        player_sticky_web: false,
        enemy_spikes_layers: 0,
        enemy_toxic_spikes_layers: 0,
        enemy_stealth_rock: false,
        enemy_sticky_web: false,
      };
    }

    // Weather
    const weatherType = arena.weather?.weatherType ?? 0;
    const weatherTurnsLeft = arena.weather?.turnsLeft ?? 0;
    const weatherIsPermanent = arena.weather ? weatherTurnsLeft === 0 : false;
    const weatherSuppressed = safe(() => (arena.weather ? arena.weather.isEffectSuppressed() : false), false);

    // Terrain
    const terrainType = arena.terrain?.terrainType ?? 0;
    const terrainTurnsLeft = arena.terrain?.turnsLeft ?? 0;
    const terrainIsPermanent = arena.terrain ? terrainTurnsLeft === 0 : false;

    // Count player teras used
    const playerTerasUsed = arena.playerTerasUsed ?? 0;

    // Arena tags
    const arenaTags = (arena.tags ?? []).filter(Boolean).map(buildArenaTagDict);

    // Positional tags (Future Sight, Wish) — from arena.positionalTagManager.tags
    const positionalTags: Record<string, unknown>[] = [];
    try {
      const posTags = arena.positionalTagManager?.tags ?? [];
      for (const pt of posTags) {
        const ptDict: Record<string, unknown> = {
          tag_type: pt.tagType ?? "",
          countdown: pt.turnCount ?? 0,
          target_index: pt.targetIndex ?? 0,
          source_id: null,
          move_id: null,
          heal_hp: null,
        };
        if (pt instanceof DelayedAttackTag) {
          ptDict.source_id = pt.sourceId ?? null;
          ptDict.move_id = pt.sourceMove ?? null;
        }
        if (pt instanceof WishTag) {
          ptDict.heal_hp = pt.healHp ?? null;
        }
        positionalTags.push(ptDict);
      }
    } catch {
      /* ignore positional tag errors */
    }

    return {
      biome_id: arena.biomeId ?? 0,
      biome_name: BiomeId[arena.biomeId] ?? "",
      weather_type: weatherType,
      weather_turns_left: weatherTurnsLeft,
      weather_is_permanent: weatherIsPermanent,
      weather_suppressed: weatherSuppressed,
      terrain_type: terrainType,
      terrain_turns_left: terrainTurnsLeft,
      terrain_is_permanent: terrainIsPermanent,
      player_teras_used: playerTerasUsed,
      arena_tags: arenaTags,
      positional_tags: positionalTags,
      is_double_battle: battle?.double ?? false,
      trick_room_active: hasArenaTag(ArenaTagType.TRICK_ROOM, ArenaTagSide.BOTH),
      gravity_active: hasArenaTag(ArenaTagType.GRAVITY, ArenaTagSide.BOTH),
      ignore_abilities: arena.ignoreAbilities ?? false,
      player_spikes_layers: getHazardLayers(ArenaTagType.SPIKES, ArenaTagSide.PLAYER),
      player_toxic_spikes_layers: getHazardLayers(ArenaTagType.TOXIC_SPIKES, ArenaTagSide.PLAYER),
      player_stealth_rock: hasArenaTag(ArenaTagType.STEALTH_ROCK, ArenaTagSide.PLAYER),
      player_sticky_web: hasArenaTag(ArenaTagType.STICKY_WEB, ArenaTagSide.PLAYER),
      enemy_spikes_layers: getHazardLayers(ArenaTagType.SPIKES, ArenaTagSide.ENEMY),
      enemy_toxic_spikes_layers: getHazardLayers(ArenaTagType.TOXIC_SPIKES, ArenaTagSide.ENEMY),
      enemy_stealth_rock: hasArenaTag(ArenaTagType.STEALTH_ROCK, ArenaTagSide.ENEMY),
      enemy_sticky_web: hasArenaTag(ArenaTagType.STICKY_WEB, ArenaTagSide.ENEMY),
    };
  } catch (err) {
    console.error("[state-builder] Error building field state:", err);
    return {
      biome_id: 0,
      biome_name: "",
      weather_type: 0,
      weather_turns_left: 0,
      weather_is_permanent: false,
      weather_suppressed: false,
      terrain_type: 0,
      terrain_turns_left: 0,
      terrain_is_permanent: false,
      player_teras_used: 0,
      arena_tags: [],
      positional_tags: [],
      is_double_battle: false,
      trick_room_active: false,
      gravity_active: false,
      ignore_abilities: false,
      player_spikes_layers: 0,
      player_toxic_spikes_layers: 0,
      player_stealth_rock: false,
      player_sticky_web: false,
      enemy_spikes_layers: 0,
      enemy_toxic_spikes_layers: 0,
      enemy_stealth_rock: false,
      enemy_sticky_web: false,
    };
  }
}

// ─── Trainer Info ────────────────────────────────────────────────────

function buildTrainerInfo(trainer: any): Record<string, unknown> | null {
  if (!trainer) {
    return null;
  }
  try {
    return {
      trainer_type: trainer.config?.trainerType ?? 0,
      trainer_name: safe(() => trainer.getName?.() ?? "", ""),
      is_double: trainer.config?.doubleOnly ?? false,
      is_boss: trainer.config?.isBoss ?? false,
      party_template_size: trainer.getPartyTemplate?.()?.size ?? 0,
      specialty_type: trainer.config?.specialtyType ?? null,
      tera_mode: trainer.config?.trainerAI?.teraMode ?? null,
    };
  } catch (err) {
    console.error("[state-builder] Error building trainer info:", err);
    return null;
  }
}

// ─── Battle State ────────────────────────────────────────────────────

function buildBattleState(): Record<string, unknown> {
  try {
    const battle = globalScene.currentBattle;
    const arena = globalScene.arena;

    // Pokeball counts
    const pokeballCounts = globalScene.pokeballCounts ?? {};
    const pokeballCountsDict: Record<string, unknown> = {
      pokeball: pokeballCounts[0] ?? 0,
      great_ball: pokeballCounts[1] ?? 0,
      ultra_ball: pokeballCounts[2] ?? 0,
      rogue_ball: pokeballCounts[3] ?? 0,
      master_ball: pokeballCounts[4] ?? 0,
    };

    // Player alive count
    const playerAliveCount = safe(
      () => (globalScene.getPlayerParty() ?? []).filter(p => p && !p.isFainted()).length,
      0,
    );

    // Enemy alive count
    const enemyAliveCount = safe(
      () => (globalScene.getEnemyParty?.() ?? []).filter(p => p && !p.isFainted()).length,
      0,
    );

    // Player faints in battle
    const playerFaintsBattle = battle?.playerFaintsHistory?.length ?? 0;

    // Can tera: the real gate is Tera Orb possession + an unused arena tera
    // (mirrors canTerastallize minus the per-pokemon form checks, which live
    // in the action mask). The old "no party member is terastallized" check
    // reported true from wave 1 with no orb.
    const teraAvailable = safe(
      () =>
        globalScene.findModifier(m => m.is("TerastallizeAccessModifier")) != null
        && (globalScene.arena?.playerTerasUsed ?? 0) < MAX_TERAS_PER_ARENA,
      false,
    );

    // Can run: wild battle outside the END biome (per-pokemon trapping lives
    // in the action mask, not this battle-level feature)
    const canRun = battle?.battleType === BattleType.WILD && arena?.biomeId !== BiomeId.END;

    // Can catch: some ball throw is actually available right now — the
    // game-faithful gating (boss shields, END biome, exactly one visible
    // target; shared with the action mask via getLegalBallTypes) ANDed with
    // ball inventory, i.e. "at least one BALL action is currently legal".
    const canCatch = safe(() => {
      const counts = globalScene.pokeballCounts ?? {};
      return getLegalBallTypes().some((legal, i) => legal && (counts[i] ?? 0) > 0);
    }, false);

    // Challenges
    const challenges: Record<string, unknown>[] = [];
    try {
      const gameChallenges = globalScene.gameMode?.challenges ?? [];
      for (const c of gameChallenges) {
        if (c.value > 0) {
          challenges.push({
            challenge_type: c.id ?? 0,
            challenge_name: Challenges[c.id] ?? "",
            value: c.value ?? 0,
            severity: c.severity ?? 0,
          });
        }
      }
    } catch {
      /* ignore */
    }

    // Mystery encounter
    let mysteryEncounter: Record<string, unknown> | null = null;
    try {
      const me = battle?.mysteryEncounter;
      if (me) {
        const options = (me.options ?? []).map((opt: any, i: number) => ({
          index: i,
          label: opt.dialogue?.buttonLabel ?? "",
          has_requirements: (opt.requirements?.length ?? 0) > 0,
          is_available: opt.meetsRequirements?.() ?? true,
        }));
        mysteryEncounter = {
          encounter_type: me.encounterType ?? 0,
          encounter_name: me.constructor?.name ?? "",
          options,
        };
      }
    } catch {
      /* ignore */
    }

    return {
      biome_id: arena?.biomeId ?? 0,
      wave_index: battle?.waveIndex ?? 0,
      turn: battle?.turn ?? 0,
      battle_type: battle?.battleType ?? 0,
      battle_spec: battle?.battleSpec ?? 0,
      is_double: battle?.double ?? false,
      escape_attempts: battle?.escapeAttempts ?? 0,
      player_alive_count: playerAliveCount,
      enemy_alive_count: enemyAliveCount,
      player_faints_battle: playerFaintsBattle,
      enemy_faints_battle: battle?.enemyFaints ?? 0,
      last_move_id: battle?.lastMove ?? null,
      money: globalScene.money ?? 0,
      score: globalScene.score ?? 0,
      pokeball_counts: pokeballCountsDict,
      can_run: canRun,
      can_catch: canCatch,
      tera_available: teraAvailable,
      game_mode: globalScene.gameMode?.modeId ?? 0,
      seed: globalScene.seed ?? "",
      trainer: buildTrainerInfo(battle?.trainer),
      mystery_encounter: mysteryEncounter,
      battle_style: (globalScene as any).battleStyle ?? 0,
      time_of_day: safe(() => arena?.getTimeOfDay() ?? 0, 0),
      player_faints_biome: arena?.playerFaints ?? 0,
      money_scattered: battle?.moneyScattered ?? 0,
      challenges,
      lock_modifier_tiers: globalScene.lockModifierTiers ?? false,
      reroll_count: safe(() => {
        const phase = globalScene.phaseManager?.getCurrentPhase();
        if (phase?.is("SelectModifierPhase")) {
          return (phase as any).getRerollCount?.() ?? 0;
        }
        return 0;
      }, 0),
      failed_run_away: globalScene.currentBattle?.failedRunAway ?? false,
      has_no_shop: globalScene.gameMode?.hasNoShop ?? false,
      has_trainers: globalScene.gameMode?.hasTrainers ?? true,
      is_spliced_only: (globalScene.gameMode as any)?.isSplicedOnly ?? false,
      seen_enemy_count: battle?.seenEnemyPartyMemberIds?.size ?? 0,
      enemy_switch_counter: battle?.enemySwitchCounter ?? 0,
      offset_gym: (globalScene as any)?.offsetGym ?? false,
      is_classic: globalScene.gameMode?.isClassic ?? false,
      is_endless: globalScene.gameMode?.isEndless ?? false,
      is_daily: globalScene.gameMode?.isDaily ?? false,
      is_challenge: (globalScene.gameMode as any)?.isChallenge ?? false,
      has_mystery_encounters: globalScene.gameMode?.hasMysteryEncounters ?? false,
      has_short_biomes: globalScene.gameMode?.hasShortBiomes ?? false,
      has_random_biomes: globalScene.gameMode?.hasRandomBiomes ?? false,
      has_random_bosses: (globalScene.gameMode as any)?.hasRandomBosses ?? false,
      inverse_battle: challenges.some(
        c => (c.challenge_type as number) === Challenges.INVERSE_BATTLE && (c.value as number) > 0,
      ),
    };
  } catch (err) {
    console.error("[state-builder] Error building battle state:", err);
    return {
      biome_id: 0,
      wave_index: 0,
      turn: 0,
      battle_type: 0,
      battle_spec: 0,
      is_double: false,
      escape_attempts: 0,
      player_alive_count: 0,
      enemy_alive_count: 0,
      player_faints_battle: 0,
      enemy_faints_battle: 0,
      last_move_id: null,
      money: 0,
      score: 0,
      pokeball_counts: { pokeball: 0, great_ball: 0, ultra_ball: 0, rogue_ball: 0, master_ball: 0 },
      can_run: false,
      can_catch: false,
      tera_available: false,
      game_mode: 0,
      seed: "",
      trainer: null,
      mystery_encounter: null,
      battle_style: 0,
      time_of_day: 0,
      player_faints_biome: 0,
      money_scattered: 0,
      challenges: [],
      lock_modifier_tiers: false,
      reroll_count: 0,
      failed_run_away: false,
      has_no_shop: false,
      has_trainers: true,
      is_spliced_only: false,
      seen_enemy_count: 0,
      enemy_switch_counter: 0,
      offset_gym: false,
      is_classic: false,
      is_endless: false,
      is_daily: false,
      is_challenge: false,
      has_mystery_encounters: false,
      has_short_biomes: false,
      has_random_biomes: false,
      has_random_bosses: false,
      inverse_battle: false,
    };
  }
}

// ─── Party Modifier ──────────────────────────────────────────────────

function buildPartyModifier(modifier: PersistentModifier): Record<string, unknown> {
  const result: Record<string, unknown> = {
    modifier_class: modifier.constructor.name,
    modifier_id: modifier.type?.id ?? "",
    name: modifier.type?.name ?? "",
    stack_count: modifier.stackCount ?? 0,
    max_stack_count: safe(() => modifier.getMaxStackCount(), 0),
    type_id: null,
    stat_id: null,
    status_effect: null,
  };

  try {
    // EnemyAttackStatusEffectChanceModifier has .effect (StatusEffect)
    if (modifier instanceof EnemyAttackStatusEffectChanceModifier) {
      result.status_effect = modifier.effect ?? null;
    }
    // TempStatStageBoosterModifier has .stat
    if (modifier instanceof TempStatStageBoosterModifier) {
      result.stat_id = (modifier as any).stat ?? null;
    }
    // AttackTypeBoosterModifier has .moveType (on held items; check for party-level too)
    if ((modifier as any).moveType !== undefined) {
      result.type_id = (modifier as any).moveType ?? null;
    }
    // Generic stat access for any modifier with a .stat property
    if (result.stat_id === null && (modifier as any).stat !== undefined) {
      result.stat_id = (modifier as any).stat ?? null;
    }
    // Generic effect access for any modifier with an .effect property
    if (result.status_effect === null && (modifier as any).effect !== undefined) {
      result.status_effect = (modifier as any).effect ?? null;
    }
  } catch {
    /* ignore subclass extraction errors */
  }

  return result;
}

// ─── Lapsing Modifier ────────────────────────────────────────────────

function buildLapsingModifier(modifier: LapsingPersistentModifier): Record<string, unknown> {
  const result: Record<string, unknown> = {
    modifier_class: modifier.constructor.name,
    modifier_id: modifier.type?.id ?? "",
    name: modifier.type?.name ?? "",
    stack_count: modifier.stackCount ?? 0,
    battles_remaining: (modifier as any).battleCount ?? 0,
    stat_id: null,
    boost: null,
  };

  if (modifier instanceof TempStatStageBoosterModifier) {
    result.stat_id = (modifier as any).stat ?? null;
    result.boost = (modifier as any).boost ?? null;
  }

  return result;
}

// ─── Modifier Inventory ──────────────────────────────────────────────

function buildModifierInventory(): Record<string, unknown> {
  try {
    const playerParty = globalScene.getPlayerParty() ?? [];
    const heldItemsBySlot: Record<string, unknown[]> = {};

    for (let i = 0; i < playerParty.length; i++) {
      try {
        heldItemsBySlot[String(i)] = (playerParty[i]?.getHeldItems() ?? []).map(buildHeldItem);
      } catch {
        heldItemsBySlot[String(i)] = [];
      }
    }

    // Party-wide modifiers (excluding held items and lapsing)
    const partyModifiers: Record<string, unknown>[] = [];
    const lapsingModifiers: Record<string, unknown>[] = [];

    for (const mod of globalScene.modifiers ?? []) {
      try {
        if (mod instanceof LapsingPersistentModifier) {
          lapsingModifiers.push(buildLapsingModifier(mod));
        } else if (!(mod instanceof PokemonHeldItemModifier)) {
          partyModifiers.push(buildPartyModifier(mod));
        }
      } catch {
        /* ignore individual modifier errors */
      }
    }

    // Enemy modifiers
    const enemyModifiers: Record<string, unknown>[] = [];
    for (const mod of (globalScene as any).enemyModifiers ?? []) {
      try {
        enemyModifiers.push(buildPartyModifier(mod));
      } catch {
        /* ignore */
      }
    }

    return {
      held_items: heldItemsBySlot,
      party_modifiers: partyModifiers,
      lapsing_modifiers: lapsingModifiers,
      enemy_modifiers: enemyModifiers,
    };
  } catch (err) {
    console.error("[state-builder] Error building modifier inventory:", err);
    return { held_items: {}, party_modifiers: [], lapsing_modifiers: [], enemy_modifiers: [] };
  }
}

// ─── Phase Info ──────────────────────────────────────────────────────

function buildPhaseInfo(phaseState: PhaseState | null): Record<string, unknown> {
  if (!phaseState) {
    return {
      current_phase: "unknown",
      command_field_index: -1,
      command_pokemon_species: null,
      action_mask: new Array(58).fill(false),
      valid_actions: [],
      learn_move_id: null,
      learn_move_name: null,
      learn_move_stats: null,
      learn_move_party_index: null,
      learn_move_current: null,
      biome_options: null,
      mystery_option_count: null,
      is_game_over: null,
      is_victory: null,
    };
  }

  const meta = phaseState.metadata ?? {};

  // Metadata keys must match what the phase-router mask builders actually
  // set (newMoveName/currentMoveNames, biomeNames, optionCount) — the old
  // reads used keys nothing writes, so these fields were always null.
  return {
    current_phase: phaseState.phase ?? "unknown",
    command_field_index: (meta.fieldIndex as number) ?? -1,
    command_pokemon_species: (meta.pokemonSpecies as string) ?? null,
    action_mask: phaseState.actionMask ?? new Array(58).fill(false),
    valid_actions: phaseState.validActions ?? [],
    learn_move_id: (meta.learnMoveId as number) ?? null,
    learn_move_name: (meta.newMoveName as string) ?? null,
    learn_move_stats: meta.learnMoveStats ? buildMoveSlot(meta.learnMoveStats as any, null) : null,
    learn_move_party_index: (meta.learnMovePartyIndex as number) ?? null,
    learn_move_current: (meta.currentMoveNames as string[]) ?? null,
    biome_options: (meta.biomeNames as string[]) ?? null,
    mystery_option_count: phaseState.phase === "mystery" ? ((meta.optionCount as number) ?? null) : null,
    is_game_over: (meta.gameOver as boolean) ?? (meta.isGameOver as boolean) ?? null,
    is_victory: (meta.isVictory as boolean) ?? null,
  };
}

// ─── Modifier Type Helpers ────────────────────────────────────────────

function extractModifierTypeId(modType: any): number | null {
  if (!modType) {
    return null;
  }
  try {
    // AttackTypeBoosterModifierType has .moveType
    if (modType.moveType !== undefined) {
      return modType.moveType;
    }
    // TerastallizeModifierType has .teraType
    if (modType.teraType !== undefined) {
      return modType.teraType;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function extractModifierStatId(modType: any): number | null {
  if (!modType) {
    return null;
  }
  try {
    // Stat-boosting modifier types may have .stat
    if (modType.stat !== undefined) {
      return modType.stat;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function extractModifierMoveId(modType: any): number | null {
  if (!modType) {
    return null;
  }
  try {
    // TmModifierType has .moveId
    if (modType.moveId !== undefined) {
      return modType.moveId;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function extractModifierBerryType(modType: any): number | null {
  if (!modType) {
    return null;
  }
  try {
    // BerryModifierType has .berryType
    if (modType.berryType !== undefined) {
      return modType.berryType;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function extractModifierDescription(modType: any): string {
  if (!modType) {
    return "";
  }
  try {
    if (typeof modType.getDescription === "function") {
      return modType.getDescription() ?? "";
    }
  } catch {
    /* ignore */
  }
  return "";
}

// ─── Shop State ──────────────────────────────────────────────────────

function buildShopState(): Record<string, unknown> | null {
  try {
    const modifiers = getAvailableModifiers();
    if (!modifiers) {
      return null;
    }

    const rewardOptions = modifiers.rewards.map((r, i) => {
      const modType = r.raw?.type;
      return {
        index: i,
        tier: r.tier ?? 0,
        upgrade_count: r.upgradeCount ?? 0,
        name: r.name ?? "",
        modifier_id: r.id ?? "",
        modifier_class: modType?.constructor?.name ?? "",
        target_kind: r.targetKind ?? "none",
        is_pokemon_modifier: r.targetKind !== "none",
        type_id: extractModifierTypeId(modType),
        stat_id: extractModifierStatId(modType),
        move_id: extractModifierMoveId(modType),
        berry_type: extractModifierBerryType(modType),
        description: extractModifierDescription(modType),
      };
    });

    const shopOptions = modifiers.shop.map((s, i) => {
      const modType = s.raw?.type;
      return {
        index: i,
        cost: s.cost ?? 0,
        tier: s.tier ?? 0,
        name: s.name ?? "",
        modifier_id: s.id ?? "",
        modifier_class: modType?.constructor?.name ?? "",
        target_kind: s.targetKind ?? "none",
        affordable: (globalScene.money ?? 0) >= (s.cost ?? 0),
        type_id: extractModifierTypeId(modType),
        stat_id: extractModifierStatId(modType),
        move_id: extractModifierMoveId(modType),
        berry_type: extractModifierBerryType(modType),
        description: extractModifierDescription(modType),
      };
    });

    return {
      reward_options: rewardOptions,
      shop_options: shopOptions,
      can_reroll: modifiers.canReroll,
      reroll_cost: modifiers.rerollCost,
      money: modifiers.money,
    };
  } catch (err) {
    console.error("[state-builder] Error building shop state:", err);
    return null;
  }
}

// ─── Top-Level: Build Game State ─────────────────────────────────────

/**
 * Build the complete game state dict matching the GameState TypedDict schema.
 *
 * @param phaseState - Current phase state from the phase router (null if unknown)
 * @param step - Decision step counter
 * @returns GameState dict matching state_schema.py
 */
export function buildGameState(phaseState: PhaseState | null, step: number): Record<string, unknown> {
  // ── Gather parties ──
  // IMPORTANT: Do NOT .filter() on getPlayerField/getEnemyField — it shifts indices.
  const playerField = safe(() => globalScene.getPlayerField() ?? [], []);
  const enemyField = safe(() => globalScene.getEnemyField() ?? [], []);
  const playerParty = safe(() => globalScene.getPlayerParty() ?? [], []);
  const enemyParty = safe(() => globalScene.getEnemyParty?.() ?? [], []);

  // ── Build active field pokemon IDs for bench exclusion ──
  const playerFieldIds = new Set<number>();
  for (const p of playerField) {
    if (p) {
      playerFieldIds.add(p.id);
    }
  }
  const enemyFieldIds = new Set<number>();
  for (const p of enemyField) {
    if (p) {
      enemyFieldIds.add(p.id);
    }
  }

  // ── Player bench: party members NOT on the active field ──
  const playerBench: (Pokemon | null)[] = [];
  for (const p of playerParty) {
    if (p && !playerFieldIds.has(p.id)) {
      playerBench.push(p);
    }
  }

  // ── Enemy bench: party members NOT on the active field ──
  const enemyBench: (Pokemon | null)[] = [];
  for (const p of enemyParty) {
    if (p && !enemyFieldIds.has(p.id)) {
      enemyBench.push(p);
    }
  }

  // ── Build 12 Pokemon slots ──
  // v9 slot mapping: slot 1 = second active in DOUBLES, first bench
  // member in SINGLES. Before v9 the ally slot sat empty in singles and
  // bench held only 4 of the 5 reserves — the 6th party member (both
  // sides) was structurally invisible to the agent.
  const isDouble = safe(() => !!globalScene.currentBattle?.double, false);
  const playerSlot1 = isDouble ? (playerField[1] ?? null) : (playerBench[0] ?? null);
  const enemySlot1 = isDouble ? (enemyField[1] ?? null) : (enemyBench[0] ?? null);
  const benchBase = isDouble ? 0 : 1;

  const player0 = buildPokemonState(playerField[0] ?? null, true, 0);
  const player1 = buildPokemonState(playerSlot1, true, 1);
  const player2 = buildPokemonState(playerBench[benchBase] ?? null, true, 2);
  const player3 = buildPokemonState(playerBench[benchBase + 1] ?? null, true, 3);
  const player4 = buildPokemonState(playerBench[benchBase + 2] ?? null, true, 4);
  const player5 = buildPokemonState(playerBench[benchBase + 3] ?? null, true, 5);

  const enemy0 = buildPokemonState(enemyField[0] ?? null, false, 0);
  const enemy1 = buildPokemonState(enemySlot1, false, 1);
  const enemy2 = buildPokemonState(enemyBench[benchBase] ?? null, false, 2);
  const enemy3 = buildPokemonState(enemyBench[benchBase + 1] ?? null, false, 3);
  const enemy4 = buildPokemonState(enemyBench[benchBase + 2] ?? null, false, 4);
  const enemy5 = buildPokemonState(enemyBench[benchBase + 3] ?? null, false, 5);

  return {
    // Pokemon slots
    player_0: player0,
    player_1: player1,
    player_2: player2,
    player_3: player3,
    player_4: player4,
    player_5: player5,
    enemy_0: enemy0,
    enemy_1: enemy1,
    enemy_2: enemy2,
    enemy_3: enemy3,
    enemy_4: enemy4,
    enemy_5: enemy5,

    // Field state
    field: buildFieldState(),

    // Battle / run state
    battle: buildBattleState(),

    // Modifier inventory
    modifiers: buildModifierInventory(),

    // Phase / decision info
    phase: buildPhaseInfo(phaseState),

    // Shop state (only during SelectModifierPhase)
    shop: buildShopState(),

    // Action labels (populated by caller if needed)
    action_labels: null,

    // Protocol metadata
    step,
    timestamp: Date.now() / 1000,
  };
}
