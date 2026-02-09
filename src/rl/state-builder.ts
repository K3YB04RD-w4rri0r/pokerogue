/**
 * Comprehensive game state serializer for the RL framework.
 *
 * Serializes the full live game state into a JSON dict matching the
 * GameState TypedDict defined in state_schema.py (35 TypedDicts, 364 fields).
 *
 * Replaces the minimal 6-field buildGameState() previously in cli.ts.
 */

import { globalScene } from "#app/global-scene";
import type { Pokemon } from "#field/pokemon";
import type { PokemonMove } from "#moves/pokemon-move";
import { MoveFlags } from "#enums/move-flags";
import { StatusEffect } from "#enums/status-effect";
import { Stat, EFFECTIVE_STATS, PERMANENT_STATS } from "#enums/stat";
import { ArenaTagType } from "#enums/arena-tag-type";
import { ArenaTagSide } from "#enums/arena-tag-side";
import { getNatureStatMultiplier } from "#data/nature";
import { getLevelTotalExp } from "#data/exp";
import { getAvailableModifiers } from "#rl/modifier-api";
import type { PhaseState } from "#rl/phase-router";
import { BiomeId } from "#enums/biome-id";

// BattlerTag subclass imports for instanceof checks
import {
  SubstituteTag,
  StockpilingTag,
  EncoreTag,
  DisabledTag,
  TypeBoostTag,
  CritBoostTag,
  GorillaTacticsTag,
  HighestStatBoostTag,
  SupremeOverlordTag,
  AutotomizedTag,
} from "#data/battler-tags";
import type { BattlerTag } from "#data/battler-tags";
import type { ArenaTag } from "#data/arena-tag";

// Modifier subclass imports for instanceof checks
import {
  PokemonHeldItemModifier,
  AttackTypeBoosterModifier,
  BaseStatModifier,
  BerryModifier,
  TurnStatusEffectModifier,
  PokemonBaseStatTotalModifier,
  PokemonFormChangeItemModifier,
  LapsingPersistentModifier,
  TempStatStageBoosterModifier,
} from "#modifiers/modifier";
import type { PersistentModifier } from "#modifiers/modifier";

import type { PokemonTurnData, PokemonBattleData, PokemonWaveData } from "#data/pokemon/pokemon-data";
import type { AttackMoveResult } from "#app/@types/attack-move-result";
import type { TurnMove } from "#app/@types/turn-move";

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
  if (!arena?.tags) return 0;
  for (const tag of arena.tags) {
    if (tag.tagType === tagType && (tag.side === side || tag.side === ArenaTagSide.BOTH)) {
      return (tag as any).layers ?? 1;
    }
  }
  return 0;
}

function hasArenaTag(tagType: ArenaTagType, side: ArenaTagSide): boolean {
  const arena = globalScene.arena;
  if (!arena?.tags) return false;
  return arena.tags.some(
    t => t.tagType === tagType && (t.side === side || t.side === ArenaTagSide.BOTH),
  );
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
  if (!td) return emptyTurnData();
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
    status_effect: null,
    berry_type: null,
    consumed: null,
    stat_modifier: null,
    form_change_item: null,
    form_change_active: null,
  };

  try {
    if (modifier instanceof AttackTypeBoosterModifier) {
      result.type_id = modifier.moveType;
    }
    if (modifier instanceof BaseStatModifier) {
      result.stat_id = (modifier as any).stat ?? null;
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

function buildMoveSlot(
  pokemonMove: PokemonMove | null | undefined,
  pokemon: Pokemon | null,
): Record<string, unknown> {
  if (!pokemonMove) return emptyMoveSlot();

  try {
    const move = pokemonMove.getMove();
    if (!move) return emptyMoveSlot();

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
    } catch { /* ignore */ }

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
    } catch { /* ignore */ }

    // Drain ratio (HitHealAttr)
    let drainRatio = 0.0;
    try {
      const hitHealAttrs = move.getAttrs("HitHealAttr");
      if (hitHealAttrs.length > 0) {
        drainRatio = (hitHealAttrs[0] as any).healRatio ?? 0.5;
      }
    } catch { /* ignore */ }

    // Recoil ratio (RecoilAttr)
    let recoilRatio = 0.0;
    try {
      const recoilAttrs = move.getAttrs("RecoilAttr");
      if (recoilAttrs.length > 0) {
        recoilRatio = (recoilAttrs[0] as any).damageRatio ?? 0.25;
      }
    } catch { /* ignore */ }

    // Heal ratio (HealAttr)
    let healRatio = 0.0;
    try {
      const healAttrs = move.getAttrs("HealAttr");
      if (healAttrs.length > 0) {
        healRatio = (healAttrs[0] as any).healRatio ?? 0.5;
      }
    } catch { /* ignore */ }

    // Multi-hit
    let isMultiHit = false;
    let multiHitType = -1;
    try {
      const multiHitAttrs = move.getAttrs("MultiHitAttr");
      if (multiHitAttrs.length > 0) {
        isMultiHit = true;
        multiHitType = (multiHitAttrs[0] as any).intrinsicMultiHitType ?? (multiHitAttrs[0] as any).multiHitType ?? -1;
      }
    } catch { /* ignore */ }

    // Crit stage boost
    let critStageBoost = 0;
    try {
      if (move.hasAttr("CritOnlyAttr")) {
        critStageBoost = 99;
      } else if (move.hasAttr("HighCritAttr")) {
        critStageBoost = 1;
      }
    } catch { /* ignore */ }

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
    } catch { /* ignore */ }

    // Traps target
    const trapsTarget = safe(() => move.hasAttr("TrapAttr"), false);

    // Protect
    const isProtect = safe(() => move.hasAttr("ProtectAttr"), false);

    // Sacrifice
    const isSacrifice = safe(() =>
      move.hasAttr("SacrificialAttr") ||
      move.hasAttr("SacrificialAttrOnHit") ||
      move.hasAttr("HalfSacrificialAttr"),
    false);

    // OHKO
    const isOhko = safe(() => move.hasAttr("OneHitKOAttr"), false);

    // Fixed damage
    let fixedDamage = 0;
    try {
      const fixedDmgAttrs = move.getAttrs("FixedDamageAttr");
      if (fixedDmgAttrs.length > 0) {
        fixedDamage = (fixedDmgAttrs[0] as any).damage ?? 0;
      }
    } catch { /* ignore */ }

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
  slotIndex: number,
): Record<string, unknown> {
  if (!pokemon) return emptyPokemonState();

  try {
    const maxHp = safe(() => pokemon.getMaxHp(), 0);
    const hasSummonData = !!pokemon.summonData;

    // Base stats from species form
    const baseStats = safe(() => pokemon.getSpeciesForm(true).baseStats.slice(), [0, 0, 0, 0, 0, 0]);

    // Computed stats (from IVs + EVs + nature + level)
    const computedStats = safe(() => pokemon.getStats(true), [0, 0, 0, 0, 0, 0]);

    // IVs
    const ivs = safe(() => Array.from(pokemon.ivs), [0, 0, 0, 0, 0, 0]);

    // Stat stages (only meaningful for on-field Pokemon with summonData)
    const statStages = hasSummonData
      ? safe(() => Array.from(pokemon.getStatStages()), [0, 0, 0, 0, 0, 0, 0])
      : [0, 0, 0, 0, 0, 0, 0];

    // Nature multipliers for the 5 effective stats
    const natureMults = safe(
      () => EFFECTIVE_STATS.map(s => getNatureStatMultiplier(pokemon.nature, s)),
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
    const volatileTags = hasSummonData
      ? safe(() => (pokemon.summonData.tags ?? []).map(buildVolatileTag), [])
      : [];

    // Move queue
    const moveQueue = hasSummonData
      ? safe(() => (pokemon.summonData.moveQueue ?? []).map(buildQueuedMove), [])
      : [];

    // Boss detection
    const isBoss = "bossSegments" in pokemon && (pokemon as any).bossSegments > 0;

    // Held items
    const heldItems = safe(() => pokemon.getHeldItems().map(buildHeldItem), []);

    // Turn data and battle data
    const turnData = buildTurnData(pokemon.turnData);
    const battleData = buildBattleDataDict(pokemon.battleData, pokemon.waveData);

    // Attacks received (convenience copy from turnData)
    const attacksReceived = safe(
      () => (pokemon.turnData?.attacksReceived ?? []).map(buildAttackReceived),
      [],
    );

    // Exp to next level
    let expToNextLevel = 0;
    try {
      const nextLevelExp = getLevelTotalExp(pokemon.level + 1, pokemon.species.growthRate);
      expToNextLevel = Math.max(0, nextLevelExp - pokemon.exp);
    } catch { /* ignore */ }

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
      tera_type: pokemon.teraType ?? -1,
      is_terastallized: !!pokemon.isTerastallized,
      added_type: addedType,
      ability_id: safe(() => pokemon.getAbility()?.id ?? 0, 0),
      ability_name: safe(() => pokemon.getAbility()?.name ?? "", ""),
      passive_ability_id: safe(() => pokemon.getPassiveAbility()?.id ?? 0, 0),
      passive_ability_name: safe(() => pokemon.getPassiveAbility()?.name ?? "", ""),
      has_passive: safe(() => pokemon.hasPassive(), false),
      ability_suppressed: hasSummonData ? (pokemon.summonData.abilitySuppressed ?? false) : false,
      ability_revealed: pokemon.waveData?.abilityRevealed ?? false,
      nature: pokemon.nature ?? 0,
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
        biome_id: 0, biome_name: "", weather_type: 0, weather_turns_left: 0,
        weather_is_permanent: false, weather_suppressed: false, terrain_type: 0,
        terrain_turns_left: 0, terrain_is_permanent: false, player_teras_used: 0,
        arena_tags: [], positional_tags: [], is_double_battle: false,
        trick_room_active: false, gravity_active: false, ignore_abilities: false,
        player_spikes_layers: 0, player_toxic_spikes_layers: 0,
        player_stealth_rock: false, player_sticky_web: false,
        enemy_spikes_layers: 0, enemy_toxic_spikes_layers: 0,
        enemy_stealth_rock: false, enemy_sticky_web: false,
      };
    }

    // Weather
    const weatherType = arena.weather?.weatherType ?? 0;
    const weatherTurnsLeft = arena.weather?.turnsLeft ?? 0;
    const weatherIsPermanent = arena.weather ? weatherTurnsLeft === 0 : false;
    const weatherSuppressed = safe(
      () => arena.weather ? arena.weather.isEffectSuppressed() : false,
      false,
    );

    // Terrain
    const terrainType = arena.terrain?.terrainType ?? 0;
    const terrainTurnsLeft = arena.terrain?.turnsLeft ?? 0;
    const terrainIsPermanent = arena.terrain ? terrainTurnsLeft === 0 : false;

    // Count player teras used
    const playerTerasUsed = arena.playerTerasUsed ?? 0;

    // Arena tags
    const arenaTags = (arena.tags ?? []).filter(Boolean).map(buildArenaTagDict);

    // Positional tags (Future Sight, Wish) — not easily accessible, default to empty
    const positionalTags: Record<string, unknown>[] = [];

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
      ignore_abilities: false, // Complex to detect (Mold Breaker etc.), default false
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
      biome_id: 0, biome_name: "", weather_type: 0, weather_turns_left: 0,
      weather_is_permanent: false, weather_suppressed: false, terrain_type: 0,
      terrain_turns_left: 0, terrain_is_permanent: false, player_teras_used: 0,
      arena_tags: [], positional_tags: [], is_double_battle: false,
      trick_room_active: false, gravity_active: false, ignore_abilities: false,
      player_spikes_layers: 0, player_toxic_spikes_layers: 0,
      player_stealth_rock: false, player_sticky_web: false,
      enemy_spikes_layers: 0, enemy_toxic_spikes_layers: 0,
      enemy_stealth_rock: false, enemy_sticky_web: false,
    };
  }
}

// ─── Trainer Info ────────────────────────────────────────────────────

function buildTrainerInfo(trainer: any): Record<string, unknown> | null {
  if (!trainer) return null;
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

    // Can tera
    const teraAvailable = safe(() => {
      const party = globalScene.getPlayerParty() ?? [];
      return !party.some(p => p?.isTerastallized);
    }, false);

    // Can run
    const canRun = battle?.battleType === 0; // WILD

    // Can catch
    const canCatch = battle?.battleType === 0 && !(battle?.double);

    // Challenges
    const challenges: Record<string, unknown>[] = [];
    try {
      const gameChallenges = globalScene.gameMode?.challenges ?? [];
      for (const c of gameChallenges) {
        if (c.value > 0) {
          challenges.push({
            challenge_type: c.challengeType ?? 0,
            challenge_name: c.constructor?.name ?? "",
            value: c.value ?? 0,
            severity: c.severity ?? 0,
          });
        }
      }
    } catch { /* ignore */ }

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
    } catch { /* ignore */ }

    return {
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
      reroll_count: 0, // rerollCount is private on SelectModifierPhase, not accessible here
      failed_run_away: safe(
        () => {
          const party = globalScene.getPlayerParty() ?? [];
          return party.some(p => p?.turnData?.failedRunAway);
        },
        false,
      ),
      has_no_shop: globalScene.gameMode?.hasNoShop ?? false,
      has_trainers: globalScene.gameMode?.hasTrainers ?? true,
      is_spliced_only: (globalScene.gameMode as any)?.isSplicedOnly ?? false,
      seen_enemy_count: battle?.seenEnemyPartyMemberIds?.size ?? 0,
      enemy_switch_counter: battle?.enemySwitchCounter ?? 0,
      offset_gym: (globalScene as any)?.offsetGym ?? false,
    };
  } catch (err) {
    console.error("[state-builder] Error building battle state:", err);
    return {
      wave_index: 0, turn: 0, battle_type: 0, battle_spec: 0, is_double: false,
      escape_attempts: 0, player_alive_count: 0, enemy_alive_count: 0,
      player_faints_battle: 0, enemy_faints_battle: 0, last_move_id: null,
      money: 0, score: 0, pokeball_counts: { pokeball: 0, great_ball: 0, ultra_ball: 0, rogue_ball: 0, master_ball: 0 },
      can_run: false, can_catch: false, tera_available: false,
      game_mode: 0, seed: "", trainer: null, mystery_encounter: null,
      battle_style: 0, time_of_day: 0, player_faints_biome: 0, money_scattered: 0,
      challenges: [], lock_modifier_tiers: false, reroll_count: 0, failed_run_away: false,
      has_no_shop: false, has_trainers: true, is_spliced_only: false,
      seen_enemy_count: 0, enemy_switch_counter: 0, offset_gym: false,
    };
  }
}

// ─── Party Modifier ──────────────────────────────────────────────────

function buildPartyModifier(modifier: PersistentModifier): Record<string, unknown> {
  return {
    modifier_class: modifier.constructor.name,
    modifier_id: modifier.type?.id ?? "",
    name: modifier.type?.name ?? "",
    stack_count: modifier.stackCount ?? 0,
    max_stack_count: safe(() => modifier.getMaxStackCount(), 0),
    type_id: null,
    stat_id: null,
    status_effect: null,
  };
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
      } catch { /* ignore individual modifier errors */ }
    }

    // Enemy modifiers
    const enemyModifiers: Record<string, unknown>[] = [];
    for (const mod of (globalScene as any).enemyModifiers ?? []) {
      try {
        enemyModifiers.push(buildPartyModifier(mod));
      } catch { /* ignore */ }
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
      action_mask: Array(58).fill(false),
      valid_actions: [],
      learn_move_id: null,
      learn_move_name: null,
      learn_move_stats: null,
      learn_move_current: null,
      biome_options: null,
      mystery_option_count: null,
      is_game_over: null,
      is_victory: null,
    };
  }

  const meta = phaseState.metadata ?? {};

  return {
    current_phase: phaseState.phase ?? "unknown",
    command_field_index: (meta.fieldIndex as number) ?? -1,
    command_pokemon_species: (meta.pokemonSpecies as string) ?? null,
    action_mask: phaseState.actionMask ?? Array(58).fill(false),
    valid_actions: phaseState.validActions ?? [],
    learn_move_id: (meta.learnMoveId as number) ?? null,
    learn_move_name: (meta.learnMoveName as string) ?? null,
    learn_move_stats: meta.learnMoveStats ? buildMoveSlot(meta.learnMoveStats as any, null) : null,
    learn_move_current: (meta.currentMoveNames as string[]) ?? null,
    biome_options: (meta.biomeOptions as string[]) ?? null,
    mystery_option_count: (meta.mysteryOptionCount as number) ?? null,
    is_game_over: (meta.isGameOver as boolean) ?? null,
    is_victory: (meta.isVictory as boolean) ?? null,
  };
}

// ─── Shop State ──────────────────────────────────────────────────────

function buildShopState(): Record<string, unknown> | null {
  try {
    const modifiers = getAvailableModifiers();
    if (!modifiers) return null;

    const rewardOptions = modifiers.rewards.map((r, i) => ({
      index: i,
      tier: r.tier ?? 0,
      upgrade_count: r.upgradeCount ?? 0,
      name: r.name ?? "",
      modifier_id: r.id ?? "",
      modifier_class: r.raw?.type?.constructor?.name ?? "",
      target_kind: r.targetKind ?? "none",
      is_pokemon_modifier: r.targetKind !== "none",
      type_id: null as number | null,
      stat_id: null as number | null,
      description: "",
    }));

    const shopOptions = modifiers.shop.map((s, i) => ({
      index: i,
      cost: s.cost ?? 0,
      tier: s.tier ?? 0,
      name: s.name ?? "",
      modifier_id: s.id ?? "",
      modifier_class: s.raw?.type?.constructor?.name ?? "",
      target_kind: s.targetKind ?? "none",
      affordable: (globalScene.money ?? 0) >= (s.cost ?? 0),
      type_id: null as number | null,
      stat_id: null as number | null,
      description: "",
    }));

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
export function buildGameState(
  phaseState: PhaseState | null,
  step: number,
): Record<string, unknown> {
  // ── Gather parties ──
  // IMPORTANT: Do NOT .filter() on getPlayerField/getEnemyField — it shifts indices.
  const playerField = safe(() => globalScene.getPlayerField() ?? [], []);
  const enemyField = safe(() => globalScene.getEnemyField() ?? [], []);
  const playerParty = safe(() => globalScene.getPlayerParty() ?? [], []);
  const enemyParty = safe(() => globalScene.getEnemyParty?.() ?? [], []);

  // ── Build active field pokemon IDs for bench exclusion ──
  const playerFieldIds = new Set<number>();
  for (let i = 0; i < playerField.length; i++) {
    if (playerField[i]) playerFieldIds.add(playerField[i].id);
  }
  const enemyFieldIds = new Set<number>();
  for (let i = 0; i < enemyField.length; i++) {
    if (enemyField[i]) enemyFieldIds.add(enemyField[i].id);
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
  // player_0 = active slot 0, player_1 = active slot 1, player_2..5 = bench
  const player0 = buildPokemonState(playerField[0] ?? null, true, 0);
  const player1 = buildPokemonState(playerField[1] ?? null, true, 1);
  const player2 = buildPokemonState(playerBench[0] ?? null, true, 2);
  const player3 = buildPokemonState(playerBench[1] ?? null, true, 3);
  const player4 = buildPokemonState(playerBench[2] ?? null, true, 4);
  const player5 = buildPokemonState(playerBench[3] ?? null, true, 5);

  const enemy0 = buildPokemonState(enemyField[0] ?? null, false, 0);
  const enemy1 = buildPokemonState(enemyField[1] ?? null, false, 1);
  const enemy2 = buildPokemonState(enemyBench[0] ?? null, false, 2);
  const enemy3 = buildPokemonState(enemyBench[1] ?? null, false, 3);
  const enemy4 = buildPokemonState(enemyBench[2] ?? null, false, 4);
  const enemy5 = buildPokemonState(enemyBench[3] ?? null, false, 5);

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
