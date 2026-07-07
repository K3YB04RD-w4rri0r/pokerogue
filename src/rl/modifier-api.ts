import { globalScene } from "#app/global-scene";
import type { ModifierTier } from "#enums/modifier-tier";
import type { Modifier } from "#modifiers/modifier";
import { HealShopCostModifier } from "#modifiers/modifier";
import type { ModifierType, ModifierTypeOption } from "#modifiers/modifier-type";
import {
  FusePokemonModifierType,
  getPlayerShopModifierTypeOptionsForWave,
  PokemonModifierType,
  PokemonMoveModifierType,
  PokemonPpRestoreModifierType,
  PokemonPpUpModifierType,
  RememberMoveModifierType,
} from "#modifiers/modifier-type";
import type { SelectModifierPhase } from "#phases/select-modifier-phase";
import { SHOP_OPTIONS_ROW_LIMIT } from "#ui/modifier-select-ui-handler";
import { NumberHolder } from "#utils/common";

/**
 * What kind of target a modifier requires for the RL agent.
 * - "none": Applies directly, no target needed.
 * - "pokemon": Needs a party Pokemon index (0-5).
 * - "move": Needs a party Pokemon index AND a move index.
 * - "pokemon_pair": Needs two Pokemon indices (for fusion).
 */
export type ModifierTargetKind = "none" | "pokemon" | "move" | "pokemon_pair";

/** Structured info about a single modifier option for RL consumption. */
export interface ModifierInfo {
  index: number;
  source: "reward" | "shop";
  id: string;
  name: string;
  tier: ModifierTier;
  upgradeCount: number;
  cost: number;
  targetKind: ModifierTargetKind;
  raw: ModifierTypeOption;
}

/** Result returned after attempting a modifier action. */
export interface ModifierActionResult {
  success: boolean;
  error?: string;
}

function getCurrentSelectModifierPhase(): SelectModifierPhase | null {
  const phase = globalScene.phaseManager.getCurrentPhase();
  if (phase?.is("SelectModifierPhase")) {
    return phase;
  }
  return null;
}

function getTargetKind(modifierType: ModifierType): ModifierTargetKind {
  if (modifierType instanceof FusePokemonModifierType) {
    return "pokemon_pair";
  }
  if (
    modifierType instanceof PokemonMoveModifierType
    || modifierType instanceof PokemonPpRestoreModifierType
    || modifierType instanceof PokemonPpUpModifierType
    || modifierType instanceof RememberMoveModifierType
  ) {
    return "move";
  }
  if (modifierType instanceof PokemonModifierType) {
    return "pokemon";
  }
  return "none";
}

function buildModifierInfo(
  opt: ModifierTypeOption,
  index: number,
  source: "reward" | "shop",
  cost: number,
): ModifierInfo {
  return {
    index,
    source,
    id: opt.type.id,
    name: opt.type.name,
    tier: opt.type.tier,
    upgradeCount: opt.upgradeCount,
    cost,
    targetKind: getTargetKind(opt.type),
    raw: opt,
  };
}

/**
 * Query the currently available modifiers during a SelectModifierPhase.
 * Returns null if not currently in a SelectModifierPhase.
 */
export function getAvailableModifiers(): {
  rewards: ModifierInfo[];
  shop: ModifierInfo[];
  rerollCost: number;
  money: number;
  canReroll: boolean;
} | null {
  const phase = getCurrentSelectModifierPhase();
  if (!phase) {
    return null;
  }

  const typeOptions = phase.getTypeOptions();
  const rewards = typeOptions.map((opt, i) => buildModifierInfo(opt, i, "reward", 0));

  const shopTypeOptions = getPlayerShopModifierTypeOptionsForWave(
    globalScene.currentBattle.waveIndex,
    globalScene.getWaveMoneyAmount(1),
  );
  // Report the TRUE purchase price: the game applies HealShopCostModifier
  // (Black Sludge) on top of the option's base cost at buy time
  // (selectShopModifierOption). Reporting the raw cost made the action mask
  // and the observation's cost/affordable features disagree with what the
  // purchase actually charges.
  const shop = shopTypeOptions.map((opt, i) => {
    const adjustedCost = new NumberHolder(opt.cost);
    globalScene.applyModifier(HealShopCostModifier, true, adjustedCost);
    return buildModifierInfo(opt, i, "shop", adjustedCost.value);
  });

  const rerollCost = phase.getRerollCost(globalScene.lockModifierTiers);
  const money = globalScene.money;

  return {
    rewards,
    shop,
    rerollCost,
    money,
    canReroll: rerollCost >= 0 && money >= rerollCost,
  };
}

/**
 * Select a reward modifier by index (free items shown after battle).
 * For pokemon-targeting modifiers, provide pokemonIndex (defaults to 0).
 * For move-targeting modifiers, also provide moveIndex.
 */
export function selectRewardModifier(index: number, pokemonIndex?: number, moveIndex?: number): ModifierActionResult {
  const phase = getCurrentSelectModifierPhase();
  if (!phase) {
    return { success: false, error: "Not in SelectModifierPhase" };
  }

  const typeOptions = phase.getTypeOptions();
  if (index < 0 || index >= typeOptions.length) {
    return { success: false, error: `Invalid reward index: ${index}. Available: 0-${typeOptions.length - 1}` };
  }

  const modifierType = typeOptions[index].type;
  const targetKind = getTargetKind(modifierType);

  if (targetKind === "none") {
    const callback = phase.getModifierSelectCallback();
    if (!callback) {
      return { success: false, error: "Modifier select callback not available" };
    }
    callback(1, index);
    return { success: true };
  }

  return applyPokemonModifier(phase, modifierType, targetKind, -1, pokemonIndex, moveIndex);
}

/**
 * Select a shop modifier by index (purchasable items).
 * Checks money before purchase. For pokemon-targeting modifiers, provide pokemonIndex.
 */
export function selectShopModifier(index: number, pokemonIndex?: number, moveIndex?: number): ModifierActionResult {
  const phase = getCurrentSelectModifierPhase();
  if (!phase) {
    return { success: false, error: "Not in SelectModifierPhase" };
  }

  const shopOptions = getPlayerShopModifierTypeOptionsForWave(
    globalScene.currentBattle.waveIndex,
    globalScene.getWaveMoneyAmount(1),
  );

  if (index < 0 || index >= shopOptions.length) {
    return { success: false, error: `Invalid shop index: ${index}. Available: 0-${shopOptions.length - 1}` };
  }

  const shopOption = shopOptions[index];
  const modifierType = shopOption.type;

  const healingItemCost = new NumberHolder(shopOption.cost);
  globalScene.applyModifier(HealShopCostModifier, true, healingItemCost);
  const cost = healingItemCost.value;

  if (globalScene.money < cost) {
    return { success: false, error: `Insufficient money. Need ${cost}, have ${globalScene.money}` };
  }

  const targetKind = getTargetKind(modifierType);

  if (targetKind === "none") {
    const callback = phase.getModifierSelectCallback();
    if (!callback) {
      return { success: false, error: "Modifier select callback not available" };
    }
    // The phase's shop callback resolves (rowCursor, cursor) as:
    //   shopOptions[rowCursor > 2 || length <= ROW_LIMIT ? cursor : cursor + ROW_LIMIT]
    // i.e. rowCursor 3 = FIRST row (indices 0..LIMIT-1), rowCursor 2 = LAST row
    // (indices LIMIT..) when two rows exist, and rowCursor 2 = the only row
    // otherwise. The old mapping had the rows swapped, so with a two-row shop
    // (> SHOP_OPTIONS_ROW_LIMIT items, mid-game onward) "buy item i" silently
    // purchased the item i±LIMIT instead.
    const twoRows = shopOptions.length > SHOP_OPTIONS_ROW_LIMIT;
    const onFirstRow = index < SHOP_OPTIONS_ROW_LIMIT;
    const rowCursor = twoRows && onFirstRow ? 3 : 2;
    const cursor = onFirstRow ? index : index - SHOP_OPTIONS_ROW_LIMIT;
    callback(rowCursor, cursor);
    return { success: true };
  }

  return applyPokemonModifier(phase, modifierType, targetKind, cost, pokemonIndex, moveIndex);
}

/** Skip modifier selection entirely, ending the phase. */
export function skipModifiers(): ModifierActionResult {
  const phase = getCurrentSelectModifierPhase();
  if (!phase) {
    return { success: false, error: "Not in SelectModifierPhase" };
  }
  phase.skipPhase();
  return { success: true };
}

/** Reroll the available modifiers if money allows. */
export function rerollModifiers(): ModifierActionResult {
  const phase = getCurrentSelectModifierPhase();
  if (!phase) {
    return { success: false, error: "Not in SelectModifierPhase" };
  }

  const rerollCost = phase.getRerollCost(globalScene.lockModifierTiers);
  if (rerollCost < 0) {
    return { success: false, error: "Reroll is disabled for this phase" };
  }
  if (globalScene.money < rerollCost) {
    return { success: false, error: `Insufficient money for reroll. Need ${rerollCost}, have ${globalScene.money}` };
  }

  const callback = phase.getModifierSelectCallback();
  if (!callback) {
    return { success: false, error: "Modifier select callback not available" };
  }
  callback(0, 0);
  return { success: true };
}

/**
 * Check which party Pokemon are eligible to receive a modifier.
 * Uses the modifier type's selectFilter to determine eligibility.
 * Returns a boolean array of length 6 (one per party slot).
 */
export function getEligiblePokemon(modifierType: ModifierType): boolean[] {
  const party = globalScene.getPlayerParty();
  const eligible = new Array<boolean>(6).fill(false);
  if (!(modifierType instanceof PokemonModifierType)) {
    return eligible;
  }
  for (let i = 0; i < party.length; i++) {
    if (modifierType.selectFilter) {
      eligible[i] = modifierType.selectFilter(party[i]) === null;
    } else {
      eligible[i] = true;
    }
  }
  return eligible;
}

/**
 * Internal helper: apply a pokemon-targeting modifier directly,
 * bypassing the party menu UI.
 */
function applyPokemonModifier(
  phase: SelectModifierPhase,
  modifierType: ModifierType,
  targetKind: ModifierTargetKind,
  cost: number,
  pokemonIndex?: number,
  moveIndex?: number,
): ModifierActionResult {
  if (targetKind === "pokemon_pair") {
    return { success: false, error: "Fusion modifiers not yet supported in RL API" };
  }

  const party = globalScene.getPlayerParty();
  const pIdx = pokemonIndex ?? 0;

  if (pIdx < 0 || pIdx >= party.length) {
    return { success: false, error: `Invalid pokemonIndex: ${pIdx}. Party size: ${party.length}` };
  }

  const pokemon = party[pIdx];

  if (modifierType instanceof PokemonModifierType && modifierType.selectFilter) {
    const filterResult = modifierType.selectFilter(pokemon);
    if (filterResult !== null) {
      return { success: false, error: `Pokemon not eligible: ${filterResult}` };
    }
  }

  let modifier: Modifier | null;
  if (targetKind === "move") {
    // Move-targeting modifiers (e.g. Memory Mushroom / RememberMoveModifierType)
    // index into the pokemon's learnable-move list. The RL action space has no
    // move sub-selection, so default to the first learnable move — passing
    // undefined would build a modifier with moveId=undefined and crash
    // LearnMovePhase.start (allMoves[undefined].id).
    modifier = modifierType.newModifier(pokemon, moveIndex ?? 0);
  } else {
    modifier = modifierType.newModifier(pokemon);
  }

  if (!modifier) {
    return { success: false, error: "Failed to create modifier instance" };
  }

  phase.applyModifierDirectly(modifier, cost);

  // Update the pokemon's battle info (HP bar, status, etc.) to reflect the modifier's effect.
  // In headless mode this may be a no-op if battleInfo is mocked.
  try {
    pokemon.updateInfo(true);
  } catch {
    /* ignore in headless mode */
  }

  return { success: true };
}
