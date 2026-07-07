/**
 * Human-readable labels for the 58-action space, shared by the headless CLI
 * (cli.ts, JSON-lines protocol) and the rendered browser bridge
 * (browser-bridge.ts, WebSocket) so both transports describe actions
 * identically. Labels are display-only — never parsed by agents.
 *
 * NOTE: imports live game modules (globalScene). The headless runner must
 * import this dynamically AFTER initHeadless() installs the jsdom globals
 * (same rule as state-builder.ts); the browser imports it statically.
 */

import { globalScene } from "#app/global-scene";
import { MoveTarget } from "#enums/move-target";
import { getAvailableModifiers } from "#rl/modifier-api";
import type { PhaseState } from "#rl/phase-router";
import { DecisionPhase } from "#rl/phase-router";
import {
  ACTION_BALL_START,
  ACTION_BUY_SHOP_START,
  ACTION_FIGHT_ALLY_START,
  ACTION_FIGHT_ENEMY_START,
  ACTION_FIGHT_ENEMY2_START,
  ACTION_PARTY_TARGET_START,
  ACTION_REROLL,
  ACTION_RUN,
  ACTION_SELECT_REWARD_START,
  ACTION_SKIP,
  ACTION_SWITCH_START,
  ACTION_TERA_ALLY_START,
  ACTION_TERA_ENEMY_START,
  ACTION_TERA_ENEMY2_START,
  MAX_MOVES,
} from "#rl/spaces";

export interface ActionInfo {
  index: number;
  label: string;
}

const BALL_NAMES = ["Poké Ball", "Great Ball", "Ultra Ball", "Rogue Ball", "Master Ball"];
const TIER_NAMES = ["COMMON", "GREAT", "ULTRA", "ROGUE", "MASTER", "LUXURY"];

function getMoveName(moveIndex: number): string {
  try {
    const phase = globalScene.phaseManager.getCurrentPhase();
    if (phase?.is("CommandPhase")) {
      const pokemon = (
        phase as unknown as {
          getPokemon(): {
            getMoveset(
              hide: boolean,
            ): Array<{ getMove(): { name: string; power: number }; getMovePp(): number; ppUsed: number }>;
          };
        }
      ).getPokemon();
      const moveset = pokemon.getMoveset(false);
      if (moveIndex < moveset.length) {
        const move = moveset[moveIndex].getMove();
        const ppLeft = moveset[moveIndex].getMovePp() - moveset[moveIndex].ppUsed;
        const ppMax = moveset[moveIndex].getMovePp();
        return `${move.name} (${ppLeft}/${ppMax} PP, pow:${move.power || "-"})`;
      }
    }
  } catch {
    /* ignore */
  }
  return `Move ${moveIndex}`;
}

function getPartyName(slot: number): string {
  try {
    const party = globalScene.getPlayerParty();
    if (slot < party.length) {
      const p = party[slot];
      const hpPct = Math.round((p.hp / p.getMaxHp()) * 100);
      return `${p.species?.name ?? "?"} Lv${p.level} (${hpPct}% HP)`;
    }
  } catch {
    /* ignore */
  }
  return `Slot ${slot}`;
}

function getEnemyName(slot: 0 | 1): string {
  try {
    // Access by slot index directly — do NOT filter, as filtering loses slot position
    const enemyField = globalScene.getEnemyField() ?? [];
    const enemy = enemyField[slot];
    if (enemy?.isActive()) {
      return enemy.species?.name ?? (slot === 0 ? "Enemy" : "Enemy 2");
    }
  } catch {
    /* ignore */
  }
  return slot === 0 ? "Enemy" : "Enemy 2";
}

function getAllyName(): string {
  try {
    // Access by slot index directly — do NOT filter. The ally is the OTHER
    // player slot relative to the acting pokemon (executeCommandAction
    // targets PLAYER_2 when commanding slot 0 and PLAYER when slot 1).
    const playerField = globalScene.getPlayerField() ?? [];
    const phase = globalScene.phaseManager?.getCurrentPhase();
    const fieldIndex = phase?.is("CommandPhase")
      ? (phase as unknown as { getFieldIndex(): number }).getFieldIndex()
      : 0;
    const ally = playerField[fieldIndex === 0 ? 1 : 0];
    if (ally?.isActive()) {
      return ally.species?.name ?? "Ally";
    }
  } catch {
    /* ignore */
  }
  return "Ally";
}

/** Get the move's target suffix for labels (e.g., "→ Rattata", "(all enemies)", "(self)") */
function getMoveTargetLabel(moveIndex: number, enemySlot: 0 | 1): string {
  try {
    const phase = globalScene.phaseManager.getCurrentPhase();
    if (phase?.is("CommandPhase")) {
      const pokemon = (phase as unknown as { getPokemon(): any }).getPokemon();
      const moveset = pokemon.getMoveset(false);
      if (moveIndex < moveset.length) {
        const mt = moveset[moveIndex].getMove().moveTarget;
        switch (mt) {
          case MoveTarget.USER:
          case MoveTarget.USER_AND_ALLIES:
          case MoveTarget.PARTY:
            return "(self)";
          case MoveTarget.ALL_NEAR_OTHERS:
            return "(all nearby)";
          case MoveTarget.ALL_NEAR_ENEMIES:
          case MoveTarget.ALL_ENEMIES:
          case MoveTarget.ALL_OTHERS:
            return "(all enemies)";
          case MoveTarget.ALL:
          case MoveTarget.BOTH_SIDES:
            return "(field)";
          case MoveTarget.USER_SIDE:
            return "(team)";
          case MoveTarget.ENEMY_SIDE:
            return "(enemy side)";
          case MoveTarget.RANDOM_NEAR_ENEMY:
            return "(random enemy)";
          case MoveTarget.ATTACKER:
            return "(counter)";
          case MoveTarget.CURSE:
            return "(curse)";
          default:
            return `→ ${getEnemyName(enemySlot)}`;
        }
      }
    }
  } catch {
    /* ignore */
  }
  return `→ ${getEnemyName(enemySlot)}`;
}

/**
 * Phase-specific label function: for non-command phases where an action index
 * carries phase-specific meaning (e.g. action 0 = "accept switch"), returns a
 * label, or null to fall through to the generic labeler.
 */
function getPhaseSpecificLabel(idx: number, phase: string, metadata: Record<string, unknown> = {}): string | null {
  switch (phase) {
    case DecisionPhase.SELECT_GENDER:
      return "Continue";
    case DecisionPhase.TITLE:
      return "Start Game";
    case DecisionPhase.EVOLUTION:
    case DecisionPhase.FORM_CHANGE:
      return "Continue";
    case DecisionPhase.CHECK_SWITCH:
      if (idx === 0) {
        return "Accept switch";
      }
      if (idx === ACTION_SKIP) {
        return "Decline switch";
      }
      return null;
    case DecisionPhase.SWITCH:
      if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
        return `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
      }
      return null;
    case DecisionPhase.LEARN_MOVE: {
      const newMove = metadata.newMoveName as string | undefined;
      const currentMoves = metadata.currentMoveNames as string[] | undefined;
      if (idx === ACTION_SKIP) {
        return `Don't learn ${newMove ?? "move"}`;
      }
      if (idx >= 0 && idx < MAX_MOVES) {
        const current = currentMoves?.[idx] ?? `slot ${idx}`;
        return `Replace ${current} with ${newMove ?? "new move"}`;
      }
      return null;
    }
    case DecisionPhase.GAME_OVER:
      return idx === 0 ? "Continue (retry)" : "Quit";
    case DecisionPhase.REVIVAL_BLESSING:
      if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
        return `Revive: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
      }
      return null;
    case DecisionPhase.MODIFIER_TARGET:
      if (idx === ACTION_SKIP) {
        return "Cancel (back to items)";
      }
      if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
        return `Apply to: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
      }
      return null;
    case DecisionPhase.SELECT_BIOME: {
      const biomeNames = metadata.biomeNames as string[] | undefined;
      if (biomeNames && idx < biomeNames.length) {
        return `Go to: ${biomeNames[idx]}`;
      }
      return `Go to: Biome ${idx}`;
    }
    case DecisionPhase.MYSTERY_ENCOUNTER:
      return `Encounter option ${idx}`;
    default:
      return null; // Fall through to generic labels
  }
}

/**
 * Build human-readable action labels for every valid action in the state.
 * Works in both transports; requires live game access (post-init).
 */
export function buildActionLabels(state: PhaseState): ActionInfo[] {
  const actions: ActionInfo[] = [];

  for (const idx of state.validActions) {
    // Try phase-specific label first
    const phaseLabel = getPhaseSpecificLabel(idx, state.phase, state.metadata);
    if (phaseLabel !== null) {
      actions.push({ index: idx, label: phaseLabel });
      continue;
    }

    let label = `Action ${idx}`;

    // Fight → Enemy (0-3) — default slot (may be single-target, multi-target, or self-target)
    if (idx >= ACTION_FIGHT_ENEMY_START && idx < ACTION_FIGHT_ENEMY_START + MAX_MOVES) {
      const mi = idx - ACTION_FIGHT_ENEMY_START;
      label = `Fight: ${getMoveName(mi)} ${getMoveTargetLabel(mi, 0)}`;
    }
    // Fight → Enemy 2 (4-7) — always single-target moves
    else if (idx >= ACTION_FIGHT_ENEMY2_START && idx < ACTION_FIGHT_ENEMY2_START + MAX_MOVES) {
      label = `Fight: ${getMoveName(idx - ACTION_FIGHT_ENEMY2_START)} → ${getEnemyName(1)}`;
    }
    // Fight → Ally (8-11)
    else if (idx >= ACTION_FIGHT_ALLY_START && idx < ACTION_FIGHT_ALLY_START + MAX_MOVES) {
      label = `Fight: ${getMoveName(idx - ACTION_FIGHT_ALLY_START)} → ${getAllyName()}`;
    }
    // Switch (12-16)
    else if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
      label = `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
    }
    // Ball (17-21)
    else if (idx >= ACTION_BALL_START && idx < ACTION_BALL_START + 5) {
      label = `Throw: ${BALL_NAMES[idx - ACTION_BALL_START] ?? "Ball"}`;
    }
    // Run (22)
    else if (idx === ACTION_RUN) {
      label = "Run away";
    }
    // Tera → Enemy (23-26) — default slot
    else if (idx >= ACTION_TERA_ENEMY_START && idx < ACTION_TERA_ENEMY_START + MAX_MOVES) {
      const mi = idx - ACTION_TERA_ENEMY_START;
      label = `Tera + ${getMoveName(mi)} ${getMoveTargetLabel(mi, 0)}`;
    }
    // Tera → Enemy 2 (27-30)
    else if (idx >= ACTION_TERA_ENEMY2_START && idx < ACTION_TERA_ENEMY2_START + MAX_MOVES) {
      label = `Tera + ${getMoveName(idx - ACTION_TERA_ENEMY2_START)} → ${getEnemyName(1)}`;
    }
    // Tera → Ally (31-34)
    else if (idx >= ACTION_TERA_ALLY_START && idx < ACTION_TERA_ALLY_START + MAX_MOVES) {
      label = `Tera + ${getMoveName(idx - ACTION_TERA_ALLY_START)} → ${getAllyName()}`;
    }
    // Select reward (35-37)
    else if (idx >= ACTION_SELECT_REWARD_START && idx < ACTION_SELECT_REWARD_START + 3) {
      const ri = idx - ACTION_SELECT_REWARD_START;
      try {
        const mods = getAvailableModifiers();
        if (mods && ri < mods.rewards.length) {
          label = `Select reward ${ri}: ${mods.rewards[ri].name} [${TIER_NAMES[mods.rewards[ri].tier] ?? "?"}]`;
        } else {
          label = `Select reward ${ri}`;
        }
      } catch {
        label = `Select reward ${ri}`;
      }
    }
    // Reroll (38)
    else if (idx === ACTION_REROLL) {
      try {
        const mods = getAvailableModifiers();
        label = mods ? `Reroll modifiers (cost: $${mods.rerollCost})` : "Reroll modifiers";
      } catch {
        label = "Reroll modifiers";
      }
    }
    // Skip (39)
    else if (idx === ACTION_SKIP) {
      label = "Skip / Decline";
    }
    // Buy shop (40-51)
    else if (idx >= ACTION_BUY_SHOP_START && idx < ACTION_BUY_SHOP_START + 12) {
      const si = idx - ACTION_BUY_SHOP_START;
      try {
        const mods = getAvailableModifiers();
        if (mods && si < mods.shop.length) {
          label = `Buy: ${mods.shop[si].name} ($${mods.shop[si].cost})`;
        } else {
          label = `Buy shop item ${si}`;
        }
      } catch {
        label = `Buy shop item ${si}`;
      }
    }
    // Party target (52-57)
    else if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
      label = `Apply to: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
    }

    actions.push({ index: idx, label });
  }

  return actions;
}
