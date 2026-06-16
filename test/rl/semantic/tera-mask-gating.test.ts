/**
 * Semantic teeth-test for the command-phase Tera action mask.
 *
 * Root-cause regression guard (rl-framework). buildCommandActionMask() used to
 * offer the Tera actions (23-34) whenever nobody on the player field was already
 * terastallized — it NEVER checked the real game gate. So the mask offered Tera
 * from wave 1 with no Tera Orb (visible in rendered play as "Tera + <move>"
 * options at the very first command). The fix mirrors the game's own
 * CommandUiHandler.canTera() EXACTLY: Tera requires a TerastallizeAccessModifier
 * (the Tera Orb, checked via canTerastallize) plus an unused arena tera.
 *
 * Teeth: the first test FAILS on the old mask (which set the Tera bits with no
 * orb); the second FAILS if the gate became too strict (no Tera with an orb).
 * Both inspect the SAME mask the live env serves (PhaseRouter.getCurrentPhaseState).
 */
import { AbilityId } from "#enums/ability-id";
import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { createPhaseRouter, DecisionPhase } from "#rl/phase-router";
import { ACTION_FIGHT_ENEMY_START, ACTION_TERA_ENEMY_START, MAX_MOVES } from "#rl/spaces";
import { GameManager } from "#test/test-utils/game-manager";
import { canSpeciesTera } from "#utils/pokemon-utils";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, expect, it } from "vitest";

describe("RL Semantic - Tera command-mask gating", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
    game.override
      .battleStyle("single")
      .moveset([MoveId.TACKLE])
      .enemySpecies(SpeciesId.SHUCKLE)
      .enemyMoveset(MoveId.SPLASH)
      .enemyAbility(AbilityId.BALL_FETCH)
      .ability(AbilityId.BALL_FETCH);
  });

  it("does NOT offer Tera at wave 1 without a Tera Orb", async () => {
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    // The game's own gate agrees: with no Tera Orb the species cannot tera.
    expect(canSpeciesTera(game.field.getPlayerPokemon())).toBe(false);

    const router = createPhaseRouter();
    try {
      const state = router.getCurrentPhaseState();
      expect(state?.phase).toBe(DecisionPhase.COMMAND);
      // The mask is real: the damaging move IS offered...
      expect(state?.actionMask[ACTION_FIGHT_ENEMY_START]).toBe(true);
      // ...but NOT a single Tera variant of it.
      for (let i = 0; i < MAX_MOVES; i++) {
        expect(state?.actionMask[ACTION_TERA_ENEMY_START + i]).toBe(false);
      }
    } finally {
      router.destroy();
    }
  });

  it("offers Tera at wave 1 once a Tera Orb is held, mirroring the fight actions", async () => {
    game.override.startingModifier([{ name: "TERA_ORB" }]);
    await game.classicMode.startBattle(SpeciesId.MAGIKARP);

    // With the orb, the game's gate now allows it.
    expect(canSpeciesTera(game.field.getPlayerPokemon())).toBe(true);

    const router = createPhaseRouter();
    try {
      const state = router.getCurrentPhaseState();
      expect(state?.phase).toBe(DecisionPhase.COMMAND);
      // Each Tera action exactly mirrors the matching fight action's availability.
      for (let i = 0; i < MAX_MOVES; i++) {
        expect(state?.actionMask[ACTION_TERA_ENEMY_START + i]).toBe(state?.actionMask[ACTION_FIGHT_ENEMY_START + i]);
      }
      // And at least one Tera action is actually offered.
      const teraBits = state?.actionMask.slice(ACTION_TERA_ENEMY_START, ACTION_TERA_ENEMY_START + MAX_MOVES);
      expect(teraBits?.some(Boolean)).toBe(true);
    } finally {
      router.destroy();
    }
  });
});
