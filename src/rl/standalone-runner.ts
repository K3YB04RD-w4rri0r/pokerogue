/**
 * Standalone headless runner for PokéRogue.
 *
 * Provides an API for the RL environment to create game sessions and run battles
 * programmatically without any Vitest (`vi.*`) dependencies.
 *
 * **Module resolution**: This module uses the project's TypeScript path aliases
 * (`#app/*`, `#test/*`, etc.) which require a bundler-compatible module resolver
 * (Vite, Vitest, or equivalent). It cannot be run directly with bare Node.js/tsx.
 *
 * Usage from the RL environment:
 * ```ts
 * import { createGameManager } from "#app/rl/standalone-runner";
 * import { SpeciesId } from "#enums/species-id";
 * import { MoveId } from "#enums/move-id";
 * import Phaser from "phaser";
 *
 * // Create a Phaser game instance (once)
 * const phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
 *
 * // Create a game manager for each episode
 * const game = createGameManager(phaserGame);
 * game.override
 *   .enemySpecies(SpeciesId.RATTATA)
 *   .startingLevel(100)
 *   .battleStyle("single")
 *   .moveset(MoveId.TACKLE);
 *
 * await game.classicMode.startBattle([SpeciesId.MEWTWO]);
 * game.move.select(MoveId.TACKLE);
 * await game.toNextWave();
 * ```
 */

import { restoreAllMocks } from "#app/rl/mocks/spy";
import { GameManager } from "#test/test-utils/game-manager";
import type Phaser from "phaser";

/**
 * Create a new GameManager instance for a headless battle session.
 * Restores all mocks from the previous session before creating a new one.
 *
 * @param phaserGame - A Phaser.Game instance (use `Phaser.HEADLESS` type).
 * @param bypassLogin - Whether to bypass the login phase (default: true).
 * @returns A fully initialized GameManager ready for `startBattle()`.
 */
export function createGameManager(phaserGame: Phaser.Game, bypassLogin = true): GameManager {
  restoreAllMocks();
  return new GameManager(phaserGame, bypassLogin);
}

export { restoreAllMocks } from "#app/rl/mocks/spy";
export { GameManager } from "#test/test-utils/game-manager";
