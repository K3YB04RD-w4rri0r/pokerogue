/**
 * Verification test: Ensures the standalone runner can initialize and run
 * a 3-wave battle without any direct Vitest (`vi.*`) dependencies.
 *
 * This test runs within Vitest's infrastructure (for module transforms),
 * but the GameManager, GameWrapper, and all helpers use only the standalone
 * mock utilities from `src/rl/mocks/spy.ts` — not `vi.fn()` or `vi.spyOn()`.
 */

import { MoveId } from "#enums/move-id";
import { SpeciesId } from "#enums/species-id";
import { GameManager } from "#test/test-utils/game-manager";
import Phaser from "phaser";
import { beforeAll, beforeEach, describe, it } from "vitest";

describe("Standalone Runner", () => {
  let phaserGame: Phaser.Game;
  let game: GameManager;

  beforeAll(() => {
    phaserGame = new Phaser.Game({ type: Phaser.HEADLESS });
  });

  beforeEach(() => {
    game = new GameManager(phaserGame);
  });

  it("runs a 3-wave battle without vi.* utilities", async () => {
    game.override
      .enemySpecies(SpeciesId.RATTATA)
      .startingLevel(2000)
      .battleStyle("single")
      .startingWave(1)
      .moveset(MoveId.TACKLE);

    await game.classicMode.startBattle([SpeciesId.MEWTWO]);

    // Wave 1
    game.move.select(MoveId.TACKLE);
    await game.toNextWave();

    // Wave 2
    game.move.select(MoveId.TACKLE);
    await game.toNextWave();

    // Wave 3
    game.move.select(MoveId.TACKLE);
    await game.toNextWave();

    console.log(`[standalone-runner] 3-wave battle completed! Wave: ${game.scene.currentBattle.waveIndex}`);
  }, 30_000);
});
