/**
 * CLI entry point for the headless RL runner.
 *
 * Boots the game in headless mode, runs a simple random-action agent loop,
 * and prints a summary. This is a minimal integration test for the build --
 * Task #18 provides a more comprehensive dummy agent.
 *
 * Usage:
 *   node dist/rl/cli.js [--seed=123] [--waves=10] [--log]
 *   node dist/rl/cli.js --interactive [--seed=123] [--waves=50]
 *
 * Options:
 *   --seed=<string>       RNG seed for deterministic replay (default: "rl-cli")
 *   --waves=<number>      Maximum waves to run before stopping (default: 5)
 *   --log                 Enable verbose logging of phase decisions
 *   --interactive         JSON-line protocol for external control (Python bridge)
 *   --help                Show usage information
 */

// IMPORTANT: Only headless-boot can be statically imported here.
// All other game/RL imports must be dynamic (after initHeadless installs jsdom globals)
// because they transitively import Phaser, which accesses `window` at load time.
import { initHeadless, destroyHeadless } from "#rl/headless-boot";

// These types are safe to import (erased at runtime by TypeScript)
import type { PhaseRouter, PhaseState } from "#rl/phase-router";

// ─── Argument Parsing ──────────────────────────────────────────────

interface CliOptions {
  seed: string;
  maxWaves: number;
  verbose: boolean;
  interactive: boolean;
}

function parseArgs(): CliOptions {
  const args = process.argv.slice(2);
  const options: CliOptions = {
    seed: "rl-cli",
    maxWaves: 5,
    verbose: false,
    interactive: false,
  };

  for (const arg of args) {
    if (arg === "--help" || arg === "-h") {
      console.log(`
PokeRogue Headless RL Runner

Usage: node dist/rl/cli.js [options]

Options:
  --seed=<string>       RNG seed for deterministic replay (default: "rl-cli")
  --waves=<number>      Maximum waves before stopping (default: 5)
  --log                 Enable verbose phase decision logging
  --interactive         JSON-line protocol for external control (Python bridge)
  --help                Show this help message
`);
      process.exit(0);
    }

    if (arg.startsWith("--seed=")) {
      options.seed = arg.split("=")[1];
    } else if (arg.startsWith("--waves=")) {
      const n = Number.parseInt(arg.split("=")[1], 10);
      if (!Number.isNaN(n) && n > 0) options.maxWaves = n;
    } else if (arg === "--log") {
      options.verbose = true;
    } else if (arg === "--interactive") {
      options.interactive = true;
    }
  }

  return options;
}

// ─── JSON Protocol (interactive mode) ───────────────────────────────

/** Write a JSON message to stdout (protocol channel). */
function sendJson(obj: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

/**
 * Buffered line reader for stdin.
 * Queues all incoming lines so they're not lost while the game processes.
 */
class LineReader {
  private queue: string[] = [];
  private closed = false;
  private waiting: ((line: string | null) => void) | null = null;

  constructor(rl: import("readline").Interface) {
    rl.on("line", (line: string) => {
      if (this.waiting) {
        const resolve = this.waiting;
        this.waiting = null;
        resolve(line);
      } else {
        this.queue.push(line);
      }
    });
    rl.on("close", () => {
      this.closed = true;
      if (this.waiting) {
        const resolve = this.waiting;
        this.waiting = null;
        resolve(null);
      }
    });
  }

  /** Read the next line. Returns null on EOF. */
  next(): Promise<string | null> {
    if (this.queue.length > 0) {
      return Promise.resolve(this.queue.shift()!);
    }
    if (this.closed) {
      return Promise.resolve(null);
    }
    return new Promise(resolve => {
      this.waiting = resolve;
    });
  }
}

/** Read an action from stdin via JSON. Returns -1 on EOF. */
async function readAction(reader: LineReader): Promise<number> {
  const line = await reader.next();
  if (line === null) return -1; // EOF
  try {
    const msg = JSON.parse(line);
    return typeof msg.action === "number" ? msg.action : 0;
  } catch {
    return 0;
  }
}

// ─── Game State Helpers ─────────────────────────────────────────────

interface ActionInfo {
  index: number;
  label: string;
}

/**
 * Build human-readable action labels for the current state.
 * Dynamically imports game modules to access live state.
 */
async function buildActionLabels(state: PhaseState): Promise<ActionInfo[]> {
  const { globalScene } = await import("#app/global-scene");
  const {
    ACTION_FIGHT_ENEMY_START,
    ACTION_FIGHT_ENEMY2_START,
    ACTION_FIGHT_ALLY_START,
    ACTION_SWITCH_START,
    ACTION_BALL_START,
    ACTION_RUN,
    ACTION_TERA_ENEMY_START,
    ACTION_TERA_ENEMY2_START,
    ACTION_TERA_ALLY_START,
    ACTION_SELECT_REWARD_START,
    ACTION_REROLL,
    ACTION_SKIP,
    ACTION_BUY_SHOP_START,
    ACTION_PARTY_TARGET_START,
    MAX_MOVES,
  } = await import("#rl/spaces");

  const { MoveTarget } = await import("#enums/move-target");
  const actions: ActionInfo[] = [];

  function getMoveName(moveIndex: number): string {
    try {
      const phase = globalScene.phaseManager.getCurrentPhase();
      if (phase?.is("CommandPhase")) {
        const pokemon = (phase as unknown as { getPokemon(): { getMoveset(hide: boolean): Array<{ getMove(): { name: string; power: number }; getMovePp(): number; ppUsed: number }> } }).getPokemon();
        const moveset = pokemon.getMoveset(false);
        if (moveIndex < moveset.length) {
          const move = moveset[moveIndex].getMove();
          const ppLeft = moveset[moveIndex].getMovePp() - moveset[moveIndex].ppUsed;
          const ppMax = moveset[moveIndex].getMovePp();
          return `${move.name} (${ppLeft}/${ppMax} PP, pow:${move.power || "-"})`;
        }
      }
    } catch { /* ignore */ }
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
    } catch { /* ignore */ }
    return `Slot ${slot}`;
  }

  const ballNames = ["Poké Ball", "Great Ball", "Ultra Ball", "Rogue Ball", "Master Ball"];

  function getEnemyName(slot: 0 | 1): string {
    try {
      // Access by slot index directly — do NOT filter, as filtering loses slot position
      const enemyField = globalScene.getEnemyField() ?? [];
      const enemy = enemyField[slot];
      if (enemy?.isActive()) {
        return enemy.species?.name ?? (slot === 0 ? "Enemy" : "Enemy 2");
      }
    } catch { /* ignore */ }
    return slot === 0 ? "Enemy" : "Enemy 2";
  }

  function getAllyName(): string {
    try {
      // Access by slot index directly — do NOT filter
      const playerField = globalScene.getPlayerField() ?? [];
      if (playerField.length > 1 && playerField[1]?.isActive()) {
        return playerField[1].species?.name ?? "Ally";
      }
    } catch { /* ignore */ }
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
    } catch { /* ignore */ }
    return `→ ${getEnemyName(enemySlot)}`;
  }

  // Phase-specific label function (handles non-command phases where action 0
  // doesn't mean "fight move 0")
  function getPhaseSpecificLabel(idx: number, phase: string, metadata: Record<string, unknown> = {}): string | null {
    switch (phase) {
      case "select_gender":
        return "Continue";
      case "title":
        return "Start Game";
      case "check_switch":
        if (idx === 0) return "Accept switch";
        if (idx === ACTION_SKIP) return "Decline switch";
        return null;
      case "switch":
        if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
          return `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
        }
        return null;
      case "learn_move": {
        const newMove = metadata.newMoveName as string | undefined;
        const currentMoves = metadata.currentMoveNames as string[] | undefined;
        if (idx === ACTION_SKIP) return `Don't learn ${newMove ?? "move"}`;
        if (idx >= 0 && idx < MAX_MOVES) {
          const current = currentMoves?.[idx] ?? `slot ${idx}`;
          return `Replace ${current} with ${newMove ?? "new move"}`;
        }
        return null;
      }
      case "game_over":
        return idx === 0 ? "Continue" : "Quit";
      case "modifier_target":
        if (idx === ACTION_SKIP) return "Cancel (back to items)";
        if (idx >= ACTION_PARTY_TARGET_START && idx < ACTION_PARTY_TARGET_START + 6) {
          return `Apply to: ${getPartyName(idx - ACTION_PARTY_TARGET_START)}`;
        }
        return null;
      case "select_biome": {
        const biomeNames = metadata.biomeNames as string[] | undefined;
        if (biomeNames && idx < biomeNames.length) {
          return `Go to: ${biomeNames[idx]}`;
        }
        return `Go to: Biome ${idx}`;
      }
      default:
        return null; // Fall through to generic labels
    }
  }

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
      label = `Throw: ${ballNames[idx - ACTION_BALL_START] ?? "Ball"}`;
    }
    // Run (22)
    else if (idx === ACTION_RUN) {
      label = "Run away";
    }
    // Tera → Enemy (23-26) — default slot (may be single-target, multi-target, or self-target)
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
        const { getAvailableModifiers } = await import("#rl/modifier-api");
        const mods = getAvailableModifiers();
        if (mods && ri < mods.rewards.length) {
          const tierNames = ["COMMON", "GREAT", "ULTRA", "ROGUE", "MASTER", "LUXURY"];
          label = `Select reward ${ri}: ${mods.rewards[ri].name} [${tierNames[mods.rewards[ri].tier]}]`;
        } else {
          label = `Select reward ${ri}`;
        }
      } catch {
        label = `Select reward ${ri}`;
      }
    }
    // Reroll (38)
    else if (idx === ACTION_REROLL) {
      label = "Reroll modifiers";
    }
    // Skip (39)
    else if (idx === ACTION_SKIP) {
      label = "Skip / Decline";
    }
    // Buy shop (40-51)
    else if (idx >= ACTION_BUY_SHOP_START && idx < ACTION_BUY_SHOP_START + 12) {
      const si = idx - ACTION_BUY_SHOP_START;
      try {
        const { getAvailableModifiers } = await import("#rl/modifier-api");
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

// buildGameState is now imported from state-builder.ts after headless init
// (see dynamic import in runInteractiveEpisode and runEpisode)

// ─── Agent Loop ────────────────────────────────────────────────────

interface EpisodeStats {
  totalSteps: number;
  wavesCleared: number;
  decisionsPerPhase: Record<string, number>;
  startTime: number;
  endTime: number;
}

/**
 * Run a single episode: reset the game, then step through decisions
 * using random valid actions until game over or max waves reached.
 */
async function runEpisode(
  router: PhaseRouter,
  options: CliOptions,
  pickDefaultAction: (state: PhaseState) => number,
  DecisionPhase: Record<string, string>,
): Promise<EpisodeStats> {
  const stats: EpisodeStats = {
    totalSteps: 0,
    wavesCleared: 0,
    decisionsPerPhase: {},
    startTime: Date.now(),
    endTime: 0,
  };

  const MAX_STEPS = options.maxWaves * 50; // Safety limit: ~50 decisions per wave

  try {
    while (stats.totalSteps < MAX_STEPS) {
      // Wait for the next decision point
      let state: PhaseState;
      try {
        state = await router.advanceToNextDecision();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("Timeout")) {
          console.log(`[cli] Timeout waiting for decision at step ${stats.totalSteps}. Ending episode.`);
          break;
        }
        throw err;
      }

      // Log any info messages from auto-skipped phases (e.g., IV Scanner)
      for (const msg of router.drainInfoMessages()) {
        console.log(`[info] ${msg}`);
      }

      // Track decision statistics
      const phaseName = state.phase;
      stats.decisionsPerPhase[phaseName] = (stats.decisionsPerPhase[phaseName] ?? 0) + 1;

      // Check for game over (arrives as a 'title' phase after GameOverPhase → TitlePhase)
      if (state.phase === DecisionPhase.GAME_OVER || router.isGameOver()) {
        if (options.verbose) {
          console.log(`[cli] Game over at step ${stats.totalSteps}. Victory: ${router.isVictory()}`);
        }
        break;
      }

      // Check max waves
      try {
        const { globalScene: gs } = await import("#app/global-scene");
        if (gs?.currentBattle?.waveIndex > options.maxWaves) {
          if (options.verbose) {
            console.log(`[cli] Reached max waves (${options.maxWaves}). Ending episode.`);
          }
          break;
        }
      } catch { /* ignore */ }

      // Select an action: use pickDefaultAction for a simple baseline agent
      const action = pickDefaultAction(state);

      if (options.verbose) {
        console.log(
          `[cli] Step ${stats.totalSteps}: phase=${phaseName}, ` +
          `validActions=${state.validActions.length}, chosen=${action}` +
          (state.metadata.pokemonSpecies ? `, pokemon=${state.metadata.pokemonSpecies}` : "")
        );
      }

      // Execute the action
      await router.executeAction(action);
      stats.totalSteps++;

      // Track wave progress from globalScene (imported dynamically to avoid
      // loading before headless init)
      try {
        const { globalScene } = await import("#app/global-scene");
        const currentWave = globalScene?.currentBattle?.waveIndex ?? 0;
        if (currentWave > stats.wavesCleared) {
          stats.wavesCleared = currentWave;
          if (options.verbose) {
            console.log(`[cli] Reached wave ${currentWave}`);
          }
        }
      } catch {
        // globalScene may not be available yet during setup phases
      }
    }
  } catch (err) {
    console.error("[cli] Error during episode:", err);
  }

  stats.endTime = Date.now();
  return stats;
}

/**
 * Run interactive episode: JSON-line protocol over stdin/stdout.
 * Game state is written to stdout, actions are read from stdin.
 */
async function runInteractiveEpisode(
  router: PhaseRouter,
  options: CliOptions,
  DecisionPhase: Record<string, string>,
  reader: LineReader,
): Promise<void> {
  const { buildGameState } = await import("#rl/state-builder");
  const MAX_STEPS = options.maxWaves * 50;
  let step = 0;

  try {
    while (step < MAX_STEPS) {
      let state: PhaseState;
      try {
        state = await router.advanceToNextDecision();
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("Timeout")) {
          sendJson({ type: "error", message: `Timeout at step ${step}` });
          break;
        }
        throw err;
      }

      // Send any info messages from auto-skipped phases (e.g., IV Scanner)
      for (const msg of router.drainInfoMessages()) {
        sendJson({ type: "info", message: msg });
      }

      // Check for game over (arrives as a 'title' phase after GameOverPhase → TitlePhase)
      if (state.phase === DecisionPhase.GAME_OVER || router.isGameOver()) {
        const gameState = buildGameState(state, step);
        sendJson({
          type: "game_over",
          step,
          victory: router.isVictory(),
          gameState,
        });
        break;
      }

      // Build action labels and game state
      const actions = await buildActionLabels(state);
      const gameState = buildGameState(state, step);

      // Send state to Python
      sendJson({
        type: "state",
        step,
        phase: state.phase,
        actions,
        gameState,
        metadata: state.metadata,
      });

      // Read action from Python
      const action = await readAction(reader);

      // EOF: stdin closed
      if (action === -1) {
        sendJson({ type: "error", message: "stdin closed (EOF)" });
        break;
      }

      // Validate and execute
      if (!state.actionMask[action]) {
        sendJson({
          type: "warning",
          message: `Invalid action ${action}, falling back to first valid action`,
        });
        await router.executeAction(state.validActions[0] ?? 0);
      } else {
        await router.executeAction(action);
      }

      step++;
    }
  } catch (err) {
    sendJson({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }

  sendJson({ type: "done", steps: step });
}

// ─── Main ──────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const options = parseArgs();

  // In interactive mode: set up stdin reader early (before boot) so piped
  // input is buffered during the ~2s boot, and redirect console to stderr.
  let stdinReader: LineReader | null = null;
  if (options.interactive) {
    const readline = await import("readline");
    const rl = readline.createInterface({ input: process.stdin, terminal: false });
    stdinReader = new LineReader(rl);

    const toStderr = (...args: unknown[]) => {
      process.stderr.write(args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" ") + "\n");
    };
    console.log = toStderr;
    console.warn = toStderr;
    console.info = toStderr;
    console.debug = toStderr;
    // console.error already goes to stderr
  }

  if (!options.interactive) {
    console.log("=== PokeRogue Headless RL Runner ===");
    console.log(`  Seed:      ${options.seed}`);
    console.log(`  Max waves: ${options.maxWaves}`);
    console.log(`  Verbose:   ${options.verbose}`);
    console.log();
  }

  // Phase 1: Boot the game in headless mode
  if (!options.interactive) {
    console.log("[cli] Initializing headless game...");
  }
  const bootStart = Date.now();

  try {
    await initHeadless({ seed: options.seed });
  } catch (err) {
    if (options.interactive) {
      sendJson({ type: "error", message: `Boot failed: ${err}` });
    } else {
      console.error("[cli] Failed to initialize headless game:", err);
    }
    process.exit(1);
  }

  const bootTime = Date.now() - bootStart;
  if (!options.interactive) {
    console.log(`[cli] Headless game initialized in ${bootTime}ms`);
  }

  // Phase 2: Now that headless is initialized, dynamically import phase-router
  const { createPhaseRouter, pickDefaultAction, DecisionPhase } = await import("#rl/phase-router");

  // Phase 3: Create the phase router
  const router = createPhaseRouter({ verbose: options.verbose });

  if (options.interactive) {
    // Interactive mode: JSON protocol over stdin/stdout
    sendJson({ type: "ready", seed: options.seed, maxWaves: options.maxWaves, bootTime });
    await runInteractiveEpisode(router, options, DecisionPhase, stdinReader!);
  } else {
    // Auto mode: run with default action picker
    console.log("[cli] Starting episode...");
    const stats = await runEpisode(router, options, pickDefaultAction, DecisionPhase);

    // Print summary
    const elapsed = stats.endTime - stats.startTime;
    const stepsPerSec = elapsed > 0 ? (stats.totalSteps / (elapsed / 1000)).toFixed(1) : "N/A";

    console.log();
    console.log("=== Episode Summary ===");
    console.log(`  Total steps:    ${stats.totalSteps}`);
    console.log(`  Waves cleared:  ${stats.wavesCleared}`);
    console.log(`  Duration:       ${elapsed}ms`);
    console.log(`  Steps/sec:      ${stepsPerSec}`);
    console.log(`  Boot time:      ${bootTime}ms`);
    console.log();
    console.log("  Decisions by phase:");
    for (const [phase, count] of Object.entries(stats.decisionsPerPhase).sort(
      (a, b) => b[1] - a[1]
    )) {
      console.log(`    ${phase}: ${count}`);
    }
  }

  // Cleanup
  router.destroy();
  await destroyHeadless();
  if (!options.interactive) {
    console.log();
    console.log("[cli] Done.");
  }

  // Force exit: Phaser and game internals may leave pending timers/intervals
  // that prevent Node.js from exiting naturally.
  process.exit(0);
}

main().catch(err => {
  console.error("[cli] Fatal error:", err);
  process.exit(1);
});
