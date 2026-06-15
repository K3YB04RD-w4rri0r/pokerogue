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

import { destroyHeadless, initHeadless, resetHeadless } from "#rl/headless-boot";
// These types are safe to import (erased at runtime by TypeScript)
import type { PhaseRouter, PhaseState } from "#rl/phase-router";
// IMPORTANT: Only headless-boot can be statically imported here.
// All other game/RL imports must be dynamic (after initHeadless installs jsdom globals)
// because they transitively import Phaser, which accesses `window` at load time.
import fs from "node:fs";

// ─── Argument Parsing ──────────────────────────────────────────────

interface CliOptions {
  seed: string;
  maxWaves: number;
  verbose: boolean;
  interactive: boolean;
  dumpObs: string | null;
  lean: boolean;
  /** Partial RewardConfig overrides parsed from --reward-config */
  rewardConfig: Record<string, number> | null;
  /** Game override values from repeated --override KEY=VALUE flags */
  overrides: Record<string, unknown> | null;
  /** Print per-step section timings to stderr at episode end */
  profile: boolean;
}

function parseArgs(): CliOptions {
  const args = process.argv.slice(2);
  const options: CliOptions = {
    seed: "rl-cli",
    maxWaves: 5,
    verbose: false,
    interactive: false,
    dumpObs: null,
    lean: false,
    rewardConfig: null,
    overrides: null,
    profile: false,
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
  --dump-obs=<path>     Write per-step JSONL records (gameState + encoded observation)
                        for the TS<->Python parity harness (tools/verify)
  --lean                Omit the full gameState from interactive state messages
                        (obsB64/mask/wave are always included; training fast path)
  --reward-config=<json|@path>
                        Partial RewardConfig overrides, e.g. '{"turnPenalty":-1}'
                        (see src/rl/rewards.ts for the field list)
  --override=KEY=VALUE  Game override (repeatable; KEY is a DefaultOverrides
                        property, VALUE is JSON or a raw string), e.g.
                        --override=STARTING_WAVE_OVERRIDE=20
  --help                Show this help message
`);
      process.exit(0);
    }

    if (arg.startsWith("--seed=")) {
      options.seed = arg.split("=")[1];
    } else if (arg.startsWith("--waves=")) {
      const n = Number.parseInt(arg.split("=")[1], 10);
      if (!Number.isNaN(n) && n > 0) {
        options.maxWaves = n;
      }
    } else if (arg === "--log") {
      options.verbose = true;
    } else if (arg === "--interactive") {
      options.interactive = true;
    } else if (arg.startsWith("--dump-obs=")) {
      options.dumpObs = arg.slice("--dump-obs=".length);
    } else if (arg === "--lean") {
      options.lean = true;
    } else if (arg.startsWith("--reward-config=")) {
      // Inline JSON ({"turnPenalty":-1}) or @path to a JSON file
      const raw = arg.slice("--reward-config=".length);
      try {
        const text = raw.startsWith("@") ? fs.readFileSync(raw.slice(1), "utf8") : raw;
        const parsed = JSON.parse(text);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          throw new Error("must be a JSON object of RewardConfig overrides");
        }
        options.rewardConfig = parsed as Record<string, number>;
      } catch (err) {
        process.stderr.write(`Invalid --reward-config: ${err}\n`);
        process.exit(1);
      }
    } else if (arg.startsWith("--override=") || arg.startsWith("--override ")) {
      // --override=KEY=VALUE (repeatable). VALUE is JSON if parseable, else a
      // raw string. KEY is a DefaultOverrides property, e.g.
      // --override=BATTLE_STYLE_OVERRIDE='"double"' --override=STARTING_WAVE_OVERRIDE=20
      const body = arg.slice("--override=".length);
      const eq = body.indexOf("=");
      if (eq <= 0) {
        process.stderr.write(`Invalid --override (expected KEY=VALUE): ${body}\n`);
        process.exit(1);
      }
      const key = body.slice(0, eq);
      const rawValue = body.slice(eq + 1);
      let value: unknown;
      try {
        value = JSON.parse(rawValue);
      } catch {
        value = rawValue; // plain string (e.g. double)
      }
      options.overrides = { ...(options.overrides ?? {}), [key]: value };
    } else if (arg === "--profile") {
      options.profile = true;
    }
  }

  return options;
}

// ─── JSON Protocol (interactive mode) ───────────────────────────────

/** Write a JSON message to stdout (protocol channel). */
function sendJson(obj: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

// ─── Observation Dump (--dump-obs) ──────────────────────────────────

/** Append-only JSONL writer for the TS<->Python parity harness. */
interface ObsDumper {
  write(rec: Record<string, unknown>): void;
}

function createObsDumper(path: string): ObsDumper {
  fs.writeFileSync(path, ""); // truncate any previous dump
  return {
    write(rec) {
      // Synchronous append: survives the process.exit(0) at the end of main()
      fs.appendFileSync(path, JSON.stringify(rec) + "\n");
    },
  };
}

/** Scan an observation for non-finite values. Returns a description or null. */
function scanObservation(obs: Float32Array): string | null {
  for (let i = 0; i < obs.length; i++) {
    if (!Number.isFinite(obs[i])) {
      return `non-finite value ${obs[i]} at dim ${i}`;
    }
  }
  return null;
}

/** Encode a Float32Array bit-exactly (little-endian on x86/ARM; decoded as "<f4"). */
function obsToBase64(obs: Float32Array): string {
  return Buffer.from(obs.buffer, obs.byteOffset, obs.byteLength).toString("base64");
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
  if (line === null) {
    return -1; // EOF
  }
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

  const ballNames = ["Poké Ball", "Great Ball", "Ultra Ball", "Rogue Ball", "Master Ball"];

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
      // Access by slot index directly — do NOT filter
      const playerField = globalScene.getPlayerField() ?? [];
      if (playerField.length > 1 && playerField[1]?.isActive()) {
        return playerField[1].species?.name ?? "Ally";
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

  // Phase-specific label function (handles non-command phases where action 0
  // doesn't mean "fight move 0")
  function getPhaseSpecificLabel(idx: number, phase: string, metadata: Record<string, unknown> = {}): string | null {
    switch (phase) {
      case "select_gender":
        return "Continue";
      case "title":
        return "Start Game";
      case "check_switch":
        if (idx === 0) {
          return "Accept switch";
        }
        if (idx === ACTION_SKIP) {
          return "Decline switch";
        }
        return null;
      case "switch":
        if (idx >= ACTION_SWITCH_START && idx < ACTION_SWITCH_START + 5) {
          return `Switch to: ${getPartyName(idx - ACTION_SWITCH_START + 1)}`;
        }
        return null;
      case "learn_move": {
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
      case "game_over":
        return idx === 0 ? "Continue" : "Quit";
      case "modifier_target":
        if (idx === ACTION_SKIP) {
          return "Cancel (back to items)";
        }
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
  dumper: ObsDumper | null,
): Promise<EpisodeStats> {
  const stats: EpisodeStats = {
    totalSteps: 0,
    wavesCleared: 0,
    decisionsPerPhase: {},
    startTime: Date.now(),
    endTime: 0,
  };

  // Parity-dump dependencies (only loaded when --dump-obs is active)
  let dumpDeps: {
    buildGameState: (s: PhaseState | null, step: number) => Record<string, unknown>;
    encodeObservation: (gs: Record<string, unknown>) => Float32Array;
  } | null = null;
  if (dumper) {
    const { buildGameState } = await import("#rl/state-builder");
    const { encodeObservation } = await import("#rl/spaces");
    dumpDeps = { buildGameState, encodeObservation };
  }

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
      } catch {
        /* ignore */
      }

      // Select an action: use pickDefaultAction for a simple baseline agent
      const action = pickDefaultAction(state);

      if (options.verbose) {
        console.log(
          `[cli] Step ${stats.totalSteps}: phase=${phaseName}, `
            + `validActions=${state.validActions.length}, chosen=${action}`
            + (state.metadata.pokemonSpecies ? `, pokemon=${state.metadata.pokemonSpecies}` : ""),
        );
      }

      // Parity dump: record the exact encoder input/output for this decision
      if (dumper && dumpDeps) {
        const gameState = dumpDeps.buildGameState(state, stats.totalSteps);
        const obs = dumpDeps.encodeObservation(gameState);
        dumper.write({
          v: 1,
          kind: "step",
          seed: options.seed,
          step: stats.totalSteps,
          phase: state.phase,
          wave: (gameState as { battle?: { wave_index?: number } }).battle?.wave_index ?? 0,
          gameState,
          obsB64: obsToBase64(obs),
          actionMask: state.actionMask,
          validActions: state.validActions,
          chosenAction: action,
          actionWasValid: !!state.actionMask[action],
          invariantError: scanObservation(obs),
        });
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
          // RL_DIAG_FIELD=1 -> per-wave histogram of globalScene.field.list to
          // localise the intra-episode field-sprite leak (stderr only).
          if (process.env.RL_DIAG_FIELD) {
            const fieldList: any[] = Array.isArray((globalScene as any)?.field?.list)
              ? (globalScene as any).field.list
              : [];
            const hist: Record<string, number> = {};
            let destroyed = 0;
            let liveCount = 0;
            for (const c of fieldList) {
              const isDestroyed = (c as any)?.__rlDestroyed === true;
              isDestroyed ? destroyed++ : liveCount++;
              const cn = (c as any)?.constructor?.name ?? "?";
              const tex = (c as any)?.texture?.key ?? "";
              const nm = (c as any)?.name ? `#${(c as any).name}` : "";
              const key = `${isDestroyed ? "DESTROYED:" : ""}${cn}${tex ? `[${tex}]` : ""}${nm}`;
              hist[key] = (hist[key] ?? 0) + 1;
            }
            const live = liveCount;
            const dead = destroyed;
            // Localise any residual (non-field) growth: count UI/fieldUI subtree
            // nodes and the global AnimationManager 'remove' listeners (one per
            // live real Sprite — a direct proxy for un-destroyed sprites anywhere).
            const countNodes = (l: any[] | undefined, depth = 0): number => {
              if (!Array.isArray(l) || depth > 8) {
                return 0;
              }
              let n = l.length;
              for (const c of l) {
                n += countNodes(c?.list, depth + 1);
              }
              return n;
            };
            const sc: any = globalScene;
            const uiNodes = countNodes(sc?.ui?.list);
            // Top UI children by recursive node count, to localise ui growth.
            const uiSizes = (sc?.ui?.list ?? [])
              .map((c: any, i: number) => [
                `${i}:${c?.constructor?.name ?? "?"}${c?.name ? `#${c.name}` : ""}`,
                countNodes(c?.list),
              ])
              .sort((a: [string, number], b: [string, number]) => b[1] - a[1])
              .slice(0, 4)
              .map(([l, n]: [string, number]) => `${l}=${n}`)
              .join(" ");
            const fieldUINodes = countNodes(sc?.fieldUI?.list);
            const animMgr: any = sc?.sys?.anims ?? sc?.anims;
            const evRemove = animMgr?._events?.remove;
            const animListeners = Array.isArray(evRemove) ? evRemove.length : evRemove ? 1 : 0;
            const modifiers = (sc?.modifiers?.length ?? 0) + (sc?.enemyModifiers?.length ?? 0);
            (globalThis as { gc?: () => void }).gc?.();
            const memNow = process.memoryUsage();
            const rssMB = Math.round(memNow.rss / 1048576);
            const heapMB = Math.round(memNow.heapUsed / 1048576);
            const top = Object.entries(hist)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 24)
              .map(([k, v]) => `${k}=${v}`)
              .join("  ");
            console.error(
              `[fielddiag] wave=${currentWave} rss=${rssMB}MB heap=${heapMB}MB field=${fieldList.length}(live=${live},dead=${dead}) `
                + `ui=${uiNodes}[${uiSizes}] fieldUI=${fieldUINodes} animLs=${animListeners} mods=${modifiers} :: ${top}`,
            );
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
  dumper?.write({
    v: 1,
    kind: "summary",
    seed: options.seed,
    steps: stats.totalSteps,
    wavesCleared: stats.wavesCleared,
  });
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
  dumper: ObsDumper | null,
): Promise<void> {
  const { buildGameState } = await import("#rl/state-builder");
  const { encodeObservation, ACTION_RUN, ACTION_SPACE_SIZE } = await import("#rl/spaces");
  const { RewardCalculator } = await import("#rl/rewards");
  const gsModule = await import("#app/global-scene");
  const { getAvailableModifiers } = await import("#rl/modifier-api");

  const MAX_STEPS = options.maxWaves * 50;
  let step = 0;

  // --profile: accumulated per-section wall time (ms) for the step hot path
  const prof = {
    advance: 0,
    execute: 0,
    buildState: 0,
    encode: 0,
    labels: 0,
    reward: 0,
    send: 0,
    dump: 0,
    waitAction: 0,
  };
  const now = () => performance.now();

  // Reward bookkeeping: mirrors RLRunner (runner.ts) so the protocol reward
  // is the same single source of truth as headless training would see.
  const rewardCalc = new RewardCalculator(options.rewardConfig ?? undefined);
  let lastFled = false;
  let lastTier = -1;
  // Pre-action snapshot of the final step, kept for the terminal reward: at
  // game over the live scene is already post-reset (cleared party, starting
  // money), so snapshotting it would inject spurious deltas (e.g. a positive
  // money delta whenever the run ended with less than starting money).
  let lastSnapshot: ReturnType<typeof rewardCalc.snapshot> | null = null;

  // Last decision-point state: reused as the terminal snapshot. By the time
  // game over is detected (GameOverPhase -> TitlePhase) the scene has already
  // reset — party cleared and the NEXT battle generated from an unseeded RNG
  // (random time_of_day/offset_gym/seed), which would make the terminal
  // observation non-deterministic and meaningless.
  let lastGameState: Record<string, unknown> | null = null;

  const takeSnapshot = () => {
    const scene = gsModule.globalScene;
    const playerParty = scene?.getPlayerParty?.() ?? [];
    const enemyParty = scene?.getEnemyParty?.() ?? [];
    return rewardCalc.snapshot(
      playerParty,
      enemyParty,
      scene?.currentBattle?.enemyFaints ?? 0,
      playerParty.filter((p: { isFainted: () => boolean }) => p.isFainted()).length,
      scene?.currentBattle?.waveIndex ?? 0,
      scene?.money ?? 0,
    );
  };

  /** Tier of the modifier a reward/shop action would select, or -1. */
  const getModifierTier = (action: number): number => {
    try {
      const modifiers = getAvailableModifiers();
      if (!modifiers) {
        return -1;
      }
      if (action >= 35 && action < 38 && action - 35 < modifiers.rewards.length) {
        return modifiers.rewards[action - 35].tier;
      }
      if (action >= 40 && action < 52 && action - 40 < modifiers.shop.length) {
        return modifiers.shop[action - 40].tier;
      }
    } catch {
      /* not in a modifier phase */
    }
    return -1;
  };

  try {
    while (step < MAX_STEPS) {
      let state: PhaseState;
      const tAdvance = now();
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
      prof.advance += now() - tAdvance;

      // Send any info messages from auto-skipped phases (e.g., IV Scanner)
      for (const msg of router.drainInfoMessages()) {
        sendJson({ type: "info", message: msg });
      }

      // Reward earned by the previous action (0 on the very first state).
      // At terminal the post-action scene is already reset, so the pre-action
      // snapshot stands in: all deltas zero, only terminal/fled/tier apply.
      const tReward = now();
      const terminal = state.phase === DecisionPhase.GAME_OVER || router.isGameOver();
      const postSnap = terminal && lastSnapshot ? lastSnapshot : takeSnapshot();
      const reward =
        step > 0 ? rewardCalc.computeReward(postSnap, terminal, router.isVictory(), lastFled, lastTier) : 0;
      prof.reward += now() - tReward;

      // Check for game over (arrives as a 'title' phase after GameOverPhase → TitlePhase)
      if (terminal) {
        // Reuse the last decision state (the live scene is already post-reset),
        // but patch the phase sub-dict to terminal truth so the state is
        // self-consistent: game_over phase one-hot, all-false action mask.
        const base = lastGameState ?? buildGameState(state, step);
        const terminalMask = new Array<boolean>(ACTION_SPACE_SIZE).fill(false);
        const gameState: Record<string, unknown> = {
          ...base,
          phase: {
            ...((base.phase as Record<string, unknown>) ?? {}),
            current_phase: "game_over",
            action_mask: terminalMask,
            valid_actions: [],
            is_game_over: true,
            is_victory: router.isVictory(),
          },
        };
        const obs = encodeObservation(gameState);
        const wave = (gameState as { battle?: { wave_index?: number } }).battle?.wave_index ?? 0;
        sendJson({
          type: "game_over",
          step,
          victory: router.isVictory(),
          gameState,
          reward,
          obsB64: obsToBase64(obs),
          mask: terminalMask,
          wave,
        });
        if (dumper) {
          dumper.write({
            v: 1,
            kind: "step",
            seed: options.seed,
            step,
            phase: "game_over",
            wave,
            gameState,
            obsB64: obsToBase64(obs),
            actionMask: state.actionMask,
            validActions: state.validActions,
            chosenAction: null,
            actionWasValid: null,
            victory: router.isVictory(),
            invariantError: scanObservation(obs),
          });
        }
        break;
      }

      // Build action labels and game state; encode once (message + dump reuse it).
      // Labels are human-display strings — skipped on the lean training path.
      const tLabels = now();
      const actions = options.lean
        ? state.validActions.map(i => ({ index: i, label: "" }))
        : await buildActionLabels(state);
      prof.labels += now() - tLabels;
      const tBuild = now();
      const gameState = buildGameState(state, step);
      prof.buildState += now() - tBuild;
      lastGameState = gameState;
      const tEncode = now();
      const obs = encodeObservation(gameState);
      prof.encode += now() - tEncode;
      const wave = (gameState as { battle?: { wave_index?: number } }).battle?.wave_index ?? 0;

      // Send state to Python. The TS encoding is the wire authority: obsB64 +
      // mask are always present; --lean drops the bulky gameState JSON.
      const tSend = now();
      sendJson({
        type: "state",
        step,
        phase: state.phase,
        actions,
        ...(options.lean ? {} : { gameState }),
        metadata: state.metadata,
        reward,
        obsB64: obsToBase64(obs),
        mask: state.actionMask,
        wave,
      });
      prof.send += now() - tSend;

      // Read action from Python
      const tWait = now();
      const action = await readAction(reader);
      prof.waitAction += now() - tWait;

      // EOF: stdin closed
      if (action === -1) {
        sendJson({ type: "error", message: "stdin closed (EOF)" });
        break;
      }

      // Validate: invalid actions fall back to the first valid action
      const actionWasValid = !!state.actionMask[action];
      const executed = actionWasValid ? action : (state.validActions[0] ?? 0);
      if (!actionWasValid) {
        sendJson({
          type: "warning",
          message: `Invalid action ${action}, falling back to first valid action`,
        });
      }

      // Pre-action bookkeeping for the next step's reward (must run before
      // executeAction: the modifier phase is gone once the action resolves)
      lastSnapshot = takeSnapshot();
      rewardCalc.savePreActionSnapshot(lastSnapshot);
      lastFled = executed === ACTION_RUN;
      lastTier = state.phase === DecisionPhase.SELECT_MODIFIER ? getModifierTier(executed) : -1;

      // Parity dump: the exact state JSON sent to Python plus the TS encoding
      if (dumper) {
        const tDump = now();
        dumper.write({
          v: 1,
          kind: "step",
          seed: options.seed,
          step,
          phase: state.phase,
          wave,
          gameState,
          obsB64: obsToBase64(obs),
          actionMask: state.actionMask,
          validActions: state.validActions,
          chosenAction: action,
          actionWasValid,
          invariantError: scanObservation(obs),
        });
        prof.dump += now() - tDump;
      }

      const tExec = now();
      await router.executeAction(executed);
      prof.execute += now() - tExec;

      step++;
    }
  } catch (err) {
    sendJson({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }

  if (options.profile && step > 0) {
    const total = Object.values(prof).reduce((a, b) => a + b, 0);
    const lines = Object.entries(prof)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}=${(v / step).toFixed(2)}ms/step (${((100 * v) / total).toFixed(0)}%)`)
      .join(" ");
    process.stderr.write(
      `[profile] steps=${step} instrumented-total=${(total / step).toFixed(2)}ms/step | ${lines}\n`
        + "[profile] note: `advance` = game simulation between decisions (includes executeAction continuation); "
        + "`waitAction` = time blocked on the agent\n",
    );
  }

  dumper?.write({ v: 1, kind: "summary", seed: options.seed, steps: step });
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

    if (options.lean && !options.verbose) {
      // Training fast path: the game logs every phase transition, AI move
      // scores and multi-line Pokemon dumps — formatting them costs real
      // time even when the consumer discards stderr. Keep warn/error.
      const noop = () => {};
      console.log = noop;
      console.info = noop;
      console.debug = noop;
      const toStderr = (...args: unknown[]) => {
        process.stderr.write(args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ") + "\n");
      };
      console.warn = toStderr;
    } else {
      const toStderr = (...args: unknown[]) => {
        process.stderr.write(args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ") + "\n");
      };
      console.log = toStderr;
      console.warn = toStderr;
      console.info = toStderr;
      console.debug = toStderr;
    }
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
    await initHeadless({ seed: options.seed, overrides: options.overrides ?? undefined });
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

  const dumper = options.dumpObs ? createObsDumper(options.dumpObs) : null;

  if (options.interactive) {
    // Interactive mode: JSON protocol over stdin/stdout, multiple episodes
    // per process. After each episode's `done`, the CLI waits for a
    // lifecycle command:
    //   {"cmd":"reset","seed"?,"waves"?} -> in-process reset (resetHeadless,
    //       ~10x faster than a respawn), fresh `ready`, new episode
    //   {"cmd":"quit"} or stdin EOF      -> exit (old clients that just close
    //       stdin keep the original one-episode lifecycle)
    // obsDim/actionDim in `ready` let the Python side reject a stale build.
    const { OBSERVATION_DIM, ACTION_SPACE_SIZE } = await import("#rl/spaces");
    let episodeSeed = options.seed;
    let episodeWaves = options.maxWaves;
    let initMs = bootTime;
    let router = createPhaseRouter({ verbose: options.verbose });

    episodeLoop: for (;;) {
      sendJson({
        type: "ready",
        seed: episodeSeed,
        maxWaves: episodeWaves,
        bootTime: initMs,
        obsDim: OBSERVATION_DIM,
        actionDim: ACTION_SPACE_SIZE,
        protocolVersion: 4,
        ...(options.rewardConfig ? { rewardConfig: options.rewardConfig } : {}),
      });
      await runInteractiveEpisode(
        router,
        { ...options, seed: episodeSeed, maxWaves: episodeWaves },
        DecisionPhase,
        stdinReader!,
        dumper,
      );

      // Await the next lifecycle command
      for (;;) {
        const line = await stdinReader!.next();
        if (line === null) {
          break episodeLoop; // EOF
        }
        let cmd: Record<string, unknown> | null = null;
        try {
          cmd = JSON.parse(line);
        } catch {
          continue;
        }
        if (cmd?.cmd === "quit") {
          break episodeLoop;
        }
        if (cmd?.cmd === "reset") {
          if (typeof cmd.seed === "string" && cmd.seed.length > 0) {
            episodeSeed = cmd.seed;
          }
          if (typeof cmd.waves === "number" && cmd.waves > 0) {
            episodeWaves = Math.floor(cmd.waves);
          }
          break;
        }
        sendJson({ type: "warning", message: 'Expected {"cmd":"reset"} or {"cmd":"quit"} after episode end' });
      }

      // In-process reset: unhook the router's prototype patches, reset the
      // scene (reuses the Phaser.Game; createScene re-applies the seed
      // override and clears localStorage), then re-hook a fresh router so all
      // per-episode router state (titleActionExecuted, gameOverFlag, pending
      // modifier action) starts clean.
      const resetStart = Date.now();
      router.destroy();
      try {
        await resetHeadless({ seed: episodeSeed });
      } catch (err) {
        sendJson({ type: "error", message: `Reset failed: ${err}` });
        break;
      }
      router = createPhaseRouter({ verbose: options.verbose });
      initMs = Date.now() - resetStart;

      if (options.verbose) {
        // Leak diagnostics for soak debugging (stderr only)
        const handles = (process as { _getActiveHandles?: () => unknown[] })._getActiveHandles?.()?.length ?? -1;
        const mem = process.memoryUsage();
        // Force a full GC first when available (NODE_OPTIONS=--expose-gc) so
        // heapMB reflects RETAINED memory, not collection lag
        (globalThis as { gc?: () => void }).gc?.();
        let growth = "";
        try {
          const { globalScene } = await import("#app/global-scene");
          const scene = globalScene as any;
          const countNodes = (list: any[] | undefined, depth = 0): number => {
            if (!Array.isArray(list) || depth > 8) {
              return 0;
            }
            let n = list.length;
            for (const c of list) {
              n += countNodes(c?.list, depth + 1);
            }
            return n;
          };
          const fieldNodes = countNodes(scene?.field?.list);
          const uiNodes = countNodes(scene?.ui?.list);
          const fxNodes = countNodes(scene?.fieldUI?.list);
          // TEMP leak probe: global AnimationManager 'remove' listener count —
          // every live Sprite registers one; orphaned (un-destroyed) sprites
          // pin theirs forever. Direct proxy for the cross-episode sprite leak.
          const animMgr: any = (scene as any)?.sys?.anims ?? (scene as any)?.anims;
          const ev = animMgr?._events?.remove;
          const animListeners = Array.isArray(ev) ? ev.length : ev ? 1 : 0;
          // Per-top-level-ui-child subtree sizes (top 5) to localize growth
          const sizes = (scene?.ui?.list ?? []).map((c: any, i: number) => [
            i,
            countNodes(c?.list),
            `${c?.constructor?.name ?? "?"}${c?.name ? `(${c.name})` : ""}`,
          ]);
          sizes.sort((a: [number, number, string], b: [number, number, string]) => b[1] - a[1]);
          const top = sizes
            .slice(0, 5)
            .map(([i, n, label]: [number, number, string]) => `${i}:${label}:${n}`)
            .join(" ");
          growth = ` fieldNodes=${fieldNodes} uiNodes=${uiNodes} fieldUI=${fxNodes} animListeners=${animListeners} topUi=[${top}]`;
          // RL_DIAG_UI_DETAIL=name1,name2 -> histogram of recursive child
          // constructor(name) inside matching top-level ui children
          const detail = process.env.RL_DIAG_UI_DETAIL;
          if (detail) {
            const wanted = detail.split(",");
            for (const c of scene?.ui?.list ?? []) {
              if (!wanted.some((w: string) => (c?.name ?? "") === w)) {
                continue;
              }
              const hist: Record<string, number> = {};
              const walk = (list: any[], depth = 0) => {
                if (!Array.isArray(list) || depth > 8) {
                  return;
                }
                for (const ch of list) {
                  const key = `${ch?.constructor?.name ?? "?"}${ch?.name ? `(${ch.name})` : ""}`;
                  hist[key] = (hist[key] ?? 0) + 1;
                  walk(ch?.list, depth + 1);
                }
              };
              walk(c?.list);
              const topEntries = Object.entries(hist)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8)
                .map(([k, v]) => `${k}=${v}`)
                .join(" ");
              console.log(`[cli] ui-detail ${c.name}: ${topEntries}`);
              // Probe liveness flags of duplicated children (sweep-predicate tuning)
              const flat: any[] = [];
              const collect = (list: any[], depth = 0) => {
                if (!Array.isArray(list) || depth > 8) {
                  return;
                }
                for (const ch of list) {
                  flat.push(ch);
                  collect(ch?.list, depth + 1);
                }
              };
              collect(c?.list);
              const dups = flat.filter(ch => ch?.name === "text-option-select").slice(0, 3);
              for (const d of dups) {
                console.log(
                  `[cli] ui-probe ${d.constructor?.name}: active=${(d as any).active} scene=${(d as any).scene == null ? "null" : "set"} `
                    + `displayList=${(d as any).displayList == null ? "null" : "set"} type=${(d as any).type} visible=${(d as any).visible}`,
                );
              }
            }
          }
        } catch {
          /* diagnostics only */
        }
        console.log(
          `[cli] reset diag: handles=${handles} heapMB=${Math.round(mem.heapUsed / 1048576)} `
            + `extMB=${Math.round(mem.external / 1048576)} rssMB=${Math.round(mem.rss / 1048576)}${growth}`,
        );
      }
    }

    router.destroy();
  } else {
    const router = createPhaseRouter({ verbose: options.verbose });
    // Auto mode: run with default action picker
    console.log("[cli] Starting episode...");
    const stats = await runEpisode(router, options, pickDefaultAction, DecisionPhase, dumper);

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
    for (const [phase, count] of Object.entries(stats.decisionsPerPhase).sort((a, b) => b[1] - a[1])) {
      console.log(`    ${phase}: ${count}`);
    }
    router.destroy();
  }

  // Cleanup
  await destroyHeadless();
  if (!options.interactive) {
    console.log();
    console.log("[cli] Done.");
  }

  // Drain stdout before exiting: process.exit() discards buffered async
  // writes, and the final game_over message (full gameState JSON) can exceed
  // the pipe buffer. A zero-length write's callback fires only after every
  // queued write before it has flushed.
  await new Promise<void>(resolve => process.stdout.write("", () => resolve()));

  // Force exit: Phaser and game internals may leave pending timers/intervals
  // that prevent Node.js from exiting naturally.
  process.exit(0);
}

main().catch(err => {
  console.error("[cli] Fatal error:", err);
  process.exit(1);
});
