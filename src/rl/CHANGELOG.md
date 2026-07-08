# RL Framework Changelog

All changes to the RL framework are documented here. Every file created or modified
must be recorded with reasoning, before/after behavior, and verification steps.

---

## 2026-02-06 — Phase 1: Diagnosis (Read-Only)

### Team Lead
**What**: Conducted comprehensive codebase investigation with 6 parallel agents.
No code changes were made — this phase was purely diagnostic.

**Investigation Reports Created**:
- `src/rl/diagnosis/investigation-1-vitest-deps.md` — Vitest dependency map
- `src/rl/diagnosis/investigation-2-phaser-coupling.md` — Phaser coupling audit
- `src/rl/diagnosis/investigation-3-phase-control.md` — Phase system control flow
- `src/rl/diagnosis/investigation-4-init-chain.md` — Initialization chain trace
- `src/rl/diagnosis/investigation-5-modifier-system.md` — Modifier system deep dive
- `src/rl/diagnosis/investigation-6-rendering-boundary.md` — Rendering boundary map
- `src/rl/diagnosis/DIAGNOSIS.md` — Unified synthesis document

**Key Findings**:
1. Dual-mode RL runner is feasible — existing test mock infrastructure solves ~90% of Phaser decoupling
2. Vitest spy/mock layer already decoupled to standalone implementations in src/rl/mocks/spy.ts
3. Module resolution (path aliases) and jsdom are the two hard blockers — both solved by Vite bundling
4. 18 decision phases identified, ~6 handled by test harness, rest need new handlers
5. Pokemon extends Phaser.GameObjects.Container — must keep Phaser loaded, not separable
6. No core game files need modification — all RL framework code is additive

**Side Effects**: None. No code was modified, only new files in src/rl/diagnosis/ were created.

**Verification**: All investigation reports can be read for detailed findings with specific code references.

---

## 2026-02-06 — Task #14: Headless Bootstrap

### Files Created
- **`src/rl/headless-boot.ts`** — The headless bootstrap module

### What Each Function Does

| Export | Purpose |
|--------|---------|
| `initHeadless(config?)` | Main entry point. Performs full headless init: jsdom globals, i18n, static game data, Phaser.HEADLESS, mock injection, speed settings. Returns a ready-to-use `BattleScene`. First call is heavy (one-time init); subsequent calls reuse the Phaser.Game and reset the scene. |
| `destroyHeadless()` | Full teardown. Restores all mocks, clears MockClock intervals, destroys the Phaser.Game instance, resets module state so `initHeadless()` can be called fresh. |
| `resetHeadless(config?)` | Lightweight episode reset. Restores mocks, re-seeds RNG, resets BattleScene state without destroying the Phaser.Game. Fastest way to start a new RL episode. |
| `HeadlessConfig` | Configuration interface: `seed` (RNG seed string), `bypassLogin` (boolean), `quietConsole` (boolean). |

### Internal Functions (not exported)

| Function | Purpose |
|----------|---------|
| `installJsdomGlobals()` | Installs jsdom browser globals (window, document, FontFace, matchMedia, canvas context, etc.) required by Phaser at module load time. Idempotent. |
| `runOneTimeInit()` | Delegates to `standalone-setup.ts:initStandalone()` for locale fetch, i18n import, overrides reset, setupStubs, and `initializeGame()`. Idempotent. |
| `getOrCreatePhaserGame(seed)` | Creates a `Phaser.Game({ type: Phaser.HEADLESS })` instance, seeded with the given RNG seed. Reuses existing instance on subsequent calls. |
| `createScene(game, config)` | Creates or reuses a BattleScene. Applies `GameWrapper.injectMandatory()` mocks (tweens, sound, renderer, clock, loader, textures), sets RL speed settings (`moveAnimations=false`, `gameSpeed=5`, etc.), and installs MockFetch. |

### Before vs After

**Before**: No way to boot the game in headless mode without Vitest. The test harness (`vitest.setup.ts` + `GameManager` + `GameWrapper`) was the only working headless path, but it required Vitest's `vi.mock()`, `beforeAll`/`afterEach` lifecycle hooks, and MSW for locale loading.

**After**: `initHeadless()` provides a single async function call that boots the entire game in headless mode from any Node.js context (via Vite bundling). No Vitest dependency. The `resetHeadless()` function allows fast episode resets for RL training loops.

### Design Decisions

1. **Reuse over duplication**: Imports `GameWrapper` directly from `test/test-utils/game-wrapper.ts` for mock injection rather than duplicating its 200 lines. This is safe because `GameWrapper` only depends on `src/rl/mocks/spy.ts` (not Vitest).

2. **Reuse `standalone-setup.ts`**: Delegates one-time init to the existing `initStandalone()` which handles locale fetch, i18n, overrides, and `initializeGame()`.

3. **jsdom globals duplicated from `interactive-boot.ts`**: The global installation logic is extracted (not imported) because `interactive-boot.ts` runs `await import(...)` at module scope immediately after setting globals, which is incompatible with the two-phase init needed here. This is documented in the code.

4. **Dynamic imports throughout**: All heavy imports (Phaser, BattleScene, GameWrapper, enums) are dynamic to ensure jsdom globals are installed before any Phaser code loads. This is critical because Phaser checks for DOM APIs at module load time.

5. **Three-tier API**: `initHeadless()` for first boot, `resetHeadless()` for fast episode resets, `destroyHeadless()` for full cleanup. This gives the RL training loop control over the performance/cleanup tradeoff.

### Dependencies Introduced
- `jsdom` (already a project devDependency via Vitest config)
- Imports from `test/test-utils/` (GameWrapper, MockFetch) -- these are already Vitest-free

### Side Effects
- No existing files modified (only new file + CHANGELOG update)
- Does NOT modify any files in `test/` or `src/` outside `src/rl/`

### How to Verify
1. The module should be importable and type-check cleanly via the project's TypeScript config
2. When run through a Vite bundler (which resolves `#app/*` path aliases), `initHeadless()` should:
   - Install jsdom globals without errors
   - Load all static game data (species, moves, abilities, etc.)
   - Create a Phaser.Game in HEADLESS mode
   - Return a BattleScene with `moveAnimations === false`
3. `resetHeadless()` should return a fresh BattleScene without re-initializing static data
4. `destroyHeadless()` should clean up without errors
5. Full integration testing will be done in Task #18 (dummy agent)

### Known Limitations
- Requires Vite bundling (cannot run with raw Node.js/tsx due to path aliases)
- Single BattleScene per process (globalScene singleton)
- MockClock uses real `setInterval(1ms)` which may accumulate timers across episodes -- `resetHeadless()` creates a new MockClock each time via `injectMandatory()`, but old intervals from previous MockClocks are not explicitly cleaned up. This is tracked as a known issue for long-running training.

---

## 2026-02-06 — Task #15: Phase Decision Router

### Files Created
- **`src/rl/phase-router.ts`** — Event-driven phase detection and action routing system

### What Each Export Does

| Export | Purpose |
|--------|---------|
| `DecisionPhase` (enum) | All 16 decision phases the RL agent may encounter: COMMAND, SELECT_TARGET, SELECT_MODIFIER, SWITCH, CHECK_SWITCH, LEARN_MOVE, EVOLUTION, SELECT_STARTER, MYSTERY_ENCOUNTER, GAME_OVER, SELECT_BIOME, REVIVAL_BLESSING, FORM_CHANGE, TITLE, SELECT_GENDER, UNKNOWN |
| `PhaseState` (interface) | Describes a decision point: which phase, valid action indices, boolean action mask over the 58-action space, and phase-specific metadata |
| `PhaseRouter` (interface) | Core API: `isAtDecisionPoint()`, `getCurrentPhaseState()`, `executeAction(action)`, `advanceToNextDecision()`, `onDecision(callback)`, `isGameOver()`, `isVictory()`, `destroy()` |
| `createPhaseRouter()` | Factory function that creates a PhaseRouter instance, installs UI.prototype.setMode and Phase.prototype.end hooks |
| `pickDefaultAction(state)` | Utility to select a sensible default action for any PhaseState (used for auto-handling non-RL phases) |

### Phase Detection Strategy

The router hooks two prototype methods (following the same pattern as `test/test-utils/phase-interceptor.ts`):

1. **`UI.prototype.setMode`** — When a decision phase calls `setMode()` with a decision-relevant UiMode, the hook detects the decision point and resolves the pending promise. This covers `endBySetMode` phases (CommandPhase, SelectModifierPhase, etc.) AND callback-based phases (SwitchPhase, CheckSwitchPhase, SelectTargetPhase, etc.).

2. **`Phase.prototype.end`** — Detects GameOverPhase completion and other terminal states.

A 50ms polling fallback runs during `advanceToNextDecision()` to catch edge cases where the setMode hook fires before the wait begins.

### Decision Phase Handlers

| Phase | Priority | Detection | Action Execution |
|-------|----------|-----------|-----------------|
| **CommandPhase** | CRITICAL | `setMode(UiMode.COMMAND)` | `commandPhase.handleCommand(cmd, cursor, ...)` directly |
| **SelectTargetPhase** | CRITICAL | `setMode(UiMode.TARGET_SELECT)` | `handler.setCursor(target)` + `processInput(ACTION)` |
| **SelectModifierPhase** | HIGH | `setMode(UiMode.MODIFIER_SELECT)` | Delegates to `src/rl/modifier-api.ts` (selectRewardModifier, selectShopModifier, skipModifiers, rerollModifiers) |
| **SwitchPhase** | HIGH | `setMode(UiMode.PARTY)` | `handler.setCursor(slot)` + `processInput(ACTION)` x2 |
| **CheckSwitchPhase** | MEDIUM | `setMode(UiMode.CONFIRM)` | `processInput(ACTION)` to accept, `processInput(CANCEL)` to decline |
| **LearnMovePhase** | MEDIUM | `setMode(UiMode.CONFIRM/SUMMARY)` | Multi-step: CONFIRM->Yes/No, SUMMARY->move slot selection |
| **EvolutionPhase** | MEDIUM | `setMode(UiMode.EVOLUTION_SCENE)` | Auto-accept (let evolution proceed) |
| **GameOverPhase** | EPISODE END | `setMode(UiMode.CONFIRM)` | ACTION to retry, CANCEL to quit |
| **MysteryEncounterPhase** | LOW | `setMode(UiMode.MYSTERY_ENCOUNTER)` | `setCursor(optionIndex)` + `processInput(ACTION)` |
| **SelectBiomePhase** | LOW | `setMode(UiMode.OPTION_SELECT)` | `setCursor(biomeIndex)` + `processInput(ACTION)` |
| **RevivalBlessingPhase** | RARE | `setMode(UiMode.PARTY)` | `setCursor(faintedSlot)` + `processInput(ACTION)` x2 |
| **TitlePhase** | SETUP | `setMode(UiMode.TITLE)` | Auto-handle (bypassed by runner setup) |
| **SelectGenderPhase** | SETUP | `setMode(UiMode.OPTION_SELECT)` | Auto-handle |
| **SelectStarterPhase** | SETUP | `setMode(UiMode.STARTER_SELECT)` | Auto-handle (bypassed by runner setup) |
| **FormChangePhase** | AUTO | `setMode(UiMode.EVOLUTION_SCENE)` | Auto-accept |

### Action Space Integration

The router uses the same action space indices defined in `src/rl/spaces.ts`:
- Actions 0-34: Battle actions (fight with targeting, switch, ball, run, tera)
- Actions 35-39: Modifier actions (select reward, reroll, skip)
- Actions 40-51: Shop actions (buy items)
- Actions 52-57: Party target actions (for modifier application and revival blessing)

Action validation is enforced: if an invalid action is provided, the router logs a warning and falls back to the first valid action.

### Before vs After

**Before**: The test harness used `PhaseInterceptor` with Vitest-dependent `setInterval` polling and `vi.waitUntil` to detect and intercept phases. The RL environment (`environment.ts`) had basic phase detection via `phase.is("CommandPhase")` but no framework for handling non-battle decision phases (SwitchPhase, LearnMovePhase, CheckSwitchPhase, etc.).

**After**: `createPhaseRouter()` provides a complete, event-driven decision router that:
- Detects ALL 16 decision phases without Vitest
- Builds action masks for each phase using the shared action space
- Executes actions via direct method calls (following the test harness patterns)
- Provides `advanceToNextDecision()` for the runner's step loop
- Handles unsupported phases with safe defaults

### Design Decisions

1. **Event-driven with polling fallback**: The primary detection uses UI.prototype.setMode hooks (same as PhaseInterceptor), but includes a 50ms polling fallback for robustness. This avoids the pure-polling approach of PhaseInterceptor while handling timing edge cases.

2. **No Vitest dependency**: Zero imports from 'vitest'. All phase detection is done through prototype hooking and game state inspection.

3. **Reuse modifier-api.ts**: Modifier actions delegate entirely to the existing `src/rl/modifier-api.ts` instead of reimplementing the complex modifier selection pipeline.

4. **Action validation with fallback**: Invalid actions trigger a warning and fall back to the first valid action rather than crashing. This is critical for RL training where the agent may occasionally select invalid actions.

5. **Prototype restoration on destroy**: `destroy()` restores both `UI.prototype.setMode` and `Phase.prototype.end` to their originals, preventing leaks between RL episodes or test runs.

6. **pickDefaultAction utility**: For phases the RL agent shouldn't control (evolution, title, setup phases), the `pickDefaultAction` function selects a sensible default (skip items, accept evolution, pick first option).

### Dependencies
- `src/rl/modifier-api.ts` — For modifier phase handling
- `src/rl/spaces.ts` — For action space constants
- Game phase imports — Types only for CommandPhase, SelectTargetPhase, SelectModifierPhase
- `#ui/ui.ts` — For UI prototype hooking
- `#app/phase.ts` — For Phase prototype hooking

### Side Effects
- No existing files modified (only new file + CHANGELOG update)
- Does NOT modify any files outside `src/rl/`
- Prototype hooks are installed when `createPhaseRouter()` is called and cleaned up on `destroy()`

### How to Verify
1. TypeScript compilation: The file should compile cleanly with the project's tsconfig.json
2. Import validation: All imports use the project's path aliases (`#app/*`, `#enums/*`, `#rl/*`, etc.)
3. Prototype safety: `destroy()` restores original prototypes, preventing interference with tests
4. Integration: Task #16 (runner.ts) will use `createPhaseRouter()` to drive the game loop
5. Full integration testing in Task #18 (dummy agent)

### Known Limitations
1. **LearnMovePhase is multi-step**: The current implementation handles only the initial CONFIRM dialog. The full flow (CONFIRM -> SUMMARY -> CONFIRM) may require the router to be called multiple times for a single learn-move decision. The runner should handle this by calling advanceToNextDecision() again after the first step.
2. **SelectTargetPhase action mapping**: Uses ACTION_FIGHT_ENEMY_START (0) as proxy for BattlerIndex.ENEMY target selection. The runner/environment should translate these correctly.
3. **Single router per process**: Since the router patches global prototypes, only one PhaseRouter should be active at a time (consistent with the globalScene singleton constraint).
4. **Timeout-based polling**: The 50ms polling interval and 30s timeout in advanceToNextDecision() may need tuning for very fast or very slow game scenarios.

---

## 2026-02-06 -- Task #16: Dual-Mode RL Runner

### Files Created
- **`src/rl/runner.ts`** -- The dual-mode RL runner (main entry point)

### What Each Export Does

| Export | Purpose |
|--------|---------|
| `RLRunner` (class) | Main RL environment class. Orchestrates init -> reset -> step -> close lifecycle. Ties together headless-boot.ts, phase-router.ts, spaces.ts, and rewards.ts. |
| `RunnerMode` (type) | `"headless" \| "rendered"` -- which mode to run in |
| `RunnerConfig` (interface) | Configuration: mode, seed, numStarters, starterSpecies, logDecisions, rewardConfig, maxDecisionLogSize, advanceTimeoutMs |
| `StepResult` (interface) | Return type of `step()`: observation, reward, done, truncated, info (wave, turn, phase, action, actionValid, pokemonFainted, enemyFainted, etc.) |
| `DecisionLogEntry` (interface) | Structure for decision logging: wave, turn, phase, action, actionName, metadata, timestamp |

### RLRunner API

| Method | Purpose |
|--------|---------|
| `constructor(config)` | Create runner with configuration. Does not start the game. |
| `init()` | One-time initialization: boots Phaser in HEADLESS mode, loads all game data. Call once before any reset/step calls. |
| `reset()` | Start a new episode: resets scene, creates PhaseRouter, auto-handles TitlePhase/SelectGenderPhase/SelectStarterPhase, advances to first CommandPhase. Returns initial observation and info. |
| `step(action)` | Execute one RL action: validates action against mask, snapshots state, executes via router, advances to next RL decision (auto-handling evolutions/form-changes), computes reward. Returns StepResult. |
| `getObservation()` | Build observation vector (Float32Array of 2951 dims) from current game state without stepping. |
| `getActionMask()` | Get boolean[58] action mask from current PhaseState. |
| `isDone()` | Check if episode has ended. |
| `close()` | Destroy router, teardown headless environment, release resources. |
| `getDecisionLog()` | Get a copy of all logged decisions in this episode. |
| `flushDecisionLog()` | Return and clear the decision log (for memory management in long episodes). |
| `getEpisodeStats()` | Get step count, wave, turn, faints, and other episode metadata. |

### Game Start Sequence

The runner automates the full startup sequence that the test harness handles via `classicMode.startBattle()`:

1. After `resetHeadless()`, the scene starts at TitlePhase
2. At TitlePhase: sets game mode to CLASSIC, generates starters via `generateStarters()` (same function as test harness), calls `selectStarterPhase.initBattle(starters)`, pushes EncounterPhase
3. Auto-handles any SelectGenderPhase, SelectStarterPhase, CheckSwitchPhase that appear during setup
4. Advances through EncounterPhase, SummonPhase, TurnInitPhase to the first CommandPhase
5. Returns initial observation at the first real RL decision point

### Phase Classification

The runner classifies phases into two categories:

**RL Decision Phases** (returned to the agent for action selection):
- COMMAND, SELECT_TARGET, SELECT_MODIFIER, SWITCH, LEARN_MOVE, MYSTERY_ENCOUNTER, SELECT_BIOME, REVIVAL_BLESSING, GAME_OVER

**Auto-Handled Phases** (handled with sensible defaults):
- TITLE, SELECT_GENDER, SELECT_STARTER, EVOLUTION, FORM_CHANGE, CHECK_SWITCH (during setup only; during gameplay CHECK_SWITCH is currently auto-handled too)

### Integration with Existing Modules

| Module | How Runner Uses It |
|--------|--------------------|
| `headless-boot.ts` | `initHeadless()` for first boot, `resetHeadless()` for episode resets, `destroyHeadless()` for cleanup |
| `phase-router.ts` | `createPhaseRouter()` for decision detection, `executeAction()` for action routing, `advanceToNextDecision()` for game advancement, `pickDefaultAction()` for auto-phases |
| `spaces.ts` | `buildObservation()` for the 2951-dim observation vector, action space constants for mask/validation |
| `rewards.ts` | `RewardCalculator` for pre/post snapshot delta rewards |
| `modifier-api.ts` | `getAvailableModifiers()` for modifier tier detection (used in reward computation) |
| `game-manager-utils.ts` (test) | `generateStarters()` for starter Pokemon generation (same as test harness) |

### Design Decisions

1. **Dynamic imports for game code**: `handleTitlePhase()` uses dynamic `await import()` for game mode, phase classes, and starter generation. This ensures game code is not loaded before jsdom globals are installed.

2. **Router DecisionPhase vs. spaces.ts DecisionPhase**: The `phase-router.ts` `DecisionPhase` enum has 16 values (all possible phases). The `spaces.ts` `DecisionPhase` type is a narrow `"command" | "modifier" | "modifier_target"` union used only for observation encoding. The runner maps between them via `mapToSpacesPhase()`.

3. **Auto-advance loop with safety limits**: `advanceToNextRLDecision()` loops through non-RL phases with a cap of 50 iterations. `handleGameStartSequence()` has a 20-iteration cap. Both throw/return-done on exceeding the limit to prevent infinite loops.

4. **Decision log with bounded size**: The log is capped at `maxDecisionLogSize` (default 10000). When full, the oldest 10% is dropped in a batch to minimize array manipulation overhead. A `flushDecisionLog()` method allows explicit memory management.

5. **Reward computation timing**: Pre-action snapshots are taken before `executeAction()`, post-action snapshots after `advanceToNextRLDecision()`. This means the reward for a CommandPhase action captures all consequences through to the next RL decision point (including enemy actions, damage, faints, wave transitions).

6. **Rendered mode stub**: Throws "not yet implemented" for now. The architecture supports it -- the PhaseRouter works identically in both modes since it hooks the same prototype methods. The only difference is whether Phaser renders real graphics.

7. **Action validation with fallback**: Invalid actions trigger a warning and fall back to the first valid action. The `actionValid` flag in `StepResult.info` tells the caller whether their action was used as-is.

### Before vs After

**Before**: The `environment.ts` `RLEnvironment` class provided a Gymnasium-compatible API but relied on an external `GameRunner` interface for game advancement. There was no implementation of `GameRunner` that could drive the game from init through reset to step. The test harness (`GameManager` + `PhaseInterceptor`) was the only way to drive the game, but it required Vitest.

**After**: `RLRunner` is a self-contained, end-to-end RL environment that:
- Boots the game in headless mode (via headless-boot.ts)
- Handles the full startup sequence (TitlePhase -> first CommandPhase)
- Detects all 16 decision phases (via phase-router.ts)
- Builds 2951-dim observations (via spaces.ts)
- Computes shaped rewards (via rewards.ts)
- Provides a clean init/reset/step/close API
- Logs every decision with human-readable action names
- Has no Vitest dependency

### Dependencies
- `src/rl/headless-boot.ts` -- Headless initialization (dynamic import)
- `src/rl/phase-router.ts` -- Phase detection and action routing
- `src/rl/spaces.ts` -- Observation and action space encoding
- `src/rl/rewards.ts` -- Reward computation
- `src/rl/modifier-api.ts` -- Modifier info for reward computation
- `test/test-utils/game-manager-utils.ts` -- Starter generation (dynamic import)
- Game enums: `BattleType`, `BiomeId` (for canRun/canCatch checks)

### Side Effects
- No existing files modified (only new file + CHANGELOG update)
- Does NOT modify any files outside `src/rl/`

### How to Verify
1. TypeScript compilation: The file should compile cleanly with the project's tsconfig.json
2. Import validation: All imports use the project's path aliases
3. The `environment.ts` `GameRunner` interface is compatible with the runner's internal approach (though the runner provides its own higher-level API rather than implementing `GameRunner` directly)
4. Full integration testing will be done in Task #18 (dummy agent)

### Known Limitations
1. **Rendered mode not implemented**: Throws an error if mode is "rendered"
2. **CHECK_SWITCH always auto-declined**: During both setup and gameplay, CheckSwitchPhase is auto-handled by declining. A future version could expose this as an RL decision.
3. **Single runner per process**: Due to the `globalScene` singleton and prototype hooks, only one RLRunner can be active at a time.
4. **Starter generation imports test code**: `generateStarters()` is imported from `test/test-utils/game-manager-utils.ts`. This works because that file is Vitest-free, but it means the RL runner depends on test infrastructure.
5. **Observation encoding for non-command/modifier phases**: When at SwitchPhase, LearnMovePhase, etc., the observation's phase indicator is null (neither "command" nor "modifier"). The observation still captures full game state, but the phase indicator does not distinguish between these phases.

---

## 2026-02-06 -- Task #17: Vite Build Config for Headless Node.js Bundle

### Files Created
- **`vite.headless.config.ts`** (project root) -- Vite build configuration for the headless RL bundle
- **`src/rl/cli.ts`** -- Minimal CLI entry point for testing the build

### Files Modified
- **`package.json`** -- Added `rl:build` and `rl:run` scripts

### What Was Built

**`vite.headless.config.ts`** -- A Vite configuration that bundles the RL runner for Node.js execution:

| Feature | Implementation |
|---------|---------------|
| Build mode | Vite SSR mode (`build.ssr: true`) which targets Node.js by default |
| Entry point | `src/rl/cli.ts` via `rollupOptions.input` |
| Path aliases | `vite-tsconfig-paths` plugin resolves all `#app/*`, `#enums/*`, `#rl/*`, `#test/*`, etc. |
| External deps | Phaser, jsdom, i18next, and 15+ other npm packages externalized (not bundled) |
| Shader files | Custom `glslShaderPlugin` returns empty string for `.frag`, `.vert`, `.glsl` imports |
| import.meta.env | All references replaced at build time via Vite `define` (MODE=production, BYPASS_LOGIN=1, etc.) |
| Output | `dist/rl/cli.js` in ESM format (matches package.json `"type": "module"`) |
| Source maps | Enabled (`sourcemap: true`) for stack trace debugging |
| Minification | Disabled (`minify: false`) for readable output and better debugging |
| Class/function names | Preserved (`keepNames: true`) for phase name detection at runtime |
| Phaser resolution | Aliases `phaser` to `node_modules/phaser/src/phaser.js` (source build, same as `vite.interactive.config.ts`) |

**`src/rl/cli.ts`** -- Minimal CLI that:
1. Parses `--seed`, `--waves`, `--log` arguments
2. Calls `initHeadless()` to boot the game
3. Creates a `PhaseRouter` and runs a simple loop using `pickDefaultAction()` for each decision
4. Prints an episode summary (steps, waves cleared, duration, decisions by phase type)
5. Cleans up via `router.destroy()` + `destroyHeadless()`

### Package.json Scripts Added
```
"rl:build": "vite build --config vite.headless.config.ts"
"rl:run": "node dist/rl/cli.js"
```

### External Dependencies Strategy

Dependencies are externalized (not bundled) when they are:
- Large packages that would bloat the bundle (phaser, jsdom, phaser3-rex-plugins)
- Packages with complex sub-dependencies (i18next ecosystem)
- Node.js built-in modules (node:fs, node:path, etc.)
- Any package installable via `npm install` at deployment time

Everything in `src/` and `test/test-utils/` IS bundled, since those files contain:
- Game logic with path aliases that must be resolved
- Mock infrastructure (MockLoader, MockClock, MockFetch, etc.)
- RL framework code (spaces, rewards, environment, phase-router)

### Design Decisions

1. **SSR build mode over library mode**: Vite's SSR mode is specifically designed for Node.js output. It correctly handles `node:` protocol imports, avoids browser-specific transforms, and produces output compatible with Node.js module resolution. Library mode was considered but SSR is more appropriate for an executable entry point.

2. **ESM output format**: The project has `"type": "module"` in package.json, so the output uses ESM format. This means the built file uses `import`/`export` statements that Node.js v18+ can run natively.

3. **No minification**: The headless bundle is not served to browsers. Keeping it unminified ensures stack traces are readable, console.log output makes sense, and the bundle can be inspected for debugging.

4. **Phaser source alias**: Following the precedent of `vite.interactive.config.ts`, we alias `phaser` to its source entry point (`phaser/src/phaser.js`). This gives Vite better tree-shaking opportunities and avoids issues with the pre-built Phaser dist files in a Node.js context.

5. **Explicit define for every import.meta.env reference**: Rather than a catch-all replacement, each `import.meta.env.*` reference found in the codebase is explicitly defined. This prevents runtime errors from undefined env vars and documents exactly what the headless bundle assumes.

6. **Circular dependency warnings suppressed**: The game codebase has many circular imports (common in large TypeScript projects). These work correctly at runtime due to ESM live bindings, but Rollup warns about them. Suppressing these warnings keeps the build output clean.

### Before vs After

**Before**: The RL framework code could only run within Vitest (using `vitest.interactive.config.ts` + jsdom environment). There was no way to produce a standalone Node.js bundle that could be invoked with `node dist/rl/cli.js`.

**After**: `pnpm rl:build` produces a complete Node.js bundle at `dist/rl/cli.js` that includes all game logic, mock infrastructure, and RL framework code. External npm dependencies are resolved at runtime from `node_modules/`. The bundle can be run with `pnpm rl:run` or `node dist/rl/cli.js [--seed=X] [--waves=N] [--log]`.

### How to Verify

1. **Build succeeds**: `pnpm rl:build` should complete without errors
2. **Output exists**: `dist/rl/cli.js` and `dist/rl/cli.js.map` should be present
3. **Run succeeds**: `pnpm rl:run -- --seed=test --waves=1 --log` should:
   - Initialize the headless game
   - Run through at least one decision point
   - Print an episode summary
   - Exit cleanly
4. **Existing workflows unaffected**: `pnpm dev`, `pnpm build`, `pnpm test` all still work
5. **Source maps work**: Stack traces from errors in `node dist/rl/cli.js` should point to original .ts files

### Known Concerns

1. **Bundle size**: The game codebase is large (~10MB+ of TypeScript). The resulting bundle will be substantial. This is acceptable for an RL training runner that runs on servers, not in browsers.

2. **Dynamic imports in headless-boot.ts**: The headless bootstrap uses dynamic `import()` calls throughout (to ensure jsdom globals are installed before Phaser loads). Vite/Rollup will try to resolve these and may create separate chunks. The `inlineDynamicImports: false` setting allows this chunking for better code splitting.

3. **Phaser window/document access at import time**: Even with jsdom globals installed, Phaser may attempt to access `window` or `document` during module evaluation. The headless-boot.ts code handles this by installing jsdom globals first via `installJsdomGlobals()` before any Phaser import. In the bundled output, this ordering is preserved because all Phaser imports in headless-boot.ts are dynamic.

4. **test/ directory files in bundle**: The build includes files from `test/test-utils/` (GameWrapper, MockLoader, etc.). This is intentional -- these mock classes are essential for headless operation and are already Vitest-free.

---

## Phase 2: Implementation (In Progress)

Implementation tasks:
- Task #14: Headless bootstrap (src/rl/headless-boot.ts) -- DONE
- Task #15: Phase decision router (src/rl/phase-router.ts) -- DONE
- Task #16: Dual-mode runner (src/rl/runner.ts) -- DONE
- Task #17: Vite build config (vite.headless.config.ts) -- DONE
- Task #18: Integration test (dummy-agent.ts)

---

## 2026-06-09/10 — Verification Sprint (Task #18 superseded)

**Goal**: systematic correctness verification of the full RL stack, a
gymnasium wrapper, and restored/updated documentation.

### New infrastructure
- `tools/verify/` — verification harness: `check_parity.py` (TS↔Python
  element-wise observation/mask compare with feature-name reporting),
  `run_episodes.py` (masked-random smoke driver + hang watchdog +
  invalid-action probes), `check_determinism.py` (same-seed bitwise replay),
  `bench_throughput.py` (steps/s + soak), `check_parser_completeness.py`
  (static AST sweep), `fixture_parity.py` (cross-language goldens),
  `invariants.py`, `common.py`.
- `src/rl/feature_names.py` — dim→name table for all 9,875 dims + 199
  one-hot groups (import-time asserted against observation.py constants).
- `test/rl/` — `rewards.test.ts` (20 unit tests), `spaces-encoding.test.ts`
  (golden fixtures, `UPDATE_RL_GOLDEN=1` regen), `semantic/` (11 files, 53
  game-truth audit tests: arena tag sides, move-flag extraction, volatile
  tags, weather/terrain, stats/status, doubles slots, KO/switch, type
  effectiveness, ability features, held items, positional tags, shop state).
- `src/rl/pokerogue_env.py` — `PokeRogueEnv(gymnasium.Env)` with
  `action_masks()`, version guard, subprocess lifecycle management.
- `examples/rl/random_agent.py`, `examples/rl/train_maskable_ppo.py`,
  `requirements-rl.txt`, `scripts/rl-verify.sh`, `pnpm rl:verify[:quick]`.

### Protocol changes (cli.ts, backward compatible)
- `--dump-obs=<path>`: per-step JSONL records (gameState + base64 float32
  observation + mask) for the parity harness.
- `state`/`game_over` messages now carry `reward` (computed by the TS
  RewardCalculator, mirroring runner.ts snapshot logic).
- `ready` message carries `obsDim`/`actionDim`/`protocolVersion` (stale-build
  guard, enforced by the wrapper).
- Terminal `game_over` reuses the last decision state (patched to game_over
  phase + all-false mask) instead of the post-reset scene; stdout drained
  before `process.exit`.

### Bugs fixed (14)
See `docs/VERIFICATION.md` §"Bugs found & fixed" — 7 Python parser/encoder
parity bugs, `arena_tag_self_side` semantics, 2 missing `MoveAttrs` registry
entries (game code), full-party capture hang, i18n boot-ordering crash on
rival encounters, title-phase determinism break, terminal-state
non-determinism, stdout flush loss.

### Results
All checks green: 91/91 tests, bitwise TS↔Python parity on 700+ live states,
bitwise determinism in both modes, 9/9 smoke episodes clean, `check_env` +
MaskablePPO smoke pass, ~96 steps/s interactive throughput.

---

## 2026-06-11 — Architecture Improvements (protocolVersion 3)

- **TS-authoritative wire**: state/game_over carry obsB64 + mask + wave;
  `--lean` drops gameState JSON (wrapper default). Python encoder stays the
  parity-checked reference.
- **In-process episode reset**: `{"cmd":"reset","seed","waves"}` after `done`
  (~3-10ms vs ~2s respawn; first reset captures a display baseline, later
  resets purge episode debris). `{"cmd":"quit"}`/EOF exits. Gated by
  check_determinism `--mode inprocess` (in-process == fresh-process, bitwise)
  and a 100-episode soak.
- **Wrapper**: respawn_every=50 recycling (bounds a residual ~4.5MB/episode
  UI-handler leak), overrides kwarg, mystery_encounters=False default
  (ME phases aren't routable yet — U5 — and would burn step-timeouts).
- **--reward-config / --override** CLI flags (reward experimentation,
  scripted scenarios).
- **Coverage assurance**: check_obs_coverage (every battler tag / move attr /
  modifier type / move flag / ability classified: encoded, excluded-with-
  reason, or fails), check_dim_exercise (never-varied dims must be ledger-
  classified; scenario-ran-but-flat = suspected-bug gate),
  gen_coverage_corpus (doubles/trainers/longrun/weather/fullparty/status
  scenarios that assert their target situations).
- **7 more bugs fixed** (terminal-reward post-reset sampling, MockClock
  interval leak, two stacked-wrapper leaks, field/fieldUI display-list
  accumulation, ScanIvsPhase nameless-sprite crash, Memory Mushroom
  moveId=undefined crash, stdout flush loss). Details: docs/VERIFICATION.md.
- Throughput: ~96-105 steps/s sustained; reset 3ms; PPO end-to-end 59 fps.

### 2026-06-11 (later) — Hot-path audit + dead code removal
- `--profile` flag: per-step section timings. Measured (lean, ~10ms/step @ ~96
  steps/s): game simulation 80% (execute 4.6ms + advance 3.7ms), buildGameState
  1.2ms (12%), encode 0.3ms, send 0.2ms, labels/reward/dump ~0. The RL layer is
  NOT the bottleneck; further speedups are game-engine surgery.
- Lean mode now skips human-only action labels and silences console.log/info/
  debug (warn/error kept).
- DELETED dead modules (zero importers, not in any runtime path):
  src/rl/runner.ts (854 lines, superseded by cli.ts + pokerogue_env.py) and
  src/rl/standalone-runner.ts (53 lines). References in older CHANGELOG
  sections are historical.

---

## 2026-06-13 — Headless leak fix + Observation v8

### Cross-episode memory/throughput leak (the big one) — fixed
- **Root cause**: `MockContainer` never set `this.scene`. Phaser's
  `Container.destroy()` only cascades into children whose `.scene` is truthy
  (`removeAll`: `if (list[i] && list[i].scene) list[i].destroy()`), so every
  real container holding mock-container children (each `BattleInfo`'s
  statsContainer → statValuesContainer → stat sprites) left them undestroyed,
  stranding ~575 sprites/episode on the process-global `AnimationManager`
  `remove` event. That listener array grew unbounded → per-step sprite churn
  went O(n²) — the real throughput collapse. The game never leaks in the
  browser; this was purely a headless-mock artifact.
- **Fixes** (faithful to Phaser, no monkey-patching): `MockContainer` sets
  `this.scene` + recursive `destroy()`; `MockTextureManager` dropped its
  write-only object registry; `BattleScene.reset()` clears the sparkle
  handler's sprite Set; deduped duplicate `addedToScene`/`removedFromScene`.
- **Result** (pure 100-episode in-process soak, no recycling): FAIL → OK;
  steps/s collapse 75% → 19% drift; anim-listener growth 575/ep → ~6/ep;
  retained heap unbounded → 1.08 MB/ep. KNOWN RESIDUALS in docs/VERIFICATION.md
  (per-episode CanvasPool ~1MB/ep, respawn-bounded; per-wave intra-episode
  ~1.5MB/wave field-sprite accumulation, UNSOLVED, not respawn-bounded).

### Observation v8 (9,875 → 10,403; protocolVersion 4)
- **CURATED_VOLATILE_TAGS 48 → 76** (+28×12 = +336): partial-trap family,
  charge/crit/boost states, exposure/ignore states, paradox/overlord boosts,
  NIGHTMARE, TRUANT, ALWAYS_GET_HIT.
- **MOVE_BLOCK_DIM 132 → 136** (+4×4×12 = +192): survives_at_1hp
  (SurviveDamageAttr), matches_user_hp (MatchHpAttr), hp_cost_stat_boost
  (CutHpStatStageBoostAttr), hits_semi_invulnerable (HitsTagAttr — instanceof
  catches HitsTagForDoubleDamageAttr).
- POKEMON_BLOCK_DIM 771 → 815. Lockstep TS (spaces.ts/state-builder.ts) +
  Python (observation.py/enums.py/state_schema.py/feature_names.py), bitwise
  parity 0 diffs. Canary re-pinned, goldens regenerated, 6 new semantic tests
  (4 flags + Bind→BIND, Focus Energy→CRIT_BOOST). Coverage manifests shrunk to
  0 backlog (28 tags + 8 move-attrs reclassified encoded/covered).
- Mystery Encounters removed env-wide (MYSTERY_ENCOUNTER_RATE_OVERRIDE=0).
- `rl-verify.sh` wipes `$ART` each run (a stale 9875-dim corpus dump broke
  parity); corpus longrun step_timeout 60→150s.
- KNOWN: rl:verify's longrun corpus intermittently hits a PRE-EXISTING game bug
  (berry/ability infinite recursion → stack overflow at deep waves), unrelated
  to v8 and tracked separately. All v8-observation gates pass.

---

## 2026-07-07 — Full-framework bug audit: mask/game-logic soft-locks + observation↔action misalignments

Top-to-bottom review of the RL framework (phase-router, cli, headless-boot,
browser-bridge, modifier-api, state-builder, spaces.ts ↔ observation.py, env,
tools). Nine fixes, all verified (`rl-verify.sh quick` green, 99/99 vitest,
bitwise TS↔Python parity incl. 4×14-wave random soak, determinism ×3, plus a
targeted repro script proven to FAIL pre-fix / PASS post-fix).

### Game-logic soft-locks (timeout class)
- **`phase-router.ts` — ball mask now mirrors the game's gating exactly**
  (new exported `getLegalBallTypes()`, replicating `checkCanUseBall` +
  `handleBallCommand`): shielded bosses (every 10th wave) only accept a
  Master Ball (never on a challenge-mode final boss), END-biome rules, and
  "exactly one visible target". Pre-fix the mask offered regular balls at
  wave-10 bosses; throwing one made CommandPhase show a prompt-blocking
  error message that nothing dismisses → 30s router timeout killed the
  episode. Also UNblocks the legal catch in doubles with one enemy left
  (old mask forbade all catching in doubles).
- **`phase-router.ts` — Struggle**: when an enemy is up but no move is
  usable (PP/Disable/Torment), the mask now offers fight slot 0 and the
  executor issues a plain FIGHT (the game auto-substitutes Struggle and
  computes targets). Pre-fix the mask could go fight-empty — and fully
  empty when also trapped in a trainer battle → unresolvable decision →
  timeout. Tera mirroring is suppressed on the synthetic Struggle bit.

### Wrong-action execution
- **`modifier-api.ts` — two-row shop purchases bought the wrong item**: the
  (rowCursor, cursor) mapping was inverted vs the phase callback's
  convention (rowCursor 2 = LAST row, 3 = first). With > 7 shop options,
  "buy item i" purchased item i±7. Affects non-targeted items (Sacred Ash).

### Observation ↔ action misalignment (blindspots)
- **`spaces.ts` + `observation.py` — shop options encoded in NATURAL order**
  (was: sorted by cost). Encoded slot k now describes BUY action 40+k,
  restoring the framework's "the observation describes that slot" contract.
  Goldens regenerated (`full.golden.b64`); bitwise parity re-verified.
- **`modifier-api.ts` — shop costs are now the TRUE purchase price**
  (HealShopCostModifier / Black Sludge applied), so mask affordability,
  observation cost features, and the actual charge agree.
- **`state-builder.ts` — battle features aligned with reality**:
  `can_catch` = some ball action currently legal AND owned (shares
  `getLegalBallTypes`), `can_run` = wild && biome ≠ END, `tera_available` =
  Tera Orb owned && arena tera unused (was: "nobody terastallized yet",
  true from wave 1 with no orb).
- **`state-builder.ts` — phase-info fields resurrected**: metadata key
  mismatches left `learn_move_name`/`learn_move_id`/`biome_options`/
  `mystery_option_count` permanently null (mask builders set
  `newMoveName`/`biomeNames`/`optionCount`; the router now also records
  `learnMoveId`).

### Display / tooling correctness
- **`browser-bridge.ts` — `getEnemyName` indexes by slot** (was: filtered
  actives, mislabeling targets once enemy slot 0 fainted — the exact bug
  cli.ts already documented). `getAllyName` (bridge + cli) now names the
  OTHER slot relative to the acting pokemon, matching the executor.
- **`tools/run_policy.py` — maxdamage reads the ACTING slot's moveset**
  via `phase.command_field_index` (was: always `player_0`, wrong for the
  second pokemon in doubles).

### Known gaps documented, deliberately NOT changed (need a protocol bump / design call)
- No fog of war: enemy movesets, IVs, exact stats, nature, ability are fully
  exposed in the observation from wave start (`ability_revealed` is captured
  but unused). Masking to player-visible information would be a protocol-
  level change (retraining impact).
- LEARN_MOVE decisions don't encode the offered move's features (agent picks
  a replace slot blind); shop options 7-12 are buyable but unencoded
  (valid=0); shop items 13-14 (wave 171+, e.g. Sacred Ash) exceed the 12
  BUY actions; in singles, a 6-mon enemy trainer's 5th bench member doesn't
  fit the enemy bench slots (enemy_1 stays reserved for doubles).

---

## 2026-07-07 — Unify headless/rendered orchestration; bring the bridge to config parity

The two transports (cli.ts stdio loop, browser-bridge.ts WebSocket loop)
duplicated their orchestration — action labels, setup-phase set, reward
bookkeeping, invalid-action fallback, terminal-state handling — and the
duplication had already produced real divergence (the bridge's stale
`getEnemyName`, no rewards, a meaningless post-reset game_over state, no
game overrides). One implementation now lives in three shared modules:

### Files created
- **`src/rl/action-labels.ts`** — `buildActionLabels()` (the richer cli.ts
  version, merged with the bridge's extra phase labels). Used by both loops.
- **`src/rl/episode-runtime.ts`** — `SETUP_PHASES`, `resolveExecutedAction()`
  (mask-validated fallback), `buildTerminalGameState()` (patches the last
  decision state to terminal truth), `EpisodeRewardTracker` (RewardCalculator
  snapshots + fled/tier bookkeeping, verbatim from cli.ts).
- **`src/rl/apply-overrides.ts`** — browser-safe override applier (no node:fs);
  standalone-setup.ts now delegates to it.

### Behavior changes (bridge only — headless is bit-identical)
- Rendered episodes now report **rewards** (per-step + terminal) using the
  same tracker headless training uses.
- The bridge's `game_over` message now carries the last decision state
  patched to terminal truth instead of the post-reset (nondeterministic,
  meaningless) scene, plus `reward` and `wave`.
- Bridge `state` messages additionally carry `reward`, `wave`, `obsB64`,
  `mask` (TS-encoded, additive — Python tools may keep encoding locally;
  the encoders are parity-verified).
- New URL params matching the headless CLI's config surface:
  `&override=KEY=VALUE` (repeatable, applied pre-battle), `&rewardConfig=`
  (URL-encoded JSON), `&waves=N` (N*50-decision step budget → `done`).
- `run_policy.py --rendered` prints reward/wave when present.

### Verification
- Headless path proven behavior-identical: `rl-verify.sh quick` green
  (bitwise parity, determinism ×3), and the verify-q1 smoke episode is
  bit-for-bit the same trajectory and total reward (74 steps, 50.74) before
  and after the refactor.
- New **`tools/verify/check_mask_gating.py`** (V15, full mode): end-to-end
  regression guard for the shielded-boss ball gate and the Struggle
  fallback — both FAIL on the pre-audit mask code.
- Rendered loop compiles under the same typecheck (no new errors); the
  browser path could not be end-to-end tested headlessly here — the shared
  modules are exercised by the headless gates, the bridge-only glue
  (WebSocket handling, URL parsing) should be smoke-tested with
  `npx vite --config vite.interactive.config.ts` + `run_policy.py --rendered`.

---

## 2026-07-07 — Run configs (YAML), importable policy API, shop reveal-chain cancellation

### Run configs — `src/rl/run_config.py`
One YAML/JSON file describes a run (seed, waves, starters, starting
wave/level/money/biome, starting modifiers/held items, pokeballs,
battle_style, raw DefaultOverrides passthrough, reward weights, env
plumbing, free `train:` section). Consumed everywhere, CLI flags override:
`PokeRogueEnv.from_config()`, `run_policy.py --config`, `play.py --config`
(play.py also gained `--starters`), `train_maskable_ppo.py --config`
(reads `train.timesteps` / `train.save`). Sugar keys expand to the game's
own override hooks; raw `overrides:` wins on conflict with a warning;
unknown top-level/env keys are rejected, unknown reward keys warn.
`examples/rl/legendary.yaml`: ready-made full-legendary level-200 run.
pyyaml added to requirements-rl.txt (JSON configs work without it).
E2E-verified: seed/wave/money/level/starters/modifiers/pokeballs all land
in the live game state.

### Policy API — `src/rl/policy.py`
`Policy` protocol (`act(obs, mask, info) -> int`) with importable built-ins
(`RandomPolicy`, `FirstLegalPolicy`, `ScriptedSkipPolicy`, `MaxDamagePolicy`,
`Sb3Policy`) and `PhaseRoutedPolicy` for per-decision-phase dispatch (mix
learned battle nets with scripted shop/switch handling). run_policy.py now
uses these instead of its inline lambdas (maxdamage keeps the doubles
command_field_index fix). `examples/rl/phase_routed_policy.py`: runnable
bring-your-own-algorithm template incl. a Python-side custom reward
(gym.RewardWrapper over info["game_state"]). README gained "Run configs",
"Bring your own algorithm" and "Where the reward lives" sections.

### Shop reveal-chain cancellation (rendered-mode overlay bug)
ModifierSelectUiHandler.show() schedules an async reveal chain — a 1250ms
counter tween, a delayedCall for shop options, and (created asynchronously
after the counter resolves) a delayedCall(500) that fades in the button
containers (Reroll / Check Team / Transfer / lock-rarity / the continue
ARROW) — plus scene-level tweens for the shop overlay and luck text. An RL
agent acts faster than the chain, so the stale callbacks fired mid-battle:
the "random overlay arrow / luck / money / check team popping up" bug in
rendered mode. Fix (phase-router.ts): the show() patch now wraps
time.delayedCall + tweens.addCounter to CAPTURE the chain's work
(handler.__rlShopEphemera) and executeModifierAction /
executeModifierTargetAction cancel it FIRST — before the action's own
phase-end work schedules real game timers. cleanupModifierUI additionally
killTweensOf()s the shop chrome (overlay, luck texts, button containers)
so in-flight fades can't re-raise them. Headless-verified behavior-neutral
(quick suite green; verify-q1 smoke bit-identical: 74 steps, 50.74).

### Noted (upgraded priority for a future protocol bump)
The bench-capacity limitation hides the PLAYER's own 6th party member in
singles (1 active + 4 bench slots; player_1 is doubles-only): with a full
party, one member is switchable (action 16) but entirely unobserved. Cannot
be fixed by stuffing bench[4] into player_1 without corrupting the derived
active-matchup features — needs the obs-v9 slot redesign.

---

## 2026-07-07 — Rendered-mode catch timeout: root cause and fix (headless-Chromium repro harness)

User symptom: rendered runs died with "Timeout at step N" right after a Poké
Ball throw, screen frozen "dimmed very opaque". Reproduced end-to-end with a
headless-Chromium harness (playwright driving the real vite server + WS
relay + bridge; new-headless mode needed — headless-shell throttles rAF).

### Root cause (proven, not the earlier overlay theory)
On slow / software-rendered browsers the game loop runs at a few fps; a
successful catch → victory → shop-entry sequence that takes ~15s at 60fps
legitimately ran ~150s at 2fps. The router's FIXED 30s decision timeout
fired mid-sequence — the "very opaque dim" is the shop overlay fading in at
the moment the session froze. With a 300s timeout the identical seed's
catch completed normally, which pinpointed the timeout itself as the bug.

### Fixes
- **Progress-aware sliding timeout** (phase-router.ts): the timeout now
  measures time WITHOUT GAME PROGRESS (phase changes, pending timer events,
  running finite-duration tweens — infinite/paused sentinel tweens
  excluded), not wall time to the next decision. Real soft-locks (e.g. an
  undismissed error prompt) still time out after 30s of no activity; slow
  legitimate sequences no longer do. Verified: the previously-timing-out
  catch completes on the DEFAULT budget; headless quick suite green with a
  bit-identical smoke episode (74 steps, 50.74).
- **`&timeout=N` URL param / `advanceToNextDecision(timeoutMs?)`**: explicit
  no-progress budget override (escape hatch for waits the activity probe
  cannot observe, e.g. long promise-based asset loads).
- **Awaited SwitchPhase callback retry**: in the browser the party UI
  installs its select callback asynchronously; executing the switch
  immediately hit the blind-UI fallback and the still-open decision was
  re-detected (4+ redundant "switch" steps per battle start). The executor
  now awaits up to 1s for the callback (headless resolves on the first try
  — bit-identical). At extreme slowness one redundant step can remain
  (self-recovering).
- **vite.interactive.config.ts: don't file-watch assets/locales/dist** —
  with the assets submodule present, vite's watcher exhausted the Linux
  inotify budget and the dev server CRASHED mid-session
  (ENOSPC ... assets/images/pokemon/...). Static publicDir needs no watch.
- **`window.__rlDebug`** (bridge): live debug handle — `__rlDebug.state()`
  returns current phase / UI mode / handler flags; used by the repro
  harness and handy in devtools while bug hunting.

### Honest note on the earlier overlay fix
The shop reveal-chain capture (previous entry) addresses a real, probe-
confirmed stale-callback chain, but the user's reported "dim overlay"
turned out to be THIS timeout freeze, not that chain. The capture stays
because the delayedCall(500) chrome pop-in it cancels was directly
observed; if rendered sessions ever show chrome ghosts again, suspect a
stale browser tab first (hard-reload — vite serves live source).

---

## 2026-07-07 — Shop tween-chain crash on destroyed sprites; reveal-cancel rescope; relay staleness

User-reported crash with stack trace (rendered mode, after shopping):
`TypeError: Cannot read properties of undefined (reading 'sys') at
Sprite.setTexture / Tween.onComplete (modifier-select-ui-handler.ts:941)`,
followed by the shop no longer rendering.

- **Root cause**: ModifierOption.show() starts tween CHAINS with multi-second
  delays for upgraded rewards whose onStart/onComplete call setTexture /
  setPosition on this.pb / this.pbTint. Phaser tweens do NOT stop when their
  targets are destroyed, so a fast shop dismissal (RL pacing) let a chain
  fire during a later phase on torn-down sprites — crashing inside
  TweenManager.update and corrupting subsequent shop rendering. Fix at the
  source: ModifierOption.destroy() now killTweensOf() its whole sprite tree
  (extends the earlier revealTimer cancellation, same crash family).
- **Reveal-cancel rescope** (phase-router): cancelShopEphemera previously ran
  on EVERY modifier/modifier-target action — including ones that keep the
  shop UI alive (entering the two-step target flow, non-re-showing
  purchases), freezing the reveal mid-animation for the human watching. Now
  cancelled only on phase-leaving / re-showing paths (skip, reroll, reward
  commit, target commits).
- **WS relay staleness** (vite-ws-plugin): a page reload (e.g. vite
  hot-reload while a session runs) created a SECOND browser socket while the
  old one kept relaying — two games interleaving onto one Python client
  ("Game ready -> sending start" twice, step counter continuing across a
  restarted game). New connections now close/replace the previous socket of
  the same role, and stale sockets stop relaying. run_policy.py / play.py
  print a loud "browser session restarted — fresh episode" warning on a
  mid-session `ready`.
- **Verified**: headless-Chromium E2E (24 steps, 5 waves, buys + target
  flows + reroll) with ZERO page errors, shop renders at the decision
  (screenshot-checked); headless quick suite green, smoke episode
  bit-identical (74 steps, 50.74).

---

## 2026-07-07 — Ghost sprites (headless-only killAll), cinematic fast-forward, stable policy seeds

Three user reports from rendered bug hunting, each root-caused:

- **Ghost pokemon (two sprites stacked, ~wave 30)**: the shop show() patch
  ran `tweens.killAll()` + `time.removeAllEvents()` on shop entry. Safe in
  HEADLESS (battle anims resolved synchronously by then — and still done
  there), but in the BROWSER the previous battle's return/faint animations
  genuinely overlap the shop opening; killing them strands the onComplete
  that hides/removes the outgoing sprite. Now gated to headless
  (`inNodeEnv`); the browser relies on the targeted cleanup that has since
  landed (reveal-chain capture + ModifierOption.destroy tween kill).
- **Evolutions hang the game (rendered)**: the evolution cinematic is
  15-20s of clock-driven beats that stretch into minutes on slow browsers,
  plus an asset-load await with zero clock activity (invisible to the
  sliding timeout). Per "keep the logic, cut the animations": the bridge
  now TIME-WARPS the scene clocks (x50) while a cinematic phase is current
  (EvolutionPhase, EndEvolutionPhase, FormChangePhase, EggHatchPhase,
  EggSummaryPhase) and restores them after — the exact same game code runs
  (evolve(), learn-move queueing, dex updates), just compressed. Bridge
  only; headless untouched.
- **Same-seed runs diverging from wave 1**: RandomPolicy derived its RNG
  seed via Python's per-process-salted `hash(str)` (PYTHONHASHSEED), so
  "seeded" random runs picked different ACTIONS every launch — the game
  seed was fine. Now zlib.crc32 (stable across processes). Rendered
  determinism = seeded game + deterministic action sequence; with this fix
  two same-seed random runs replay identically (validated E2E: identical
  step/phase/wave/action/reward traces).

Ops note: RL test servers now run on a separate port from the user's dev
server (vite auto-increments; repro scripts must target their own port —
earlier probes on :8000 collided with the user's live session).


---

## 2026-07-07 — Shop renders instantly ("show everything at once, no animations")

User-suggested and strictly better than managing the reveal chain: the
shop's staggered reveal (reward counter tween -> per-option reveal timers
-> shop-row delayedCall -> button fades, 3-9s of presentation) is now
skipped entirely in RL mode.

- New game-side helpers: `ModifierOption.revealInstantly()` (cancel the
  option's pending reveal work; jump sprites to the exact final state) and
  `ModifierSelectUiHandler.revealAllInstantly()` (all rewards + shop rows +
  buttons, cursor target + awaitingActionInput/onActionInput armed —
  mirrors the chain's completion step).
- The RL show() patch runs the original show() inside a SYNCHRONOUS-window
  scheduler capture (`captureScheduledDuring`): the counter tween and the
  shop-row delayedCall are created synchronously and the entire async
  remainder spawns from them, so cancelling those two at birth kills the
  whole chain with zero foreign-capture risk. Then `revealAllInstantly()`.
- SUPERSEDED AND REMOVED (per the failed-fixes policy): the persistent
  ephemera-capture system (`captureShopEphemera`/`cancelShopEphemera`, the
  handler-attached `__rlShopEphemera`, and per-action `cancelShopReveal()`
  call-sites). It kept wrappers installed between show() and the next agent
  action; it prevented the chrome pop-ins but is the prime suspect for the
  "buy options don't render" report (reveal timers suppressed/cancelled in
  legitimate flows) and carried foreign-capture risk by design. With the
  chain cancelled at birth and the UI rendered instantly, there is nothing
  left for it to manage.
- Structurally closes: invisible shop/buy rows, mid-battle chrome pop-ins,
  and the destroyed-sprite tween-crash class (the chain never lives long
  enough to dangle; ModifierOption.destroy() hardening stays as a belt).

Also in this batch: rendered E2E harness (`tools/verify/check_rendered.py`,
opt-in V16 via RL_VERIFY_RENDERED=1 — self-hosts vite on a strict isolated
port), obs-audit Phase 1 tooling (`tools/verify/audit_obs_redundancy.py`)
and the audit plan (`src/rl/docs/OBS_AUDIT_PLAN.md`).

---

## 2026-07-07 — Instant evolution (logic-only), obs-audit Phases 1-2 delivered

- **Evolution cinematic REPLACED by its logic in rendered mode** (bridge
  patches EvolutionPhase.start): pokemon.evolve() + EVOLVE_MOVE
  LearnMovePhases + EndEvolutionPhase, extracted verbatim from
  handleSuccessEvolution/postEvolve. The temp display-pokemon asset load
  (a hang-forever hazard on broken assets) is skipped entirely; the real
  mon's sprite load is raced against 15s. Validated E2E: "Instant
  evolution: Caterpie -> Metapod", lv8, no errors, run continues.
  SUPERSEDES the x50 time-warp for EvolutionPhase (kept for
  FormChange/EggHatch): user reported a freeze under the warp, and the
  A/B/C validation showed a silent no-evolution — root cause of the
  latter was the test policy losing the battle (status move slot 0), but
  the logic-only path is strictly more robust than any cinematic pacing.
- Earlier user reports of "no evolution happened in my test" were the
  test's fault, not the game's — document for posterity: FirstLegal picks
  move slot 0; give the test mon a damaging slot-0 move (MOVESET_OVERRIDE)
  or it may never win a battle.
- **Obs audit Phase 1 (redundancy) delivered**: docs/AUDIT_FINDINGS_P1.md —
  only ~29% of 10,403 dims carry independent signal on a 3,769-decision
  interactive corpus; move blocks = 63% of the obs and majority-dead;
  85 tera one-hot dims duplicate type1. Tools: gen_audit_corpus.sh
  (run_episodes --cli-arg passthrough added), audit_obs_redundancy.py.
- **Phase 2 (sufficiency) delivered**: docs/SUFFICIENCY_MATRIX.md, fully
  code-grounded (accuracy/weather/multi-hit/AI/PP pipelines audited with
  citations). NEW encoder gaps found: enemy ai_type and multi_hit_type are
  serialized but never encoded; plus the known learn-move/6th-member/
  shop-tail gaps. v9 direction: ~3.5-4.5k dims with MORE content.

---

## 2026-07-08 — Observation protocol v9 (4 → 5): 10,403 → 6,991 dims

Evidence-based redesign per the Phase 1-2 audit (docs/OBS_V9_LAYOUT.md has
the full design; docs/AUDIT_FINDINGS_P1.md + SUFFICIENCY_MATRIX.md the
evidence). Approved layout decisions: fog-of-war as a toggle defaulting to
FULL info; move vector cut to the evidence keep-list; ONE fixed layout
(no layout toggles — escape hatch is ObservationWrapper over
info["game_state"], README example added; env now exposes `last_info`).

CUT (−3,732): per-move vector 136→60 (77 near-dead flags → one
has_other_effect catch-all; exact OR-list in the design doc); volatile
tags 76→69 (TURN_END-transient tags — structurally unobservable at
decision boundaries, now excluded in the coverage manifest with reasons).

ADDED (+320): per-pokemon ai_type_onehot(3) + move_known(4) +
ability_known + was_seen (revealed-indicators; constant-truthy under full
obs); multi_hit_count (replaces the is_multi_hit boolean — 2-5 hit count
was serialized but dropped); shop options encoded 6→12 (all buy actions
observable); field +8 Wish/Future-Sight dims per side; NEW learn_move
block (66): the OFFERED move as a compact move vector + learner
party-index (phase-router now writes learnMoveStats/learnMovePartyIndex —
previously never populated).

FIXED (0 dims): singles slot mapping — slot 1 = second active in doubles,
FIRST BENCH in singles; the 6th party member (both sides) was silently
invisible before v9.

FOG OF WAR (value-mask toggle, default OFF): PokeRogueEnv(fog_of_war=True)
/ --fog-of-war / &fog=1 zeroes enemy unseen moves (move_history-derived,
conservative: resets on re-summon), unrevealed abilities, nature_mults +
computed_stats, and never-seen bench members; indicators carry reveal
state. Parity verified bitwise in BOTH modes; dump records carry fogOfWar
so check_parity re-encodes faithfully.

Verification: quick suite fully green (99/99, coverage OK, fixture parity
OK incl. new ai_type default alignment, live parity OK) and the smoke
anchor SURVIVED — verify-q1 bit-identical at 74 steps / reward 50.74
(auto-mode reads mask/gameState, not the obs), proving the protocol bump
changed representation, not behavior. Goldens regenerated. Trained v8
checkpoints are invalidated by design (none exist yet).

Honest notes: the audit-tool mishap during implementation (a careless
multi-line edit deleted entries from _BATTLER_TAG_VALUES instead of
CURATED_VOLATILE_TAGS) was caught by the tag-parity check and reverted —
the lockstep gates work. state_schema.py's stale v7 constant fixed.

---

## 2026-07-08 — First trained agent (milestone)

MaskablePPO, 200k timesteps on protocol v9, 6 parallel envs, 53 min wall
on an 8-core box, default 64x64 net. Held-out eval (10 episodes, 20-wave
budget, seeds disjoint from training):

| policy | mean reward | median | mean wave |
|---|---|---|---|
| random | 6.8 | 7.1 | 6.7 |
| maxdamage (scripted heuristic) | 93.2 | 76.8 | 9.5 |
| **ppo_v9_first (200k)** | **300.7** | **311.0** | **23.7** |

3.2x the scripted heuristic's reward and 2.5x its depth (best episode:
wave 34) after less than an hour of training — the observation carries
learnable signal, the mask/reward/protocol/trainer/eval stack works end
to end. Next: horizon-extension curriculum at scale (docs/
TRAINING_SERVER.md).

---

## 2026-07-08 — Audit round 2: wave cap is real, truncation carries the true final obs

Self-directed audit (3 parallel reviews: env lifecycle, rendered path,
Python API). Confirmed and fixed:

- **--waves was a STEP cap (waves*50), not a wave cap** — the wave check
  existed only on the unused auto path. Training/eval episodes ran past
  the configured horizon (eval reached wave 34 under waves=20), and the
  curriculum docs' promises were false. Now: the episode ends as
  truncated at the FIRST DECISION of wave N+1 (plays through wave N);
  waves*50 stays as a safety backstop. Both transports.
- **Truncated episodes handed sb3 a ZERO final observation** — `done`
  carried only {steps}; the env zero-filled obs and reward, so the
  TimeLimit bootstrap used V(zeros) on every capped episode. Now `done`
  carries the final decision's obsB64/mask/reward/wave (+gameState when
  not lean) and the env consumes them. Verified E2E: waves=3 truncates at
  wave 4's first decision with a real obs and the wave-clear reward.
- **SMOKE ANCHOR RE-BASELINED (deliberate)**: verify-q1/waves=6 was
  74 steps / reward 50.74 (game_over — the loose cap let it run to a
  death at wave 7+). Under correct semantics: **49 steps / reward 101.83 /
  wave_cap truncation**. Representation unchanged (goldens untouched);
  SEMANTICS deliberately fixed.
- eval_policy: read info["victory"] (was a never-set key), record
  terminated/truncated, add win%/budget% columns.
- run_config: unknown train.* keys now warn (typos fell to defaults).
- CI: setup-python cache needed cache-dependency-path (first GitHub run
  failed there). stderr_log now truncates per spawn. Doc-drift fixes
  (observation.py header, README protocol line).
