# RL Environment Verification

What "verified" means for this stack, how to re-run every check, and the
latest results. Re-run the suite after ANY change to: `state-builder.ts`,
`spaces.ts`, `observation.py`, `cli.ts`, `phase-router.ts`, `rewards.ts`,
`ability-features.ts`, `modifier-features.ts`, `episode-runtime.ts`,
`modifier-api.ts`, `browser-bridge.ts`, `headless-boot.ts`, or the Python
lockstep files `enums.py` / `state_schema.py` / `feature_names.py`.

## Prerequisites

```bash
pnpm install
pnpm rl:build                      # builds dist/rl/cli.js (REQUIRED before node-side checks)
pip install -r requirements-rl.txt # installs the rl package (-e .): numpy, gymnasium, pyyaml
```

## Running

```bash
pnpm rl:verify        # full suite (V-1..V16), ~30-60 min, artifacts in .rl-verify/
pnpm rl:verify:quick  # 1 smoke seed, fewer waves; skips corpus (V14), dim-gate
                      # (V13b), mask gating (V15), bench/soak (V12/V12b)
```

## Check matrix

| ID  | What it verifies | Command | Pass criterion |
|-----|------------------|---------|----------------|
| V0  | Every observation.py parser assigns every dataclass field (static AST sweep — catches silent-default bugs like the is_mega/game-mode-flag gaps) | `python3 tools/verify/check_parser_completeness.py` | zero missing fields |
| V1  | Headless bundle builds | `pnpm rl:build` | exit 0, dist/rl/cli.js exists |
| V2  | Auto-mode episode completes | `node dist/rl/cli.js --seed=test --waves=3 --log` | exit 0, episode summary printed |
| V3  | Protocol conformance + episode completion under a masked-random policy | `python3 tools/verify/run_episodes.py --seeds 5 --waves 10 --probe-invalid 0.02 --dump-dir .rl-verify/smoke` | every episode reaches game_over/step-cap; zero hangs |
| V4  | Layout version guard | automatic at every wrapper reset | ready message protocolVersion/obsDim/actionDim == 5/6991/58 |
| V5a | TS↔Python encoder parity on hand-built fixtures (golden files) | `pnpm exec vitest run test/rl/spaces-encoding.test.ts` + `python3 tools/verify/fixture_parity.py` | bitwise-identical to goldens on both sides |
| V5b | TS↔Python parity on real gameplay states | `python3 tools/verify/check_parity.py ".rl-verify/**/*.jsonl"` | 0 mismatches at atol 1e-6 (observed: bitwise-identical), exact mask equality |
| V6  | Action-mask validity | part of V3 | zero unexpected `warning` messages (mask-approved actions never rejected); every `--probe-invalid` probe acknowledged |
| V7  | Determinism | `python3 tools/verify/check_determinism.py --seed S --waves 6 --mode auto` and `--mode interactive --action-seed 42` | both runs bitwise-identical per step (state hash + obs bytes + action + phase) |
| V8  | gymnasium API contract | `python3 -c "import sys; sys.path.insert(0,'src'); from gymnasium.utils.env_checker import check_env; from rl.pokerogue_env import PokeRogueEnv; check_env(PokeRogueEnv(waves=3))"` | no errors |
| V9  | MaskablePPO training smoke | `python3 examples/rl/train_maskable_ppo.py --timesteps 600 --waves 8` | completes without exception |
| V10 | Vectorized envs + no orphan processes | see `examples/rl/random_agent.py` with multiple envs / SubprocVecEnv | all envs step; `pgrep -f dist/rl/cli.js` empty after close |
| V11 | Unit + semantic test suite (rewards, golden encodings, 11 semantic audit files) | `pnpm exec vitest run test/rl/` | all pass |
| V12 | Throughput + soak | `python3 tools/verify/bench_throughput.py --episodes 3 --waves 8` / `--soak 30 --waves 3` | numbers recorded below; soak drift < 50% |
| V7b | In-process reset leakage gate: episode after `{"cmd":"reset"}` bitwise == fresh-process episode | `python3 tools/verify/check_determinism.py --mode inprocess --seed S --waves 6 --action-seed 42` | both comparisons bitwise-identical |
| V12b | In-process soak under the shipped wrapper policy (`respawn_every=50`) | `python3 tools/verify/bench_throughput.py --inprocess --soak 100 --waves 4 --recycle-every 50` | RSS + steps/s drift < 50% |
| V13a | Observation coverage: every game-side enumerable (96 battler tags, 214 move attrs, 109 modifier types, 20 move flags, 311 abilities) is encoded or excluded-with-reason | `python3 tools/verify/check_obs_coverage.py` | 0 UNREVIEWED |
| V13b | Dim exercise: every never-varying dim group classified (unexercised → names its corpus scenario / structurally-constant → code reason); scenario-ran-but-dim-stayed-flat = suspected bug | `python3 tools/verify/check_dim_exercise.py ".rl-verify/**/*.jsonl" --gate` | 0 UNREVIEWED, 0 suspected-bug |
| V14 | Deep-coverage corpus: scripted scenarios (doubles, trainer parties, megas, long runs, weather, full-party, status) each ASSERT their target situation occurred | `python3 tools/verify/gen_coverage_corpus.py` | all scenarios + asserts pass |
| V15 | Mask soft-lock guards end-to-end (shielded-boss ball gating + can_catch, PP-exhaustion Struggle, shop reroll-keeps-open/skip-exits) | `python3 tools/verify/check_mask_gating.py` | all scenarios pass |
| V16 | Rendered E2E (real vite dev server + WS relay + browser bridge under headless Chromium): determinism, evolution, shop, headless-vs-rendered reset-obs equivalence | `RL_VERIFY_RENDERED=1 python3 tools/verify/check_rendered.py` (opt-in; needs playwright + chromium) | all scenarios pass |
| V-1 | Python lint (pyflakes/bugbear/import order via ruff) | `python3 -m ruff check src/rl tools examples/rl` | zero findings |

Golden regeneration (ONLY after an intentional encoding/layout change):
`UPDATE_RL_GOLDEN=1 pnpm exec vitest run test/rl/spaces-encoding.test.ts`

## Historical results — 2026-06-10 (v8 era, protocol 2; kept for the audit trail)

> Current numbers: protocolVersion 5 / 6,991 dims, 105 tests / 20 files,
> ~95-105 steps/s interactive, ~140-150 steps/s in-process soak — see the
> nightly `rl-verify` CI runs for the living record.

| Check | Result | Notes |
|-------|--------|-------|
| V0 | PASS | 30 parser/dataclass pairs complete |
| V1 | PASS | builds in ~6s |
| V2 | PASS | ~80-90 steps/s auto mode |
| V3 | PASS | 5/5 episodes clean, ~60-105 steps each |
| V4 | PASS | protocolVersion 2 |
| V5a | PASS | 3 fixtures bitwise-identical TS↔Python |
| V5b | PASS | 700+ real records, bitwise-identical, masks exact |
| V6 | PASS | 0 unexpected warnings across all runs; all probes acknowledged |
| V7 | PASS | auto + interactive (incl. terminal record) |
| V8 | PASS | check_env clean |
| V9 | PASS | 600 timesteps MaskablePPO |
| V11 | PASS | 91 tests / 16 files |
| V12 | PASS | see .rl-verify/bench.json; ~15-25 steps/s interactive incl. Python encode, boot ~2-2.5s |

## Bugs found & fixed by this suite (2026-06-09/10)

Python encoder (observation.py) — all were silent parity breaks vs the TS reference:
1. 9 game-mode flags (`is_classic`…`inverse_battle`) never parsed → battle dims 31-39 always 0.
2. `status_effect`/`berry_type` dropped from reward/shop options → modifier-phase feature dims wrong for berries/status items.
3. `is_mega`/`is_max` never parsed (latent: triggers when megas appear).
4. `turn_data.move_effectiveness`/`hits_left`/`single_hit_damage_dealt` never parsed.
5. `stack_ratio` semantics: explicit `max_stack_count: 0` must encode 0 (TS), missing key must default 1 — Python did `max(ms,1)` for both.
6. `biome_id` missing the `biome_type` fallback the TS encoder has.
7. `is_grounded` parser default (True) disagreed with TS missing-key default (0).

Game-side (state-builder / phase-router / boot / game code):
8. `arena_tag_self_side` read the attr's `selfSideTarget` constructor param; the real side semantic is the move TARGET (Reflect/Light Screen reported false). Fixed to use `move.moveTarget`.
9. `WishAttr` + `SuppressAbilitiesIfActedAttr` missing from the game's `MoveAttrs` registry → `hasAttr()` silently false → `is_wish`/`suppresses_if_acted` never set.
10. Full-party capture hang: the CONFIRM→PARTY→options-submenu→goodbye-prompt flow needed a poll-driven state machine (was: 30s timeout + truncated episode whenever the agent caught with 6 mons).
11. i18n initialized AFTER game modules evaluated in headless boot (static imports in standalone-setup.ts) → trainer names undefined → process crash on every rival/gendered-trainer encounter.
12. TitlePhase presented a timing-dependent number of times (initBattle is async) → action-RNG desync → non-deterministic episodes. Now presented exactly once.
13. Terminal game_over state captured the post-reset NEXT battle (unseeded `time_of_day`/`offset_gym`/`seed`, cleared party) → non-deterministic terminal observation. Now reuses the last real decision state.
14. `process.exit(0)` discarded the buffered terminal message once it exceeded the pipe buffer → EOF without game_over/done. stdout now drained before exit.

Known asymmetries / documented non-bugs:
- Malformed mask lengths: TS returns all-false; Python pads/truncates. Unreachable with real states; pinned by tests on both sides.
- `turn_data.move_effectiveness` is transient by game design (nulled at MoveEffectPhase end) — the dim is ~always 0 at decision points; the derived type-effectiveness block is the usable signal.
- RewardCalculator revive quirk: faint → revive → faint again yields no second penalty (cumulative counter vs current-fainted-count comparison). Documented in test/rl/rewards.test.ts; fix would change reward semantics — deferred deliberately.

## Architecture changes — 2026-06-11 (protocolVersion 3)

1. **TS is the wire encoding authority**: `state`/`game_over` messages carry
   `obsB64` + `mask` + `wave`; `--lean` omits the bulky gameState JSON
   (wrapper default). observation.py remains the verified reference,
   kept honest by V5a/V5b on every run.
2. **In-process episode reset**: after `done` the CLI awaits
   `{"cmd":"reset","seed","waves"}` (fresh `ready` follows) or
   `{"cmd":"quit"}`/EOF. Reset latency ~3-10ms vs ~2s respawn. Wrapper
   default; `respawn=True` restores the old lifecycle.
3. **Reward config**: `--reward-config=<json|@path>` /
   `PokeRogueEnv(reward_config=...)`; echoed in `ready`.
4. **Game overrides**: `--override=KEY=VALUE` (DefaultOverrides keys) for
   scripted scenarios; unknown keys warn to stderr.

Bugs found & fixed by the V7b/V12b gates while building this:
- Terminal reward sampled the post-reset scene (spurious positive money delta
  whenever the run ended below starting money) — now uses the pre-action
  snapshot at terminal.
- `MockClock` leaked a permanent 1ms `setInterval` per episode/test (also
  affected the vitest suite); now destroyed on replacement.
- `patchModifierHandler` + the `showText` null-guard re-wrapped the previous
  wrapper every episode (stacked closures pinned every router ever created);
  both are idempotent now.
- `scene.field`/`scene.fieldUI` accumulated ~500 display children per episode
  (tween onComplete destroys never fire under mock tweens); a baseline-diff
  purge at each reset clears episode debris.
- `ScanIvsPhase` crashed on nameless mock sprites (`m.name.includes`);
  Memory Mushroom via the RL party-target flow built a modifier with
  `moveId=undefined` and crashed LearnMovePhase — both fixed.
- `process.exit(0)` could discard the buffered terminal message (>64KB pipe);
  stdout is drained before exit.

## Heap/throughput soak (2026-06-13)

The dominant cross-episode leak was root-caused and fixed. `MockContainer`
never set `this.scene`, and Phaser's `Container.destroy()` only cascades into
children for which `child.scene` is truthy (`removeAll` →
`if (list[i] && list[i].scene) list[i].destroy()`). So a real container
holding mock-container children (e.g. every `BattleInfo`'s
`statsContainer → statValuesContainer → stat sprites`) never destroyed them,
stranding ~575 sprites/episode on the process-global `AnimationManager`
`remove` event. Beyond memory, that listener array grew without bound and
turned per-step sprite create/destroy into O(n²) — the real throughput killer.

Fix (faithful-to-Phaser, no monkey-patching): `MockContainer` sets
`this.scene = textureManager.scene` (as `MockSprite` already did) and its
`destroy()` recurses into children. Plus: `MockTextureManager` dropped its
write-only object registry, and `BattleScene.reset()` clears the
`PokemonSpriteSparkleHandler` Set (its lazy `!sprite.scene` prune never runs
under mock tweens).

Pure 100-episode in-process soak (NO recycling), `--waves 4`:

| metric | before | after |
| --- | --- | --- |
| anim-listener growth | ~575 / episode | **~6 / episode** |
| steps/s (first→last 10-ep avg) | 102 → 26 (75% collapse, hit step-cap) | 72 → 59 (19%) |
| retained heap (post-GC linear fit) | unbounded | **1.08 MB/episode** |
| soak verdict | **FAIL** | **OK** |

KNOWN RESIDUALS — two distinct leaks remain, both headless-mock artifacts (the
game does NOT leak in the browser, where real tweens fire cleanup on schedule
and destroyed GameObjects detach from their parent via `parentContainer`):

1. PER-EPISODE: ~1.08 MB/episode retained heap from rexBBCodeText option-select
   menu texts (`text-option-select`) whose jsdom canvases are not returned to
   Phaser's global `CanvasPool` on `destroy()`. Bounded for EPISODIC training by
   `respawn_every=50` (caps growth at ~54MB, then a fresh process).

2. PER-WAVE / INTRA-EPISODE (UNSOLVED): `scene.field` accumulates ~61 LIVE
   (un-destroyed) unnamed sprites per wave (~1.5 MB RSS/wave). This is NOT
   bounded by `respawn_every` — that recycles between EPISODES, and a single
   long endless run (e.g. 200 floors) never resets, so it grows ~290MB over 200
   floors (survivable, not crash-level, but real). A `sweepDestroyedFieldChildren`
   attempt was tried and REVERTED: it only removes already-destroyed children,
   but these sprites are live, so it had zero effect. Source not yet confirmed
   (suspect: `addFieldSprite` sprites from summon/anim phases). Tracked for a
   proven fix; do not re-attempt without first confirming the source and
   measuring per-wave RSS. Only affects single very-long runs, not episodic
   training.

Diag for both: `RL_VERIFY_CLI_LOG=1 RL_VERIFY_EXPOSE_GC=1` + the per-reset
`reset diag` stderr line (includes `animListeners=` as a regression canary for
the throughput-leak class fixed above).

## Verifying by hand (play it yourself)

```bash
pnpm rl:build
python3 tools/play.py --waves=10 --seed=myrun     # headless TUI: full battle
                                                  # state, HP bars, movesets,
                                                  # numbered actions; q quits
# Rendered (watch the actual game in a browser):
npx vite --config vite.interactive.config.ts      # terminal 1
python3 tools/play.py --rendered                  # terminal 2
```
The TUI shows exactly what the agent sees (same protocol, non-lean) — if the
displayed state matches what the game does, the observation pipeline is doing
its job in front of your eyes.

## Hot-path profile (2026-06-11, `--profile`)

Per step at ~96 steps/s (lean): game simulation **80%** (execute 4.6ms +
advance 3.7ms — phases, damage calc, enemy AI), buildGameState 1.2ms (12%),
TS encode 0.3ms, send 0.2ms, everything else ~0. The RL layer costs ~2ms;
the rest is the game engine. Dead modules removed: runner.ts (854 lines),
standalone-runner.ts (53) — zero importers, never in the runtime path.
Remaining game-side levers (unexplored, diminishing returns): per-wave
session saves, MockText message simulation, phase-manager overhead.

## Adding a new check

Put scripts in `tools/verify/` (import `common.py` for subprocess/parity
helpers — it inserts `<repo>/src` on sys.path; never insert `src/rl`
directly), wire them into `scripts/rl-verify.sh`, and add a row here.
