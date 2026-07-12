# RL Framework Verification Campaign — 2026-07-11/12

Full software-engineering audit and hardening of the `rl-framework` branch:
baseline gate ladder → 10-dimension multi-agent deep review (adversarially
cross-checked) → 16 fix commits → full gate ladder re-run. This document is
the campaign record; the per-check reference lives in `VERIFICATION.md`.

## Verdict

The branch's core is sound — the observation/reward/action pipeline, the
TS↔Python mirror (all static tables programmatically diffed: zero
differences), the gymnasium contract, and the verify suite's anti-vacuity
all checked out under adversarial review. The campaign found and fixed one
deep pre-existing correctness defect (load-sensitive determinism flake),
completed three red mechanical gates (lint/types/architecture), hardened
the transports and the shared test-harness shims, and reconciled every
stale documentation claim.

## Gate matrix (baseline → final)

| Gate | Baseline | Final |
|------|----------|-------|
| biome-ci | ❌ 21 errors | ✅ 0 |
| tsc --noEmit | ❌ 42 errors | ✅ 0 |
| depcruise | ❌ 2 violations | ✅ 0 |
| rl:build | ✅ | ✅ |
| vitest test/rl | ✅ 99 tests | ✅ 105 tests (+6 shim-contract) |
| full vitest | ✅ 4,287 tests | ✅ 4,293 tests |
| rl-verify quick | ✅ | ✅ (superseded by full) |
| rl-verify full | ✅ (lucky seed — see below) | ✅ |
| in-process determinism, loaded machine, adversarial seed | ❌ ~50–80% failure rate | ✅ 20/20 trials bitwise identical |
| rendered E2E | ❌ wedged mid-scenario | (final result in gates log) |
| ruff (new gate) | — | ✅ 0 findings |
| interactive throughput | ~63 steps/s (load-adjusted) | ~67 steps/s |

## The headline finding: wall-clock determinism escape

The branch promises bitwise same-seed reproducibility, and its determinism
gate passed at baseline — but only by seed luck. Under machine load, an
adversarial seed (`det-flake-1`) failed the in-process gate in ~5 of 6
trials on the ORIGINAL build (A/B-verified against the campaign's changes).
Mechanism: headless time is a MockClock driven by a REAL 1 ms `setInterval`;
delayedCall-driven phase transitions (shop dismissal/reveal chains) therefore
completed on wall-clock cadence, and the decision loop could catch a phase
mid-transition in one run and post-transition in another — extra skip-only
shop decisions, diverging trajectories, and the same failure family as a
rendered-mode session observed wedging at a shop step. Three-layer fix
(commits `eb73e2a7f6e`, `3d01b74f1e4`):

1. shop-livelock counters tick/reset only on EXECUTED decisions, using the
   mask-build-time progress signature (mask building is side-effect-free);
2. `drainMockTimers()` pumps the mock timer queue TO EMPTY at every decision
   boundary (detect / hook resolution / post-execute);
3. RL headless sets `__rlDeterministicClock`: MockClock skips the real
   interval entirely, and waiting loops poll at 2 ms — wall cadence affects
   latency only, never outcomes.

Also closed: `trainerId`/`secretId` were `Math.random` per episode and gate
every shiny roll (obs bit + RNG-stream shift + luck→shop tiers) — now
seed-derived (FNV-1a) in both transports; the evil-grunt gender roll (reaches
`trainer_name` → state hash at waves 35+) is now seeded like the double roll.

## Fix commits (16, on top of `d87f998fe47`)

See `git log d87f998fe47..` — one commit per area: lint, types ×2, mock-shim
completion (+contract tests), verify tooling, Python tools, env robustness,
router correctness, transport, encoding/self-play parity, determinism,
CI+scripts+packaging, models, docs, perf.

Highlights beyond the headline:
- **Self-play observation bugs** (parity-invisible: both encoders agreed on
  the wrong answer): `enemy_command` missing from both phase one-hot maps;
  arena/positional tag sides unswapped in the enemy view (74 dims); `ai_type`
  positively asserting RANDOM for unknown controllers; fog+enemy-perspective
  producing a fully-blind opponent side (now warned, pending player-side
  seen-tracking).
- **Transport**: mid-episode `{"cmd":"reset"}` executed as action 0
  (systematic one-step desync in the buffered case) — now aborts cleanly and
  takes the fast in-process reset path (verified live); `play.py` stderr-PIPE
  deadlock at 64 KB; unguarded env stdin writes killing SB3 workers; strict
  version handshake on all clients; WS relay mid-step handover no longer
  mutually deadlocks; WS URL derived from `location`.
- **Router**: enemy-command one-shot delivery could park the phase forever on
  an empty mask; enemy turns ignored skipTurn/move queues and bypassed the
  Struggle substitution; rendered victory could report as loss (poll racing
  `GameOverPhase.end()`); stale-decision guards on five mask builders;
  challenge-legality in switch masks.
- **Test infra**: the shim now implements the full MockInstance surface tests
  use, installs stamped recording getter spies (closing a worker-global
  property-poisoning interleave with `vi.spyOn` + `restoreMocks`), records
  throw results, and has a contract test pinning vitest interop.
- **CI**: path filters watched a nonexistent file; the branch had NO standard
  CI at all (tests/linting now trigger on it); nightly cron documented as
  inert until the workflow exists on the default branch; setup-deps composite;
  `pip install -e .` smoke; ruff gate (V-1).

## Accepted/intentional (flagged, not changed — owner decisions)

- Player-facing changes stay: Game Speed default 5, Tutorials off,
  `MYSTERY_ENCOUNTER_RATE_OVERRIDE: 0`, seeded battle RNG (note: the seeded
  grunt rolls make all same-run grunt encounters share one double/gender
  outcome — a within-run distribution change vs vanilla's independent rolls).
- Checkpoints stay in git (see `models/README.md` for the policy).
- `ruff format` is deliberately NOT gated (would split the encoder's
  one-dim-per-line parity idiom); `ruff check` is.

## Phase 2 — RL-methodology audit (2026-07-12)

A second adversarially-verified review wave targeted the LEARNING layer
(MDP semantics, reward design, observation quality for training, and
training/eval methodology) — 43 unique findings, every sampled verifier
verdict confirmed. Full ledger: campaign artifacts. The load-bearing ones:

**Reward exploits (CRIT, confirmed):**
- *Flee-wave farming*: `waveCleared` (+10) pays on ANY waveIndex advance —
  including a successful RUN — while the flee penalty is only −2: net
  ≈ +7.99 per fled wild wave, risk-free. A dense-shaping PPO run can lock
  onto flee-spam as a local optimum, and short-wave curriculum stages
  truncate before the underleveling cost materializes. FIXED: wave rewards
  are now gated on not-fled (see below).
- *Stalling beats losing*: truncation carries no terminal penalty while a
  loss costs −50, so an agent about to lose prefers any truncation path
  (~−0.4 in step penalties). Left as a DESIGN decision (wave-cap truncation
  is the intended curriculum outcome and must stay penalty-free) — owners
  should consider a penalty on livelock-truncation specifically.

**Reward accounting bugs (MAJ, confirmed):**
- *Terminal shaping dropped*: at game over the post-snapshot was the
  pre-action snapshot itself, so every delta component of the final
  transition (the killing blow / the losing hit) was exactly zero, and
  terminal semantics differed from wave-cap truncation semantics. FIXED via
  a pre-reset snapshot hook on GameOverPhase.
- *Slot-indexed HP deltas*: HP diffs compared party slots by index, but
  switches physically reorder slots — a same-wave switch between mons with
  unequal HP fabricated damage-dealt/taken reward. FIXED: deltas now match
  by pokemon id.
- *Money reward non-stationarity*: `moneyGained` (0.01/unit) against a
  super-linear money curve means a wave-50 Relic Gold pick ≈ +77 reward —
  dwarfing combat shaping. FLAGGED (weight redesign is an owner decision).
- *Self-play reward perspective*: in --enemy-controlled mode, enemy-decision
  steps receive the PLAYER-perspective reward stream. FLAGGED prominently:
  self-play reward semantics are undefined until an enemy-perspective reward
  is designed; the mode is fine for data generation, wrong for training P2.

**Observation gaps for learning (MAJ, confirmed — design decisions, flagged):**
- 155 of 311 abilities share the identical default feature vector and
  ability_id is not otherwise encoded → half of all abilities unobservable.
- Level/computed-stat/wave normalizations saturate late-game (level/100
  clamps at cap ≈104 by wave 120).
- `speed_rank` ignores paralysis/Tailwind/Trick Room — asserting the wrong
  turn order exactly when it matters.
- Volatile-tag DURATIONS are dropped to 0/1 presence (Perish Song count,
  Encore/Taunt remaining turns) — a genuine non-Markov gap, undocumented.
- TM purchases are blind: the taught move's identity is serialized but not
  encoded in the offer features.

**Training/eval methodology (MAJ, confirmed):**
- The documented headline eval (300.7 mean reward / wave 23.7) was measured
  under two since-fixed semantics bugs and never re-measured. Re-measured
  this campaign on the fixed engine + fixed reward semantics (10 held-out
  episodes, 20-wave budget, ±SEM):

  | policy | mean reward | median | mean wave | full-budget% |
  |--------|------------|--------|-----------|--------------|
  | random | −17.76 ± 6.72 | −20.61 | 6.4 | 0% |
  | maxdamage | 88.94 ± 14.89 | 78.08 | 9.6 | 0% |
  | ppo_v9_first.zip | **272.46 ± 18.86** | 277.54 | **19.8** | 50% |

  The checkpoint decisively beats both baselines — the framework trains
  agents that genuinely learn. A fresh 600-step training smoke also passes
  (seeded learner, monitored, ep_rew_mean rising within the smoke).
- Training was learner-unseeded (torch init/sampling) → irreproducible;
  SIGTERM lost the entire run (no save); no Monitor wrapper → no learning
  curve; gamma/lr/entropy not configurable; default 64×64 MLP on 6,991 dims.
  FIXED: train.seed + salvage-on-interrupt + VecMonitor + exposed knobs
  (sb3 defaults unchanged).
- eval_policy: protocol-error exclusion is policy-dependent censoring (now
  reports excluded seeds), livelock truncations were counted as "budget"
  outcomes (now a separate column), no dispersion (now ±SEM), win% is
  structurally 0 under wave budgets < 200 (now annotated). maxdamage
  baseline ignores type effectiveness/accuracy and declines every shop —
  FLAGGED as a weak baseline.

## Phase 2 addenda (found while validating)

- **Rendered runtime overrides were silently inert**: the applier patched a
  dynamically-imported module instance that, under the Vite dev server, can
  differ from the instance game code reads. Fixed (the bridge passes its
  game-graph instance); this also explains the committed
  MYSTERY_ENCOUNTER_RATE_OVERRIDE belt-and-suspenders.
- **Starter moveset ORDER diverges headless-vs-rendered for the same seed**
  (observed: slot 0 = Tackle headless, String Shot rendered, seed
  verify-rendered-evo with STARTING_LEVEL_OVERRIDE=6). The reset-observation
  equivalence check passes, so the divergence is in a post-reset generation
  draw. FLAGGED — root cause not yet isolated; scenario made
  moveset-independent meanwhile.
- **Pick-cancel modifier farming** reproduced live by a first-legal policy
  (+0.49 per bounce) — fixed via applied-gated modifier bonus.
- The rendered gate's wedge diagnosis also hardened the harness itself:
  port-poll startup (the "Local:" stdout scan could block forever), relay +
  browser-console evidence capture, deterministic evolution scenario.

## Deferred roadmap (not this campaign)

- Player-side `seenPlayerPartyMemberIds` so fog-of-war works under the enemy
  perspective (cli warns today).
- Generating the Python encoder/schema from the TS source (or a shared spec)
  — the correct end-state for the dual-encoder maintenance hazard.
- Full internal retyping of `state-builder.ts` (48 verified-but-unchecked
  `as any` casts onto private game internals; highest-risk on refactor:
  `LapsingPersistentModifier.battleCount`, `BattleScene.enemyModifiers`).
- Rendered E2E in CI (needs playwright + chromium + xvfb; nightly only).
- Offer-slot `status_effect` emission (feature dims 10/17-19 structurally
  zero for shop/reward options — both encoders agree, documented).
- Wave-35+ (grunt-wave) seed in the determinism gate's default matrix, and a
  `--waves 40` variant in CI.
