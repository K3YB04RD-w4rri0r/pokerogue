# Contributing to the PokeRogue RL environment

Welcome! This directory turns PokeRogue into a reinforcement-learning
environment: a headless game, a Gymnasium API, a browser-rendered twin,
and a verification harness that keeps them all honest. This guide is what
you need to change it safely.

## Setup

```bash
bash scripts/rl-setup.sh          # env + build + smoke checks
bash scripts/rl-setup.sh --train  # + sb3/torch training extras
```

## The one rule: run the gate

```bash
bash scripts/rl-verify.sh quick
```

Everything must be green before a PR. It runs the RL test suite (99
tests), rebuilds, checks TypeScript↔Python **bitwise** encoder parity
against golden fixtures, replays a deterministic smoke episode, and
verifies action-mask gating. CI runs the same gate.

## The invariants (why the gate is strict)

1. **`spaces.ts` is the wire authority; `observation.py` is its bitwise
   mirror.** Every encoder change lands in BOTH, byte-for-byte. The
   golden fixtures (`test/rl/fixtures/*.golden.b64`) pin the encoding;
   `tools/verify/fixture_parity.py` and `check_parity.py` enforce parity.
2. **The layout canary** (`test/rl/semantic/layout-canary.test.ts` +
   `obs-layout.ts`) hard-asserts every block's absolute offsets. If you
   move a dim, the canary must be updated deliberately — that is the
   point.
3. **The smoke anchor**: seed `verify-q1` must replay bit-identically
   (74 steps, reward 50.74). It proves behavior didn't change when only
   representation should have. If your change legitimately alters game
   behavior, say so in the PR and update the anchor consciously.
4. **Coverage manifests** (`tools/verify/coverage-manifests/`): every
   game mechanic that is deliberately NOT encoded carries a written
   reason. Adding a battler tag / move attr / modifier to the game side
   without classifying it fails the gate — by design.

## Changing the observation (protocol bump checklist)

Any dim added, removed, or moved = a protocol version bump. The full
worked example is `docs/OBS_V9_LAYOUT.md` (v8→v9). Lockstep order:

1. Design doc in `src/rl/docs/` with exact dim arithmetic (sums must
   check).
2. `spaces.ts` constants + encoders → 3. `observation.py` mirror →
4. `feature_names.py` (names + `BLOCK_RANGES`; its asserts pin totals) →
5. `enums.py` if tag/order lists changed → 6. `obs-layout.ts` +
   `layout-canary.test.ts` offsets → 7. bump `protocolVersion` (cli.ts)
   and `PROTOCOL_VERSION` (pokerogue_env.py) together →
8. regenerate goldens: `UPDATE_RL_GOLDEN=1 pnpm exec vitest run
   test/rl/spaces-encoding.test.ts` → 9. `rl-verify.sh quick` fully
   green → 10. update `INPUT.md` + `CHANGELOG.md`.

Cheap alternative to a bump: cut dims are still serialized in
`info["game_state"]` — a Python `ObservationWrapper` can add any feature
without touching the protocol (see README "Bring your own features").

## Repo map (RL surface)

| Path | What it is |
|---|---|
| `src/rl/phase-router.ts` | decision detection, action masks, action execution (game-logic-faithful) |
| `src/rl/state-builder.ts` | serializes the COMPLETE game state dict |
| `src/rl/spaces.ts` / `observation.py` | dict → 6,991-dim float32 (wire authority / bitwise mirror) |
| `src/rl/cli.ts` | headless runner (jsdom + mock Phaser, JSON-lines stdio) |
| `src/rl/browser-bridge.ts` + `vite-ws-plugin.ts` | rendered twin (same modules, WebSocket relay) |
| `src/rl/pokerogue_env.py` | Gymnasium env (in-process resets, masks, `last_info`) |
| `src/rl/policy.py`, `run_config.py` | policy protocol + YAML run configs |
| `tools/` | run_policy, play, eval_policy, verify/* gates |
| `examples/rl/` | training stub (vectorized), phase-routed policy, configs |
| `src/rl/docs/` | design docs, audits, verification guide, server guide |

## House policies

- **Failed fixes come out.** If an attempted fix doesn't work, remove it
  rather than leaving dead code — and document the attempt in
  `CHANGELOG.md` so the next person doesn't retry it. Negative results
  are recorded, not erased.
- **Honest changelog.** Root causes, wrong turns, and superseded
  approaches are written down (see CHANGELOG for the house style).
- **Game logic is sacred.** The mask/executor must mirror the game's own
  rules (`handleCommand`, `checkCanUseBall`, ...) — never approximate
  them. When the game updates upstream, the E2E gates
  (`tools/verify/check_mask_gating.py`, `check_rendered.py`) catch
  drift.
- **Rendered = headless.** Both transports share phase-router /
  state-builder / encoders. Never fork logic per transport; fix shared
  modules.

## Testing tiers

| Command | When |
|---|---|
| `pnpm exec vitest run test/rl` | fast iteration |
| `bash scripts/rl-verify.sh quick` | before every PR (CI parity) |
| `bash scripts/rl-verify.sh` | full: soak, corpus, dim-exercise, bench |
| `RL_VERIFY_RENDERED=1 bash scripts/rl-verify.sh` | + browser E2E (needs playwright) |
