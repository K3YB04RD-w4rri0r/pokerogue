# PokeRogue RL Environment — Play & Watch Guide

A reinforcement-learning environment wrapped around the PokeRogue battle game. The
*same* game logic runs three ways, all sharing one observation encoder, one action
space, and one reward function:

| Mode | What it is | Use for |
|---|---|---|
| **Headless** | `cli.js` under Node + jsdom, no rendering | training, CI, fast eval |
| **Interactive (terminal)** | the headless game driven turn-by-turn from a text TUI | debugging, playing by hand |
| **Rendered (browser)** | the real game in your browser, driven over a WebSocket | **watching** a policy play |

A *policy* is just `f(observation, action_mask) → action`. Because the observation
encoder (`observation.py`) and the 58-action space are identical in every mode, a
policy — random, the built-in max-damage heuristic, or a trained MaskablePPO
checkpoint — runs **unchanged** whether headless (fast) or in the browser (watchable).
Only the transport differs (stdio subprocess vs WebSocket).

---

## Quick start — watch a bot play

```bash
# 0. one-time: install JS deps and the Python WebSocket client
pnpm install
pip install websocket-client numpy

# 1. terminal A — serve the game + RL bridge (real browser, real rendering)
npx vite --config vite.interactive.config.ts        # http://localhost:8000

# 2. terminal B — drive it with the max-damage bot, opens your browser
python3 tools/run_policy.py --rendered --policy maxdamage \
    --starters MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS --delay 0.8
```

You'll watch a full legendary team fight through waves, picking its strongest move
each turn. Drop `--starters` for the default team, change `--delay` for pacing.

---

## Setup

One command from a fresh clone (checks toolchain, inits submodules,
builds, installs python deps, smoke-tests the round trip):

```bash
bash scripts/rl-setup.sh            # environment
bash scripts/rl-setup.sh --train    # + sb3/torch training extras
```

Prereqs: Node ≥ 24.9, Python ≥ 3.10, git. Details per use-case:

Library users can also `pip install -e .` from the repo root (extras:
`.[train]`, `.[rendered]`) and `from rl.pokerogue_env import PokeRogueEnv`
anywhere — set `POKEROGUE_RL_CLI=/path/to/dist/rl/cli.js` when running
outside the repo checkout.

| You want to… | Need |
|---|---|
| Play/bot **headless** or **terminal** | the setup script (or `pnpm install` + **`pnpm rl:build`**) |
| Play/bot **rendered** (browser) | the Vite interactive server (below) + `pip install websocket-client` |
| Run **`run_policy.py`** | the setup script (`sb3-contrib` only for `--model`) |
| **Train** | `bash scripts/rl-setup.sh --train` (see `docs/TRAINING_SERVER.md` for big boxes) |
| **Contribute** | `src/rl/CONTRIBUTING.md` — the invariants + the `rl-verify.sh quick` gate (also run by CI) |

`pnpm rl:build` = `vite build --config vite.headless.config.ts`. Rebuild it after
changing any TypeScript the headless runner uses. The **rendered** server serves
TypeScript source directly, so it picks up edits on browser reload — no rebuild.

---

## Ways to run

### 1. Watch a bot (browser) — `run_policy.py --rendered`
See Quick start. Terminal A runs the Vite interactive server; terminal B runs the
policy and opens the browser.

### 2. Play it yourself (browser) — `play.py --rendered`
```bash
npx vite --config vite.interactive.config.ts        # terminal A
python3 tools/play.py --rendered --port 8000         # terminal B
```
The browser renders; you choose each action from the terminal TUI (see
[Playing in the terminal](#playing-in-the-terminal)).

### 3. Play it yourself (terminal only) — `play.py`
```bash
pnpm rl:build                  # once
python3 tools/play.py          # spawns cli.js --interactive, text-only
```
No browser; the game state and legal actions are printed as text.

### 4. Run a bot headless (fast, no browser) — `run_policy.py`
```bash
python3 tools/run_policy.py --policy maxdamage --waves 10
```

### 5. Train an agent
The Gym env is `src/rl/pokerogue_env.py` (`PokeRogueEnv`, a `gymnasium.Env` that
spawns `cli.js --interactive`). A MaskablePPO stub lives at
`examples/rl/train_maskable_ppo.py`. A checkpoint trained this way drops straight
into `run_policy.py --model run.zip` (headless **or** `--rendered`).

---

## Policies (`run_policy.py`)

| Flag | Behaviour |
|---|---|
| `--policy random` | uniformly-random **legal** action (a baseline / fuzzer) |
| `--policy maxdamage` | highest base-power damaging move in battle; skips the shop and declines the battle-start switch; first legal action otherwise |
| `--model <path.zip>` | a trained sb3-contrib **MaskablePPO** checkpoint |

Common flags: `--rendered`, `--starters NAMES`, `--seed S`, `--delay SECONDS`
(rendered pacing), `--waves N` (headless), `--port` (rendered). `--policy random`
is intentionally a *fuzzer*: it drives legal-but-degenerate paths and is great for
finding bugs, but it's not fun to watch — use `maxdamage` (or a model) for that.

The rendered URL accepts the same config surface as the headless CLI:
`&override=KEY=VALUE` (repeatable game override), `&rewardConfig=<json>`
(partial RewardConfig), `&waves=N` (wave cap; ends at the first decision of wave N+1), alongside `&seed=`,
`&starters=`, `&delay=`. Rendered state/game_over messages carry the same
`reward`/`wave`/`obsB64`/`mask` fields the headless protocol has, so a rendered
episode reports the rewards headless training would.

---

## Run configs (YAML)

One file describes a run — seed, episode budget, party, starting
wave/level/money/items, game overrides, reward shaping — and every entry
point consumes it (CLI flags override file values):

```bash
python3 tools/run_policy.py --config examples/rl/legendary.yaml --rendered --policy maxdamage
python3 tools/play.py --config examples/rl/legendary.yaml
python3 examples/rl/train_maskable_ppo.py --config examples/rl/legendary.yaml
```
```python
env = PokeRogueEnv.from_config("examples/rl/legendary.yaml", waves=5)
```

Full schema in `src/rl/run_config.py` (module docstring); the sugar keys
(`starting_wave`, `starting_level`, `starting_money`, `starting_modifiers`,
`starting_held_items`, `pokeballs`, `battle_style`, ...) map onto the game's
own `DefaultOverrides` hooks, and a raw `overrides:` section reaches anything
else in `src/overrides.ts`. `examples/rl/legendary.yaml` is a ready-made
full-legendary bug-hunting run.

## Bring your own algorithm

A policy is anything with `act(obs, mask, info) -> int` — see `rl.policy`
(`RandomPolicy`, `MaxDamagePolicy`, `Sb3Policy`, ...).

> **New here? Start with [`docs/WRITING_AN_AGENT.md`](docs/WRITING_AN_AGENT.md)** —
> a from-scratch walkthrough (the contract, the mask, the phases, obs vs
> game_state, the run loop, and growing a heuristic into a learned policy),
> with a runnable `examples/rl/custom_agent.py` that beats the random baseline.

**Phase routing** mixes learned and scripted behavior per decision phase:

```python
from rl.policy import PhaseRoutedPolicy, Sb3Policy, ScriptedSkipPolicy
policy = PhaseRoutedPolicy(
    routes={"command": Sb3Policy("battle.zip"), "target": Sb3Policy("battle.zip")},
    default=ScriptedSkipPolicy(),          # scripted shop / switches / learn-move
)
action = policy.act(obs, env.action_masks(), info)   # info["phase"] routes
```

`examples/rl/phase_routed_policy.py` is a runnable template, including a
custom Python-side reward as a `gym.RewardWrapper` over `info["game_state"]`.

**Where the reward lives — two levels of control:**

1. **Reweight the built-in reward** (no code): `src/rl/rewards.ts`
   (`RewardCalculator`, 20 weighted components — reward v2, see
   `docs/REWARD_V2.md`) is the DEFAULT, a sane documented baseline — not
   "the one true reward." Retune any component's magnitude via the
   `reward:` section of a run config / `--reward-config`
   / `PokeRogueEnv(reward_config=...)`; set a weight to 0 to drop a
   component (unknown or removed keys are a hard error). Both transports
   report it per step. Note the default's editorial choices (all
   overridable): money gains pay on a CUMULATIVE log scale (a bounded
   tiebreaker; spending is priced per-delta), enemy damage pays only
   new-low HP per enemy, a faint/KO is counted by BOTH its HP-delta and
   the ±KO event (dense + sparse shaping), stalling is priced at the
   stalling steps, and only the once-per-shop reward *pick* scores, not
   repeated buys.

2. **Write your own reward in Python** (`src/rl/reward.py`): implement a
   `RewardFn` over the full game state and wrap the env with
   `CustomReward` — it REPLACES the built-in reward, so none of the
   default's choices above are load-bearing. Compose weighted components
   or write from scratch:

   ```python
   from rl.reward import CustomReward, ComponentReward, delta_component, wave, money
   reward = ComponentReward({
       "depth":  (10.0,  delta_component(wave)),    # +10 per wave cleared
       "wealth": (0.0005, delta_component(money)),  # YOUR money scale
   })
   env = CustomReward(PokeRogueEnv.from_config(cfg, lean=False), reward)
   ```

   The module documents the state accessors (`wave`, `money`,
   `player_hp_fraction`, `player_faints`, …); the full `game_state` schema
   is in `INPUT.md`. `examples/rl/custom_reward.py` is a runnable template.

**Bring your own features:** the v9 observation is a curated 6,991-dim
vector (`src/rl/docs/OBS_V9_LAYOUT.md`), but `info["game_state"]` carries
the COMPLETE serialized state every step (with `lean=False`) — including
every field the encoder cut (all 100+ per-move effect flags, exact IVs,
full move history, …). Appending a custom feature needs no protocol
change, no TypeScript, and breaks nobody else:

```python
import gymnasium as gym
import numpy as np

class WithSelfSwitchFlags(gym.ObservationWrapper):
    """Append 4 dims: does player_0's move i self-switch (U-turn etc.)?"""
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n + 4,), np.float32)

    def observation(self, obs):
        gs = self.env.unwrapped.last_info.get("game_state") or {}
        moves = (gs.get("player_0") or {}).get("moves") or []
        extra = [1.0 if (m or {}).get("self_switch") else 0.0 for m in (moves + [{}] * 4)[:4]]
        return np.concatenate([obs, np.asarray(extra, np.float32)])
```

**Fog of war (optional):** `PokeRogueEnv(fog_of_war=True)` (or
`--fog-of-war` / `&fog=1`) masks enemy private info to what a human could
know — unseen moves, unrevealed abilities, IV/nature-derived stats and
never-seen bench members are zeroed, and per-enemy `move_known` /
`ability_known` / `was_seen` indicator dims carry the reveal state.
Default OFF: full information (the right baseline to train first; fog is
a controlled experiment on top).

## Custom starting team

Set the starting party to any species, in all three modes, using **SpeciesId names**:

```bash
# headless CLI
node dist/rl/cli.js --starters=MEWTWO,LUGIA,RAYQUAZA --waves=5
# rendered URL
http://localhost:8000/?rl=true&starters=MEWTWO,LUGIA,RAYQUAZA
# Gym env
PokeRogueEnv(starters="MEWTWO,LUGIA,RAYQUAZA")
# run_policy (either mode)
python3 tools/run_policy.py --starters MEWTWO,LUGIA,RAYQUAZA ...
```

Names are case-insensitive and parsed by `parseStarterCsv` (phase-router.ts);
unknown names are ignored. Omit it for the default daily-run starters.

---

## The action space

The agent always sees a **fixed 58-slot discrete action space** plus a per-state
boolean **mask** of which slots are legal *right now*. Each action **id** means
exactly one thing, always — what changes between phases is only the mask.

| ids | meaning | active in |
|---|---|---|
| `0–3` | use move slot 0–3 → enemy 0 | command |
| `4–7` / `8–11` | move 0–3 → enemy 1 / ally | command (doubles) |
| `12–16` | switch to party slot 1–5 | command / switch |
| `17–21` | throw poké-ball type 0–4 | command |
| `22` | run | command |
| `23–34` | **Tera + move** (enemy0 / enemy1 / ally) | command |
| `35–37` | pick reward 0–2 | shop |
| `38` / `39` | reroll / skip | shop |
| `40–51` | buy shop item 0–11 | shop |
| `52–57` | apply modifier to party slot 0–5 | modifier-target |

Command actions (0–34) and shop actions (35–57) occupy **disjoint ranges**, so an
id is never ambiguous: `38` is always "reroll", `0` is always "move slot 0 vs enemy".

**How the policy "understands" an action.** It doesn't symbolically — each action is
*positional* ("move slot 0", "reward slot 0"), and the **observation describes that
slot** (move power/type/effectiveness/flags; the reward's feature vector). So the
agent learns *"pick the slot whose features are good"*, which generalises across
movesets and shops.

### Why "Tera + move" is one action

Terastallizing isn't a standalone toggle — like Mega/Dynamax it's committed *with*
your move on that turn. So `23–34` mean "this turn, Tera **and** use move N". It's a
**one-time** decision (`MAX_TERAS_PER_ARENA = 1`, reset at each party heal): after
you Tera once, the Pokémon stays terastallized for the battle, so your later attacks
use the **normal** move actions `0–3` (already Tera-boosted) and `23–34` are masked
off. The Tera **Shard** item only sets *which type* you'll Tera into — it does not
terastallize on its own.

> Note: in the terminal TUI you type the **action id** directly (e.g. `26`), so it's
> stable across phases. In the rendered display the bridge also prints a readable
> label next to each.

---

## Playing in the terminal

`play.py` (with or without `--rendered`) prints the current field, your party, and
the **legal action ids** with readable labels:

```
Available Actions (type the id):
   0  Fight: Confusion (pow 50) -> enemy
   1  Fight: Psystrike (pow 100) -> enemy
  12  Switch to: Lugia
  17  Throw: Poké Ball
  22  Run away
```

Type the **id** of the action you want. Inspection commands are available at the
prompt — type `h` for the list (inspect modifiers, field, party, etc.). `q` quits.

---

## Developer reference

- **Observation:** 6,991 float32 dims (protocolVersion 6 / obs v9 — v6 changed
  the wire REWARD semantics, not the layout). The TypeScript
  encoder (`spaces.ts`) is the wire authority; `observation.py` mirrors it bitwise.
  Layout reference: `docs/OBS_V9_LAYOUT.md`; audit trail:
  `docs/AUDIT_FINDINGS_P1.md`, `docs/SUFFICIENCY_MATRIX.md`.
- **Reward:** `rewards.ts` (reward v2: 18 base + 2 shaped components,
  exploit-hardened — `docs/REWARD_V2.md`).
- **Verify everything:** `pnpm rl:verify` — build, TS+Python parity (bitwise),
  determinism, a deep-coverage corpus, and an in-process soak. See
  `docs/VERIFICATION.md`.
- **History:** `CHANGELOG.md` (design decisions, bug post-mortems, eval runs).
- **Key files:** `cli.ts` (headless runner + interactive protocol),
  `browser-bridge.ts` + `vite-ws-plugin.ts` (rendered transport),
  `phase-router.ts` (decision detection, action mask, action execution),
  `state-builder.ts` (game-state JSON), `pokerogue_env.py` (Gym env),
  `tools/play.py` / `tools/run_policy.py` (terminal players).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `--rendered`: "Failed to connect" | start `npx vite --config vite.interactive.config.ts` first; check `--port` |
| `--rendered`: `websocket-client required` | `pip install websocket-client` |
| headless: `cli.js` not found | run `pnpm rl:build` |
| browser shows Tera at wave 1 / stale behaviour | hard-reload the tab (Ctrl+Shift+R) so Vite serves the latest source |
| obs dim / protocol mismatch | rebuild (`pnpm rl:build`); TS and Python must agree on protocolVersion |
