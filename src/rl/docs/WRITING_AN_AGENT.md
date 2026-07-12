# Writing your own agent from scratch

This guide takes you from nothing to a working agent that plays PokeRogue and
beats the random baseline — then points at how to grow it into a learned policy.
The companion runnable is [`examples/rl/custom_agent.py`](../../../examples/rl/custom_agent.py)
(a phase-aware greedy attacker); everything here is grounded in that file.

Prerequisite: a built headless bundle — `pnpm rl:build` (see the top-level
[README](../README.md) for setup).

---

## 1. The whole contract in one line

An agent is a function:

```python
act(obs, mask, info) -> int      # return one of the 58 action ids
```

- **`obs`** — a `float32` NumPy vector, `OBSERVATION_DIM` (6991) dims. The
  normalized, curated view of the game. *Always present, in every mode.*
- **`mask`** — a `bool` NumPy vector, `ACTION_SPACE_SIZE` (58) long. `mask[i]`
  is True iff action `i` is legal *right now*.
- **`info`** — a dict with `phase`, `wave`, `game_state` (see §4), and more.

The action space is **fixed at 58 slots**. Each id means exactly one thing,
always; what changes between decisions is only *which ids are legal* (the mask).
So you never parse a variable action list — you pick an index into a constant
space and let the mask tell you what's allowed.

That's the entire interface. No base class, no registration — any object (or
closure) with an `act` method is an agent.

---

## 2. The two rules you cannot break

**Rule 1 — never return an illegal action.** If you return `i` where
`mask[i]` is False, the environment rejects it and silently executes a
*different* action (the first legal one) in its place. Your agent then gets
credit/blame for an action it didn't choose — corrupt training signal. The env
counts these in `info["invalid_action_count"]`; a correct agent never triggers it (the key only appears once nonzero).
Always intersect your choice with the mask.

**Rule 2 — branch on `info["phase"]`.** The same run visits battle commands,
shop screens, forced switches, move-learning, and more. A move-selection id
(`0`) is meaningless on a shop screen. Look at the phase first, then decide.

The minimum viable agent obeys both rules in three lines:

```python
import numpy as np

class RandomLegal:
    def act(self, obs, mask, info):
        legal = np.flatnonzero(mask)          # ids where mask is True
        return int(np.random.choice(legal))   # never returns an illegal id
```

That already runs end-to-end (it's the built-in `RandomPolicy`). Everything
past here is about choosing *better* among the legal ids.

---

## 3. A real agent, walked through

`examples/rl/custom_agent.py` defines `GreedyAttacker`. It clears the early game
comfortably (reaches the wave cap in ~6 steps/wave with 0 invalid actions),
where random stalls or dies around wave 5-7. The shape:

```python
class GreedyAttacker:
    SKIP_PHASES = frozenset({"check_switch", "learn_move"})  # id 39 = decline

    def act(self, obs, mask, info):
        phase = info.get("phase")

        if phase == "command":                              # in battle
            move = self._best_damaging_move(mask, info)
            return move if move is not None else self._first_legal(mask)

        if phase == "modifier":                             # the shop: grab a free reward
            return self._first_in(mask, range(35, 38)) or (39 if mask[39] else ...)
        if phase == "modifier_target":                      # apply it to a party member
            return self._first_in(mask, range(52, 58)) or ...   # 52-57, NOT the cancel (39)

        if phase in self.SKIP_PHASES and mask[39]:          # optional switch / learn-move
            return 39

        return self._first_legal(mask)                      # forced switch, target, ... stay legal
```

(The real file adds a small counter so an un-applyable reward can't loop — see
below. `_first_in` returns the lowest legal id within a range, or None.)

Four ideas do all the work:

1. **In `command`**, pick the highest-base-power *damaging* move against enemy 0
   (ids 0-3), reading move power from `info["game_state"]` (see §4). Fall back
   to the first legal id if nothing damaging is usable (out of PP, all-status
   moveset → the game substitutes Struggle for you).
2. **Take the free shop reward.** A free reward each wave is how you get
   stronger, so engage the shop, don't skip it: pick a reward (`35-37`) in
   `modifier`, then — this is the subtle part — apply it to a **party-target id
   (`52-57`)** in the follow-up `modifier_target` phase.
3. **Decline the optional prompts.** `check_switch` (pre-battle "want to
   switch?") and `learn_move` (new move, full moveset) both accept id `39` =
   *decline*.
4. **Never get stuck.** For anything unhandled (a forced switch after a faint, a
   target select, a biome choice), take the first legal id — always progresses,
   never violates the mask.

> **The one shop trap.** In `modifier_target` the legal ids are e.g.
> `[39, 54]`: `54` = *apply the reward to party slot 2*, `39` = *cancel/back*.
> The naive "first legal action" picks the **lowest** id — `39` — which cancels
> the reward and bounces you back to the shop, where you pick it again... a
> silent infinite loop until the step cap. Always pick a real party-target id
> (`52-57`) there, not the cancel. (A few rewards can't be applied at all —
> DNA Splicers is a no-op — so the real agent also caps reward attempts per shop
> and then skips; an agent must always make progress.)

Run it:

```bash
python3 examples/rl/custom_agent.py --episodes 2 --waves 10
python3 examples/rl/custom_agent.py --starters MEWTWO,LUGIA,RAYQUAZA
```

---

## 4. What you get to look at each step

### `obs` — the curated vector (always available)

6991 normalized floats: every party & enemy slot (stats, types, HP, status,
moves with power/type/effectiveness/PP), field state, battle meta, your
modifier inventory, the shop, derived matchup/speed features, and a one-hot of
the current phase. It is **complete enough to play well** and is **identical
across headless / terminal / browser** (a bitwise parity gate enforces this). A
*learned* policy consumes `obs` directly — it must, because that's its only
input. Layout: [`OBS_V9_LAYOUT.md`](OBS_V9_LAYOUT.md); dim names:
`rl.feature_names.dim_to_name(i)`.

### `info` — the structured extras

| key | meaning |
|---|---|
| `phase` | decision-phase string (table below) — **branch on this** |
| `wave` | current wave number |
| `game_state` | the FULL raw state dict — **only when the env runs `lean=False`** (else `{}`) |
| `action_mask` | same mask `env.action_masks()` returns |
| `victory` | present only on a terminal game-over: True = won the run |
| `invalid_action_count` | appears if you ever picked an illegal id (should stay absent) |

### `obs` vs `game_state` — which to read

They are **not** the same channel, and this trips people up:

- The **observation** is the curated, normalized 6991-dim vector. Always sent.
  Cheap. What a network learns from.
- **`game_state`** is the complete, human-readable JSON dump of the game
  (~100 KB+). It carries everything the encoder deliberately cut — exact IVs,
  every per-move effect flag, full move history, raw `move.power`. It rides
  along **only with `lean=False`**.

So: read `obs` for anything a trained model needs (recommended, and required for
the learned part). Reach into `game_state` when you want a raw field that's
tedious to locate in the vector — `GreedyAttacker` reads `move["power"]` from it
rather than reverse-engineering which dims hold move power. That's why it
constructs the env with `lean=False`. An obs-only agent keeps the faster
default `lean=True`. (See "What lean does" in the README.)

### The decision phases (`info["phase"]`)

| phase string | when | typical handling |
|---|---|---|
| `command` | your turn in battle | choose a move / switch / ball / run / tera |
| `target` | pick a target (mostly doubles) | first legal, or your target logic |
| `modifier` | the shop / reward screen | pick reward `35-37`, buy `40-51`, reroll `38`, or skip `39` |
| `modifier_target` | apply a chosen item to a party member | `52-57` (slot 0-5) |
| `switch` | **forced** switch (your active fainted) | you must pick a replacement (`12-16`) |
| `check_switch` | **optional** pre-battle switch prompt | accept or skip `39` |
| `learn_move` | full moveset, new move offered | replace a slot, or skip `39` to keep |
| `revival_blessing` | choose a fainted mon to revive | `12-16` |
| `select_biome` | branching path choice | first legal, or your routing |
| `game_over` | terminal | episode ends; read `info["victory"]` |

(Setup phases — `title`, `starter`, `evolution`, … — are auto-played for you and
never reach your agent.)

---

## 5. The 58 action ids

| ids | meaning | legal in |
|---|---|---|
| `0-3` | use move slot 0-3 → enemy 0 | command |
| `4-7` / `8-11` | move → enemy 1 / ally | command (doubles) |
| `12-16` | switch to party slot 1-5 | command / switch |
| `17-21` | throw poké-ball type 0-4 | command |
| `22` | run away | command |
| `23-34` | Tera + move (enemy0 / enemy1 / ally) | command |
| `35-37` | pick reward 0-2 | modifier |
| `38` / `39` | reroll / **skip** | modifier |
| `40-51` | buy shop item 0-11 | modifier |
| `52-57` | apply modifier to party slot 0-5 | modifier_target |

Command ids (`0-34`) and shop ids (`35-57`) are disjoint, so an id is never
ambiguous. Because each action is *positional* ("move slot 0", "reward slot 0")
and the observation describes that slot, an agent learns "pick the slot whose
features are good," which generalizes across movesets and shops. Full rationale
(incl. why Tera is bundled with the move): [README](../README.md#the-action-space).

---

## 6. Driving it — the run loop

Any of the three transports run the identical `act`. Headless is the one you'll
use for development and training:

```python
import sys; sys.path.insert(0, "src")
from rl.pokerogue_env import PokeRogueEnv

env = PokeRogueEnv(waves=10, seed="demo", lean=False)   # lean=False: agent reads game_state
agent = GreedyAttacker()
obs, info = env.reset()
done = False
while not done:
    action = agent.act(obs, env.action_masks(), info)   # mask is a method call
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

`env.action_masks()` returns the current mask (it's also `info["action_mask"]`).
This is the standard `gymnasium` API — `reset()`/`step()` return the 5-tuple,
`terminated` = the run ended (win or loss), `truncated` = hit the wave/step cap.

**Watch it in the browser.** The built-in policies are wired into
`tools/run_policy.py --rendered`; to watch *your* agent, either add it there or
reuse the same loop against the rendered transport (the README "Ways to run"
section covers the Vite server). The obs/mask/reward your agent sees are
identical to headless, so behavior matches.

**Score it.** Compare against baselines over held-out seeds with
`tools/eval_policy.py` (built-in policies), or wrap your own loop to report mean
reward / waves / win% the way it does. Keep eval seeds disjoint from any
training seeds.

---

## 7. Shaping the reward (optional)

Each step's `reward` comes from the built-in `RewardCalculator` (a documented
default, retunable via the run config's `reward:` section). To define a
*structurally different* objective, write a `RewardFn` over `game_state` and
wrap the env — it **replaces** the built-in reward:

```python
from rl.reward import CustomReward, ComponentReward, delta_component, wave, money

reward = ComponentReward({
    "depth":  (10.0,   delta_component(wave)),    # +10 per wave cleared
    "wealth": (0.0005, delta_component(money)),   # your own money scale
})
env = CustomReward(PokeRogueEnv.from_config(cfg, lean=False), reward)
```

`CustomReward` **requires `lean=False`** (it reads `game_state`) and now fails
loudly if you forget. Runnable template: `examples/rl/custom_reward.py`; the
accessor list and `RewardFn` protocol live in `src/rl/reward.py`.

---

## 8. Multiple agents: routing by state

Want a different agent in different situations — a learned battle net but a
scripted shop, an aggressive attacker but a defensive switcher when low on HP, a
specialist on boss waves? You don't need any new machinery: **a router is just an
agent that delegates.** It reads the state and calls a sub-agent's `act`. Because
sub-agents and routers share the one `act(obs, mask, info) -> int` contract, they
compose freely — you can even nest routers inside routers.

**Route by phase — the built-in.** `rl.policy.PhaseRoutedPolicy` dispatches on
`info["phase"]`: a learned policy for the battle, a scripted skipper for the rest.

```python
from rl.policy import PhaseRoutedPolicy, Sb3Policy, ScriptedSkipPolicy
policy = PhaseRoutedPolicy(
    routes={"command": Sb3Policy("battle.zip"), "target": Sb3Policy("battle.zip")},
    default=ScriptedSkipPolicy(),          # shop / switches / learn-move
)
```

**Route by anything — a general router.** Phase is just one condition. To route
on HP, boss waves, which Pokémon is active, wave number — anything in the state —
write a tiny router with a list of `(name, predicate, policy)` rules, first match
wins. `examples/rl/routed_agents.py` is the runnable version; the core is:

```python
class StateRouter:
    def __init__(self, rules, default):   # rules: [(name, predicate, policy), ...]
        self.rules, self.default = rules, default

    def act(self, obs, mask, info):
        for name, predicate, policy in self.rules:
            if predicate(obs, mask, info):        # first matching rule wins
                return policy.act(obs, mask, info)
        return self.default.act(obs, mask, info)

def in_trouble(obs, mask, info):                  # a predicate is just (obs,mask,info)->bool
    hp = ((info.get("game_state") or {}).get("player_0") or {}).get("hp_ratio", 1.0)
    return info.get("phase") == "command" and hp < 0.35 and any(mask[a] for a in range(12, 17))

router = StateRouter(
    rules=[("retreat", in_trouble, RetreatPolicy())],   # low HP + can switch → pull out
    default=GreedyAttacker(),                            # everything else
)
```

Running `routed_agents.py` prints which sub-agent handled each decision, e.g.
`routing={'default': 45, 'retreat': 7}` — proof the router is dispatching by
state. Rule order is priority; keep a catch-all `default` so every state is
handled. This is also how you grow a learned agent incrementally: start with
`GreedyAttacker` as the default, then peel off states (`command`, then boss
waves, then …) to a trained policy as it earns its keep.

---

## 9. From hand-written to learned

The same interface hosts a trained network — a policy is still
`act(obs, mask, info) -> int`. The env exposes `action_masks()` specifically so
`sb3-contrib`'s **MaskablePPO** only ever samples legal actions:

- `examples/rl/train_maskable_ppo.py` — a training stub (`SubprocVecEnv` for
  parallel game processes, mask wiring via `ActionMasker`).
- `rl.policy.Sb3Policy("model.zip")` — load a checkpoint as an `act`-able policy.
- `rl.policy.PhaseRoutedPolicy(routes, default)` — **mix learned and scripted**
  by phase, e.g. a trained network for `command`/`target` and a scripted skip
  for the shop. This is the recommended way to grow `GreedyAttacker`: keep the
  scripted shop/switch handling, drop a learned policy into the battle phases.

```python
from rl.policy import PhaseRoutedPolicy, Sb3Policy, ScriptedSkipPolicy
policy = PhaseRoutedPolicy(
    routes={"command": Sb3Policy("battle.zip"), "target": Sb3Policy("battle.zip")},
    default=ScriptedSkipPolicy(),
)
```

A checkpoint trained this way drops straight into `run_policy.py --model run.zip`
(headless or `--rendered`) with no code changes.

---

## 10. Pitfalls checklist

- **Illegal action → silent corruption.** Always `if mask[i]`. Watch
  `info["invalid_action_count"]` — it must stay 0.
- **`game_state` empty?** You're running `lean=True` (the default). A heuristic
  or custom reward that reads `game_state` needs `lean=False`.
- **"First legal action" can pick a cancel/back and loop.** The lowest legal id
  isn't always a safe default: in `modifier_target` id `39` = *cancel*, so
  `first_legal` cancels your reward and re-enters the shop forever. Pick the
  *right* id for the phase (a party target `52-57` here), not just any legal one.
- **Non-determinism.** Seed everything: `PokeRogueEnv(seed=...)` fixes the game;
  seed your own RNG from a stable integer (string `hash()` is salted per
  process — use `zlib.crc32`, as `RandomPolicy` does).
- **Forced vs optional.** `switch` (forced, active fainted) has no skip — you
  must pick a replacement; `check_switch` (optional) accepts `39`.
- **A router needs a catch-all.** In a `StateRouter`, keep a `default` so every
  unmatched state is still handled — otherwise a state with no matching rule
  has no action.

---

## See also

- [`examples/rl/custom_agent.py`](../../../examples/rl/custom_agent.py) — the runnable agent from §3
- [`examples/rl/routed_agents.py`](../../../examples/rl/routed_agents.py) — the state router from §8
- [`examples/rl/phase_routed_policy.py`](../../../examples/rl/phase_routed_policy.py) — phase routing + a custom reward wrapper
- [`src/rl/policy.py`](../policy.py) — the built-in policies to copy from
- [README](../README.md) — modes, the action space, "what lean does"
- [`OBS_V9_LAYOUT.md`](OBS_V9_LAYOUT.md) — the observation vector, dim by dim
