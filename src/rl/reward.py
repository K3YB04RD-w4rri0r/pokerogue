"""
Bring-your-own reward for the PokeRogue RL environment.

Two ways to shape reward, in increasing order of control:

1. **Reweight the built-in reward** (no Python needed): the TS
   ``RewardCalculator`` (src/rl/rewards.ts, 16 weighted components) is the
   default. Set weights from the run config's ``reward:`` section, or
   ``PokeRogueEnv(reward_config={...})``. The env reports that reward per
   step. Use this when the built-in components are the right *shape* and you
   only want to retune magnitudes.

2. **Write your own reward in Python** (this module): implement a
   ``RewardFn`` over the full serialized game state and wrap the env with
   ``CustomReward``. Your function REPLACES the env reward. This is the path
   when you want a structurally different objective (e.g. survival-only,
   depth-only, catch-focused, curriculum-shaped).

A ``RewardFn`` sees, each step, the game-state dict BEFORE and AFTER the
action plus the step ``info`` — everything the observation is built from and
more. The full ``game_state`` schema is documented in src/rl/INPUT.md; the
helpers below (``player_hp_fraction``, ``wave``, ``money``, ...) cover the
common fields so most reward functions are a few lines.

REQUIREMENT: custom rewards read ``info["game_state"]``, which is only
populated when the env runs with ``lean=False`` (``CustomReward`` sets this
expectation; construct the env with ``lean=False``).

Example — a depth-and-survival reward in ~10 lines::

    from rl.reward import CustomReward, RewardFn, wave, player_alive_count

    class DepthSurvival(RewardFn):
        def __call__(self, prev, cur, info):
            r = 10.0 * max(0, wave(cur) - wave(prev))      # +10 per wave cleared
            r -= 3.0 * max(0, player_alive_count(prev) - player_alive_count(cur))  # -3 per faint
            return r

    env = CustomReward(PokeRogueEnv.from_config(cfg, lean=False), DepthSurvival())

Or compose weighted components without writing a class::

    from rl.reward import ComponentReward, delta_component, wave, money
    reward = ComponentReward({
        "depth":  (10.0, delta_component(wave)),
        "wealth": (0.001, delta_component(money)),  # your own money scale
    })
    env = CustomReward(env, reward)
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Protocol, Tuple

import gymnasium as gym

# ── State accessors (thin, null-safe readers over the game_state dict) ──
# game_state schema: src/rl/INPUT.md. These cover the common fields; read
# any other field directly from the dict in your RewardFn.

GameState = Mapping[str, object]


def _battle(gs: GameState) -> Mapping[str, object]:
    return (gs.get("battle") or {}) if gs else {}  # type: ignore[union-attr]


def wave(gs: GameState) -> int:
    return int(_battle(gs).get("wave_index", 0) or 0)


def money(gs: GameState) -> int:
    return int(_battle(gs).get("money", 0) or 0)


def player_alive_count(gs: GameState) -> int:
    return int(_battle(gs).get("player_alive_count", 0) or 0)


def enemy_alive_count(gs: GameState) -> int:
    return int(_battle(gs).get("enemy_alive_count", 0) or 0)


def player_faints(gs: GameState) -> int:
    """Cumulative player faints this arena (the game's own counter)."""
    return int(_battle(gs).get("player_faints_battle", 0) or 0)


def enemy_faints(gs: GameState) -> int:
    return int(_battle(gs).get("enemy_faints_battle", 0) or 0)


def _slot(gs: GameState, key: str) -> Mapping[str, object]:
    return (gs.get(key) or {}) if gs else {}  # type: ignore[union-attr]


def player_hp_fraction(gs: GameState) -> float:
    """Mean HP fraction over valid player party members (0..1)."""
    fracs = [
        float(_slot(gs, f"player_{i}").get("hp_ratio", 0.0) or 0.0)
        for i in range(6)
        if _slot(gs, f"player_{i}").get("valid")
    ]
    return sum(fracs) / len(fracs) if fracs else 0.0


# ── The reward-function interface ──────────────────────────────────────


class RewardFn(Protocol):
    """A custom reward. ``__call__`` returns the scalar reward for the
    transition prev_state --(action)--> cur_state. ``reset`` is called at
    the start of each episode for stateful rewards (e.g. potentials)."""

    def __call__(self, prev_state: GameState, cur_state: GameState, info: Mapping[str, object]) -> float: ...

    def reset(self) -> None:  # optional; default no-op via RewardBase
        ...


class RewardBase:
    """Convenience base: implement ``__call__``; ``reset`` defaults to no-op."""

    def reset(self) -> None:
        return None

    def __call__(self, prev_state: GameState, cur_state: GameState, info: Mapping[str, object]) -> float:
        raise NotImplementedError


# ── Composable components ──────────────────────────────────────────────

ComponentFn = Callable[[GameState, GameState, Mapping[str, object]], float]


def delta_component(accessor: Callable[[GameState], float], *, only_positive: bool = True) -> ComponentFn:
    """Component = change in ``accessor`` between prev and cur. With
    ``only_positive`` (default) negatives are floored to 0 (e.g. money gained
    but not money spent; waves advanced but not reset)."""

    def component(prev: GameState, cur: GameState, _info: Mapping[str, object]) -> float:
        d = float(accessor(cur)) - float(accessor(prev))
        return max(0.0, d) if only_positive else d

    return component


class ComponentReward(RewardBase):
    """Weighted sum of named components: ``{name: (weight, component_fn)}``.
    Fully user-editable — add/remove/reweight entries to taste."""

    def __init__(self, components: Dict[str, Tuple[float, ComponentFn]]):
        self.components = components

    def __call__(self, prev: GameState, cur: GameState, info: Mapping[str, object]) -> float:
        return sum(w * fn(prev, cur, info) for (w, fn) in self.components.values())


# ── The gym wrapper that installs a RewardFn ───────────────────────────


class CustomReward(gym.Wrapper):
    """Replace the env reward with ``reward_fn`` computed from the full
    game state. Requires the env to run with ``lean=False`` so
    ``info["game_state"]`` is populated.

    The wrapped env's own (TS) reward is discarded; your function is the
    sole reward signal. Set ``keep_terminal=True`` to ADD the env's
    terminal reward (the ±win/lose bonus) on top of yours."""

    def __init__(self, env: gym.Env, reward_fn: RewardFn, *, keep_terminal: bool = False):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._keep_terminal = keep_terminal
        self._prev_state: GameState = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gs = info.get("game_state")
        # The env ALWAYS puts a "game_state" key in info, but in lean mode
        # (the default) it is an empty dict, not None — so `is None` never
        # catches the "you forgot lean=False" mistake and the user would
        # silently train on all-zero reward (every accessor reads 0 off {}).
        # Guard on falsiness so lean mode fails loud and early.
        if not gs:
            raise RuntimeError(
                "CustomReward needs a populated info['game_state'] — construct the "
                "env with lean=False (lean=True omits game_state, giving zero reward)"
            )
        self._prev_state = gs
        reset = getattr(self._reward_fn, "reset", None)
        if callable(reset):
            reset()
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        cur = info.get("game_state") or {}
        # A truncation with no post-state (the env's step-timeout path returns
        # info without game_state) has no valid s' to diff against — scoring
        # reward_fn(prev, {}) fabricates deltas (every accessor reads 0 off the
        # empty dict, e.g. a full-HP prev vs 0 "now" → spurious HP penalty).
        # Skip the transition reward on that path; keep_terminal still applies.
        reward = self._reward_fn(self._prev_state, cur, info) if cur else 0.0
        # keep_terminal ADDS the env's ±win/lose bonus, which only exists on a
        # true termination (game over). A truncation (wave/step cap, timeout)
        # carries NO terminal bonus — its env_reward is the final transition's
        # shaped reward, and adding it would double-count that step against the
        # user's own reward_fn for the same transition.
        if self._keep_terminal and terminated:
            reward += float(env_reward)  # env_reward carries the terminal bonus
        self._prev_state = cur
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Forward the sb3-contrib mask hook through the wrapper (gym.Wrapper
        does not auto-delegate it, and MaskablePPO/ActionMasker need it)."""
        return self.env.unwrapped.action_masks()
