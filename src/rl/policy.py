"""Reusable policy interfaces for the PokeRogue RL environment.

A policy is anything with ``act(obs, mask, info) -> int``:

    obs   — np.float32 observation (OBSERVATION_DIM,)
    mask  — np.bool_ legal-action mask (ACTION_SPACE_SIZE,)
    info  — the env's info dict: ``phase`` (decision phase string), ``wave``,
            ``game_state`` (full state dict when the env runs lean=False), ...

Built-ins:

    RandomPolicy()                  — uniform over legal actions (fuzzer baseline)
    FirstLegalPolicy()              — lowest legal action id
    MaxDamagePolicy()               — highest base-power damaging move in battle
    ScriptedSkipPolicy()            — decline shops/optional switches, else first legal
    Sb3Policy("ckpt.zip")           — a trained sb3-contrib MaskablePPO checkpoint
    PhaseRoutedPolicy(routes, default) — dispatch by decision phase

Phase routing is the intended way to mix learned and scripted behavior, e.g.
a trained network for battles with a scripted shop::

    policy = PhaseRoutedPolicy(
        routes={"command": Sb3Policy("battle.zip"), "target": Sb3Policy("battle.zip")},
        default=ScriptedSkipPolicy(),
    )
    action = policy.act(obs, env.action_masks(), info)

The decision-phase strings are the DecisionPhase values (phase-router.ts):
command, target, modifier, modifier_target, switch, check_switch, learn_move,
select_biome, revival_blessing, mystery, game_over, ... — the observation also
one-hots them (dims named ``phase/*`` in feature_names.py).
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import numpy as np

# Stable action-space anchors (spaces.ts). The full map is in src/rl/README.md.
ACTION_SKIP = 39
MAX_MOVES = 4
_STATUS_CATEGORY = 2  # MoveCategory.STATUS


@runtime_checkable
class Policy(Protocol):
    """Anything that maps (obs, mask, info) to an action id."""

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int: ...


class RandomPolicy:
    """Uniformly random LEGAL action — a baseline and a great fuzzer."""

    def __init__(self, seed: int | str | None = None):
        derived = abs(hash(seed)) % (2**32) if isinstance(seed, str) else seed
        self._rng = np.random.default_rng(derived)

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        valid = np.flatnonzero(mask)
        return int(self._rng.choice(valid)) if valid.size else 0


class FirstLegalPolicy:
    """Lowest legal action id (deterministic, mildly aggressive: fight first)."""

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else 0


class ScriptedSkipPolicy:
    """Decline optional engagements (shop, battle-start switch, move learning),
    first legal action otherwise. Useful as the non-battle half of a router."""

    SKIP_PHASES = frozenset({"modifier", "modifier_target", "check_switch", "learn_move"})

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        if info.get("phase") in self.SKIP_PHASES and ACTION_SKIP < len(mask) and mask[ACTION_SKIP]:
            return ACTION_SKIP
        valid = np.flatnonzero(mask)
        return int(valid[0]) if valid.size else 0


class MaxDamagePolicy:
    """Highest base-power damaging move vs an enemy; declines the shop and the
    battle-start switch; first legal action otherwise.

    Reads move power/category from ``info["game_state"]`` — run the env with
    ``lean=False`` (run_policy.py does this automatically for built-ins).
    """

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        if info.get("phase") == "command":
            game_state = info.get("game_state") or {}
            # In doubles the acting pokemon may be slot 1 — its moves live
            # under player_1 and the fight actions refer to ITS moveset.
            field_index = (game_state.get("phase") or {}).get("command_field_index") or 0
            me = game_state.get("player_1" if field_index == 1 else "player_0") or {}
            moves = me.get("moves") or []
            best_action, best_power = None, 0
            for action in range(min(2 * MAX_MOVES, len(mask))):
                if not mask[action]:
                    continue
                move = moves[action % MAX_MOVES] if action % MAX_MOVES < len(moves) else None
                if not move:
                    continue
                power = move.get("power") or 0
                if move.get("category", 0) != _STATUS_CATEGORY and power > best_power:
                    best_power, best_action = power, action
            if best_action is not None:
                return best_action
        return ScriptedSkipPolicy().act(obs, mask, info)


class Sb3Policy:
    """A trained sb3-contrib MaskablePPO checkpoint (lazy torch import)."""

    def __init__(self, model_path: str, deterministic: bool = False):
        from sb3_contrib import MaskablePPO  # imported lazily; needs sb3-contrib

        self._model = MaskablePPO.load(model_path)
        self._deterministic = deterministic

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=self._deterministic)
        return int(action)


class PhaseRoutedPolicy:
    """Dispatch to a sub-policy by the current decision phase.

    ``routes`` maps DecisionPhase strings to policies; anything unrouted goes
    to ``default``. This is how you combine a learned battle policy with
    scripted shop/switch handling (or several specialist networks).
    """

    def __init__(self, routes: Mapping[str, Policy], default: Policy):
        self.routes = dict(routes)
        self.default = default

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        policy = self.routes.get(str(info.get("phase")), self.default)
        return policy.act(obs, mask, info)


def make_builtin_policy(name: str, seed: int | str | None = None) -> Policy:
    """Resolve a built-in policy by CLI name ('random', 'maxdamage', 'firstlegal')."""
    if name == "random":
        return RandomPolicy(seed)
    if name == "maxdamage":
        return MaxDamagePolicy()
    if name == "firstlegal":
        return FirstLegalPolicy()
    raise ValueError(f"unknown policy {name!r} (expected random | maxdamage | firstlegal)")
