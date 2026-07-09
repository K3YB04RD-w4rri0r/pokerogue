#!/usr/bin/env python3
"""
Multiple agents, one controller — routing sub-agents by the current state.

Because every agent is just ``act(obs, mask, info) -> int``, a *router* is also
just an agent: it inspects the state and delegates to a sub-agent. This composes
with no special machinery — routers of routers, learned + scripted side by side.

Two ways to route, both shown here:

  1. rl.policy.PhaseRoutedPolicy — the BUILT-IN case: dispatch by
     ``info["phase"]`` (a learned battle policy, a scripted shop, ...).

  2. StateRouter (below) — the GENERAL case: a list of
     ``(name, predicate, policy)`` rules over the whole ``(obs, mask, info)``;
     the first predicate that matches wins. Route by HP, boss waves, which
     Pokemon is active, wave number — anything you can read from the state.

Run (from the repo root, after `pnpm rl:build`):
    python3 examples/rl/routed_agents.py --episodes 2 --waves 12
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling import below

from rl.policy import PhaseRoutedPolicy, ScriptedSkipPolicy  # noqa: E402
from rl.pokerogue_env import PokeRogueEnv  # noqa: E402

from custom_agent import GreedyAttacker  # noqa: E402  our own agent, reused as a sub-policy

Predicate = Callable[[np.ndarray, np.ndarray, dict], bool]


# ── The general router: route by any condition over the state ──────────────
class StateRouter:
    """Delegate to the first sub-policy whose predicate matches the state.

    ``rules`` is ``[(name, predicate, policy), ...]``; ``default`` handles
    anything unmatched. It's an agent like any other, so you can nest it inside
    another router or hand it to run_policy / a training loop unchanged.
    """

    def __init__(self, rules: list[tuple[str, Predicate, object]], default: object):
        self.rules = rules
        self.default = default
        self.fired: Counter = Counter()  # bookkeeping: which sub-agent ran, for the demo

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        for name, predicate, policy in self.rules:
            if predicate(obs, mask, info):
                self.fired[name] += 1
                return policy.act(obs, mask, info)
        self.fired["default"] += 1
        return self.default.act(obs, mask, info)


# ── A small sub-agent to route TO ──────────────────────────────────────────
class RetreatPolicy:
    """Pull the active Pokemon out to a healthy bench slot (switch ids 12-16)."""

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        switch = next((a for a in range(12, 17) if mask[a]), None)
        if switch is not None:
            return switch
        legal = np.flatnonzero(mask)  # nothing to switch to → stay legal
        return int(legal[0]) if legal.size else 0


# ── Predicates: pure functions of (obs, mask, info) -> bool ─────────────────
def _active_hp(info: dict) -> float:
    gs = info.get("game_state") or {}
    field_index = (gs.get("phase") or {}).get("command_field_index") or 0
    me = gs.get("player_1" if field_index == 1 else "player_0") or {}
    return float(me.get("hp_ratio", 1.0))


def in_trouble(obs: np.ndarray, mask: np.ndarray, info: dict) -> bool:
    """Battle turn, the active mon is low, and a switch is available."""
    return (
        info.get("phase") == "command"
        and _active_hp(info) < 0.35
        and any(mask[a] for a in range(12, 17))
    )


def build_router(verbose: bool = False) -> StateRouter:
    # Rule order = priority (first match wins). GreedyAttacker is the default:
    # it already handles command + the shop + skips, so the router only has to
    # override it in the states where a *different* agent should take over.
    return StateRouter(
        rules=[
            ("retreat", in_trouble, RetreatPolicy()),   # low HP + can switch → pull out
            # add more: ("boss", on_boss_wave, BossSpecialist()), ...
        ],
        default=GreedyAttacker(),                        # everything else
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--waves", type=int, default=12)
    ap.add_argument("--seed", type=str, default="router")
    args = ap.parse_args()

    # (Phase routing is one line — the built-in dispatches by info["phase"].
    #  We RUN the general StateRouter below, which is strictly more flexible.)
    _phase_routed = PhaseRoutedPolicy(
        routes={"command": GreedyAttacker(), "target": GreedyAttacker()},
        default=ScriptedSkipPolicy(),
    )

    env = PokeRogueEnv(waves=args.waves, seed=args.seed, lean=False)  # predicates read game_state
    router = build_router()

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total, steps = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = router.act(obs, env.action_masks(), info)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
            outcome = "victory" if info.get("victory") else ("defeat" if terminated else "truncated")
            print(
                f"episode {ep}: {outcome} at wave {info.get('wave')} in {steps} steps, "
                f"reward {total:+.2f}, routing={dict(router.fired)}"
            )
            router.fired.clear()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
