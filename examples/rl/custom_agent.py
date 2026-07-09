#!/usr/bin/env python3
"""
Your own agent, from scratch — the companion script to
``src/rl/docs/WRITING_AN_AGENT.md``.

An *agent* here is just a function ``act(obs, mask, info) -> int``: given the
observation, the legal-action mask, and the step info, return one of the 58
action ids. Nothing else is required — no base class, no registration. This
file defines one such agent (a phase-aware greedy attacker) with zero imports
from ``rl.policy``, and runs it against the real headless game.

Two hard rules the agent below follows (and yours must too):
  1. NEVER return an id whose ``mask[id]`` is False. The env will reject it and
     fall back to another action, silently corrupting your credit assignment
     (it surfaces as info["invalid_action_count"]).
  2. Branch on ``info["phase"]`` — the SAME id means different things is false;
     each id means exactly one thing, but which ids are *legal* changes per
     phase, so a battle move and a shop pick need different logic.

Run (from the repo root, after `pnpm rl:build`):
    python3 examples/rl/custom_agent.py --episodes 2 --waves 10
    python3 examples/rl/custom_agent.py --starters MEWTWO,LUGIA,RAYQUAZA
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from rl.pokerogue_env import PokeRogueEnv  # noqa: E402

# ── Action-space anchors (src/rl/spaces.ts / README action table) ──────────
FIGHT_ENEMY0 = range(0, 4)      # ids 0-3: use move slot 0-3 against enemy 0
REWARD_PICKS = range(35, 38)    # ids 35-37: take free reward 0-2
PARTY_TARGETS = range(52, 58)   # ids 52-57: apply a reward to party slot 0-5
SKIP = 39                       # id 39:    skip the shop / decline / cancel
MAX_MOVES = 4
STATUS_CATEGORY = 2             # MoveCategory.STATUS — deals no damage


class GreedyAttacker:
    """A hand-written baseline that beats random by a wide margin.

    In a battle it fires its highest-base-power *damaging* move at enemy 0; it
    grabs the free shop reward, declines optional switches, and otherwise takes
    the first legal action. It reads move power from ``info["game_state"]`` (so
    construct the env with ``lean=False``); an obs-only variant is in the docs.

    The shop is worth engaging — a free reward each wave is how you get stronger.
    The one trap: in the ``modifier_target`` phase (apply the reward to a party
    member) the *lowest* legal id is ``39`` = cancel/back, so a plain
    "first legal action" would cancel the reward and bounce back to the shop
    forever. Always pick a real party-target id (52-57) there. Some rewards
    can't be applied at all (e.g. DNA Splicers is a no-op), so we also give up
    on a shop after a few attempts and skip it — an agent must always progress.
    """

    # Optional phases where id 39 = "decline": the pre-battle switch prompt and
    # the move-learn offer (skip = keep the current moveset).
    SKIP_PHASES = frozenset({"check_switch", "learn_move"})

    def __init__(self) -> None:
        # How many rewards we've tried to take at the CURRENT shop — bounds the
        # rare un-targetable-reward loop; reset when a battle starts.
        self._reward_tries = 0

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        phase = info.get("phase")

        if phase == "command":
            self._reward_tries = 0                     # a battle -> a fresh shop next
            move_id = self._best_damaging_move(mask, info)
            if move_id is not None:
                return move_id
            # no damaging move usable (out of PP / all status) → first legal
            return self._first_legal(mask)

        if phase == "modifier":                        # the shop
            if self._reward_tries < len(REWARD_PICKS):
                pick = self._first_in(mask, REWARD_PICKS)
                if pick is not None:
                    self._reward_tries += 1
                    return pick                        # take a free reward
            return SKIP if mask[SKIP] else self._first_legal(mask)

        if phase == "modifier_target":                 # apply the reward to a mon
            target = self._first_in(mask, PARTY_TARGETS)   # 52-57, NOT the cancel (39)
            if target is not None:
                return target
            return SKIP if mask[SKIP] else self._first_legal(mask)

        if phase in self.SKIP_PHASES and mask[SKIP]:
            return SKIP

        # forced switch, revival_blessing, select_biome, target, game_over,
        # ... — no special handling, stay legal.
        return self._first_legal(mask)

    # ── helpers ────────────────────────────────────────────────────────────
    def _best_damaging_move(self, mask: np.ndarray, info: dict) -> int | None:
        """Highest-power damaging move among the legal FIGHT_ENEMY0 ids."""
        game_state = info.get("game_state") or {}
        # In doubles the acting mon may be slot 1; its moves live under player_1.
        field_index = (game_state.get("phase") or {}).get("command_field_index") or 0
        me = game_state.get("player_1" if field_index == 1 else "player_0") or {}
        moves = me.get("moves") or []

        best_id, best_power = None, 0
        for a in FIGHT_ENEMY0:
            if not mask[a]:
                continue
            move = moves[a] if a < len(moves) else None
            if not move:
                continue
            power = move.get("power") or 0
            if move.get("category") != STATUS_CATEGORY and power > best_power:
                best_id, best_power = a, power
        return best_id

    @staticmethod
    def _first_in(mask: np.ndarray, ids) -> int | None:
        """Lowest legal id within a specific range (or None)."""
        return next((a for a in ids if mask[a]), None)

    @staticmethod
    def _first_legal(mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        return int(legal[0]) if legal.size else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--waves", type=int, default=10)
    ap.add_argument("--seed", type=str, default="agent")
    ap.add_argument("--starters", default=None, help="e.g. MEWTWO,LUGIA,RAYQUAZA")
    args = ap.parse_args()

    # lean=False: GreedyAttacker reads info["game_state"] for move power. An
    # agent that only looks at obs + mask can keep the faster default lean=True.
    env = PokeRogueEnv(waves=args.waves, seed=args.seed, starters=args.starters, lean=False)
    agent = GreedyAttacker()

    try:
        for ep in range(args.episodes):
            t0 = time.time()
            obs, info = env.reset()
            total, steps = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, env.action_masks(), info)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
            outcome = "victory" if info.get("victory") else ("defeat" if terminated else "truncated")
            print(
                f"episode {ep}: {outcome} at wave {info.get('wave')} in {steps} steps, "
                f"reward {total:+.2f}, invalid-actions {info.get('invalid_action_count', 0)} "
                f"({time.time() - t0:.1f}s)"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
