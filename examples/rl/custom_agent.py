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
SKIP = 39                       # id 39:    skip the shop / decline the switch
MAX_MOVES = 4
STATUS_CATEGORY = 2             # MoveCategory.STATUS — deals no damage


class GreedyAttacker:
    """A hand-written baseline that beats random by a wide margin.

    In a battle it fires its highest-base-power *damaging* move at enemy 0; it
    SKIPS the shop and optional switches, and otherwise takes the first legal
    action. It reads move power from ``info["game_state"]`` (so construct the
    env with ``lean=False``); an obs-only variant is described in the docs.

    Why skip the shop? A free reward that targets a party member (a TM, a
    vitamin) opens a follow-up ``modifier_target`` decision, and choosing
    rewards+targets well is its own problem. The common, robust pattern is to
    LEARN or hand-tune the battle and route the shop to a skip — so this
    baseline stays focused on the part that matters most. Taking rewards is a
    good exercise once the battle loop works (see the action table for the
    reward-pick / apply-to-slot ids).
    """

    # Phases where the right move is "decline / don't engage" (id 39 = skip):
    #   modifier       — the shop (skip the free pick and the buys)
    #   check_switch   — the optional pre-battle switch prompt
    #   learn_move     — offered a new move with a full moveset (skip = keep old)
    SKIP_PHASES = frozenset({"modifier", "check_switch", "learn_move"})

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        phase = info.get("phase")

        if phase == "command":
            move_id = self._best_damaging_move(mask, info)
            if move_id is not None:
                return move_id
            # no damaging move usable (out of PP / all status) → first legal
            return self._first_legal(mask)

        if phase in self.SKIP_PHASES and mask[SKIP]:
            return SKIP

        # switch (forced), revival_blessing, select_biome, target,
        # modifier_target, game_over, ... — no special handling, stay legal.
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
