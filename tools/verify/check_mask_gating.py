#!/usr/bin/env python3
"""
Targeted end-to-end checks for action-mask gates that, when wrong, soft-lock
the game (CommandPhase error prompts nothing dismisses -> 30s router timeout):

A) Shielded boss (wave 10): ball actions 17-21 must be masked OFF while only
   regular balls are in inventory (only a Master Ball can move a shielded
   boss), and battle.can_catch must agree.
B) All moves out of PP: the mask must offer fight slot 0 (the game's
   auto-Struggle) instead of going fight-empty, and executing it must
   advance the game.
C) Shop flow: REROLL must keep the shop open (fresh modifier decision,
   reroll_count +1, money charged, cost doubled) and only SKIP exits.

A and B FAIL on the pre-2026-07-07 mask code (regression guard).

Usage: python3 tools/verify/check_mask_gating.py
Requires: pnpm rl:build (dist/rl/cli.js), numpy, gymnasium.
"""

from __future__ import annotations

import sys

from common import REPO_ROOT  # noqa: F401  (sys.path side effect adds src/)

from rl.pokerogue_env import PokeRogueEnv  # noqa: E402

BALL_ACTIONS = range(17, 22)
ACTION_SKIP = 39
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def check_boss_ball_gating() -> None:
    print("A) wave-10 shielded boss: ball actions masked off")
    env = PokeRogueEnv(waves=11, seed="verify-maskgate-boss", lean=False,
                       overrides={"STARTING_WAVE_OVERRIDE": 10})
    obs, info = env.reset()
    for _ in range(6):
        if info.get("phase") == "command":
            break
        mask = env.action_masks()
        a = ACTION_SKIP if mask[ACTION_SKIP] else int(mask.argmax())
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    gs = info.get("game_state") or {}
    enemy = gs.get("enemy_0") or {}
    battle = gs.get("battle") or {}
    mask = env.action_masks()
    check("reached a command decision", info.get("phase") == "command", str(info.get("phase")))
    check("enemy is a shielded boss",
          bool(enemy.get("is_boss")) and enemy.get("boss_segment_index", 0) >= 1,
          f"segments={enemy.get('boss_segments')} idx={enemy.get('boss_segment_index')}")
    have_master = (battle.get("pokeball_counts") or {}).get("master_ball", 0) > 0
    offered = [int(a) for a in BALL_ACTIONS if mask[a]]
    check("no regular-ball action offered", all(a == 21 for a in offered), f"offered={offered}")
    if not have_master:
        check("no ball action at all (no Master Ball owned)", not offered, f"offered={offered}")
    check("battle.can_catch is False", battle.get("can_catch") is False, str(battle.get("can_catch")))
    env.close()


def check_struggle() -> None:
    print("B) PP exhaustion: Struggle stays available, no empty mask")
    # One 5-PP move (Hydro Pump), enemy locked to Splash so the run survives.
    env = PokeRogueEnv(waves=9, seed="verify-maskgate-struggle", lean=False,
                       overrides={"MOVESET_OVERRIDE": [56],
                                  "ENEMY_MOVESET_OVERRIDE": [150],
                                  "STARTING_LEVEL_OVERRIDE": 15})
    obs, info = env.reset()
    struggle_turns = 0
    saw_pp_zero = False
    for step in range(120):
        phase = info.get("phase")
        mask = env.action_masks()
        if not mask.any():
            check("mask never empty", False, f"step {step} phase={phase}")
            break
        if phase == "command":
            gs = info.get("game_state") or {}
            fi = (gs.get("phase") or {}).get("command_field_index") or 0
            me = gs.get("player_1" if fi == 1 else "player_0") or {}
            moves = me.get("moves") or []
            usable = any(m.get("is_usable") for m in moves if m)
            if not usable:
                saw_pp_zero = True
                check(f"fight slot 0 offered with no usable move (step {step})", bool(mask[0]))
                struggle_turns += 1
                if struggle_turns >= 3:
                    break
            a = 0 if mask[0] else int(mask.argmax())
        elif phase in ("modifier", "modifier_target", "check_switch", "learn_move"):
            a = ACTION_SKIP if mask[ACTION_SKIP] else int(mask.argmax())
        else:
            a = int(mask.argmax())
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            detail = (info.get("protocol_error") or "") if trunc else "terminated"
            check("episode survived to the Struggle turns", saw_pp_zero and struggle_turns > 0, detail)
            break
    check("reached a no-usable-move command decision", saw_pp_zero)
    check("executed Struggle turns without timeout", struggle_turns >= 1, f"turns={struggle_turns}")
    env.close()


def check_reroll_keeps_shop_open() -> None:
    print("C) shop flow: reroll keeps the shop open, skip exits")
    env = PokeRogueEnv(waves=6, seed="verify-maskgate-reroll", lean=False,
                       overrides={"STARTING_MONEY_OVERRIDE": 5000})
    obs, info = env.reset()
    for _ in range(60):
        if info.get("phase") == "modifier":
            break
        mask = env.action_masks()
        a = 0 if mask[0] else int(mask.argmax())
        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    check("reached a shop decision", info.get("phase") == "modifier", str(info.get("phase")))
    if info.get("phase") != "modifier":
        env.close()
        return
    gs = info["game_state"]
    before_money = gs["battle"]["money"]
    before_count = gs["battle"]["reroll_count"]
    reroll_cost = (gs.get("shop") or {}).get("reroll_cost")
    obs, _r, term, trunc, info = env.step(38)  # reroll
    gs = info["game_state"]
    check("still at a modifier decision after reroll", info.get("phase") == "modifier", str(info.get("phase")))
    check("reroll_count incremented", gs["battle"]["reroll_count"] == before_count + 1)
    check("money charged the reroll cost", gs["battle"]["money"] == before_money - reroll_cost,
          f"{before_money}->{gs['battle']['money']} (cost {reroll_cost})")
    check("reroll cost doubled", (gs.get("shop") or {}).get("reroll_cost") == reroll_cost * 2)
    check("rewards regenerated", bool((gs.get("shop") or {}).get("reward_options")))
    obs, _r, term, trunc, info = env.step(39)  # skip
    check("shop exits on SKIP", info.get("phase") != "modifier", str(info.get("phase")))
    env.close()


def main() -> int:
    check_boss_ball_gating()
    check_struggle()
    check_reroll_keeps_shop_open()
    print(f"\nMASK GATING: {'FAIL (' + ', '.join(FAILS) + ')' if FAILS else 'OK'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
