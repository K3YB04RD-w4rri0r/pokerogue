#!/usr/bin/env python3
"""Bring-your-own-algorithm example: phase routing + a custom Python reward.

Demonstrates the two extension points an external RL project needs:

1. **Phase routing** (rl.policy.PhaseRoutedPolicy): different sub-policies per
   decision phase — here a battle policy for command/target phases and a
   scripted skipper for everything else. Swap any entry for your own
   ``act(obs, mask, info) -> int`` object (e.g. Sb3Policy("battle.zip")).

2. **Custom rewards in Python**: the protocol reward (rewards.ts, weights
   configurable via the run config's `reward:` section) arrives per step, but
   a structurally different reward can be computed here from info["game_state"]
   (run the env with lean=False) — shown as a gym RewardWrapper.

Run (after `pnpm rl:build`):
    python3 examples/rl/phase_routed_policy.py [--config examples/rl/legendary.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from rl.pokerogue_env import PokeRogueEnv  # noqa: E402
from rl.policy import MaxDamagePolicy, PhaseRoutedPolicy, ScriptedSkipPolicy  # noqa: E402
from rl.run_config import RunConfig, load_run_config  # noqa: E402


class SurvivalBonusWrapper(gym.RewardWrapper):
    """Example custom reward: protocol reward plus a bonus per living party
    member (reads the full game_state — requires lean=False)."""

    def __init__(self, env, bonus: float = 0.05):
        super().__init__(env)
        self._bonus = bonus
        self._last_alive = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        battle = (info.get("game_state") or {}).get("battle") or {}
        alive = battle.get("player_alive_count", self._last_alive)
        self._last_alive = alive
        return obs, self.reward(reward) + self._bonus * alive, terminated, truncated, info

    def reward(self, reward: float) -> float:
        return reward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="run-config YAML/JSON")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    cfg = load_run_config(args.config) if args.config else RunConfig(waves=10)
    env = SurvivalBonusWrapper(PokeRogueEnv.from_config(cfg, lean=False))

    battle_policy = MaxDamagePolicy()  # drop in Sb3Policy("battle.zip") here
    policy = PhaseRoutedPolicy(
        routes={"command": battle_policy, "target": battle_policy},
        default=ScriptedSkipPolicy(),
    )

    obs, info = env.reset()
    total = 0.0
    for step in range(args.steps):
        action = policy.act(obs, env.unwrapped.action_masks(), info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"step {step:>3} | wave {info.get('wave', '?')} | {info.get('phase', '?'):<14} "
              f"| action {action:>2} | reward {reward:+.2f}")
        if terminated or truncated:
            break
    print(f"\ntotal shaped reward: {total:+.2f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
