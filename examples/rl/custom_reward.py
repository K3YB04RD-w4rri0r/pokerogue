#!/usr/bin/env python3
"""
Bring-your-own reward — runnable template.

Shows the two ways to fully own the reward signal (src/rl/reward.py):
  A. compose weighted components without writing a class
  B. write a RewardFn from scratch over the full game state

Run (needs the training extras only for a real train loop; this demo just
drives a few steps with a scripted policy):

    python3 examples/rl/custom_reward.py --config examples/rl/first_train.yaml

The built-in TS reward is a sane DEFAULT; a CustomReward REPLACES it, so
every shaping choice (money scale, KO weighting, shop-buy farming, …) is
yours. The env must run with lean=False so info["game_state"] is present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from rl.pokerogue_env import PokeRogueEnv  # noqa: E402
from rl.policy import make_builtin_policy  # noqa: E402
from rl.reward import (  # noqa: E402
    ComponentReward,
    CustomReward,
    RewardBase,
    delta_component,
    money,
    player_alive_count,
    player_hp_fraction,
    wave,
)
from rl.run_config import RunConfig, load_run_config  # noqa: E402


# ── A. composed from weighted components ───────────────────────────────
def build_component_reward() -> ComponentReward:
    return ComponentReward(
        {
            # +10 per wave cleared (delta, positives only)
            "depth": (10.0, delta_component(wave)),
            # a gentler money scale than the built-in default
            "wealth": (0.0005, delta_component(money)),
        }
    )


# ── B. written from scratch ────────────────────────────────────────────
class DepthSurvival(RewardBase):
    """Reward depth and mean party HP; penalize losing a party member."""

    def __call__(self, prev, cur, info) -> float:
        r = 10.0 * max(0, wave(cur) - wave(prev))
        r += 2.0 * (player_hp_fraction(cur) - player_hp_fraction(prev))  # dense HP shaping
        r -= 3.0 * max(0, player_alive_count(prev) - player_alive_count(cur))  # -3 per faint
        return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="run-config YAML/JSON")
    ap.add_argument("--reward", choices=["component", "scratch"], default="scratch")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    cfg = load_run_config(args.config) if args.config else RunConfig()
    if cfg.waves is None:
        cfg.waves = 10

    # lean=False is REQUIRED: CustomReward reads info["game_state"].
    env = PokeRogueEnv.from_config(cfg, lean=False)
    reward_fn = build_component_reward() if args.reward == "component" else DepthSurvival()
    env = CustomReward(env, reward_fn)

    policy = make_builtin_policy("maxdamage")
    obs, info = env.reset()
    total = 0.0
    for _ in range(args.steps):
        action = policy.act(obs, env.action_masks(), info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"wave {info.get('wave')} | custom reward {reward:+.3f} | cumulative {total:+.3f}", flush=True)
        if terminated or truncated:
            break
    env.close()
    print(f"\ntotal custom reward over the run: {total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
