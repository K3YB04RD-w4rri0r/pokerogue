#!/usr/bin/env python3
"""
Masked-random rollout against the PokeRogue gymnasium env.

Usage (from the repo root, after `pnpm rl:build`):
    python3 examples/rl/random_agent.py [--episodes 3] [--waves 10] [--seed 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from rl.pokerogue_env import PokeRogueEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--waves", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = PokeRogueEnv(waves=args.waves)
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            t0 = time.time()
            obs, info = env.reset(seed=args.seed + ep)
            total_reward, steps = 0.0, 0
            terminated = truncated = False

            while not (terminated or truncated):
                mask = env.action_masks()
                valid = np.flatnonzero(mask)
                action = int(rng.choice(valid))
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            outcome = "victory" if info.get("victory") else ("defeat" if terminated else "truncated")
            print(
                f"episode {ep}: {outcome} after {steps} steps, wave {info.get('wave')}, "
                f"total reward {total_reward:.2f}, obs[{obs.shape[0]}] finite={np.isfinite(obs).all()}, "
                f"{time.time() - t0:.1f}s"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
