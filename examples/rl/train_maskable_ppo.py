#!/usr/bin/env python3
"""
Minimal MaskablePPO training stub for the PokeRogue env.

Requires the optional training extras:
    pip install stable-baselines3 sb3-contrib torch

Usage (from the repo root, after `pnpm rl:build`):
    python3 examples/rl/train_maskable_ppo.py [--timesteps 1000] [--waves 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    sys.exit("sb3-contrib not installed — pip install stable-baselines3 sb3-contrib torch")

from rl.pokerogue_env import PokeRogueEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=1_000)
    ap.add_argument("--waves", type=int, default=10)
    ap.add_argument("--save", type=str, default=None, help="path to save the model zip")
    args = ap.parse_args()

    env = PokeRogueEnv(waves=args.waves)
    # PokeRogueEnv exposes action_masks() directly; ActionMasker makes the
    # contract explicit and works with vectorized setups too.
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())

    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64)
    model.learn(total_timesteps=args.timesteps)

    if args.save:
        model.save(args.save)
        print(f"saved model to {args.save}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
