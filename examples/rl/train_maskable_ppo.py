#!/usr/bin/env python3
"""
Minimal MaskablePPO training stub for the PokeRogue env.

Requires the optional training extras:
    pip install stable-baselines3 sb3-contrib torch

Usage (from the repo root, after `pnpm rl:build`):
    python3 examples/rl/train_maskable_ppo.py [--timesteps 1000] [--waves 10]
    python3 examples/rl/train_maskable_ppo.py --config examples/rl/legendary.yaml

A run config (src/rl/run_config.py) describes the whole run — seed, starters,
starting wave/level/money/items, game overrides, reward shaping — and its
`train:` section can carry timesteps/save so one file defines the experiment.
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
from rl.run_config import RunConfig, load_run_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="run-config YAML/JSON; CLI flags override it")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--waves", type=int, default=None)
    ap.add_argument("--save", type=str, default=None, help="path to save the model zip")
    args = ap.parse_args()

    cfg = load_run_config(args.config) if args.config else RunConfig()
    if args.waves is not None:
        cfg.waves = args.waves
    if cfg.waves is None:
        cfg.waves = 10
    timesteps = args.timesteps if args.timesteps is not None else int(cfg.train.get("timesteps", 1_000))
    save_path = args.save if args.save is not None else cfg.train.get("save")

    env = PokeRogueEnv.from_config(cfg)
    # PokeRogueEnv exposes action_masks() directly; ActionMasker makes the
    # contract explicit and works with vectorized setups too.
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())

    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64)
    model.learn(total_timesteps=timesteps)

    if save_path:
        model.save(save_path)
        print(f"saved model to {save_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
