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
    from stable_baselines3.common.vec_env import SubprocVecEnv
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
    ap.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="parallel game processes (SubprocVecEnv); each is a full headless game "
        "(~250MB rss, ~1 core). Rule of thumb: cores - 2. Default: train.num_envs or 1",
    )
    ap.add_argument("--device", default=None, help="torch device (cpu | cuda | cuda:0); default train.device or auto")
    ap.add_argument(
        "--net-arch",
        default=None,
        help="comma-separated hidden sizes, e.g. 1024,512 — sb3's default 64,64 is undersized "
        "for the 6,991-dim observation; default train.net_arch or sb3 default",
    )
    ap.add_argument("--n-steps", type=int, default=None, help="rollout length per env (default train.n_steps or 256)")
    ap.add_argument("--checkpoint-every", type=int, default=None,
                    help="save <save>.ckpt-<steps>.zip every N timesteps (default train.checkpoint_every; 0=off)")
    ap.add_argument("--tensorboard", default=None, help="tensorboard log dir (default train.tensorboard)")
    args = ap.parse_args()

    cfg = load_run_config(args.config) if args.config else RunConfig()
    if args.waves is not None:
        cfg.waves = args.waves
    if cfg.waves is None:
        cfg.waves = 10
    timesteps = args.timesteps if args.timesteps is not None else int(cfg.train.get("timesteps", 1_000))
    save_path = args.save if args.save is not None else cfg.train.get("save")

    num_envs = args.num_envs if args.num_envs is not None else int(cfg.train.get("num_envs", 1))

    def make_env(rank: int):
        def _init():
            # per-worker seed family keeps parallel episodes decorrelated
            # while staying reproducible
            env = PokeRogueEnv.from_config(cfg, seed=f"{cfg.seed or 'train'}-w{rank}")
            return ActionMasker(env, lambda e: e.unwrapped.action_masks())

        return _init

    if num_envs > 1:
        # Each worker is its own node process — throughput scales ~linearly
        # with cores (the game, not the network, is the bottleneck).
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    else:
        env = make_env(0)()

    device = args.device or cfg.train.get("device") or "auto"
    net_arch_raw = args.net_arch or cfg.train.get("net_arch")
    policy_kwargs = {}
    if net_arch_raw:
        policy_kwargs["net_arch"] = [int(x) for x in str(net_arch_raw).split(",")]
    n_steps = args.n_steps if args.n_steps is not None else int(cfg.train.get("n_steps", 256))
    tb = args.tensorboard or cfg.train.get("tensorboard")

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=int(cfg.train.get("batch_size", 64)),
        device=device,
        policy_kwargs=policy_kwargs or None,
        tensorboard_log=tb,
    )

    callback = None
    ckpt_every = args.checkpoint_every if args.checkpoint_every is not None else int(cfg.train.get("checkpoint_every", 0))
    if ckpt_every and save_path:
        from stable_baselines3.common.callbacks import CheckpointCallback

        ckpt_dir = str(Path(save_path).parent)
        callback = CheckpointCallback(
            save_freq=max(ckpt_every // max(num_envs, 1), 1),
            save_path=ckpt_dir,
            name_prefix=Path(save_path).stem + ".ckpt",
        )
    model.learn(total_timesteps=timesteps, callback=callback)

    if save_path:
        model.save(save_path)
        print(f"saved model to {save_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
