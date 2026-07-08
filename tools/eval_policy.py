#!/usr/bin/env python3
"""
Evaluate policies over N held-out episodes and report mean reward / waves.

    python3 tools/eval_policy.py --policies random maxdamage sb3:models/ppo_v9_first.zip \
        --episodes 10 --waves 20 --seed-prefix eval-v9

Seeds are f"{seed_prefix}-{i}" — keep the prefix DISJOINT from training
seeds (the trainer used seed families derived from the run config's seed).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rl.policy import make_builtin_policy  # noqa: E402
from rl.pokerogue_env import PokeRogueEnv  # noqa: E402


def run_episode(env: PokeRogueEnv, policy, seed: str) -> dict:
    obs, info = env.reset(options={"game_seed": seed} if seed else None)
    total = 0.0
    steps = 0
    while True:
        mask = env.action_masks()
        action = policy.act(obs, mask, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break
    gs = info.get("game_state") or {}
    wave = ((gs.get("battle") or {}).get("wave_index")) or info.get("wave")
    return {"reward": total, "steps": steps, "wave": wave, "victory": info.get("is_victory")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True,
                    help="random | maxdamage | firstlegal | sb3:<model.zip>")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--waves", type=int, default=20)
    ap.add_argument("--seed-prefix", default="eval-v9")
    args = ap.parse_args()

    results: dict[str, list[dict]] = {}
    for name in args.policies:
        policy = make_builtin_policy(name, seed=args.seed_prefix)
        env = PokeRogueEnv(waves=args.waves, seed=f"{args.seed_prefix}-0", lean=True)
        runs = []
        t0 = time.time()
        for i in range(args.episodes):
            r = run_episode(env, policy, seed=f"{args.seed_prefix}-{i}")
            runs.append(r)
            print(f"  [{name}] ep {i}: reward={r['reward']:.2f} steps={r['steps']} wave={r['wave']}", flush=True)
        env.close()
        results[name] = runs
        print(f"[{name}] {args.episodes} eps in {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'policy':<40} {'mean_reward':>12} {'median':>9} {'mean_wave':>10}")
    for name, runs in results.items():
        rewards = [r["reward"] for r in runs]
        waves = [r["wave"] for r in runs if r["wave"] is not None]
        print(
            f"{name:<40} {statistics.mean(rewards):>12.2f} {statistics.median(rewards):>9.2f} "
            f"{(statistics.mean(waves) if waves else float('nan')):>10.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
