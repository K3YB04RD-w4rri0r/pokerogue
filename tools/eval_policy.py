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
    return {
        "reward": total,
        "steps": steps,
        "wave": wave,
        # env sets info["victory"] only on real game_over; a wave-budget
        # stop is truncated (no victory key) — keep the three outcomes
        # distinguishable in metrics
        "victory": bool(info.get("victory")),
        "terminated": terminated,
        "truncated": truncated,
    }


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
        # maxdamage reads move power/category from info["game_state"] (same as
        # run_policy.py). Under lean=True that dict is empty, so the policy
        # silently degrades to first-legal-action — mis-measuring the baseline.
        # sb3/random/firstlegal only need obs+mask, so they keep the faster lean.
        env = PokeRogueEnv(waves=args.waves, seed=f"{args.seed_prefix}-0", lean=(name != "maxdamage"))
        runs = []
        t0 = time.time()
        for i in range(args.episodes):
            r = run_episode(env, policy, seed=f"{args.seed_prefix}-{i}")
            runs.append(r)
            outcome = "win" if r["victory"] else ("loss" if r["terminated"] else "budget")
            print(
                f"  [{name}] ep {i}: reward={r['reward']:.2f} steps={r['steps']} wave={r['wave']} ({outcome})",
                flush=True,
            )
        env.close()
        results[name] = runs
        print(f"[{name}] {args.episodes} eps in {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'policy':<40} {'mean_reward':>12} {'median':>9} {'mean_wave':>10} {'win%':>6} {'budget%':>8}")
    for name, runs in results.items():
        rewards = [r["reward"] for r in runs]
        waves = [r["wave"] for r in runs if r["wave"] is not None]
        wins = sum(1 for r in runs if r["victory"]) / len(runs)
        budget = sum(1 for r in runs if r["truncated"]) / len(runs)
        print(
            f"{name:<40} {statistics.mean(rewards):>12.2f} {statistics.median(rewards):>9.2f} "
            f"{(statistics.mean(waves) if waves else float('nan')):>10.1f} {100 * wins:>5.0f}% {100 * budget:>7.0f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
