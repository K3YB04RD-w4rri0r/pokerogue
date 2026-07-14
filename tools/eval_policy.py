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

from rl.pokerogue_env import PokeRogueEnv  # noqa: E402
from rl.policy import make_builtin_policy  # noqa: E402


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
    done_reason = info.get("done_reason")
    return {
        "reward": total,
        "steps": steps,
        "wave": wave,
        # env sets info["victory"] only on a real game_over win (wave 200);
        # under a wave cap the surrogate win is done_reason == "wave_cap"
        # (terminated since reward v2, with the waveCapReached bonus inside).
        "victory": bool(info.get("victory")),
        "terminated": terminated,
        "truncated": truncated,
        # Episode-end reason from the done payload (reward v2): wave_cap |
        # step_cap | livelock | lifecycle-command | None (game_over).
        "done_reason": done_reason,
        # A protocol error (step timeout / CLI died) surfaces as truncated with
        # info["protocol_error"] and a partial reward — NOT a real budget stop.
        # Flag it so the stats can exclude it instead of folding it into budget%.
        "error": bool(info.get("protocol_error")),
        # A livelock ending (CLI guard or env backstop) is a POLICY failure,
        # not a completed horizon — kept out of both win% and budget%.
        "livelock": done_reason == "livelock" or bool(info.get("livelock_truncation")),
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
            outcome = (
                "error" if r["error"]
                else "win" if r["victory"]
                else "cap" if r["done_reason"] == "wave_cap"
                else "livelock" if r["livelock"]
                else "loss" if r["terminated"]
                else "budget"
            )
            print(
                f"  [{name}] ep {i}: reward={r['reward']:.2f} steps={r['steps']} wave={r['wave']} ({outcome})",
                flush=True,
            )
        env.close()
        results[name] = runs
        print(f"[{name}] {args.episodes} eps in {time.time() - t0:.0f}s", flush=True)

    print(
        f"\n{'policy':<40} {'mean_reward':>12} {'±sem':>7} {'median':>9} {'mean_wave':>10} "
        f"{'win%':>6} {'cap%':>6} {'budget%':>8} {'lvlk':>5} {'err':>4}"
    )
    for name, runs in results.items():
        # Exclude protocol-error episodes from every statistic — their partial
        # reward and spurious `truncated` flag are not a real outcome. NOTE:
        # this is policy-dependent censoring (errors correlate with long/deep
        # episodes), so the excluded seeds are LISTED so paired-seed
        # comparisons can be re-run on the intersection.
        valid = [r for r in runs if not r["error"]]
        errors = len(runs) - len(valid)
        excluded = [i for i, r in enumerate(runs) if r["error"]]
        rewards = [r["reward"] for r in valid]
        waves = [r["wave"] for r in valid if r["wave"] is not None]
        denom = len(valid)
        wins = (sum(1 for r in valid if r["victory"]) / denom) if denom else float("nan")
        # cap% — surrogate wins (reached the wave cap; terminated with the
        # clean-ratio waveCapReached bonus since reward v2)
        caps = (sum(1 for r in valid if r["done_reason"] == "wave_cap") / denom) if denom else float("nan")
        livelocks = sum(1 for r in valid if r["livelock"])
        # budget% — step_cap (or other unpenalized truncations): ran out of
        # step budget without stalling; wave caps are counted in cap% now.
        budget = (sum(1 for r in valid if r["truncated"] and not r["livelock"]) / denom) if denom else float("nan")
        mean_reward = statistics.mean(rewards) if rewards else float("nan")
        # Roguelike returns are heavy-tailed; a mean over ~10 episodes without
        # dispersion invites reading noise as signal.
        sem = (statistics.stdev(rewards) / (len(rewards) ** 0.5)) if len(rewards) > 1 else float("nan")
        median_reward = statistics.median(rewards) if rewards else float("nan")
        mean_wave = statistics.mean(waves) if waves else float("nan")
        print(
            f"{name:<40} {mean_reward:>12.2f} {sem:>7.2f} {median_reward:>9.2f} "
            f"{mean_wave:>10.1f} {100 * wins:>5.0f}% {100 * caps:>5.0f}% {100 * budget:>7.0f}% {livelocks:>5} {errors:>4}"
        )
        if excluded:
            print(f"    excluded (protocol errors) seeds: {[f'{args.seed_prefix}-{i}' for i in excluded]}")
    if args.waves < 200:
        print(
            f"\nnote: win%% requires reaching wave 200; under --waves {args.waves} it is structurally 0 — "
            "cap%% (wave cap reached, paid the waveCapReached bonus) is the surrogate win rate."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
