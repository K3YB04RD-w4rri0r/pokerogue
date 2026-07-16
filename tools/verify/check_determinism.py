#!/usr/bin/env python3
"""
Determinism check: the same seed (and, in interactive mode, the same action
seed) must produce an identical trajectory twice.

Compares per step: canonical gameState hash (timestamp stripped), the bitwise
TS observation (obsB64 string), chosen action, and phase.

Modes:
    auto        — two fresh processes, deterministic default policy
    interactive — two fresh processes, seeded random policy
    inprocess   — THE cross-episode leakage gate: one process plays the same
                  seed twice via {"cmd":"reset"}; both episodes must be
                  bitwise identical to each other AND to a fresh-process
                  reference episode

Usage:
    python3 tools/verify/check_determinism.py --seed det-1 --waves 8 --mode auto
    python3 tools/verify/check_determinism.py --seed det-1 --waves 8 --mode interactive --action-seed 42
    python3 tools/verify/check_determinism.py --seed det-1 --waves 8 --mode inprocess --action-seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (  # noqa: E402
    ACTION_RUN,
    CLI_PATH,
    REPO_ROOT,
    canonical_state_hash,
    diff_json,
    kill_proc,
    read_json,
    require_cli,
    send_action,
    spawn_cli,
)


def run_auto(seed: str, waves: int, dump_path: Path, extra_args: list[str] | None = None) -> None:
    subprocess.run(
        ["node", str(CLI_PATH), f"--seed={seed}", f"--waves={waves}", f"--dump-obs={dump_path}", *(extra_args or [])],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=600,
    )


def run_interactive(seed: str, waves: int, action_seed: str, dump_path: Path) -> None:
    import run_episodes

    r = run_episodes.run_episode(
        seed, waves, action_seed, probe_invalid=0.0, dump_dir=None, boot_timeout=180, step_timeout=60
    )
    # run_episode owns the dump path layout only when dump_dir is set; redo with explicit dir
    if r["errors"]:
        sys.exit(f"interactive episode failed: {r['errors']}")


def _play_one_episode(proc, reader, seed: str, action_seed: str) -> None:
    """Drive one episode with the seeded masked-random policy (after ready)."""
    rng = random.Random(f"{seed}|{action_seed}")
    while True:
        msg = read_json(reader, 60)
        if msg is None:
            raise RuntimeError("EOF mid-episode")
        mtype = msg.get("type")
        if mtype in ("info", "warning", "game_over"):
            continue
        if mtype == "error":
            raise RuntimeError(f"CLI error: {msg.get('message')}")
        if mtype == "done":
            return
        if mtype != "state":
            continue
        valid = [a["index"] for a in msg.get("actions", [])]
        if not valid:
            raise RuntimeError(f"no valid actions at step {msg.get('step')}")
        choices = [a for a in valid if a != ACTION_RUN]
        if not choices or (ACTION_RUN in valid and rng.random() < 0.1):
            action = ACTION_RUN if ACTION_RUN in valid else rng.choice(choices)
        else:
            action = rng.choice(choices)
        send_action(proc, action)


def run_inprocess(
    seed: str, waves: int, action_seed: str, dump_path: Path, episodes: int, extra_args: list[str] | None = None
) -> None:
    """One process, `episodes` runs of the SAME seed via {"cmd":"reset"}."""
    proc, reader, stderr_fh = spawn_cli(seed, waves, interactive=True, dump_path=dump_path, extra_args=extra_args)
    try:
        for ep in range(episodes):
            ready = read_json(reader, 180)
            # skip leftover messages until ready (none expected, but robust)
            while ready is not None and ready.get("type") != "ready":
                ready = read_json(reader, 180)
            if ready is None:
                raise RuntimeError(f"no ready before episode {ep}")
            _play_one_episode(proc, reader, seed, action_seed)
            if ep < episodes - 1:
                proc.stdin.write(json.dumps({"cmd": "reset", "seed": seed, "waves": waves}) + "\n")
                proc.stdin.flush()
        proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=30)
    finally:
        kill_proc(proc, stderr_fh)


def load_episode_steps(path: Path) -> list[list[dict]]:
    """Split a multi-episode dump into per-episode step lists (summary = boundary)."""
    episodes: list[list[dict]] = [[]]
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "step":
                episodes[-1].append(rec)
            elif rec.get("kind") == "summary":
                episodes.append([])
    return [ep for ep in episodes if ep]


def load_steps(path: Path) -> list[dict]:
    steps = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "step":
                steps.append(rec)
    return steps


def compare(a: list[dict], b: list[dict]) -> int:
    if len(a) != len(b):
        print(f"FAIL: step counts differ: {len(a)} vs {len(b)}")
        n = min(len(a), len(b))
    else:
        n = len(a)

    for i in range(n):
        ra, rb = a[i], b[i]
        fields = {
            "phase": (ra["phase"], rb["phase"]),
            "chosenAction": (ra.get("chosenAction"), rb.get("chosenAction")),
            "obsB64": (ra["obsB64"], rb["obsB64"]),
            "stateHash": (canonical_state_hash(ra["gameState"]), canonical_state_hash(rb["gameState"])),
            # Reward and mask are what training actually consumes; comparing
            # them closes the gap where calculator/guard state could drift
            # across in-process episodes while the chosen trajectory matches.
            "reward": (ra.get("reward"), rb.get("reward")),
            "actionMask": (ra.get("actionMask"), rb.get("actionMask")),
        }
        bad = {k: v for k, v in fields.items() if v[0] != v[1]}
        if bad:
            print(f"FAIL: first divergence at step {i} (phase {ra['phase']} vs {rb['phase']}):")
            for k, (va, vb) in bad.items():
                if k in ("obsB64", "stateHash"):
                    print(f"  {k}: differs")
                else:
                    print(f"  {k}: {va!r} != {vb!r}")
            if "stateHash" in bad:
                print("  gameState diff (first 20 paths):")
                for d in diff_json(ra["gameState"], rb["gameState"], limit=20):
                    print(f"    {d}")
            return 1

    if len(a) != len(b):
        return 1
    if not a:
        print("FAIL: 0 steps compared — determinism check would pass vacuously on an empty dump")
        return 1
    print(f"OK: {len(a)} steps identical (state hash, observation bytes, action, phase)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=str, default="det-1")
    ap.add_argument("--waves", type=int, default=8)
    ap.add_argument("--mode", choices=["auto", "interactive", "inprocess"], default="auto")
    ap.add_argument("--action-seed", type=str, default="42")
    # Repeatable raw CLI arg passthrough (same flag as run_episodes.py), e.g.
    # --cli-arg=--override=STARTING_WAVE_OVERRIDE=35 to pin the episode into
    # grunt-trainer territory the masked-random policy never reaches itself.
    ap.add_argument("--cli-arg", action="append", default=[])
    args = ap.parse_args()

    require_cli()

    if args.mode == "inprocess":
        # Cross-episode leakage gate: one process replays the same seed twice
        # via {"cmd":"reset"}; a fresh process provides the reference.
        with tempfile.TemporaryDirectory(prefix="rl-det-") as td:
            ip_dump = Path(td) / "inprocess.jsonl"
            ref_dump = Path(td) / "reference.jsonl"
            run_inprocess(args.seed, args.waves, args.action_seed, ip_dump, episodes=2, extra_args=args.cli_arg)
            print("in-process run complete (2 episodes, 1 process)")
            run_inprocess(args.seed, args.waves, args.action_seed, ref_dump, episodes=1, extra_args=args.cli_arg)
            print("fresh-process reference complete")

            ip_eps = load_episode_steps(ip_dump)
            ref_eps = load_episode_steps(ref_dump)
            if len(ip_eps) != 2 or len(ref_eps) != 1:
                print(f"FAIL: expected 2 in-process + 1 reference episodes, got {len(ip_eps)} + {len(ref_eps)}")
                return 1

            print("\n[1/2] in-process episode 1 vs episode 2 (same process, after reset):")
            rc = compare(ip_eps[0], ip_eps[1])
            if rc:
                return rc
            print("[2/2] in-process episode 1 vs fresh-process reference:")
            return compare(ip_eps[0], ref_eps[0])

    with tempfile.TemporaryDirectory(prefix="rl-det-") as td:
        dumps = [Path(td) / "run1.jsonl", Path(td) / "run2.jsonl"]
        for dump in dumps:
            if args.mode == "auto":
                run_auto(args.seed, args.waves, dump, extra_args=args.cli_arg)
            else:
                # Interactive determinism: same game seed + same action RNG seed
                import run_episodes

                dump_dir = dump.parent / dump.stem
                dump_dir.mkdir()
                r = run_episodes.run_episode(
                    args.seed,
                    args.waves,
                    args.action_seed,
                    probe_invalid=0.0,
                    dump_dir=dump_dir,
                    boot_timeout=180,
                    step_timeout=60,
                    extra_args=args.cli_arg,
                )
                if r["result"] not in ("game_over", "step_cap", "wave_cap", "livelock") or r["errors"]:
                    sys.exit(f"interactive episode did not finish cleanly: {r['result']} {r['errors']}")
                (dump_dir / f"{args.seed}.jsonl").rename(dump)
            print(f"run complete -> {dump.name}")

        return compare(load_steps(dumps[0]), load_steps(dumps[1]))


if __name__ == "__main__":
    sys.exit(main())
