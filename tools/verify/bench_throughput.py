#!/usr/bin/env python3
"""
Throughput + soak benchmark: training-feasibility numbers for the RL env.

Measures:
  - end-to-end steps/second of the interactive loop (masked-random policy)
  - Python-side cost split: parse_game_state vs encode_observation per step
  - episode boot time
  - soak: N sequential episodes watching boot time, steps/s and node RSS drift
    (catches the MockClock timer-accumulation leak flagged in headless-boot.ts)

Usage:
    python3 tools/verify/bench_throughput.py --episodes 5 --waves 10
    python3 tools/verify/bench_throughput.py --soak 30 --waves 3
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from common import (  # noqa: E402
    ACTION_RUN,
    REPO_ROOT,
    kill_proc,
    read_json,
    send_action,
    spawn_cli,
)
from rl.observation import encode_observation, parse_game_state  # noqa: E402


def node_rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def bench_episode(seed: str, waves: int, action_seed: str) -> dict:
    t_spawn = time.perf_counter()
    proc, reader, stderr_fh = spawn_cli(seed, waves, interactive=True)
    rng = random.Random(f"{seed}|{action_seed}")

    out = {
        "seed": seed,
        "boot_s": None,
        "steps": 0,
        "wall_s": None,
        "steps_per_s": None,
        "parse_ms_avg": None,
        "encode_ms_avg": None,
        "rss_mb_start": None,
        "rss_mb_end": None,
        "result": "incomplete",
    }
    parse_ms: list[float] = []
    encode_ms: list[float] = []

    try:
        msg = read_json(reader, 180)
        if not msg or msg.get("type") != "ready":
            out["result"] = f"no-ready: {msg}"
            return out
        out["boot_s"] = round(time.perf_counter() - t_spawn, 2)
        out["rss_mb_start"] = node_rss_mb(proc.pid)

        t_loop = time.perf_counter()
        while True:
            msg = read_json(reader, 60)
            if msg is None:
                break
            mtype = msg.get("type")
            if mtype in ("info", "warning", "error"):
                continue
            if mtype == "game_over":
                out["result"] = "game_over"
                continue
            if mtype == "done":
                if out["result"] == "incomplete":
                    out["result"] = "step_cap"
                break
            if mtype != "state":
                continue

            gs = msg.get("gameState") or {}
            t0 = time.perf_counter()
            state = parse_game_state(gs)
            t1 = time.perf_counter()
            encode_observation(state)
            t2 = time.perf_counter()
            parse_ms.append((t1 - t0) * 1000)
            encode_ms.append((t2 - t1) * 1000)

            valid = [a["index"] for a in msg.get("actions", [])]
            if not valid:
                break
            choices = [a for a in valid if a != ACTION_RUN] or valid
            send_action(proc, rng.choice(choices))
            out["steps"] += 1

        out["wall_s"] = round(time.perf_counter() - t_loop, 2)
        out["rss_mb_end"] = node_rss_mb(proc.pid)
        if out["wall_s"] and out["steps"]:
            out["steps_per_s"] = round(out["steps"] / out["wall_s"], 1)
        if parse_ms:
            out["parse_ms_avg"] = round(sum(parse_ms) / len(parse_ms), 2)
            out["encode_ms_avg"] = round(sum(encode_ms) / len(encode_ms), 2)
    finally:
        kill_proc(proc, stderr_fh)

    return out


def bench_inprocess(
    n_episodes: int, waves: int, action_seed: str, stderr_log: str | None = None, recycle_every: int = 0
) -> list[dict]:
    """One process, n episodes via {"cmd":"reset"}: reset latency + drift.

    recycle_every: respawn the node process after N episodes (mirrors the
    wrapper's respawn_every policy). 0 = never (pure leak-hunting soak).
    """
    import json as _json

    results: list[dict] = []
    proc, reader, stderr_fh = spawn_cli("bench-ip", waves, interactive=True, stderr_path=stderr_log)
    rng = random.Random(action_seed)
    try:
        for ep in range(n_episodes):
            if recycle_every and ep > 0 and ep % recycle_every == 0:
                try:
                    proc.stdin.write(_json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=15)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    pass
                kill_proc(proc, stderr_fh)
                proc, reader, stderr_fh = spawn_cli(f"bench-ip-r{ep}", waves, interactive=True, stderr_path=stderr_log)
            t0 = time.perf_counter()
            msg = read_json(reader, 180)
            while msg is not None and msg.get("type") != "ready":
                msg = read_json(reader, 180)
            if msg is None:
                results.append({"ep": ep, "result": "no-ready"})
                break
            reset_s = round(time.perf_counter() - t0, 3)

            steps = 0
            result = "incomplete"
            t_loop = time.perf_counter()
            while True:
                msg = read_json(reader, 60)
                if msg is None:
                    result = "eof"
                    break
                mtype = msg.get("type")
                if mtype in ("info", "warning"):
                    continue
                if mtype == "game_over":
                    result = "game_over"
                    continue
                if mtype in ("done", "error"):
                    if result == "incomplete":
                        result = "step_cap" if mtype == "done" else "error"
                    break
                if mtype != "state":
                    continue
                valid = [a["index"] for a in msg.get("actions", [])]
                choices = [a for a in valid if a != ACTION_RUN] or valid
                send_action(proc, rng.choice(choices))
                steps += 1
            wall = time.perf_counter() - t_loop
            results.append(
                {
                    "ep": ep,
                    "reset_s": reset_s,
                    "steps": steps,
                    "steps_per_s": round(steps / wall, 1) if wall > 0 and steps else None,
                    "rss_mb": node_rss_mb(proc.pid),
                    "result": result,
                }
            )
            if result in ("eof", "error"):
                results.append({"ep": ep + 1, "result": f"aborted (node exit={proc.poll()})"})
                break
            if ep < n_episodes - 1:
                try:
                    proc.stdin.write(_json.dumps({"cmd": "reset", "seed": f"bench-ip-{ep + 1}", "waves": waves}) + "\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    results.append({"ep": ep + 1, "result": f"broken pipe (node exit={proc.poll()})"})
                    break
        try:
            proc.stdin.write(_json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=15)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
    finally:
        kill_proc(proc, stderr_fh)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3, help="benchmark episodes")
    ap.add_argument("--soak", type=int, default=0, help="run N soak episodes (drift check)")
    ap.add_argument("--inprocess", action="store_true", help="soak/bench in ONE process via cmd:reset")
    ap.add_argument("--waves", type=int, default=10)
    ap.add_argument("--action-seed", type=str, default="bench")
    ap.add_argument("--out", type=str, default=".rl-verify/bench.json")
    ap.add_argument("--stderr-log", type=str, default=None, help="capture node stderr (crash diagnosis)")
    ap.add_argument("--recycle-every", type=int, default=0, help="respawn the process after N episodes (wrapper policy)")
    args = ap.parse_args()

    if args.inprocess:
        n = args.soak if args.soak else args.episodes
        results = bench_inprocess(
            n, args.waves, args.action_seed, stderr_log=args.stderr_log, recycle_every=args.recycle_every
        )
        print(f"{'ep':>4} {'reset_s':>8} {'steps':>6} {'steps/s':>8} {'rss_mb':>7} result")
        for r in results:
            print(
                f"{r['ep']:>4} {r.get('reset_s', -1):>8} {r.get('steps', 0):>6} "
                f"{r.get('steps_per_s') or -1:>8} {(r.get('rss_mb') or 0):>7.0f} {r.get('result')}"
            )
        done = [r for r in results if r.get("steps_per_s")]
        if len(done) >= 10:
            first, last = done[: len(done) // 3], done[-(len(done) // 3) :]
            f_rss = sum(r["rss_mb"] for r in first if r["rss_mb"]) / len(first)
            l_rss = sum(r["rss_mb"] for r in last if r["rss_mb"]) / len(last)
            f_sps = sum(r["steps_per_s"] for r in first) / len(first)
            l_sps = sum(r["steps_per_s"] for r in last) / len(last)
            resets = [r["reset_s"] for r in done[1:]]
            print(f"\n  reset latency (in-process): min={min(resets)}s median={sorted(resets)[len(resets) // 2]}s max={max(resets)}s")
            print(f"  drift: rss {f_rss:.0f}MB -> {l_rss:.0f}MB, steps/s {f_sps:.0f} -> {l_sps:.0f}")
            out_path = REPO_ROOT / args.out
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"inprocess_episodes": results}, indent=2))
            if l_rss > f_rss * 1.5 or l_sps < f_sps * 0.66:
                print("  IN-PROCESS SOAK: FAIL (rss/throughput drifted >50%)")
                return 1
            print("  IN-PROCESS SOAK: OK")
        return 0

    episodes = []
    n = args.soak if args.soak else args.episodes
    label = "soak" if args.soak else "bench"

    print(f"{'ep':>4} {'boot_s':>7} {'steps':>6} {'steps/s':>8} {'parse_ms':>9} {'enc_ms':>7} {'rss_mb':>7} result")
    for i in range(n):
        r = bench_episode(f"{label}-{i}", args.waves, args.action_seed)
        episodes.append(r)
        print(
            f"{i:>4} {r['boot_s'] or -1:>7} {r['steps']:>6} {r['steps_per_s'] or -1:>8} "
            f"{r['parse_ms_avg'] or -1:>9} {r['encode_ms_avg'] or -1:>7} "
            f"{(r['rss_mb_end'] or 0):>7.0f} {r['result']}"
        )

    done = [e for e in episodes if e["steps_per_s"]]
    if done:
        sps = [e["steps_per_s"] for e in done]
        boots = [e["boot_s"] for e in done if e["boot_s"]]
        print("\nsummary:")
        print(f"  steps/s: min={min(sps)} median={sorted(sps)[len(sps) // 2]} max={max(sps)}")
        print(f"  boot_s:  min={min(boots)} max={max(boots)}")
        print(f"  python parse+encode per step: {done[0]['parse_ms_avg']}+{done[0]['encode_ms_avg']}ms (first ep)")
        if args.soak and len(done) >= 10:
            first, last = done[: len(done) // 3], done[-len(done) // 3 :]
            f_boot = sum(e["boot_s"] for e in first) / len(first)
            l_boot = sum(e["boot_s"] for e in last) / len(last)
            f_sps = sum(e["steps_per_s"] for e in first) / len(first)
            l_sps = sum(e["steps_per_s"] for e in last) / len(last)
            print(f"  soak drift: boot {f_boot:.1f}s -> {l_boot:.1f}s, steps/s {f_sps:.0f} -> {l_sps:.0f}")
            if l_boot > f_boot * 1.5 or l_sps < f_sps * 0.66:
                print("  SOAK: FAIL (latency/throughput drifted >50%)")
                return 1
            print("  SOAK: OK")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"episodes": episodes}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
