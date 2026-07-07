#!/usr/bin/env python3
"""
Smoke / hang / mask-property driver for the headless RL CLI.

Spawns interactive episodes, plays a seeded masked-random policy, and asserts:
  - episodes terminate (game_over / step cap), no hangs (watchdog)
  - actions chosen from the mask NEVER produce a `warning` (mask validity)
  - optional probes: deliberately invalid actions DO produce a graceful warning
  - phase coverage: `command` and `modifier` phases must appear

Usage:
    python3 tools/verify/run_episodes.py --seeds 5 --waves 10 --probe-invalid 0.02 --dump-dir .rl-verify/smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from common import (  # noqa: E402
    ACTION_RUN,
    HangTimeout,
    decode_obs_b64,
    kill_proc,
    read_json,
    send_action,
    spawn_cli,
)


def run_episode(
    seed: str,
    waves: int,
    action_seed: str,
    probe_invalid: float,
    dump_dir: Path | None,
    boot_timeout: float,
    step_timeout: float,
    run_weight: float = 0.1,
    extra_args: list[str] | None = None,
) -> dict:
    """Play one episode; returns a result dict."""
    dump_path = dump_dir / f"{seed}.jsonl" if dump_dir else None
    stderr_path = dump_dir / f"{seed}.stderr.log" if dump_dir else None
    proc, reader, stderr_fh = spawn_cli(
        seed, waves, interactive=True, dump_path=dump_path, stderr_path=stderr_path, extra_args=extra_args
    )
    rng = random.Random(f"{seed}|{action_seed}")

    result = {
        "seed": seed,
        "steps": 0,
        "result": "incomplete",
        "victory": None,
        "phases": {},
        "unexpected_warnings": [],
        "probes_sent": 0,
        "probes_acknowledged": 0,
        "errors": [],
        "ready": None,
        "total_reward": 0.0,
    }
    expect_warning = False

    try:
        msg = read_json(reader, boot_timeout)
        if not msg or msg.get("type") != "ready":
            result["errors"].append(f"no ready message, got: {msg}")
            return result
        result["ready"] = {k: msg.get(k) for k in ("obsDim", "actionDim", "protocolVersion", "bootTime")}

        while True:
            msg = read_json(reader, step_timeout)
            if msg is None:
                result["errors"].append("EOF before done message")
                break
            mtype = msg.get("type")

            if mtype == "info":
                continue
            if mtype == "warning":
                if expect_warning:
                    result["probes_acknowledged"] += 1
                    expect_warning = False
                else:
                    result["unexpected_warnings"].append(msg.get("message", ""))
                continue
            if mtype == "error":
                if "Timeout" in str(msg.get("message", "")):
                    result["result"] = "router_timeout"
                else:
                    result["errors"].append(str(msg.get("message")))
                continue
            if mtype == "game_over":
                result["victory"] = bool(msg.get("victory"))
                result["total_reward"] += float(msg.get("reward", 0.0))
                result["result"] = "game_over"
                continue
            if mtype == "done":
                if result["result"] == "incomplete":
                    result["result"] = "step_cap"
                result["steps"] = int(msg.get("steps", result["steps"]))
                break
            if mtype != "state":
                result["errors"].append(f"unknown message type: {mtype}")
                continue

            # state message
            expect_warning = False
            phase = msg.get("phase", "?")
            result["phases"][phase] = result["phases"].get(phase, 0) + 1
            result["total_reward"] += float(msg.get("reward", 0.0))
            valid = [a["index"] for a in msg.get("actions", [])]
            mask = msg.get("mask") or (msg.get("gameState") or {}).get("phase", {}).get("action_mask") or []
            if not valid:
                result["errors"].append(f"step {msg.get('step')}: no valid actions in phase {phase}")
                break

            # Wire-authority spot check (~5% of steps): the message's obsB64
            # must equal a local re-encode of the message's gameState
            if "obsB64" in msg and msg.get("gameState") and rng.random() < 0.05:
                from rl.observation import encode_observation, parse_game_state

                local = encode_observation(parse_game_state(msg["gameState"]))
                wire = decode_obs_b64(msg["obsB64"])
                if local.tobytes() != wire.tobytes():
                    result["errors"].append(f"step {msg.get('step')}: wire obsB64 != local encode")

            if probe_invalid > 0 and rng.random() < probe_invalid:
                # Send a deliberately invalid action; the CLI must warn + fall back
                invalid = next((i for i in range(58) if i < len(mask) and not mask[i]), 999)
                send_action(proc, invalid)
                result["probes_sent"] += 1
                expect_warning = True
            else:
                # Masked-random policy; RUN down-weighted so episodes don't end trivially
                choices = [a for a in valid if a != ACTION_RUN]
                if not choices or (ACTION_RUN in valid and rng.random() < run_weight):
                    action = ACTION_RUN if ACTION_RUN in valid else rng.choice(choices)
                else:
                    action = rng.choice(choices)
                send_action(proc, action)

    except HangTimeout:
        result["result"] = "hang"
        result["errors"].append(f"hang: no output within {step_timeout}s")
    finally:
        exit_code = proc.poll()
        kill_proc(proc, stderr_fh)
        result["exit_code"] = exit_code

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3, help="number of generated seeds")
    ap.add_argument("--seed-list", type=str, default=None, help="comma-separated explicit seeds")
    ap.add_argument("--seed-prefix", type=str, default="smoke")
    ap.add_argument("--waves", type=int, default=10)
    ap.add_argument("--action-seed", type=str, default="42")
    ap.add_argument("--probe-invalid", type=float, default=0.0)
    ap.add_argument("--dump-dir", type=str, default=None)
    ap.add_argument("--boot-timeout", type=float, default=180)
    ap.add_argument("--step-timeout", type=float, default=60)
    ap.add_argument(
        "--cli-arg",
        action="append",
        default=None,
        help="extra cli.js flag, repeatable (e.g. --cli-arg=--override=STARTING_WAVE_OVERRIDE=25); "
        "used by the obs-audit corpus for scenario diversity",
    )
    args = ap.parse_args()

    seeds = args.seed_list.split(",") if args.seed_list else [f"{args.seed_prefix}-{i}" for i in range(args.seeds)]
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    all_phases: dict[str, int] = {}
    failures = []
    results = []

    for seed in seeds:
        t0 = time.time()
        r = run_episode(
            seed,
            args.waves,
            args.action_seed,
            args.probe_invalid,
            dump_dir,
            args.boot_timeout,
            args.step_timeout,
            extra_args=args.cli_arg,
        )
        r["duration_s"] = round(time.time() - t0, 1)
        results.append(r)
        for p, c in r["phases"].items():
            all_phases[p] = all_phases.get(p, 0) + c

        ok = (
            r["result"] in ("game_over", "step_cap")
            and not r["unexpected_warnings"]
            and not r["errors"]
            and r["probes_acknowledged"] == r["probes_sent"]
        )
        if not ok:
            failures.append(r)
        status = "OK " if ok else "FAIL"
        print(
            f"[{status}] seed={seed} result={r['result']} steps={r['steps']} "
            f"reward={r['total_reward']:.2f} probes={r['probes_acknowledged']}/{r['probes_sent']} "
            f"warnings={len(r['unexpected_warnings'])} errors={len(r['errors'])} ({r['duration_s']}s)"
        )
        for w in r["unexpected_warnings"][:3]:
            print(f"    warning: {w}")
        for e in r["errors"][:3]:
            print(f"    error: {e}")

    print("\nphase coverage:")
    for p, c in sorted(all_phases.items(), key=lambda kv: -kv[1]):
        print(f"  {p}: {c}")

    required = {"command", "modifier"}
    missing_required = required - set(all_phases)
    if missing_required:
        print(f"\nFAIL: required phases never seen: {sorted(missing_required)}")
    optional = {"switch", "target", "learn_move", "modifier_target", "check_switch", "select_biome"}
    missing_optional = optional - set(all_phases)
    if missing_optional:
        print(f"note: optional phases not seen this run: {sorted(missing_optional)}")

    if dump_dir:
        (dump_dir / "results.json").write_text(json.dumps(results, indent=2))

    ok = not failures and not missing_required
    print(f"\nSMOKE: {'OK' if ok else 'FAIL'} ({len(seeds) - len(failures)}/{len(seeds)} episodes clean)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
