#!/usr/bin/env python3
"""
TS<->Python observation parity checker (offline).

Re-encodes every gameState in --dump-obs JSONL files with the Python encoder
and compares element-wise against the TS encoding captured in the dump.

Usage:
    python3 tools/verify/check_parity.py ".rl-verify/**/*.jsonl" [--atol 1e-6] [--strict]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from common import decode_obs_b64  # noqa: E402  (sys.path side effect)
from invariants import check_mask as inv_check_mask  # noqa: E402
from invariants import check_observation  # noqa: E402

from rl.feature_names import dim_to_name  # noqa: E402
from rl.observation import (  # noqa: E402
    ACTION_SPACE_SIZE,
    OBSERVATION_DIM,
    encode_observation,
    extract_action_mask,
    parse_game_state,
)


def check_invariants(obs: np.ndarray, label: str) -> list[str]:
    if obs.shape != (OBSERVATION_DIM,):
        return [f"{label}: wrong shape {obs.shape}"]
    return [f"{label}: {e}" for e in check_observation(obs)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("globs", nargs="+", help="JSONL dump files (glob patterns)")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--strict", action="store_true", help="require bitwise equality")
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    files = sorted({f for g in args.globs for f in glob.glob(g, recursive=True)})
    if not files:
        print(f"error: no files matched {args.globs}", file=sys.stderr)
        return 2

    n_records = 0
    n_tolerance_bad = 0
    n_bitwise_bad = 0
    n_mask_bad = 0
    n_invariant_bad = 0
    dim_offenders: dict[int, dict] = defaultdict(lambda: {"count": 0, "max_delta": 0.0, "example": ""})
    mask_offenders: dict[int, int] = defaultdict(int)
    reports = 0

    for path in files:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("kind") != "step":
                    continue
                n_records += 1
                where = f"{path}:step{rec['step']}:phase={rec['phase']}"

                state = parse_game_state(rec["gameState"])
                py_obs = encode_observation(state, fog_of_war=bool(rec.get("fogOfWar", False)))
                ts_obs = decode_obs_b64(rec["obsB64"])

                # Invariants on both vectors
                inv = check_invariants(py_obs, "py") + check_invariants(ts_obs, "ts")
                if rec.get("invariantError"):
                    inv.append(f"ts-runtime: {rec['invariantError']}")
                if inv:
                    n_invariant_bad += 1
                    if reports < args.max_report:
                        print(f"INVARIANT {where}: {inv}")
                        reports += 1

                # Observation parity
                bad = np.where(~np.isclose(py_obs, ts_obs, atol=args.atol, rtol=0))[0]
                bitwise_equal = py_obs.tobytes() == ts_obs.tobytes()
                if not bitwise_equal:
                    n_bitwise_bad += 1
                if len(bad):
                    n_tolerance_bad += 1
                    for i in bad:
                        d = dim_offenders[int(i)]
                        d["count"] += 1
                        delta = abs(float(py_obs[i]) - float(ts_obs[i]))
                        if delta > d["max_delta"]:
                            d["max_delta"] = delta
                            d["example"] = f"{where} ts={ts_obs[i]!r} py={py_obs[i]!r}"
                    if reports < args.max_report:
                        i = int(bad[0])
                        print(f"MISMATCH {where}: {len(bad)} dims, first {dim_to_name(i)} ts={ts_obs[i]!r} py={py_obs[i]!r}")
                        reports += 1

                # Mask parity (exact)
                py_mask = extract_action_mask(state)
                for e in inv_check_mask(py_mask, rec.get("phase", "?")):
                    n_invariant_bad += 1
                    if reports < args.max_report:
                        print(f"MASK-INVARIANT {where}: {e}")
                        reports += 1
                ts_mask = np.array(rec["actionMask"], dtype=bool)
                if ts_mask.shape != (ACTION_SPACE_SIZE,):
                    n_mask_bad += 1
                    if reports < args.max_report:
                        print(f"MASK-SHAPE {where}: ts mask has {ts_mask.shape}")
                        reports += 1
                else:
                    md = np.where(py_mask != ts_mask)[0]
                    if len(md):
                        n_mask_bad += 1
                        for i in md:
                            mask_offenders[int(i)] += 1
                        if reports < args.max_report:
                            from common import ACTION_NAMES

                            i = int(md[0])
                            print(f"MASK {where}: action {i} ({ACTION_NAMES[i]}) ts={bool(ts_mask[i])} py={bool(py_mask[i])}")
                            reports += 1

    print()
    print(f"checked {n_records} records from {len(files)} file(s)")
    print(f"  tolerance (atol={args.atol}) mismatches: {n_tolerance_bad}")
    print(f"  bitwise differences:                     {n_bitwise_bad}")
    print(f"  mask mismatches:                         {n_mask_bad}")
    print(f"  invariant failures:                      {n_invariant_bad}")

    if dim_offenders:
        print("\ntop offending dims:")
        top = sorted(dim_offenders.items(), key=lambda kv: -kv[1]["count"])[:10]
        for i, d in top:
            print(f"  {dim_to_name(i)} (dim {i}): {d['count']}x, max|delta|={d['max_delta']:.3g}, e.g. {d['example']}")
    if mask_offenders:
        from common import ACTION_NAMES

        print("\ntop offending mask actions:")
        for i, c in sorted(mask_offenders.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  action {i} ({ACTION_NAMES[i]}): {c}x")

    # A dump with files but zero step records must not pass vacuously — a broken
    # dump pipeline (renamed "kind":"step" sentinel, empty --dump-obs) would
    # otherwise read as full parity.
    if n_records == 0:
        print("\nPARITY: FAIL (0 step records found across all files — dump pipeline broken?)")
        return 1
    failed = n_tolerance_bad or n_mask_bad or n_invariant_bad or (args.strict and n_bitwise_bad)
    print("\nPARITY: " + ("FAIL" if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
