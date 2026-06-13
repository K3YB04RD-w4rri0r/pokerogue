#!/usr/bin/env python3
"""
Dim-exercise report: which observation dims actually varied across a corpus.

"Never varied" is NOT "dead" — every never-varying dim group must carry an
explicit classification in the ledger
(tools/verify/coverage-manifests/dim-exercise-ledger.json):

  unexercised           the corpus never reached the situation; the entry
                        names the gen_coverage_corpus.py scenario that should
                        light it up (or "" with a reason if none exists yet)
  structurally-constant provable from code (e.g. turn_data.move_effectiveness
                        is nulled before every decision point); entry carries
                        the code-level reason
  sparse-categorical    one cell of a one-hot/flag bank — varies only when a
                        specific enum value lands in that slot; the review
                        unit is the bank (covered by semantic tests/goldens),
                        not the cell
  bench-structural      bench Pokemon cannot carry battle-scoped state (stat
                        stages reset on switch-out; turn-data applies on-field
                        only)
  (no entry)            UNREVIEWED -> fails
  suspected-bug         flagged AUTOMATICALLY when a ledger entry says a
                        scenario exercises the dim, that scenario's dump is in
                        the corpus, and the dim STILL never varied -> fails

Ledger format:
  { "groups": { "<group-key prefix>": {
        "status": "...", "reason": "...", "scenario": "<corpus scenario>" } } }
Group keys match by PREFIX against collapsed feature names ("player_3/" and
"enemy_4/" both collapse to "bench-slot/"); first matching ledger entry wins,
so put specific prefixes before broad ones.

Usage:
    python3 tools/verify/check_dim_exercise.py ".rl-verify/**/*.jsonl" [--gate]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import REPO_ROOT, decode_obs_b64  # noqa: E402
from rl.feature_names import FEATURE_NAMES, dim_to_name  # noqa: E402

LEDGER_PATH = Path(__file__).resolve().parent / "coverage-manifests" / "dim-exercise-ledger.json"


def load_corpus(globs: list[str]) -> tuple[np.ndarray, set[str]]:
    obs_rows = []
    scenarios: set[str] = set()
    for pattern in globs:
        for f in glob.glob(pattern, recursive=True):
            name = Path(f).stem
            m = re.match(r"([a-z]+)-corpus-", name)
            if m:
                scenarios.add(m.group(1))
            for line in open(f):
                rec = json.loads(line)
                if rec.get("kind") == "step":
                    obs_rows.append(decode_obs_b64(rec["obsB64"]))
    if not obs_rows:
        sys.exit("error: no step records found in the given globs")
    return np.stack(obs_rows), scenarios


def group_key(name: str) -> str:
    """Collapse per-slot/per-index names into reviewable groups."""
    g = re.sub(r"^(player|enemy)_[2-5]/", "bench-slot/", name)
    g = re.sub(r"^(player|enemy)_[01]/", "active-slot/", g)
    g = re.sub(r"moves\[\d\]/", "moves[]/", g)
    g = re.sub(r"\[\d+\]", "[]", g)
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--gate", action="store_true", help="fail on UNREVIEWED or suspected-bug")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    X, scenarios = load_corpus(args.globs)
    never_varied = np.where(X.var(axis=0) == 0)[0]
    print(f"corpus: {X.shape[0]} states from scenarios {sorted(scenarios) or '(unnamed dumps)'}")
    print(f"never-varying dims: {len(never_varied)} / {len(FEATURE_NAMES)}")

    ledger = json.loads(LEDGER_PATH.read_text()).get("groups", {}) if LEDGER_PATH.exists() else {}

    groups: dict[str, int] = defaultdict(int)
    for i in never_varied:
        groups[group_key(dim_to_name(int(i)))] += 1

    unreviewed: list[tuple[str, int]] = []
    suspected: list[str] = []
    counts = {"unexercised": 0, "structurally-constant": 0, "sparse-categorical": 0, "bench-structural": 0}

    for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        entry = next((v for k, v in ledger.items() if g.startswith(k)), None)
        if entry is None:
            unreviewed.append((g, n))
            continue
        status = entry.get("status", "")
        if status == "unexercised":
            counts["unexercised"] += n
            scenario = entry.get("scenario", "")
            if scenario and scenario in scenarios:
                # The exercising scenario ran and the dim group still never
                # varied -> this is what an encoding bug looks like
                suspected.append(f"{g} ({n} dims) — scenario '{scenario}' ran but the dims stayed flat")
        elif status in counts:
            counts[status] += n
        else:
            unreviewed.append((g, n))

    print(
        "classified: "
        + ", ".join(f"{v} {k}" for k, v in counts.items())
    )
    if unreviewed:
        print(f"\nUNREVIEWED groups ({len(unreviewed)}):")
        for g, n in unreviewed[: args.top]:
            print(f"  {n:>5}  {g}")
    if suspected:
        print(f"\nSUSPECTED BUGS ({len(suspected)}):")
        for s in suspected:
            print(f"  {s}")

    failed = bool(suspected) or (args.gate and bool(unreviewed))
    print(f"\nDIM EXERCISE: {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
