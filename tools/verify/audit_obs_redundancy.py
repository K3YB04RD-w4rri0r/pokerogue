#!/usr/bin/env python3
"""
Observation redundancy audit (obs-v9 plan, Phase 1).

Consumes --dump-obs JSONL dumps (the parity-harness format: one record per
decision with obsB64) and produces a per-dim redundancy ledger:

  dead            never varies across the whole corpus
  episode-const   varies across episodes but never within one (run metadata:
                  game-mode flags, seed-derived identity, party constants)
  duplicate       byte-identical to another dim across the corpus (groups)
  near-dup        |r| >= 0.999 with another varying dim (linear redundancy)
  low-info        varies but takes <= 2 distinct values in < 0.1% of steps

plus block-level rollups (by feature_names prefix) so "which blocks carry
the least information per dim" is visible at a glance.

Usage:
    python3 tools/verify/audit_obs_redundancy.py '<glob.jsonl>' [more globs]
        [--out audit-redundancy.json] [--max-steps 20000]

Interpretation rules (see src/rl/docs/OBS_AUDIT_PLAN.md):
- "dead" on a SMALL corpus just means unexercised — verdicts are only
  evidence-grade on the full forced-scenario corpus.
- duplicates/near-dups are structural candidates for merging in v9, but
  each needs a semantic justification before dropping (a dim can be
  corpus-redundant yet semantically distinct in unreached states).
"""

from __future__ import annotations

import argparse
import base64
import glob as globlib
import json
import re
import sys
from collections import defaultdict

import numpy as np
from common import REPO_ROOT  # noqa: F401  (sys.path side effect adds src/)

from rl.feature_names import dim_to_name  # noqa: E402
from rl.observation import OBSERVATION_DIM  # noqa: E402


def load_corpus(patterns: list[str], max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps x dims float32 matrix, episode-id per step)."""
    rows: list[np.ndarray] = []
    episode_ids: list[int] = []
    episode = -1
    for pattern in patterns:
        for path in sorted(globlib.glob(pattern)):
            episode += 1
            with open(path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") != "step" or "obsB64" not in rec:
                        continue
                    obs = np.frombuffer(base64.b64decode(rec["obsB64"]), dtype="<f4")
                    if obs.shape != (OBSERVATION_DIM,):
                        print(f"warn: {path}: obs dim {obs.shape} != {OBSERVATION_DIM}, skipping file")
                        break
                    rows.append(obs)
                    episode_ids.append(episode)
                    if len(rows) >= max_steps:
                        return np.stack(rows), np.array(episode_ids)
    if not rows:
        sys.exit("no steps found — pass --dump-obs JSONL globs (quote them)")
    return np.stack(rows), np.array(episode_ids)


def block_of(name: str) -> str:
    """Coarse block key from a feature name, e.g. player_0/moves[2]/... -> player_0/moves."""
    parts = name.split("/")
    head = parts[0].split("[")[0]
    if len(parts) > 1:
        second = parts[1].split("[")[0]
        return f"{head}/{second}"
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", help="JSONL dump globs (quote them)")
    ap.add_argument("--out", default="audit-redundancy.json")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--near-dup-r", type=float, default=0.999)
    args = ap.parse_args()

    matrix, episodes = load_corpus(args.globs, args.max_steps)
    n_steps, n_dims = matrix.shape
    n_episodes = len(np.unique(episodes))
    print(f"corpus: {n_steps} steps, {n_episodes} episodes, {n_dims} dims")

    # ── dead / varying ────────────────────────────────────────────────
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    dead = np.flatnonzero(mins == maxs)
    varying = np.flatnonzero(mins != maxs)
    print(f"dead (never vary in corpus): {dead.size}   varying: {varying.size}")

    # ── episode-constant (varies across, never within, episodes) ─────
    episode_const: list[int] = []
    if n_episodes > 1:
        within_varies = np.zeros(n_dims, dtype=bool)
        for ep in np.unique(episodes):
            sub = matrix[episodes == ep]
            within_varies |= sub.min(axis=0) != sub.max(axis=0)
        episode_const = [int(d) for d in varying if not within_varies[d]]
    print(f"episode-constant (run metadata): {len(episode_const)}")

    # ── exact duplicates (byte-identical columns) ─────────────────────
    col_bytes = {}
    dup_groups: dict[int, list[int]] = defaultdict(list)
    for d in varying:
        key = matrix[:, d].tobytes()
        if key in col_bytes:
            dup_groups[col_bytes[key]].append(int(d))
        else:
            col_bytes[key] = int(d)
    dup_groups = {k: v for k, v in dup_groups.items() if v}
    n_dup_dims = sum(len(v) for v in dup_groups.values())
    print(f"exact-duplicate dims: {n_dup_dims} (in {len(dup_groups)} groups)")

    # ── near-duplicates (|r| >= threshold) among varying, non-exact ──
    exact_dup_set = set()
    for _k, v in dup_groups.items():
        exact_dup_set.update(v)
    candidates = np.array([d for d in varying if d not in exact_dup_set])
    near_pairs: list[tuple[int, int, float]] = []
    if candidates.size >= 2:
        sub = matrix[:, candidates].astype(np.float64)
        sub -= sub.mean(axis=0)
        norms = np.linalg.norm(sub, axis=0)
        norms[norms == 0] = 1
        sub /= norms
        # blockwise to bound memory
        BS = 1024
        for i in range(0, candidates.size, BS):
            ci = sub[:, i : i + BS]
            corr = ci.T @ sub  # (bs, n_candidates)
            for a in range(corr.shape[0]):
                gi = i + a
                row = corr[a]
                hits = np.flatnonzero(np.abs(row) >= args.near_dup_r)
                for h in hits:
                    if h > gi:
                        near_pairs.append((int(candidates[gi]), int(candidates[h]), float(row[h])))
    print(f"near-duplicate pairs (|r|>={args.near_dup_r}): {len(near_pairs)}")

    # ── low-info: <=2 values and the rare one occurs in <0.1% steps ──
    low_info: list[int] = []
    for d in varying:
        col = matrix[:, d]
        vals, counts = np.unique(col, return_counts=True)
        if vals.size <= 2 and counts.min() < max(1, n_steps // 1000):
            low_info.append(int(d))
    print(f"low-info dims (rare binary flips): {len(low_info)}")

    # ── block rollup ──────────────────────────────────────────────────
    block_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"dims": 0, "dead": 0, "dup": 0})
    dead_set = set(int(d) for d in dead)
    for d in range(n_dims):
        b = block_of(dim_to_name(d))
        block_stats[b]["dims"] += 1
        if d in dead_set:
            block_stats[b]["dead"] += 1
        if d in exact_dup_set:
            block_stats[b]["dup"] += 1

    # ── lane-collapsed deadness (uniform-slot metric) ─────────────────
    # A LANE merges semantically-identical positions across slots (pokemon
    # slots, move slots, shop/reward slots, item channels, sides). In a
    # uniform-slot layout, a lane proves itself if ANY slot exercises it;
    # per-slot deadness merely reflects party sizes and item draws. This is
    # the v9 acceptance metric (see docs/OBS_V9_LAYOUT.md §6).
    def lane_of(name: str) -> str:
        n = re.sub(r"^(player|enemy)_\d/", "mon/", name)
        n = re.sub(r"/moves\[\d\]/", "/moves[]/", n)
        n = re.sub(r"^shop/(item|reward)\[\d+\]/", r"shop/\1[]/", n)
        n = re.sub(r"^inventory/held/[a-z_0-9]+/item\[\d\]/", "inventory/held/item[]/", n)
        n = re.sub(r"^inventory/held/[a-z_0-9]+/", "inventory/held/", n)
        n = re.sub(r"^field/(player|enemy)_", "field/side_", n)
        return n

    dead_names = set(dim_to_name(int(d)) for d in dead)
    lanes_total: dict[str, int] = defaultdict(int)
    lanes_dead: dict[str, int] = defaultdict(int)
    for di in range(n_dims):
        k = lane_of(dim_to_name(di))
        lanes_total[k] += 1
        if dim_to_name(di) in dead_names:
            lanes_dead[k] += 1
    dead_lanes = sorted(k for k in lanes_total if lanes_dead[k] == lanes_total[k])
    lane_dead_dims = sum(lanes_total[k] for k in dead_lanes)
    print(
        f"lane-collapsed: {len(dead_lanes)}/{len(lanes_total)} lanes fully dead "
        f"({lane_dead_dims} dims, {100 * lane_dead_dims / n_dims:.2f}%)"
    )

    ledger = {
        "lanes": {
            "total": len(lanes_total),
            "fully_dead": len(dead_lanes),
            "fully_dead_dims": lane_dead_dims,
            "dead_lanes": dead_lanes,
        },
        "corpus": {"steps": n_steps, "episodes": n_episodes, "globs": args.globs},
        "summary": {
            "dims": n_dims,
            "dead": int(dead.size),
            "episode_constant": len(episode_const),
            "exact_duplicate_dims": n_dup_dims,
            "near_duplicate_pairs": len(near_pairs),
            "low_info": len(low_info),
        },
        "dead": [{"dim": int(d), "name": dim_to_name(int(d))} for d in dead],
        "episode_constant": [{"dim": d, "name": dim_to_name(d)} for d in episode_const],
        "duplicate_groups": [
            {
                "keep": {"dim": k, "name": dim_to_name(k)},
                "duplicates": [{"dim": d, "name": dim_to_name(d)} for d in v],
            }
            for k, v in sorted(dup_groups.items())
        ],
        "near_duplicates": [
            {"a": {"dim": a, "name": dim_to_name(a)}, "b": {"dim": b, "name": dim_to_name(b)}, "r": round(r, 6)}
            for a, b, r in sorted(near_pairs, key=lambda t: -abs(t[2]))
        ],
        "low_info": [{"dim": d, "name": dim_to_name(d)} for d in low_info],
        "blocks": {
            k: v
            for k, v in sorted(block_stats.items(), key=lambda kv: -(kv[1]["dead"] + kv[1]["dup"]))
        },
    }
    with open(args.out, "w") as fh:
        json.dump(ledger, fh, indent=1)
    print(f"\nledger written to {args.out}")

    # Human-readable highlights
    print("\nTop blocks by dead+duplicate dims:")
    for name, st in list(ledger["blocks"].items())[:12]:
        if st["dead"] + st["dup"] == 0:
            break
        print(f"  {name:<28} dims={st['dims']:>5} dead={st['dead']:>5} dup={st['dup']:>4}")
    print("\nSample duplicate groups:")
    for g in ledger["duplicate_groups"][:8]:
        names = [g["keep"]["name"]] + [d["name"] for d in g["duplicates"]]
        print("  " + "  ==  ".join(names[:4]) + (" ..." if len(names) > 4 else ""))
    print("\nStrongest near-duplicates:")
    for p in ledger["near_duplicates"][:8]:
        print(f"  r={p['r']:+.4f}  {p['a']['name']}  ~  {p['b']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
