#!/usr/bin/env python3
"""
Deep-coverage corpus generator: scripted scenario runs that force the game
situations a masked-random agent rarely reaches (doubles, big trainer
parties, megas, long runs, ...).

Each scenario ASSERTS that its target situation actually occurred in the
dump. That assertion is what gives the dim-exercise report its teeth: if a
scenario ran, its situation provably occurred, and a target dim still never
varied — that's evidence of an encoding bug, not another corpus gap.

Output: .rl-verify/corpus/<scenario>-<seed>.jsonl — consumed by
check_parity.py and check_dim_exercise.py.

Usage:
    python3 tools/verify/gen_coverage_corpus.py [--only doubles,megas] [--out-dir .rl-verify/corpus]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_episodes  # noqa: E402
from common import REPO_ROOT  # noqa: E402


def _any_step(records: list[dict], pred) -> bool:
    return any(pred(r.get("gameState") or {}) for r in records if r.get("kind") == "step")


def _any_pokemon(gs: dict, pred) -> bool:
    slots = [f"player_{i}" for i in range(6)] + [f"enemy_{i}" for i in range(6)]
    return any(pred(gs.get(s) or {}) for s in slots if (gs.get(s) or {}).get("valid"))


SCENARIOS: dict[str, dict] = {
    "doubles": {
        "description": "Forced double battles -> player_1/enemy_1 blocks, ally targeting, speed ranks",
        "overrides": {"BATTLE_STYLE_OVERRIDE": "double"},
        "waves": 8,
        "seeds": ["corpus-dbl-1", "corpus-dbl-2"],
        "asserts": {
            "a double battle occurred": lambda recs: _any_step(recs, lambda gs: (gs.get("field") or {}).get("is_double_battle") is True),
            "enemy_1 slot populated": lambda recs: _any_step(recs, lambda gs: (gs.get("enemy_1") or {}).get("valid") is True),
        },
    },
    "trainers": {
        "description": "Start at trainer-dense waves -> multi-mon enemy parties (enemy_2+ bench blocks), trainer metadata",
        "overrides": {"STARTING_WAVE_OVERRIDE": 95, "STARTING_LEVEL_OVERRIDE": 80},
        "waves": 100,
        "seeds": ["corpus-trn-1", "corpus-trn-2"],
        "asserts": {
            "an enemy bench slot was populated (enemy party > 2)": lambda recs: _any_step(
                recs, lambda gs: (gs.get("enemy_2") or {}).get("valid") is True
            ),
            "a trainer battle occurred": lambda recs: _any_step(
                recs, lambda gs: (gs.get("battle") or {}).get("trainer") is not None
            ),
        },
    },
    # NOTE: a "megas" scenario (STARTING_MODIFIER_OVERRIDE: MEGA_BRACELET) is
    # deliberately absent: that override is applied during the title flow
    # where no battle exists yet and crashes the game's modifier application
    # ("getBattlerCount of null"). is_mega/is_max dims stay `unexercised` in
    # the dim-exercise ledger until mid-run bracelet acquisition is scripted.
    "longrun": {
        "description": "Strong start, 30+ waves -> select_biome, boss segments, lapsing modifiers, high wave dims",
        # MEs are disabled env-wide by the CLI (removed from the env by decision)
        "overrides": {"STARTING_LEVEL_OVERRIDE": 100},
        "waves": 35,
        "seeds": ["corpus-long-1", "corpus-long-2"],
        "asserts": {
            "reached wave 11+": lambda recs: _any_step(recs, lambda gs: ((gs.get("battle") or {}).get("wave_index") or 0) >= 11),
            "a boss segment appeared": lambda recs: _any_step(recs, lambda gs: _any_pokemon(gs, lambda p: (p.get("boss_segments") or 0) > 1)),
        },
    },
    "weather": {
        "description": "Permanent weather + abilities -> weather one-hots, suppression",
        "overrides": {"WEATHER_OVERRIDE": 3},  # SANDSTORM
        "waves": 6,
        "seeds": ["corpus-wthr-1"],
        "asserts": {
            "sandstorm active": lambda recs: _any_step(recs, lambda gs: (gs.get("field") or {}).get("weather_type") == 3),
        },
    },
    "fullparty": {
        "description": "Catch-heavy run -> full party, release flow, catch counters",
        "overrides": {"STARTING_LEVEL_OVERRIDE": 30},
        "waves": 12,
        "seeds": ["corpus-fp-1", "corpus-fp-2"],
        "run_weight": 0.0,  # never run away; random policy throws balls often
        "asserts": {
            "party grew beyond starters": lambda recs: _any_step(recs, lambda gs: (gs.get("player_3") or {}).get("valid") is True),
        },
    },
    "status": {
        "description": "Status-heavy battles -> status one-hots, toxic/sleep counters",
        "overrides": {"ENEMY_MOVESET_OVERRIDE": [92, 79]},  # TOXIC, SLEEP_POWDER
        "waves": 6,
        "seeds": ["corpus-sts-1"],
        "asserts": {
            "a player pokemon was statused": lambda recs: _any_step(
                recs, lambda gs: _any_pokemon(gs, lambda p: p.get("is_player") and (p.get("status_effect") or 0) > 0)
            ),
        },
    },
}


def run_scenario(name: str, spec: dict, out_dir: Path) -> tuple[bool, list[str]]:
    extra_args = [f"--override={k}={json.dumps(v)}" for k, v in spec["overrides"].items()]
    notes: list[str] = []
    all_records: list[dict] = []
    ok = True

    for seed in spec["seeds"]:
        r = run_episodes.run_episode(
            seed=seed,
            waves=spec["waves"],
            action_seed="corpus",
            probe_invalid=0.0,
            dump_dir=out_dir,
            boot_timeout=180,
            # 60s was tight for the 35-wave longrun under full-suite CPU load
            # (transient stalls -> killed -> "EOF before done"); it completes
            # well within this in isolation.
            step_timeout=150,
            run_weight=spec.get("run_weight", 0.1),
            extra_args=extra_args,
        )
        # "wave_cap" is a COMPLETE episode (played through the whole wave
        # budget), same as game_over/step_cap — it was added by the round-2 real
        # wave-cap fix but never allowed here, so any scenario reaching its wave
        # budget was wrongly marked failed and its records dropped (asserts then
        # ran on an empty list and failed). Only router_timeout/hang/incomplete
        # are genuine failures.
        if r["result"] not in ("game_over", "step_cap", "wave_cap", "livelock") or r["errors"]:
            ok = False
            notes.append(f"{seed}: episode failed ({r['result']}, errors={r['errors'][:2]})")
            continue
        dump = out_dir / f"{seed}.jsonl"
        new_name = out_dir / f"{name}-{seed}.jsonl"
        dump.rename(new_name)
        for line in open(new_name):
            all_records.append(json.loads(line))
        notes.append(f"{seed}: {r['steps']} steps, {r['result']}")

    for assert_name, pred in spec["asserts"].items():
        if not pred(all_records):
            ok = False
            notes.append(f"ASSERT FAILED: {assert_name} — the scenario did not produce its target situation")
        else:
            notes.append(f"assert ok: {assert_name}")

    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, default=".rl-verify/corpus")
    ap.add_argument("--only", type=str, default=None, help="comma-separated scenario names")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.only.split(",") if args.only else list(SCENARIOS)
    failures = 0
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario: {name}")
            failures += 1
            continue
        spec = SCENARIOS[name]
        print(f"\n=== {name}: {spec['description']}")
        ok, notes = run_scenario(name, spec, out_dir)
        for n in notes:
            print(f"  {n}")
        if not ok:
            failures += 1

    print(f"\nCORPUS: {'OK' if failures == 0 else f'FAIL ({failures} scenario(s))'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
