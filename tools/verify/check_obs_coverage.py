#!/usr/bin/env python3
"""
Observation coverage report: for every game-side enumerable, classify each
entry as ENCODED, EXCLUDED (with a documented reason in a manifest), or
UNREVIEWED. Fails while anything is UNREVIEWED — "deliberately not encoded"
and "forgotten" must never be indistinguishable.

Manifests: tools/verify/coverage-manifests/<name>.json
    { "excluded": { "ENTRY_NAME": "one-line reason" } }
Reasons starting with "BACKLOG:" mark decision-relevant entries that SHOULD
be encoded eventually (also tracked in src/rl/docs/OBSERVATION_CHUNKS.md);
they count as reviewed for gating purposes.

Usage: python3 tools/verify/check_obs_coverage.py [--list-unreviewed]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from common import REPO_ROOT  # noqa: E402

MANIFEST_DIR = Path(__file__).resolve().parent / "coverage-manifests"
SRC = REPO_ROOT / "src"


def read(p: Path) -> str:
    return p.read_text()


def strip_comments(ts: str) -> str:
    """Remove /* */ blocks and // line comments so commented-out registry
    entries (e.g. the disabled REPEL items) don't count as real entries."""
    ts = re.sub(r"/\*.*?\*/", "", ts, flags=re.S)
    return re.sub(r"(?<![:/])//[^\n]*", "", ts)


def load_manifest(name: str) -> dict[str, str]:
    p = MANIFEST_DIR / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("excluded", {})


def report(domain: str, universe: set[str], encoded: set[str], manifest: dict[str, str]) -> tuple[int, list[str]]:
    """Returns (unreviewed_count, lines)."""
    excluded = set(manifest)
    unknown_manifest = excluded - universe
    unreviewed = sorted(universe - encoded - excluded)
    backlog = sorted(k for k, v in manifest.items() if v.startswith("BACKLOG:"))
    lines = [
        f"{domain}: {len(universe)} total | {len(encoded & universe)} encoded | "
        f"{len(excluded & universe)} excluded ({len(backlog)} backlog) | {len(unreviewed)} UNREVIEWED"
    ]
    for name in unknown_manifest:
        lines.append(f"  WARNING: manifest entry {name!r} not in the {domain} universe (stale manifest?)")
    return len(unreviewed), lines


def get_unreviewed(universe: set[str], encoded: set[str], manifest: dict[str, str]) -> list[str]:
    return sorted(universe - encoded - set(manifest))


# ── Domain extractors ───────────────────────────────────────────────────


def battler_tags() -> tuple[set[str], set[str]]:
    enum_src = strip_comments(read(SRC / "enums" / "battler-tag-type.ts"))
    universe = set(re.findall(r'^\s*(\w+)\s*=\s*"', enum_src, re.M))
    spaces = read(SRC / "rl" / "spaces.ts")
    m = re.search(r"CURATED_VOLATILE_TAGS[^=]*=\s*\[(.*?)\]", spaces, re.S)
    encoded = set(re.findall(r"BattlerTagType\.(\w+)", m.group(1)))
    return universe, encoded


def move_attrs() -> tuple[set[str], set[str]]:
    mv = strip_comments(read(SRC / "data" / "moves" / "move.ts"))
    reg = re.search(r"const MoveAttrs = Object\.freeze\(\{(.*?)\}\);", mv, re.S)
    universe = set(re.findall(r"^\s*(\w+),?\s*$", reg.group(1), re.M))
    sb = read(SRC / "rl" / "state-builder.ts")
    encoded = set(re.findall(r'(?:hasAttr|getAttrs)\(\s*"(\w+)"', sb))
    return universe, encoded


def modifier_types() -> tuple[set[str], set[str]]:
    mt = strip_comments(read(SRC / "modifier" / "modifier-type.ts"))
    # the registry: `const modifierTypeInitObj = Object.freeze({ KEY: () => ..., ... })`
    reg = re.search(r"const modifierTypeInitObj\s*=\s*Object\.freeze\(\{(.*?)\n\}\);", mt, re.S)
    universe = set(re.findall(r"^\s{2}([A-Z][A-Z0-9_]+):", reg.group(1), re.M))
    mf = read(SRC / "rl" / "modifier-features.ts")
    # entries are assignments: MODIFIER_FEATURES["LEFTOVERS"] = f(...)
    encoded = set(re.findall(r'MODIFIER_FEATURES\[\s*"([A-Z][A-Z0-9_]+)"\s*\]', mf))
    return universe, encoded


def move_flags() -> tuple[set[str], set[str]]:
    mf_src = strip_comments(read(SRC / "enums" / "move-flags.ts"))
    universe = set(re.findall(r"^\s*(\w+)\s*=\s*", mf_src, re.M)) - {"NONE"}
    sb = read(SRC / "rl" / "state-builder.ts")
    encoded = set(re.findall(r"MoveFlags\.(\w+)", sb))
    return universe, encoded


def ability_rows() -> list[str]:
    """Abilities aren't name-curated — verify table size covers the enum.

    AbilityId uses implicit enum values, so the universe size is the MEMBER
    COUNT (NONE=0 .. N-1), not a parsed max value.
    """
    ab_enum = strip_comments(read(SRC / "enums" / "ability-id.ts"))
    body = re.search(r"export enum AbilityId\s*\{(.*?)\}", ab_enum, re.S)
    members = re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*,?\s*$", body.group(1), re.M)
    feat = read(SRC / "rl" / "ability-features.ts")
    rows = len(re.findall(r"^\s*/\*\s*\d+", feat, re.M))
    ok = rows >= len(members)
    return [
        f"abilities: AbilityId members={len(members)}, ABILITY_FEATURES rows={rows} -> {'OK' if ok else 'FAIL (table smaller than enum!)'}"
    ] + ([] if ok else ["  every ability past the table end silently encodes as all-zeros"])


def field_sweep() -> list[str]:
    """HEURISTIC (report-only): pokemon.* properties read by state-builder."""
    sb = read(SRC / "rl" / "state-builder.ts")
    reads = sorted(set(re.findall(r"pokemon\.(\w+)", sb)))
    return [f"field sweep (heuristic, report-only): state-builder reads {len(reads)} distinct pokemon.* properties"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-unreviewed", action="store_true")
    args = ap.parse_args()

    domains = {
        "battler-tags": battler_tags,
        "move-attrs": move_attrs,
        "modifier-types": modifier_types,
        "move-flags": move_flags,
    }

    total_unreviewed = 0
    for name, extractor in domains.items():
        universe, encoded = extractor()
        manifest = load_manifest(name)
        n, lines = report(name, universe, encoded, manifest)
        total_unreviewed += n
        for line in lines:
            print(line)
        if args.list_unreviewed and n:
            for entry in get_unreviewed(universe, encoded, manifest):
                print(f"    UNREVIEWED: {entry}")

    for line in ability_rows():
        print(line)
    for line in field_sweep():
        print(line)

    print(f"\nOBS COVERAGE: {'OK' if total_unreviewed == 0 else f'FAIL ({total_unreviewed} unreviewed entries)'}")
    return 1 if total_unreviewed else 0


if __name__ == "__main__":
    sys.exit(main())
