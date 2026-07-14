#!/usr/bin/env python3
"""
Sync check: hand-written Python mirrors vs generated/encoder-data.json.

Phase 1 of the encoder single-sourcing (docs/ENCODER_SINGLE_SOURCE_PROPOSAL.md)
replaced observation.py's tables with data LOADED from the generated JSON.
enums.py's public IntEnum classes and string-vocab lists stay hand-written
(replacing them risks silent public-API renames) — this check makes their
drift mechanical instead of silent: every hand member must exist TS-side with
the same value. TS may legitimately carry extra members (Python mirrors only
what the wire uses).

Exit 0 = in sync; non-zero = drift (fails rl-verify).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rl import encoder_data, enums  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# Python-only sentinel members that intentionally have no TS counterpart
# (they encode the WIRE's "absent" convention, e.g. multi_hit_type = -1).
PY_ONLY_SENTINELS: dict[str, set[str]] = {"MultiHitType": {"NA"}}


def check_int_enum(py_enum, ts_name: str) -> None:
    ts = encoder_data.ENUMS.get(ts_name)
    if ts is None:
        check(f"enum {ts_name}", False, "missing from encoder-data.json")
        return
    sentinels = PY_ONLY_SENTINELS.get(ts_name, set())
    bad = [
        f"{m.name}={m.value}!={ts.get(m.name)}"
        for m in py_enum
        if m.name not in sentinels and ts.get(m.name) != m.value
    ]
    check(f"enum {ts_name} ({len(list(py_enum))} members ⊆ TS)", not bad, ", ".join(bad[:5]))


def main() -> int:
    # Public IntEnums: every Python member must exist TS-side with the same value
    check_int_enum(enums.PokemonType, "PokemonType")
    check_int_enum(enums.StatusEffect, "StatusEffect")
    check_int_enum(enums.WeatherType, "WeatherType")
    check_int_enum(enums.TerrainType, "TerrainType")
    check_int_enum(enums.MoveCategory, "MoveCategory")
    check_int_enum(enums.Nature, "Nature")
    check_int_enum(enums.Gender, "Gender")
    check_int_enum(enums.BattleType, "BattleType")
    check_int_enum(enums.ModifierTier, "ModifierTier")
    check_int_enum(enums.PokeballType, "PokeballType")
    check_int_enum(enums.AiType, "AiType")
    check_int_enum(enums.Stat, "Stat")
    check_int_enum(enums.MultiHitType, "MultiHitType")
    check_int_enum(enums.ArenaTagSide, "ArenaTagSide")
    check_int_enum(enums.BattlerIndex, "BattlerIndex")
    check_int_enum(enums.GameMode, "GameModes")
    check_int_enum(enums.TimeOfDay, "TimeOfDay")
    check_int_enum(enums.BattleStyle, "BattleStyle")
    check_int_enum(enums.BerryType, "BerryType")
    check_int_enum(enums.HitResult, "HitResult")
    check_int_enum(enums.MoveResult, "MoveResult")
    check_int_enum(enums.Command, "Command")

    # String vocab lists: exact membership vs the TS enums / feature-table keys
    ts_battler = set(encoder_data.STRING_ENUMS["BattlerTagType"].values())
    py_battler = set(enums._BATTLER_TAG_VALUES)
    check(
        f"_BATTLER_TAG_VALUES ({len(py_battler)})",
        py_battler == ts_battler,
        f"py-only={sorted(py_battler - ts_battler)[:4]} ts-only={sorted(ts_battler - py_battler)[:4]}",
    )
    ts_arena = set(encoder_data.STRING_ENUMS["ArenaTagType"].values())
    py_arena = set(enums._ARENA_TAG_VALUES)
    check(
        f"_ARENA_TAG_VALUES ({len(py_arena)})",
        py_arena == ts_arena,
        f"py-only={sorted(py_arena - ts_arena)[:4]} ts-only={sorted(ts_arena - py_arena)[:4]}",
    )
    ts_mods = set(encoder_data.MODIFIER_FEATURES.keys())
    py_mods = set(enums._MODIFIER_TYPE_VALUES)
    check(
        f"_MODIFIER_TYPE_VALUES ({len(py_mods)})",
        py_mods == ts_mods,
        f"py-only={sorted(py_mods - ts_mods)[:4]} ts-only={sorted(ts_mods - py_mods)[:4]}",
    )

    # Phase map: exact equality (including the enemy_command alias)
    check(
        "PHASE_STR_TO_ID == phase_index_map",
        dict(enums.PHASE_STR_TO_ID) == dict(encoder_data.PHASE_INDEX_MAP),
        "",
    )

    # Loaded tables really are the generated objects (no re-divergence path)
    check("observation dims from generated data", __import__("rl.observation", fromlist=["OBSERVATION_DIM"]).OBSERVATION_DIM == encoder_data.DIMS["OBSERVATION_DIM"])

    # feature_names' block ranges vs the TS-emitted layout manifest (Phase 2):
    # the name table's top-level structure must tile the vector exactly as
    # the encoder writes it.
    from rl import feature_names  # noqa: PLC0415

    manifest = [(b["name"], b["base"], b["dim"]) for b in encoder_data.BLOCK_LAYOUT]
    ranges = [(label, start, size) for (label, start, size) in feature_names.BLOCK_RANGES]
    check(
        f"feature_names.BLOCK_RANGES == TS block_layout ({len(manifest)} blocks)",
        ranges == manifest,
        f"first diff: {next(((a, b) for a, b in zip(ranges, manifest, strict=False) if a != b), (len(ranges), len(manifest)))}",
    )

    if FAILURES:
        print(f"\nGENERATED-DATA SYNC: FAIL ({len(FAILURES)})")
        return 1
    print("\nGENERATED-DATA SYNC: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
