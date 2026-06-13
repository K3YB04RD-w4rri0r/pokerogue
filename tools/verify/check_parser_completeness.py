#!/usr/bin/env python3
"""
Static check: every observation.py parser must assign every field of the
dataclass it constructs.

The Python encoder reads dataclass attributes; a field the parser forgets to
assign silently keeps its default (0/False/[]) and only shows up as a parity
mismatch when the field is nonzero in real play (e.g. is_mega — rare). This
catches the whole bug class statically. Empty default constructions like
`ObsPokemon()` for invalid slots are intentionally ignored (we look at the
fullest constructor call per parser).

Usage: python3 tools/verify/check_parser_completeness.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

OBSERVATION_PY = Path(__file__).resolve().parent.parent.parent / "src" / "rl" / "observation.py"


def main() -> int:
    tree = ast.parse(OBSERVATION_PY.read_text())

    dataclass_fields: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fields = {
                s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
            if fields:
                dataclass_fields[node.name] = fields

    # Fullest constructor call per (parser function, dataclass)
    best: dict[tuple[str, str], set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_parse"):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in dataclass_fields:
                    key = (node.name, call.func.id)
                    assigned = {kw.arg for kw in call.keywords if kw.arg}
                    if key not in best or len(assigned) > len(best[key]):
                        best[key] = assigned

    issues = []
    for (fn, cls), assigned in sorted(best.items()):
        missing = dataclass_fields[cls] - assigned
        if missing and assigned:  # skip pure-default constructions
            issues.append(f"{fn} -> {cls}: parser never assigns {sorted(missing)}")

    if issues:
        print("PARSER COMPLETENESS: FAIL")
        for i in issues:
            print(f"  {i}")
        return 1
    print(f"PARSER COMPLETENESS: OK ({len(best)} parser/dataclass pairs checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
