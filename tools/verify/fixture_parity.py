#!/usr/bin/env python3
"""
Cross-language golden check: the Python encoder must reproduce the TS
encoder's golden output for every fixture, bit for bit.

Goldens are produced by the TS side:
    UPDATE_RL_GOLDEN=1 pnpm exec vitest run test/rl/spaces-encoding.test.ts

Usage: python3 tools/verify/fixture_parity.py
"""

from __future__ import annotations

import base64
import json
import sys

import numpy as np

from common import REPO_ROOT  # noqa: E402  (sys.path side effect)
from rl.feature_names import dim_to_name, name_to_dim
from rl.observation import ACTION_SPACE_SIZE, encode_observation, extract_action_mask, parse_game_state

FIXTURE_DIR = REPO_ROOT / "test" / "rl" / "fixtures"
FIXTURES = ["full", "minimal", "edge"]


def main() -> int:
    failures = 0

    for name in FIXTURES:
        state_path = FIXTURE_DIR / f"{name}.state.json"
        golden_path = FIXTURE_DIR / f"{name}.golden.b64"
        if not golden_path.exists():
            print(f"FAIL {name}: golden missing — generate with UPDATE_RL_GOLDEN=1 vitest run")
            failures += 1
            continue

        fixture = json.loads(state_path.read_text())
        ts_obs = np.frombuffer(base64.b64decode(golden_path.read_text().strip()), dtype="<f4")
        py_obs = encode_observation(parse_game_state(fixture))

        if py_obs.tobytes() == ts_obs.tobytes():
            print(f"OK   {name}: bitwise identical ({int((py_obs != 0).sum())} nonzero dims)")
        else:
            bad = np.where(py_obs != ts_obs)[0]
            print(f"FAIL {name}: {len(bad)} dims differ")
            for i in bad[:8]:
                print(f"     {dim_to_name(int(i))} (dim {i}): ts={ts_obs[i]!r} py={py_obs[i]!r}")
            failures += 1

    # Named-dim probes pin the feature-name table to encoder reality
    full = json.loads((FIXTURE_DIR / "full.state.json").read_text())
    obs = encode_observation(parse_game_state(full))
    p0 = full.get("player_0") or {}
    probes: list[tuple[str, float]] = [
        ("player_0/valid", 1.0 if p0.get("valid") else 0.0),
        ("player_0/hp_ratio", float(p0.get("hp_ratio", 0.0))),
        ("battle/is_classic", 1.0 if (full.get("battle") or {}).get("is_classic") else 0.0),
    ]
    for pname, expected in probes:
        got = float(obs[name_to_dim(pname)])
        if abs(got - expected) > 1e-5:
            print(f"FAIL probe {pname}: expected {expected}, got {got}")
            failures += 1
        else:
            print(f"OK   probe {pname} = {got}")

    # Mask-length asymmetry, pinned per side: edge fixture carries a 57-long mask.
    # TS returns all-false for any length != 58; Python pads with False.
    edge = json.loads((FIXTURE_DIR / "edge.state.json").read_text())
    edge_mask_raw = (edge.get("phase") or {}).get("action_mask") or []
    if len(edge_mask_raw) != ACTION_SPACE_SIZE:
        py_mask = extract_action_mask(parse_game_state(edge))
        expected_py = list(bool(v) for v in edge_mask_raw) + [False] * (ACTION_SPACE_SIZE - len(edge_mask_raw))
        if list(py_mask) == expected_py:
            print(f"OK   edge mask: Python pads {len(edge_mask_raw)} -> {ACTION_SPACE_SIZE} (TS returns all-false; known asymmetry)")
        else:
            print("FAIL edge mask: Python padding semantics changed")
            failures += 1

    print(f"\nFIXTURE PARITY: {'FAIL' if failures else 'OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
