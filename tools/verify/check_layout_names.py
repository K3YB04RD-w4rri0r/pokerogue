#!/usr/bin/env python3
"""
Differential layout-name probes (encoder single-sourcing Phase 2).

feature_names.py hand-mirrors the encoder's SUB-BLOCK order (its top-level
block layout is verified against the TS-emitted manifest by
check_generated_sync.py). This check binds the NAMES to the actual encoder
write order: perturb exactly one input field of the "full" fixture, re-encode
with the Python encoder (bitwise-parity-locked to TS), and require every
changed dimension's name to reference the perturbed field. A re-ordered
sub-block makes a probe light up dims with unrelated names and fails here —
the Python analog of test/rl/semantic/layout-canary.test.ts.

Usage: python3 tools/verify/check_layout_names.py
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable

import numpy as np
from common import REPO_ROOT  # noqa: E402  (sys.path side effect)

from rl.feature_names import FEATURE_NAMES
from rl.observation import encode_observation, parse_game_state

FIXTURE = REPO_ROOT / "test" / "rl" / "fixtures" / "full.state.json"


def set_path(d: dict, path: list, value) -> None:
    cur = d
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


# (probe label, json path, new value, predicate on each changed dim name)
Probe = tuple[str, list, object, Callable[[str], bool]]

PROBES: list[Probe] = [
    ("pokemon scalar", ["player_0", "hp_ratio"], 0.123,
     lambda n: n == "player_0/hp_ratio"),
    ("move scalar", ["player_0", "moves", 0, "power"], 222,
     lambda n: n == "player_0/moves[0]/power"),
    ("move type -> move one-hot + derived", ["player_0", "moves", 0, "type"], 9,
     lambda n: n.startswith("player_0/moves[0]/type_onehot") or n.startswith("derived/")),
    ("battle scalar", ["battle", "wave_index"], 137,
     lambda n: n == "battle/wave_index"),
    ("field weather", ["field", "weather_type"], 4,
     lambda n: n.startswith("field/weather")),
    ("phase one-hot", ["phase", "current_phase"], "switch",
     lambda n: n.startswith("phase_onehot[")),
    ("command_field_index patch", ["phase", "command_field_index"], 1,
     lambda n: "command_field_index" in n),
    ("reward option tier", ["shop", "reward_options", 0, "tier"], 4,
     lambda n: n.startswith("shop/reward[0]/")),
    ("learn-move block", ["phase", "learn_move_stats"],
     {"move_id": 33, "name": "Tackle", "type": 0, "category": 0, "power": 40,
      "accuracy": 100, "pp_max": 35, "pp_used": 0},
     lambda n: n.startswith("learn_move/")),
]


def main() -> int:
    base_raw = json.loads(FIXTURE.read_text())
    base_obs = encode_observation(parse_game_state(base_raw))
    failures = 0

    for label, path, value, pred in PROBES:
        mutated = copy.deepcopy(base_raw)
        try:
            set_path(mutated, path, value)
        except (KeyError, IndexError, TypeError) as err:
            print(f"FAIL {label}: fixture path {path} not settable ({err})")
            failures += 1
            continue
        obs = encode_observation(parse_game_state(mutated))
        changed = np.flatnonzero(obs != base_obs)
        names = [FEATURE_NAMES[int(i)] for i in changed]
        offenders = [n for n in names if not pred(n)]
        if len(changed) == 0:
            print(f"FAIL {label}: perturbing {path} changed NO dims (dead input or wrong path)")
            failures += 1
        elif offenders:
            print(f"FAIL {label}: unrelated dims changed: {offenders[:5]} (of {len(changed)})")
            failures += 1
        else:
            print(f"OK   {label}: {len(changed)} dim(s), names consistent (e.g. {names[0]})")

    if failures:
        print(f"\nLAYOUT NAME PROBES: FAIL ({failures}/{len(PROBES)})")
        return 1
    print(f"\nLAYOUT NAME PROBES: OK ({len(PROBES)} probes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
