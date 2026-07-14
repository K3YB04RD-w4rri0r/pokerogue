"""
Single-source encoder data, loaded from generated/encoder-data.json.

The JSON artifact is GENERATED from the live TypeScript encoder modules
(spaces.ts, ability-features.ts, modifier-features.ts, src/enums/*) by
test/rl/encoder-data-sync.test.ts, which also gates staleness in CI: the
checked-in JSON must byte-match a fresh serialization of the TS tables.
Regenerate after intentional table changes with:

    UPDATE_RL_ENCODER_DATA=1 pnpm exec vitest run test/rl/encoder-data-sync.test.ts

Python consumers import the names below instead of hand-mirroring the
tables (see docs/ENCODER_SINGLE_SOURCE_PROPOSAL.md, Phase 1). The encode
LOGIC remains hand-written in observation.py, proven bitwise-equal to
spaces.ts by the golden fixtures and tools/verify/check_parity.py.
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "generated" / "encoder-data.json"
_RAW: dict = json.loads(_PATH.read_text(encoding="utf-8"))

# Grouped sections (raw dicts)
DIMS: dict[str, int] = _RAW["dims"]
CAPS: dict[str, int] = _RAW["caps"]
ACTION_SPACE: dict[str, int] = _RAW["action_space"]
# TS numeric enums as {MEMBER_NAME: value}; string enums as {MEMBER: "value"}
ENUMS: dict[str, dict[str, int]] = _RAW["enums"]
STRING_ENUMS: dict[str, dict[str, str]] = _RAW["string_enums"]

# Top-level block layout in encoder write order: [{name, base, dim}, ...]
# tiling [0, OBSERVATION_DIM) exactly. feature_names.BLOCK_RANGES is verified
# against this by tools/verify/check_generated_sync.py.
BLOCK_LAYOUT: list[dict] = _RAW["block_layout"]

# Index maps / orderings (order is load-bearing for one-hot layouts)
PHASE_INDEX_MAP: dict[str, int] = _RAW["phase_index_map"]
CURATED_VOLATILE_TAGS: list[str] = _RAW["curated_volatile_tags"]
ARENA_TAG_ORDER: list[str] = _RAW["arena_tag_order"]
KEY_ARENA_TAGS: list[str] = _RAW["key_arena_tags"]
POKEMON_SLOT_KEYS: list[str] = _RAW["pokemon_slot_keys"]
ACTIVE_SLOT_KEYS: list[str] = _RAW["active_slot_keys"]
POKEBALL_KEYS: list[str] = _RAW["pokeball_keys"]
PARTY_FLAG_IDS: list[str] = _RAW["party_flag_ids"]
ENEMY_MOD_IDS: list[str] = _RAW["enemy_mod_ids"]
ENEMY_MOD_NORM_DIVISORS: list[float] = _RAW["enemy_mod_norm_divisors"]
MODIFIER_PRIORITY_KEYS: list[int] = _RAW["modifier_priority_keys"]

# Move-target classes and the has_other_effect OR-set
SELF_ALLY_TARGETS: frozenset[int] = frozenset(_RAW["self_ally_targets"])
SINGLE_ENEMY_TARGETS: frozenset[int] = frozenset(_RAW["single_enemy_targets"])
OTHER_EFFECT_FLAGS: tuple[str, ...] = tuple(_RAW["other_effect_flags"])

# Numeric tables
TYPE_EFFECTIVENESS: list[list[float]] = _RAW["type_effectiveness"]
STAGE_MULTIPLIERS: list[float] = _RAW["stage_multipliers"]
ABILITY_FEATURES: list[list[float]] = _RAW["ability_features"]
MODIFIER_FEATURES: dict[str, list[float]] = _RAW["modifier_features"]
DEFAULT_MODIFIER_FEATURES: list[float] = _RAW["default_modifier_features"]
