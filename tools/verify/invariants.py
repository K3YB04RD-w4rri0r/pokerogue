"""
Observation/mask invariant checks, driven by the feature-name table.

Used by check_parity.py on every record; importable anywhere.
"""

from __future__ import annotations

import numpy as np

import common  # noqa: F401  (sys.path side effect)
from rl.feature_names import FEATURE_NAMES, ONE_HOT_GROUPS, dim_to_name
from rl.observation import ACTION_SPACE_SIZE, OBSERVATION_DIM

EPS = 1e-5

# Features that legitimately leave [0, 1].
# stat stages and priority are signed (encoded /6 and /7); nature multipliers
# are raw 0.9/1.0/1.1; contrary-style ability cells can be -1.
_SIGNED_PATTERNS = ("stat_stages[", "/priority", "ability_feat[", "passive_feat[", "stat_change_")
_GT1_PATTERNS = ("nature_mult",)


def _expected_range(name: str) -> tuple[float, float]:
    if any(p in name for p in _SIGNED_PATTERNS):
        return (-1.0 - EPS, 1.0 + EPS)
    if any(p in name for p in _GT1_PATTERNS):
        return (0.0 - EPS, 1.25 + EPS)
    return (0.0 - EPS, 1.0 + EPS)


_RANGES_LO = np.array([_expected_range(n)[0] for n in FEATURE_NAMES], dtype=np.float32)
_RANGES_HI = np.array([_expected_range(n)[1] for n in FEATURE_NAMES], dtype=np.float32)


def check_observation(obs: np.ndarray) -> list[str]:
    """Return a list of human-readable invariant violations (empty = OK)."""
    errors: list[str] = []
    if obs.shape != (OBSERVATION_DIM,):
        return [f"wrong shape {obs.shape}, expected ({OBSERVATION_DIM},)"]

    bad = np.where(~np.isfinite(obs))[0]
    if len(bad):
        errors.append(f"non-finite at {[dim_to_name(int(i)) for i in bad[:5]]}")

    out_of_range = np.where((obs < _RANGES_LO) | (obs > _RANGES_HI))[0]
    if len(out_of_range):
        samples = [f"{dim_to_name(int(i))}={obs[i]:.4g}" for i in out_of_range[:5]]
        errors.append(f"{len(out_of_range)} dims out of expected range, e.g. {samples}")

    for start, size, label in ONE_HOT_GROUPS:
        s = float(obs[start : start + size].sum())
        if s > 1.0 + EPS:
            errors.append(f"one-hot group {label} (@{start},{size}) sums to {s:.4g} > 1")

    return errors


def check_mask(mask: np.ndarray, phase: str) -> list[str]:
    errors: list[str] = []
    if mask.shape != (ACTION_SPACE_SIZE,):
        return [f"mask wrong shape {mask.shape}"]
    if phase not in ("game_over", "unknown") and not mask.any():
        errors.append(f"empty action mask in decision phase {phase!r}")
    return errors
