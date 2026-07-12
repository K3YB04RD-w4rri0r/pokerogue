"""Declarative run configuration for the PokeRogue RL environment.

One YAML (or JSON) file describes a run — seed, episode budget, starting
party/wave/level/money/items, game overrides, reward shaping, env plumbing —
and every entry point consumes it:

    env = PokeRogueEnv.from_config("runs/my-run.yaml")          # training
    python3 tools/run_policy.py --config runs/my-run.yaml       # eval, either mode
    python3 tools/play.py --config runs/my-run.yaml             # play it yourself
    python3 examples/rl/train_maskable_ppo.py --config runs/my-run.yaml

Schema (all keys optional; unknown top-level keys are rejected)::

    seed: exp-42                 # game RNG seed (str). Omit -> random per episode.
    waves: 50                    # WAVE CAP: truncates at the first decision of wave 51
                                 # (waves*50 decisions remains only as a safety backstop)
    starters: MEWTWO,LUGIA       # SpeciesId names, CSV string or list
    starting_wave: 10            # -> STARTING_WAVE_OVERRIDE
    starting_level: 20           # -> STARTING_LEVEL_OVERRIDE
    starting_money: 5000         # -> STARTING_MONEY_OVERRIDE
    starting_biome: 5            # -> STARTING_BIOME_OVERRIDE (BiomeId int)
    battle_style: double         # -> BATTLE_STYLE_OVERRIDE
                                 #    (single|double|even-doubles|odd-doubles)
    starting_modifiers:          # party-wide items -> STARTING_MODIFIER_OVERRIDE
      - {name: EXP_SHARE, count: 2}
      - {name: GOLDEN_POKEBALL}
    starting_held_items:         # lead pokemon held items -> STARTING_HELD_ITEMS_OVERRIDE
      - {name: LEFTOVERS, count: 2}
    pokeballs:                   # -> POKEBALL_OVERRIDE (unlisted types: standard
      pokeball: 10               #    inventory = 5 pokeballs, 0 of the rest)
      master_ball: 1
    overrides:                   # raw DefaultOverrides passthrough (power users);
      NEVER_CRIT_OVERRIDE: true  # wins over the sugar keys above on conflict
    reward:                      # partial RewardConfig (src/rl/rewards.ts)
      turnPenalty: -0.02
    env:                         # PokeRogueEnv plumbing kwargs
      lean: true
      respawn_every: 50
    train: {}                    # free-form; consumed by training scripts

The sugar keys map onto the game's own override hooks (src/overrides.ts), so
anything expressible there also works via ``overrides:`` directly.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

# Keys of the TS RewardConfig (src/rl/rewards.ts) — used for a warning-level
# typo check on the `reward:` section (unknown keys are ignored TS-side).
REWARD_CONFIG_KEYS = frozenset({
    "hpDamageDealt", "hpDamageTaken", "enemyKo", "playerKo", "waveCleared",
    "bossWaveCleared", "runWon", "runLost", "ranAway", "moneyGained",
    "pokemonCaught", "modifierSelected", "modifierTierBonus", "turnPenalty",
    "statBoostReward", "statusInflictionReward",
})

# PokeRogueEnv constructor kwargs accepted in the `env:` section.
# Known train: section keys (consumed by examples/rl/train_maskable_ppo.py).
# Unknown keys warn — a typo like `timestep:` would otherwise silently fall
# back to the default.
TRAIN_CONFIG_KEYS = frozenset({
    "timesteps", "save", "num_envs", "n_steps", "net_arch", "device",
    "checkpoint_every", "tensorboard", "batch_size",
    # learner-level reproducibility + core PPO knobs (sb3 defaults when unset)
    "seed", "gamma", "gae_lambda", "learning_rate", "ent_coef", "clip_range",
})

ENV_KWARG_KEYS = frozenset({
    "cli_path", "node_bin", "step_timeout", "boot_timeout", "stderr_log",
    "lean", "respawn", "respawn_every", "fog_of_war",
})

# PokeballType enum order (spaces.ts NUM_POKEBALL_TYPES / PokeballCounts keys).
POKEBALL_NAMES = ("pokeball", "great_ball", "ultra_ball", "rogue_ball", "master_ball")

# The game's standard starting inventory (see POKEBALL_OVERRIDE defaults).
DEFAULT_POKEBALL_COUNTS = {0: 5, 1: 0, 2: 0, 3: 0, 4: 0}

BATTLE_STYLES = ("single", "double", "even-doubles", "odd-doubles")

TOP_LEVEL_KEYS = frozenset({
    "seed", "waves", "starters", "starting_wave", "starting_level",
    "starting_money", "starting_biome", "battle_style", "starting_modifiers",
    "starting_held_items", "pokeballs", "overrides", "reward", "env", "train",
})


class RunConfigError(ValueError):
    """A run-config file is malformed."""


@dataclass
class RunConfig:
    """Parsed, validated run configuration. See module docstring for schema."""

    seed: str | None = None
    waves: int | None = None
    starters: str | None = None
    starting_wave: int | None = None
    starting_level: int | None = None
    starting_money: int | None = None
    starting_biome: int | None = None
    battle_style: str | None = None
    starting_modifiers: list[dict] | None = None
    starting_held_items: list[dict] | None = None
    pokeballs: dict[str, int] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, float] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    source: str | None = None  # path the config was loaded from (diagnostics)

    # ── Overrides assembly ────────────────────────────────────────────

    def merged_overrides(self) -> dict[str, Any]:
        """Sugar keys expanded into DefaultOverrides keys, merged with the raw
        ``overrides:`` section (raw wins on conflict, with a warning)."""
        sugar: dict[str, Any] = {}
        if self.starting_wave is not None:
            sugar["STARTING_WAVE_OVERRIDE"] = int(self.starting_wave)
        if self.starting_level is not None:
            sugar["STARTING_LEVEL_OVERRIDE"] = int(self.starting_level)
        if self.starting_money is not None:
            sugar["STARTING_MONEY_OVERRIDE"] = int(self.starting_money)
        if self.starting_biome is not None:
            sugar["STARTING_BIOME_OVERRIDE"] = int(self.starting_biome)
        if self.battle_style is not None:
            sugar["BATTLE_STYLE_OVERRIDE"] = self.battle_style
        if self.starting_modifiers is not None:
            sugar["STARTING_MODIFIER_OVERRIDE"] = self.starting_modifiers
        if self.starting_held_items is not None:
            sugar["STARTING_HELD_ITEMS_OVERRIDE"] = self.starting_held_items
        if self.pokeballs is not None:
            counts = dict(DEFAULT_POKEBALL_COUNTS)
            for name, count in self.pokeballs.items():
                counts[POKEBALL_NAMES.index(name)] = int(count)
            sugar["POKEBALL_OVERRIDE"] = {"active": True, "pokeballs": counts}

        merged = dict(sugar)
        for key, value in self.overrides.items():
            if key in sugar and sugar[key] != value:
                warnings.warn(
                    f"run config: overrides.{key} ({value!r}) wins over the "
                    f"shortcut key's value ({sugar[key]!r})",
                    stacklevel=3,
                )
            merged[key] = value
        return merged

    # ── Entry-point adapters ──────────────────────────────────────────

    def to_env_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``PokeRogueEnv(**kwargs)``."""
        kwargs: dict[str, Any] = dict(self.env)
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.waves is not None:
            kwargs["waves"] = self.waves
        if self.starters is not None:
            kwargs["starters"] = self.starters
        overrides = self.merged_overrides()
        if overrides:
            kwargs["overrides"] = overrides
        if self.reward:
            kwargs["reward_config"] = dict(self.reward)
        return kwargs

    def to_cli_args(self) -> list[str]:
        """Flags for a direct ``node dist/rl/cli.js --interactive`` spawn."""
        args: list[str] = []
        if self.seed is not None:
            args.append(f"--seed={self.seed}")
        if self.waves is not None:
            args.append(f"--waves={self.waves}")
        if self.starters is not None:
            args.append(f"--starters={self.starters}")
        for key, value in self.merged_overrides().items():
            args.append(f"--override={key}={json.dumps(value)}")
        if self.reward:
            args.append(f"--reward-config={json.dumps(self.reward)}")
        if self.env.get("lean"):
            args.append("--lean")
        if self.env.get("fog_of_war"):
            args.append("--fog-of-war")
        return args

    def to_url_query(self) -> str:
        """Query-string fragment for the rendered bridge (appended after
        ``?rl=true``); mirrors the headless config surface."""
        pairs: list[tuple[str, str]] = []
        if self.seed is not None:
            pairs.append(("seed", self.seed))
        if self.starters is not None:
            pairs.append(("starters", self.starters))
        if self.waves is not None:
            pairs.append(("waves", str(self.waves)))
        for key, value in self.merged_overrides().items():
            pairs.append(("override", f"{key}={json.dumps(value)}"))
        if self.reward:
            pairs.append(("rewardConfig", json.dumps(self.reward)))
        if self.env.get("fog_of_war"):
            pairs.append(("fog", "1"))
        query = urlencode(pairs)
        return f"&{query}" if query else ""


def _validate(cfg: RunConfig) -> None:
    if cfg.battle_style is not None and cfg.battle_style not in BATTLE_STYLES:
        raise RunConfigError(f"battle_style must be one of {BATTLE_STYLES}, got {cfg.battle_style!r}")
    for section_name in ("starting_modifiers", "starting_held_items"):
        section = getattr(cfg, section_name)
        if section is None:
            continue
        if not isinstance(section, list):
            raise RunConfigError(f"{section_name} must be a list of {{name, count?}} entries")
        for entry in section:
            if not isinstance(entry, dict) or "name" not in entry:
                raise RunConfigError(f"{section_name} entries need a 'name' key, got {entry!r}")
    if cfg.pokeballs is not None:
        for name in cfg.pokeballs:
            if name not in POKEBALL_NAMES:
                raise RunConfigError(f"pokeballs key {name!r} not one of {POKEBALL_NAMES}")
    unknown_reward = set(cfg.reward) - REWARD_CONFIG_KEYS
    if unknown_reward:
        warnings.warn(f"run config: unknown reward keys {sorted(unknown_reward)} (ignored by rewards.ts)", stacklevel=2)
    unknown_env = set(cfg.env) - ENV_KWARG_KEYS
    if unknown_env:
        raise RunConfigError(f"env section: unknown PokeRogueEnv kwargs {sorted(unknown_env)}")
    unknown_train = set(cfg.train) - TRAIN_CONFIG_KEYS
    if unknown_train:
        warnings.warn(f"run config: unknown train keys {sorted(unknown_train)} (the trainer will ignore them)", stacklevel=2)
    if not isinstance(cfg.overrides, dict):
        raise RunConfigError("overrides must be a mapping of DefaultOverrides keys")


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML (or JSON) run config."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as err:
            raise RunConfigError("YAML run configs need pyyaml (pip install pyyaml)") from err
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RunConfigError(f"{path}: top level must be a mapping")

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise RunConfigError(f"{path}: unknown top-level keys {sorted(unknown)} (valid: {sorted(TOP_LEVEL_KEYS)})")

    starters = raw.get("starters")
    if isinstance(starters, list):
        starters = ",".join(str(s) for s in starters)
    elif starters is not None:
        starters = str(starters)

    cfg = RunConfig(
        seed=str(raw["seed"]) if raw.get("seed") is not None else None,
        waves=int(raw["waves"]) if raw.get("waves") is not None else None,
        starters=starters,
        starting_wave=raw.get("starting_wave"),
        starting_level=raw.get("starting_level"),
        starting_money=raw.get("starting_money"),
        starting_biome=raw.get("starting_biome"),
        battle_style=raw.get("battle_style"),
        starting_modifiers=raw.get("starting_modifiers"),
        starting_held_items=raw.get("starting_held_items"),
        pokeballs=raw.get("pokeballs"),
        overrides=raw.get("overrides") or {},
        reward=raw.get("reward") or {},
        env=raw.get("env") or {},
        train=raw.get("train") or {},
        source=str(path),
    )
    _validate(cfg)
    return cfg
