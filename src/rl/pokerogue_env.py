"""
Gymnasium environment wrapper for the PokeRogue headless RL runner.

Spawns `node dist/rl/cli.js --interactive` (one episode per process) and
speaks its JSON-lines protocol over stdio. Observations are encoded with the
Python encoder (observation.py); rewards come from the TypeScript
RewardCalculator via the protocol's per-step `reward` field.

Usage:
    from rl.pokerogue_env import PokeRogueEnv     # with <repo>/src on sys.path

    env = PokeRogueEnv(waves=50)
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(action)
    mask = env.action_masks()                     # sb3-contrib MaskablePPO hook
    env.close()

Build prerequisite:  pnpm rl:build   (creates dist/rl/cli.js)

Vectorization: safe — pure stdio, no ports, no disk state. Each env is a full
Phaser+jsdom node process (several hundred MB); ~4-8 envs per 16 GB RAM.
"""

from __future__ import annotations

import atexit
import base64
import json
import queue
import subprocess
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np

from .observation import (
    ACTION_SPACE_SIZE,
    OBSERVATION_DIM,
    encode_observation,
    extract_action_mask,
    parse_game_state,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CLI = REPO_ROOT / "dist" / "rl" / "cli.js"

# Setup phases auto-played by reset(); check_switch is left to the agent.
SETUP_PHASES = {"title", "select_gender", "starter"}

PROTOCOL_VERSION = 5  # v9 observation: 69 curated tags, 60-dim moves, 6991 dims


class ProtocolError(RuntimeError):
    """The CLI subprocess violated the expected protocol."""


class _LineReader:
    """Background thread draining the subprocess stdout into a queue."""

    def __init__(self, stream):
        self._queue: queue.Queue = queue.Queue()
        t = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        t.start()

    def _drain(self, stream) -> None:
        for line in stream:
            self._queue.put(line)
        self._queue.put(None)

    def next(self, timeout: float) -> str | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no CLI output within {timeout}s") from None


class PokeRogueEnv(gym.Env):
    """PokeRogue classic-mode runs as a gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cli_path: str | Path | None = None,
        waves: int = 50,
        seed: str | None = None,
        node_bin: str = "node",
        step_timeout: float = 60.0,
        boot_timeout: float = 180.0,
        stderr_log: str | Path | None = None,
        lean: bool = True,
        respawn: bool = False,
        respawn_every: int = 50,
        reward_config: dict | None = None,
        overrides: dict | None = None,
        starters: str | None = None,
        fog_of_war: bool = False,
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_DIM,), np.float32)
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        self._cli_path = Path(cli_path) if cli_path else DEFAULT_CLI
        self._waves = waves
        self._base_seed = seed
        self._node_bin = node_bin
        self._step_timeout = step_timeout
        self._boot_timeout = boot_timeout
        self._stderr_log = Path(stderr_log) if stderr_log else None
        # lean: the TS side is the encoding authority — state messages carry
        # obsB64 + mask and omit the bulky gameState JSON. Set lean=False to
        # receive full gameState and encode locally (debugging / inspection).
        self._lean = lean
        # fog_of_war: mask enemy private info to what a human could know
        # (unseen moves, unrevealed abilities, IV/nature-derived values,
        # never-seen bench members). Default False = full information.
        self._fog_of_war = fog_of_war
        # respawn: force a fresh node process per episode (the original,
        # slower lifecycle). Default False = in-process resets (~3ms vs ~2s),
        # with automatic fallback to respawn whenever the reset path errors.
        self._respawn = respawn
        # respawn_every: recycle the node process after N in-process episodes.
        # A small residual leak (~70 UI nodes / ~4.5MB heap per episode inside
        # lazily-managed UI handler containers) slowly degrades throughput in
        # very long-lived processes; periodic recycling bounds it while keeping
        # the amortized reset cost ~40ms/episode at N=50. Set 0 to disable.
        self._respawn_every = max(0, respawn_every)
        self._episodes_since_spawn = 0
        # Partial RewardConfig overrides (see src/rl/rewards.ts for fields)
        self._reward_config = reward_config
        # Game overrides (DefaultOverrides keys, e.g. BATTLE_STYLE_OVERRIDE).
        # NOTE: Mystery Encounters are removed from this environment by
        # decision — the CLI disables them at boot (their option phases are
        # outside the 58-action interface). The env's game scope is classic
        # mode without MEs; passing MYSTERY_ENCOUNTER_RATE_OVERRIDE here
        # re-enables them deliberately (expect step-timeout truncations).
        self._overrides = dict(overrides or {})
        # Comma-separated SpeciesId names for a custom starting party (e.g. a
        # legendary team), forwarded to the CLI as --starters. None -> default.
        self._starters = starters

        self._proc: subprocess.Popen | None = None
        self._reader: _LineReader | None = None
        self._stderr_fh = None
        self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        self._episode = 0
        self._needs_reset = True
        # Last step/reset info dict — includes info["game_state"] when
        # lean=False; consumed by ObservationWrappers (README: bring your
        # own features).
        self.last_info: dict = {}
        self._last_wave = 0

        if not self._cli_path.exists():
            raise FileNotFoundError(
                f"{self._cli_path} not found — build the headless bundle first:  pnpm rl:build"
            )
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config, **kwargs) -> "PokeRogueEnv":
        """Build an env from a run config (path to a YAML/JSON file, or a
        RunConfig object). Explicit ``kwargs`` win over the file's values.

        ::

            env = PokeRogueEnv.from_config("examples/rl/legendary.yaml")
            env = PokeRogueEnv.from_config("run.yaml", waves=5)   # override
        """
        from .run_config import RunConfig, load_run_config

        cfg = config if isinstance(config, RunConfig) else load_run_config(config)
        merged = {**cfg.to_env_kwargs(), **kwargs}
        return cls(**merged)

    # ── gymnasium API ──────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Game seed: explicit option > gymnasium seed > base-seed-derived > RNG
        if options and "game_seed" in options:
            game_seed = str(options["game_seed"])
        elif seed is not None:
            game_seed = f"gym-{seed}"
        elif self._base_seed is not None:
            game_seed = f"{self._base_seed}-ep{self._episode}"
        else:
            game_seed = f"ep-{self.np_random.integers(0, 2**63):x}"
        self._episode += 1

        recycle_due = self._respawn_every > 0 and self._episodes_since_spawn >= self._respawn_every
        ready = None
        if not self._respawn and not recycle_due and self._proc is not None and self._proc.poll() is None:
            # In-process reset (~3ms vs ~2s respawn): the CLI waits for a
            # lifecycle command after each episode's `done`.
            try:
                self._send_raw({"cmd": "reset", "seed": game_seed, "waves": self._waves})
                ready = self._await_ready(self._boot_timeout)
            except (ProtocolError, TimeoutError, OSError):
                ready = None  # fall through to a clean respawn
        if ready is None:
            self.close()
            self._spawn(game_seed)
            ready = self._await_ready(self._boot_timeout)
            self._episodes_since_spawn = 0
        self._episodes_since_spawn += 1
        self._check_versions(ready)

        msg = self._next_decision()
        while msg.get("type") == "state" and msg.get("phase") in SETUP_PHASES:
            valid = self._valid_actions(msg)
            self._send(valid[0] if valid else 0)
            msg = self._next_decision()

        if msg.get("type") != "state":
            raise ProtocolError(f"episode ended during setup: {msg!r}")
        if "reward" not in msg:
            raise ProtocolError(
                "state message has no `reward` field — dist/rl/cli.js is stale; rebuild with: pnpm rl:build"
            )

        self._needs_reset = False
        obs = self._obs_from(msg)
        self.last_info = self._info_from(msg)
        return obs, self.last_info

    def step(self, action):
        if self._needs_reset or self._proc is None:
            raise RuntimeError("episode is done or env not started — call reset() first")

        self._send(int(action))
        msg = self._next_decision()
        mtype = msg.get("type")

        terminated = mtype == "game_over"
        truncated = mtype in ("done", "error")
        reward = float(msg.get("reward", 0.0))

        if terminated or truncated:
            self._needs_reset = True
            has_obs = "obsB64" in msg or msg.get("gameState")
            obs = self._obs_from(msg) if has_obs else np.zeros(OBSERVATION_DIM, np.float32)
            info = self._info_from(msg)
            self.last_info = info
            if mtype == "error":
                info["protocol_error"] = msg.get("message")
            if self._respawn or mtype == "error":
                # error/timeout episodes get a fresh process; otherwise the CLI
                # is now waiting for {"cmd":"reset"} and the process is reused
                self._reap()
            return obs, reward, terminated, truncated, info

        if mtype != "state":
            raise ProtocolError(f"unexpected message type {mtype!r}")
        self.last_info = self._info_from(msg)
        return self._obs_from(msg), reward, False, False, self.last_info

    def action_masks(self) -> np.ndarray:
        """Valid-action mask for the current state (sb3-contrib MaskablePPO hook)."""
        return self._mask.copy()

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_raw({"cmd": "quit"})
            except OSError:
                pass
        self._reap()

    # ── internals ──────────────────────────────────────────────────────

    def _send_raw(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _await_ready(self, timeout: float) -> dict:
        """Read to the next `ready`, skipping leftovers from the previous
        episode (its trailing `done`, info/warning noise)."""
        while True:
            msg = self._read_message(timeout)
            if msg is None:
                raise ProtocolError("EOF while waiting for ready")
            if msg.get("type") == "ready":
                return msg
            if msg.get("type") not in ("done", "info", "warning"):
                raise ProtocolError(f"expected ready, got {msg!r}")

    def _spawn(self, game_seed: str) -> None:
        cmd = [
            self._node_bin,
            str(self._cli_path),
            "--interactive",
            f"--seed={game_seed}",
            f"--waves={self._waves}",
        ]
        if self._lean:
            cmd.append("--lean")
        if self._fog_of_war:
            cmd.append("--fog-of-war")
        if self._reward_config:
            cmd.append(f"--reward-config={json.dumps(self._reward_config)}")
        for key, value in self._overrides.items():
            cmd.append(f"--override={key}={json.dumps(value)}")
        if self._starters:
            cmd.append(f"--starters={self._starters}")
        # stderr must be discarded or drained: an unread PIPE deadlocks node at 64KB
        if self._stderr_log:
            # "w": fresh log per process generation — append mode grows
            # unbounded across respawn_every recycles on long runs
            self._stderr_fh = open(self._stderr_log, "w")
        else:
            self._stderr_fh = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            text=True,
            bufsize=1,
        )
        self._reader = _LineReader(self._proc.stdout)

    def _check_versions(self, ready: dict) -> None:
        obs_dim = ready.get("obsDim")
        action_dim = ready.get("actionDim")
        proto = ready.get("protocolVersion")
        if obs_dim != OBSERVATION_DIM or action_dim != ACTION_SPACE_SIZE:
            raise ProtocolError(
                f"layout mismatch: CLI reports obsDim={obs_dim}, actionDim={action_dim}; "
                f"Python encoder expects {OBSERVATION_DIM}/{ACTION_SPACE_SIZE}. "
                "dist/rl/cli.js and src/rl/observation.py are out of sync — rebuild: pnpm rl:build"
            )
        if proto != PROTOCOL_VERSION:
            raise ProtocolError(
                f"protocol version mismatch: CLI={proto}, wrapper expects {PROTOCOL_VERSION} — rebuild: pnpm rl:build"
            )

    def _read_message(self, timeout: float) -> dict | None:
        """Next JSON message, skipping non-JSON noise lines. None on EOF."""
        assert self._reader is not None
        while True:
            line = self._reader.next(timeout)
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def _next_decision(self) -> dict:
        """Read to the next state / game_over / done / error message."""
        while True:
            msg = self._read_message(self._step_timeout)
            if msg is None:
                return {"type": "error", "message": "EOF from CLI subprocess"}
            if msg.get("type") in ("state", "game_over", "done", "error"):
                return msg
            # info / warning are consumed; a warning means a masked action was
            # rejected — should never happen when the mask is respected

    def _send(self, action: int) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"action": action}) + "\n")
        self._proc.stdin.flush()

    @staticmethod
    def _valid_actions(msg: dict) -> list[int]:
        return [a["index"] for a in msg.get("actions", [])]

    def _obs_from(self, msg: dict) -> np.ndarray:
        # Fast path: the TS encoder is the wire authority (protocolVersion 3)
        if "obsB64" in msg:
            obs = np.frombuffer(base64.b64decode(msg["obsB64"]), dtype="<f4").copy()
            if obs.shape != (OBSERVATION_DIM,):
                raise ProtocolError(f"obsB64 decoded to shape {obs.shape}, expected ({OBSERVATION_DIM},)")
            mask_raw = msg.get("mask")
            if mask_raw is not None:
                self._mask = np.array(mask_raw, dtype=bool)
            else:
                self._mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            return obs
        # Fallback: encode locally from the full gameState (lean=False / old CLI)
        state = parse_game_state(msg.get("gameState") or {})
        self._mask = extract_action_mask(state)
        return encode_observation(state)

    def _info_from(self, msg: dict) -> dict:
        gs = msg.get("gameState") or {}
        battle = gs.get("battle") or {}
        wave = msg.get("wave") or battle.get("wave_index") or 0
        # At game_over the captured battle is the NEXT (post-reset) one with
        # wave 0 — report the last wave actually reached instead.
        if wave > 0:
            self._last_wave = wave
        info = {
            "action_mask": self._mask.copy(),
            "phase": msg.get("phase", "game_over" if msg.get("type") == "game_over" else "?"),
            "step": msg.get("step"),
            "wave": wave if wave > 0 else self._last_wave,
            # Raw gameState (populated only when lean=False) so heuristic policies
            # like max-damage can read move power/category headless, same as rendered.
            "game_state": gs,
        }
        if msg.get("type") == "game_over":
            info["victory"] = bool(msg.get("victory"))
        return info

    def _reap(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            for pipe in (self._proc.stdin, self._proc.stdout):
                try:
                    if pipe:
                        pipe.close()
                except Exception:
                    pass
        if self._stderr_fh is not None and self._stderr_fh is not subprocess.DEVNULL:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
        self._proc = None
        self._reader = None
        self._stderr_fh = None
