"""
Shared utilities for the RL verification harness.

All verify tools insert <repo>/src on sys.path and import the `rl` package
(NEVER insert src/rl directly — that double-loads `enums` as a second module).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
# Overridable so the verify suite can target an out-of-tree build,
# mirroring pokerogue_env._resolve_default_cli().
CLI_PATH = Path(os.environ.get("POKEROGUE_RL_CLI", REPO_ROOT / "dist" / "rl" / "cli.js"))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def require_cli() -> None:
    """Fail fast with a build hint if the headless bundle is missing."""
    if not CLI_PATH.exists():
        sys.exit(
            f"error: {CLI_PATH} not found.\n"
            "Build it first:  pnpm rl:build  (vite build --config vite.headless.config.ts)"
        )


def decode_obs_b64(s: str) -> np.ndarray:
    """Decode the cli.ts base64 Float32Array dump (little-endian f4)."""
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def canonical_state_hash(game_state: dict) -> str:
    """Order-independent hash of a gameState dict.

    Ignores the wall-clock timestamp and battle.seed: the seed is a debug
    string that is never encoded into the observation, and at game_over it
    belongs to the NEXT (post-reset) battle, which is generated from an
    unseeded RNG.
    """
    clean = dict(game_state)
    clean.pop("timestamp", None)
    battle = clean.get("battle")
    if isinstance(battle, dict) and "seed" in battle:
        battle = dict(battle)
        battle.pop("seed")
        clean["battle"] = battle
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def diff_json(a, b, path: str = "$", out: list[str] | None = None, limit: int = 50) -> list[str]:
    """Recursive structural diff; returns list of '<path>: a=... b=...' strings."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k == "timestamp":
                continue
            if k not in a:
                out.append(f"{path}.{k}: missing in a")
            elif k not in b:
                out.append(f"{path}.{k}: missing in b")
            else:
                diff_json(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            diff_json(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


# Mirrors getActionName in src/rl/runner.ts
def action_name(action: int) -> str:
    if 0 <= action < 4:
        return f"FIGHT_ENEMY_MOVE_{action}"
    if 4 <= action < 8:
        return f"FIGHT_ENEMY2_MOVE_{action - 4}"
    if 8 <= action < 12:
        return f"FIGHT_ALLY_MOVE_{action - 8}"
    if 12 <= action < 17:
        return f"SWITCH_SLOT_{action - 12 + 1}"
    if 17 <= action < 22:
        return f"BALL_TYPE_{action - 17}"
    if action == 22:
        return "RUN"
    if 23 <= action < 27:
        return f"TERA_ENEMY_MOVE_{action - 23}"
    if 27 <= action < 31:
        return f"TERA_ENEMY2_MOVE_{action - 27}"
    if 31 <= action < 35:
        return f"TERA_ALLY_MOVE_{action - 31}"
    if 35 <= action < 38:
        return f"SELECT_REWARD_{action - 35}"
    if action == 38:
        return "REROLL"
    if action == 39:
        return "SKIP"
    if 40 <= action < 52:
        return f"BUY_SHOP_{action - 40}"
    if 52 <= action < 58:
        return f"PARTY_TARGET_{action - 52}"
    return f"UNKNOWN_{action}"


ACTION_NAMES = [action_name(i) for i in range(58)]

ACTION_RUN = 22


class HangTimeout(Exception):
    """Raised when the CLI produces no output line within the timeout."""


class LineReader:
    """Background thread draining a pipe into a queue, with per-read timeouts."""

    def __init__(self, stream):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self._thread.start()

    def _drain(self, stream) -> None:
        for line in stream:
            self._queue.put(line)
        self._queue.put(None)  # EOF marker

    def next(self, timeout: float) -> str | None:
        """Next line, or None on EOF. Raises HangTimeout after `timeout` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise HangTimeout(f"no output within {timeout}s") from None


def spawn_cli(
    seed: str,
    waves: int,
    interactive: bool = True,
    dump_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    extra_args: list[str] | None = None,
):
    """Spawn the headless CLI. Returns (Popen, LineReader, stderr_file_handle).

    Set RL_VERIFY_CLI_LOG=1 to pass --log (verbose router logs land in stderr_path).
    extra_args: appended verbatim (e.g. --override=STARTING_WAVE_OVERRIDE=20).
    """
    require_cli()
    cmd = ["node", str(CLI_PATH), f"--seed={seed}", f"--waves={waves}"]
    if interactive:
        cmd.append("--interactive")
    if dump_path:
        cmd.append(f"--dump-obs={dump_path}")
    if os.environ.get("RL_VERIFY_CLI_LOG"):
        cmd.append("--log")
    if extra_args:
        cmd.extend(extra_args)
    # stderr must be drained or discarded — an unread PIPE deadlocks node at 64KB
    stderr_fh = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
    env = None
    if os.environ.get("RL_VERIFY_EXPOSE_GC"):
        env = dict(os.environ)
        env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "") + " --expose-gc").strip()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_fh,
        text=True,
        bufsize=1,
        env=env,
    )
    return proc, LineReader(proc.stdout), stderr_fh


def send_action(proc, action: int) -> None:
    proc.stdin.write(json.dumps({"action": int(action)}) + "\n")
    proc.stdin.flush()


def read_json(reader: LineReader, timeout: float) -> dict | None:
    """Next JSON message, skipping non-JSON noise lines. None on EOF."""
    while True:
        line = reader.next(timeout)
        if line is None:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def kill_proc(proc, stderr_fh=None) -> None:
    """terminate -> wait(5) -> kill, then close the stderr handle."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if stderr_fh is not None and stderr_fh is not subprocess.DEVNULL:
        try:
            stderr_fh.close()
        except Exception:
            pass
    # Close our ends of the pipes so respawn loops don't leak 2 fds/generation.
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
