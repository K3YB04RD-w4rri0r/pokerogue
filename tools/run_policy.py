#!/usr/bin/env python3
"""Run a policy against the PokeRogue RL env in EITHER mode — same weights, same encoder.

A policy is ``f(obs, mask, ctx) -> action``. The observation and action mask come
from the SAME ``observation.py`` encoder (``parse_game_state`` -> ``encode_observation`` /
``extract_action_mask``) whether the game runs headless or in the browser, so a
checkpoint trained headless runs unchanged when rendered — only the transport differs:

  * headless  -> the Gym env (``rl.pokerogue_env``), a ``cli.js --interactive`` subprocess
                 talking JSON lines over stdio. Fast; for training/eval.
  * rendered  -> a thin WebSocket client to the browser bridge (``ws://host/ws/rl``).
                 For watching a policy play in real time.

Policies (``--policy``):
  * ``random``     — uniformly-random LEGAL action (a baseline / fuzzer).
  * ``maxdamage``  — in battle, the highest base-power damaging move at the active
                     Pokemon; first legal action otherwise. Plays competently and
                     never wanders into menus.
  * ``--model <ckpt.zip>`` — a trained sb3-contrib MaskablePPO checkpoint.

Examples
--------
Watch a max-damage bot play a legendary team in the browser (start the rendered
server first: ``npx vite --config vite.interactive.config.ts``)::

    python3 tools/run_policy.py --rendered --policy maxdamage \
        --starters MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS --delay 0.8

The same policy headless (no browser)::

    python3 tools/run_policy.py --policy maxdamage --waves 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Match the import convention used by examples/rl/* and the Gym env: put src/ on
# the path and import the rl package. observation.py is the single source of truth
# for the obs/mask encoding, shared by headless training and this runner.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from rl.observation import (  # noqa: E402
    MAX_MOVES,
    encode_observation,
    extract_action_mask,
    parse_game_state,
)

# Move category 2 == STATUS (0 physical, 1 special) — see MoveCategory.
_STATUS_CATEGORY = 2
# ACTION_SKIP in the action space (spaces.ts) — "skip the shop" / "decline" / "no".
_ACTION_SKIP = 39
# Phases a battle bot should decline/skip rather than engage: skipping the shop
# avoids a deterministic modifier<->modifier_target loop (pick reward -> needs
# target -> skip target -> pick reward ...), and it shouldn't voluntarily switch
# out the lead at battle start (check_switch).
_SKIP_PHASES = frozenset({"check_switch", "modifier", "modifier_target"})


def maxdamage_action(mask: np.ndarray, ctx: dict | None) -> int:
    """Pick the highest base-power damaging move vs an enemy; first legal action otherwise.

    Inferred purely from the mask + gameState, so it needs no phase-specific wiring
    beyond ``ctx["phase"] == "command"``. Enemy-targeting fight actions occupy
    indices ``0 .. 2*MAX_MOVES-1`` (slot 0 then slot 1); the move slot is ``a % MAX_MOVES``.
    """
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0
    ctx = ctx or {}
    phase = ctx.get("phase")
    if phase == "command":
        game_state = ctx.get("game_state") or {}
        # In doubles the acting pokemon may be slot 1 — its moves live under
        # player_1 and the mask's fight actions refer to ITS moveset.
        field_index = (game_state.get("phase") or {}).get("command_field_index") or 0
        slot_key = "player_1" if field_index == 1 else "player_0"
        moves = (game_state.get(slot_key) or {}).get("moves") or []
        best_action, best_power = None, 0
        for a in range(min(2 * MAX_MOVES, len(mask))):
            if not mask[a]:
                continue
            slot = a % MAX_MOVES
            if slot >= len(moves):
                continue
            move = moves[slot] or {}
            power = move.get("power") or 0
            if move.get("category", 0) != _STATUS_CATEGORY and power > best_power:
                best_power, best_action = power, a
        if best_action is not None:
            return best_action
    # Optional-engagement phases: decline/skip rather than engage (see _SKIP_PHASES).
    if phase in _SKIP_PHASES and _ACTION_SKIP < len(mask) and mask[_ACTION_SKIP]:
        return _ACTION_SKIP
    # Otherwise (forced switch, target select, ...): first legal action.
    return int(valid[0])


def make_policy(args):
    """Return ``policy(obs, mask, ctx) -> int``."""
    if args.model:
        from sb3_contrib import MaskablePPO  # imported lazily; only needed for --model

        model = MaskablePPO.load(args.model)
        print(f"Loaded MaskablePPO checkpoint: {args.model}")

        def policy(obs, mask, _ctx=None):
            action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
            return int(action)

        return policy

    if args.policy == "maxdamage":
        return lambda _obs, mask, ctx=None: maxdamage_action(mask, ctx)

    # random: uniform over the LEGAL actions (a baseline / fuzzer).
    seed = abs(hash(args.seed)) % (2**32) if args.seed else None
    rng = np.random.default_rng(seed)

    def policy(_obs, mask, _ctx=None):
        valid = np.flatnonzero(mask)
        return int(rng.choice(valid)) if valid.size else 0

    return policy


def run_rendered(args, policy) -> None:
    """Drive the browser bridge over a WebSocket and watch the policy play."""
    try:
        import websocket  # websocket-client
    except ImportError:
        sys.exit("error: --rendered needs websocket-client (pip install websocket-client)")
    import webbrowser

    query = "".join(f"&{k}={v}" for k, v in (("seed", args.seed), ("starters", args.starters)) if v)
    url = f"http://localhost:{args.port}/?rl=true{query}"
    ws_url = f"ws://localhost:{args.port}/ws/rl"

    print(f"Opening {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print("(could not open a browser here — open the URL above manually)")
    print(f"Connecting to {ws_url} ...")
    ws = websocket.create_connection(ws_url, timeout=60)
    ws.settimeout(None)  # the browser may take a while to boot/animate
    print("Connected. Waiting for the game to boot...\n")

    try:
        while True:
            raw = ws.recv()
            if not raw:
                break
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "ready":
                print("Game ready -> sending start.\n")
                ws.send(json.dumps({"type": "start"}))
            elif mtype == "state":
                game_state = msg.get("gameState", {})
                state = parse_game_state(game_state)
                obs = encode_observation(state)
                mask = extract_action_mask(state)
                ctx = {"phase": msg.get("phase"), "game_state": game_state}
                action = policy(obs, mask, ctx)
                label = next((a.get("label", "") for a in msg.get("actions", []) if a.get("index") == action), "")
                reward = msg.get("reward")
                rstr = f" | reward {reward:+.2f}" if isinstance(reward, (int, float)) else ""
                wave = msg.get("wave")
                wstr = f" | wave {wave:>3}" if isinstance(wave, int) and wave > 0 else ""
                print(f"step {msg.get('step'):>3}{wstr} | {str(msg.get('phase')):<14} | action {action:>2}{rstr}  {label}")
                ws.send(json.dumps({"action": int(action)}))
                time.sleep(args.delay)
            elif mtype == "game_over":
                reward = msg.get("reward")
                rstr = f", terminal reward {reward:+.2f}" if isinstance(reward, (int, float)) else ""
                print(f"\n=== GAME OVER: {'VICTORY' if msg.get('victory') else 'defeat'} (step {msg.get('step')}{rstr}) ===")
            elif mtype == "done":
                print(f"\nEpisode complete ({msg.get('steps')} decisions).")
                break
            elif mtype in ("info", "warning"):
                print(f"  [{mtype}] {msg.get('message', '')}")
            elif mtype == "error":
                print(f"  [bridge error] {msg.get('message', '')}")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except websocket.WebSocketConnectionClosedException:
        print("\nWebSocket closed.")
    finally:
        ws.close()


def run_headless(args, policy) -> None:
    """Run the same policy against the headless Gym env (stdio subprocess)."""
    from rl.pokerogue_env import PokeRogueEnv

    # max-damage needs move power/category from the gameState -> lean=False.
    needs_state = args.policy == "maxdamage"
    env = PokeRogueEnv(waves=args.waves, seed=args.seed, starters=args.starters, lean=not needs_state)
    obs, info = env.reset()
    done = False
    step = 0
    try:
        while not done:
            mask = env.action_masks()
            decision_phase = info.get("phase", "?")
            ctx = {"phase": decision_phase, "game_state": info.get("game_state")}
            action = policy(obs, mask, ctx)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            print(f"step {step:>3} | wave {info.get('wave', '?')} | {str(decision_phase):<14} | action {action:>2} | reward {reward:+.2f}")
            done = terminated or truncated
            if args.delay:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
    print(f"\nEpisode finished after {step} steps.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a policy (random, maxdamage, or a trained MaskablePPO) headless or rendered.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rendered", action="store_true", help="drive the browser via WebSocket (watch in real time)")
    ap.add_argument("--model", default=None, help="path to a MaskablePPO .zip checkpoint (overrides --policy)")
    ap.add_argument("--policy", default="random", choices=["random", "maxdamage"], help="policy when --model is not given")
    ap.add_argument("--starters", default=None, help="comma-separated SpeciesId names, e.g. MEWTWO,LUGIA,RAYQUAZA")
    ap.add_argument("--seed", default=None, help="RNG seed (game + random policy)")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds between actions (rendered: watchability)")
    ap.add_argument("--port", type=int, default=8000, help="rendered dev-server port")
    ap.add_argument("--waves", type=int, default=20, help="max waves (headless)")
    ap.add_argument("--deterministic", action="store_true", help="deterministic predict for --model")
    args = ap.parse_args()

    policy = make_policy(args)
    if args.rendered:
        run_rendered(args, policy)
    else:
        run_headless(args, policy)


if __name__ == "__main__":
    main()
