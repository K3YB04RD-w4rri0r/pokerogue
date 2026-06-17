#!/usr/bin/env python3
"""Run a policy against the PokeRogue RL env in EITHER mode — same weights, same encoder.

A policy is just ``f(obs, mask) -> action``. The observation and action mask come
from the SAME ``observation.py`` encoder (``parse_game_state`` -> ``encode_observation`` /
``extract_action_mask``) whether the game runs headless or in the browser, so a
checkpoint trained headless runs unchanged when rendered — only the transport differs:

  * headless  -> the Gym env (``rl.pokerogue_env``), a ``cli.js --interactive`` subprocess
                 talking JSON lines over stdio. Fast; for training/eval.
  * rendered  -> a thin WebSocket client to the browser bridge (``ws://host/ws/rl``).
                 For watching a policy play in real time.

Examples
--------
Watch a random bot play a legendary team in the browser. Start the rendered server
first in another terminal::

    npx vite --config vite.interactive.config.ts

then::

    python3 tools/run_policy.py --rendered \
        --starters MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS --delay 0.8

The same policy, headless (no browser, as fast as it runs)::

    python3 tools/run_policy.py --waves 10

A trained agent, watchable — identical command, just point at a checkpoint::

    python3 tools/run_policy.py --rendered --model runs/ppo.zip
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
    encode_observation,
    extract_action_mask,
    parse_game_state,
)


def make_policy(args):
    """Return ``policy(obs, mask) -> int``: a MaskablePPO checkpoint, or a random valid action."""
    if args.model:
        from sb3_contrib import MaskablePPO  # imported lazily; only needed for --model

        model = MaskablePPO.load(args.model)
        print(f"Loaded MaskablePPO checkpoint: {args.model}")

        def policy(obs, mask):
            action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
            return int(action)

        return policy

    # Random policy over the VALID actions (respects the mask, like training would).
    seed = abs(hash(args.seed)) % (2**32) if args.seed else None
    rng = np.random.default_rng(seed)

    def policy(_obs, mask):
        valid = np.flatnonzero(mask)
        return int(rng.choice(valid)) if valid.size else 0

    return policy


def decide(game_state: dict, policy):
    """Parse a raw gameState dict, encode obs+mask, and pick an action. Returns (action, n_valid)."""
    state = parse_game_state(game_state or {})
    obs = encode_observation(state)
    mask = extract_action_mask(state)
    return policy(obs, mask), int(mask.sum())


def run_rendered(args, policy) -> None:
    """Drive the browser bridge over a WebSocket and watch the policy play."""
    try:
        import websocket  # websocket-client
    except ImportError:
        sys.exit("error: --rendered needs websocket-client (pip install websocket-client)")
    import webbrowser

    query = "".join(
        f"&{k}={v}" for k, v in (("seed", args.seed), ("starters", args.starters)) if v
    )
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
                action, n_valid = decide(msg.get("gameState", {}), policy)
                print(f"step {msg.get('step'):>3} | {msg.get('phase'):<16} | {n_valid:>2} valid -> action {action}")
                ws.send(json.dumps({"action": action}))
                time.sleep(args.delay)
            elif mtype == "game_over":
                print(f"\n=== GAME OVER: {'VICTORY' if msg.get('victory') else 'defeat'} (step {msg.get('step')}) ===")
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

    env = PokeRogueEnv(waves=args.waves, seed=args.seed, starters=args.starters)
    obs, info = env.reset()
    done = False
    step = 0
    try:
        while not done:
            mask = env.action_masks()
            action = policy(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            print(f"step {step:>3} | wave {info.get('wave', '?')} | action {action:>2} | reward {reward:+.2f}")
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
        description="Run a policy (random or a trained MaskablePPO) headless or rendered.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rendered", action="store_true", help="drive the browser via WebSocket (watch in real time)")
    ap.add_argument("--model", default=None, help="path to a MaskablePPO .zip checkpoint (default: random policy)")
    ap.add_argument("--policy", default="random", choices=["random"], help="policy when --model is not given")
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
