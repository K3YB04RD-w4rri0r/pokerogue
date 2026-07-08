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
import base64
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
    ACTION_SPACE_SIZE,
    encode_observation,
    extract_action_mask,
    parse_game_state,
)
from rl.policy import Sb3Policy, make_builtin_policy  # noqa: E402
from rl.pokerogue_env import PROTOCOL_VERSION  # noqa: E402
from rl.run_config import RunConfig, load_run_config  # noqa: E402


def make_policy(args, cfg: RunConfig):
    """Resolve the policy: --model checkpoint, else a built-in from rl.policy."""
    if args.model:
        policy = Sb3Policy(args.model, deterministic=args.deterministic)
        print(f"Loaded MaskablePPO checkpoint: {args.model}")
        return policy
    return make_builtin_policy(args.policy, seed=cfg.seed)


def run_rendered(args, policy, cfg: RunConfig) -> None:
    """Drive the browser bridge over a WebSocket and watch the policy play."""
    try:
        import websocket  # websocket-client
    except ImportError:
        sys.exit("error: --rendered needs websocket-client (pip install websocket-client)")
    import webbrowser

    # The whole run config (seed/starters/waves/overrides/reward) travels as
    # URL params — the bridge accepts the same config surface as the CLI.
    url = f"http://localhost:{args.port}/?rl=true{cfg.to_url_query()}"
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

    started = False
    try:
        while True:
            raw = ws.recv()
            if not raw:
                break
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "ready":
                # Protocol guard: a stale browser bundle (old tab, cached
                # build) mis-encodes silently — reject it loudly instead.
                proto = msg.get("protocolVersion")
                if proto is not None and proto != PROTOCOL_VERSION:
                    raise SystemExit(
                        f"browser bundle speaks protocol {proto}, this client needs {PROTOCOL_VERSION} — "
                        "hard-reload the browser tab (Ctrl+Shift+R) and rerun"
                    )
                if proto is None:
                    print("warning: browser bridge sent no protocolVersion (pre-audit bundle?) — "
                          "hard-reload the tab if observations look wrong")
                if started:
                    # A second `ready` mid-session means the PAGE RELOADED (vite
                    # hot-reload after a source edit, manual reload, second tab)
                    # and a fresh game is starting — step/reward continuity with
                    # the previous episode is broken.
                    print("\n!!! browser session restarted (page reload?) — starting a FRESH episode; "
                          "previous step/reward continuity is void\n")
                print("Game ready -> sending start.\n")
                started = True
                ws.send(json.dumps({"type": "start"}))
            elif mtype == "state":
                game_state = msg.get("gameState", {})
                # Prefer the wire encoding: it is the TS authority and already
                # reflects the bridge's observability mode (&fog=1). Local
                # re-encode is the fallback for old bundles — full-info only.
                if msg.get("obsB64"):
                    obs = np.frombuffer(base64.b64decode(msg["obsB64"]), dtype="<f4").copy()
                    mask = np.array(msg.get("mask") or [False] * ACTION_SPACE_SIZE, dtype=bool)
                else:
                    state = parse_game_state(game_state)
                    obs = encode_observation(state)
                    mask = extract_action_mask(state)
                ctx = {"phase": msg.get("phase"), "game_state": game_state}
                action = policy.act(obs, mask, ctx)
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


def run_headless(args, policy, cfg: RunConfig) -> None:
    """Run the same policy against the headless Gym env (stdio subprocess)."""
    from rl.pokerogue_env import PokeRogueEnv

    env_kwargs = cfg.to_env_kwargs()
    # max-damage needs move power/category from the gameState -> lean=False.
    if args.policy == "maxdamage" and not args.model:
        env_kwargs["lean"] = False
    env = PokeRogueEnv(**env_kwargs)
    obs, info = env.reset()
    done = False
    step = 0
    try:
        while not done:
            mask = env.action_masks()
            decision_phase = info.get("phase", "?")
            ctx = {"phase": decision_phase, "game_state": info.get("game_state"), "wave": info.get("wave")}
            action = policy.act(obs, mask, ctx)
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
    ap.add_argument("--config", default=None, help="run-config YAML/JSON (see src/rl/run_config.py); CLI flags override it")
    ap.add_argument("--rendered", action="store_true", help="drive the browser via WebSocket (watch in real time)")
    ap.add_argument("--model", default=None, help="path to a MaskablePPO .zip checkpoint (overrides --policy)")
    ap.add_argument(
        "--policy",
        default="random",
        choices=["random", "maxdamage", "firstlegal"],
        help="policy when --model is not given",
    )
    ap.add_argument("--starters", default=None, help="comma-separated SpeciesId names, e.g. MEWTWO,LUGIA,RAYQUAZA")
    ap.add_argument("--seed", default=None, help="RNG seed (game + random policy)")
    ap.add_argument("--delay", type=float, default=0.8, help="seconds between actions (rendered: watchability)")
    ap.add_argument("--port", type=int, default=8000, help="rendered dev-server port")
    ap.add_argument("--waves", type=int, default=None, help="max waves (headless default: 20)")
    ap.add_argument("--deterministic", action="store_true", help="deterministic predict for --model")
    args = ap.parse_args()

    # Config file first, individual CLI flags override its values.
    cfg = load_run_config(args.config) if args.config else RunConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.starters is not None:
        cfg.starters = args.starters
    if args.waves is not None:
        cfg.waves = args.waves
    if cfg.waves is None:
        cfg.waves = 20

    policy = make_policy(args, cfg)
    if args.rendered:
        run_rendered(args, policy, cfg)
    else:
        run_headless(args, policy, cfg)


if __name__ == "__main__":
    main()
