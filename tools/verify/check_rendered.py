#!/usr/bin/env python3
"""
Rendered-mode E2E gate: drives the REAL browser stack (vite dev server +
WebSocket relay + browser bridge) with headless Chromium and asserts the
rendered-only behaviors that headless mocks can't exercise. This harness
root-caused the 2026-07-07 rendered bugs (catch-flow timeout, shop
tween-chain crash, ghost sprites, evolution hang, seed instability).

Scenarios (--scenario all|determinism|evolution|shop):
  determinism  two same-seed RandomPolicy runs must produce IDENTICAL
               (step, phase, wave, action, reward) traces — catches both
               unstable action seeding and game-side nondeterminism.
  evolution    Caterpie lv6 wins a battle and must evolve (Metapod in
               party) without a decision timeout — exercises the bridge's
               cinematic fast-forward.
  shop         buy (two-step target flow) + reroll + skip across waves with
               ZERO page errors — guards the ModifierOption tween-chain
               crash class and shop rendering.

Requirements:
  pip install playwright websocket-client && playwright install chromium
  Node >= 20 on PATH (the tool starts its own vite dev server).

The tool binds its OWN vite server (--port, default 8123, strict) so it
never collides with a developer's session on 8000. On software-rendered
machines the game runs at a few fps — each scenario can take several
minutes; this is a deep gate, not part of the quick suite. Optional hook:
  RL_VERIFY_RENDERED=1 bash scripts/rl-verify.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from common import REPO_ROOT  # noqa: F401  (sys.path side effect adds src/)

from rl.policy import FirstLegalPolicy, RandomPolicy  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)


def start_vite(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["npx", "vite", "--config", "vite.interactive.config.ts", "--port", str(port), "--strictPort"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 90
    ready = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if "Local:" in line:
            ready = True
            break
        if time.time() > deadline or proc.poll() is not None:
            break
    if not ready:
        proc.terminate()
        sys.exit(f"vite dev server failed to start on :{port} (is the port free? is Node >= 20 on PATH?)")
    # Drain further output so the pipe can't fill and block vite
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    print(f"vite dev server ready on :{port}", flush=True)
    return proc


def run_session(pw, port: int, query: str, policy, max_steps: int, tag: str, timeout_s: int = 240):
    """Open the game with the bridge, drive it via the WS relay, return
    (trace, last_game_state, page_errors)."""
    trace: list = []
    last_gs: dict = {}
    done = {"v": False}

    def drive() -> None:
        try:
            ws = None
            deadline = time.time() + 120
            while ws is None and time.time() < deadline:
                try:
                    import websocket

                    ws = websocket.create_connection(f"ws://localhost:{port}/ws/rl?role=python", timeout=5)
                except Exception:
                    time.sleep(1)
            if ws is None:
                trace.append(("exc", "could not reach WS relay"))
                return
            ws.settimeout(1800)
            while True:
                msg = json.loads(ws.recv())
                t = msg.get("type")
                if t == "ready":
                    ws.send(json.dumps({"type": "start"}))
                elif t == "state":
                    step, phase = msg.get("step"), msg.get("phase")
                    acts = sorted(a["index"] for a in msg.get("actions", []))
                    mask = np.zeros(58, dtype=bool)
                    mask[acts] = True
                    a = policy.act(None, mask, {"phase": phase})
                    trace.append((step, phase, msg.get("wave"), a, round(msg.get("reward") or 0, 4)))
                    last_gs.clear()
                    last_gs.update(msg.get("gameState") or {})
                    print(
                        f"  [{tag}] step {step:>3} | wave {msg.get('wave')} | {str(phase):<14} | -> {a}",
                        flush=True,
                    )
                    ws.send(json.dumps({"action": a}))
                    if step is not None and step >= max_steps:
                        return
                elif t in ("error", "done", "game_over"):
                    trace.append(("terminal", t, msg.get("message", "")))
                    return
        except Exception as e:  # noqa: BLE001
            trace.append(("exc", str(e)))
        finally:
            done["v"] = True

    browser = pw.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
    page = browser.new_page(viewport={"width": 640, "height": 400})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://localhost:{port}/?rl=true&delay=0&timeout={timeout_s}{query}")
    th = threading.Thread(target=drive, daemon=True)
    th.start()
    while not done["v"]:
        page.wait_for_timeout(1500)
    th.join(timeout=30)
    browser.close()
    return trace, dict(last_gs), errors


def scenario_determinism(pw, port: int) -> None:
    print("determinism) two same-seed random runs must replay identically", flush=True)
    q = "&seed=verify-rendered-det"
    trace_a, _, err_a = run_session(pw, port, q, RandomPolicy("verify-rendered-det"), 8, "A")
    trace_b, _, err_b = run_session(pw, port, q, RandomPolicy("verify-rendered-det"), 8, "B")
    check("run A clean", not err_a, str(err_a[:1]))
    check("run B clean", not err_b, str(err_b[:1]))
    check("traces identical", trace_a == trace_b, f"lenA={len(trace_a)} lenB={len(trace_b)}")
    if trace_a != trace_b:
        for i, (x, y) in enumerate(zip(trace_a, trace_b)):
            if x != y:
                print(f"  first divergence at index {i}: A={x} B={y}", flush=True)
                break


def scenario_evolution(pw, port: int) -> None:
    print("evolution) Caterpie lv6 must evolve without a decision timeout", flush=True)
    q = "&seed=verify-rendered-evo&starters=CATERPIE&override=STARTING_LEVEL_OVERRIDE%3D6"
    trace, gs, errors = run_session(pw, port, q, FirstLegalPolicy(), 10, "C")
    timed_out = any(x[0] == "terminal" and x[1] == "error" for x in trace)
    party = [((gs.get(f"player_{i}") or {}).get("species_name") or "") for i in range(6)]
    check("no decision timeout", not timed_out, str([x for x in trace if x[0] == "terminal"]))
    check("evolved (Metapod in party)", "Metapod" in party, str(party))
    check("no page errors", not errors, str(errors[:1]))


class ShopPolicy:
    """Buy up to two items (exercising the target flow), reroll once, skip."""

    def __init__(self) -> None:
        self.bought = 0
        self.rerolled = False

    def act(self, obs, mask, info) -> int:
        phase = info.get("phase")
        legal = np.flatnonzero(mask)
        if phase == "modifier":
            buy = next((a for a in range(40, 52) if mask[a]), None)
            if buy is not None and self.bought < 2:
                self.bought += 1
                return buy
            if mask[38] and not self.rerolled:
                self.rerolled = True
                return 38
            return 39 if mask[39] else int(legal[0])
        if phase == "modifier_target":
            return next((a for a in range(52, 58) if mask[a]), 39)
        fight = next((a for a in range(0, 4) if mask[a]), None)
        if fight is not None:
            return fight
        return 39 if mask[39] else int(legal[0])


def scenario_shop(pw, port: int) -> None:
    print("shop) buy (target flow) + reroll + skip with zero page errors", flush=True)
    trace, _, errors = run_session(pw, port, "&seed=verify-rendered-shop", ShopPolicy(), 16, "S")
    phases = [x[1] for x in trace if isinstance(x[0], int)]
    check("reached shop decisions", "modifier" in phases, str(sorted(set(phases))))
    check("no page errors (tween-chain crash guard)", not errors, str(errors[:1]))
    timed_out = any(x[0] == "terminal" and x[1] == "error" for x in trace)
    check("no decision timeout", not timed_out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8123, help="dedicated vite port (strict; default 8123)")
    ap.add_argument("--scenario", default="all", choices=["all", "determinism", "evolution", "shop"])
    args = ap.parse_args()

    os.environ.setdefault("PLAYWRIGHT_CHROMIUM_USE_HEADLESS_NEW", "1")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("check_rendered needs playwright (pip install playwright && playwright install chromium)")

    vite = start_vite(args.port)
    try:
        with sync_playwright() as pw:
            if args.scenario in ("all", "determinism"):
                scenario_determinism(pw, args.port)
            if args.scenario in ("all", "evolution"):
                scenario_evolution(pw, args.port)
            if args.scenario in ("all", "shop"):
                scenario_shop(pw, args.port)
    finally:
        vite.terminate()
        try:
            vite.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vite.kill()

    print(f"\nRENDERED E2E: {'FAIL (' + ', '.join(FAILS) + ')' if FAILS else 'OK'}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
