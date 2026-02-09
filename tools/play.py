#!/usr/bin/env python3
"""
Interactive PokeRogue player.

Supports two modes:
  Headless:  Spawns the headless RL runner (default).
  Rendered:  Connects to the browser game via WebSocket (--rendered).

Usage:
    python tools/play.py [--seed=abc123] [--waves=10]
    python tools/play.py --rendered [--port=8000]

Requirements:
    Headless mode:
        - Node.js installed
        - Headless build: npx vite build --config vite.headless.config.ts
    Rendered mode:
        - pip install websocket-client
        - Vite dev server: npx vite --config vite.interactive.config.ts
"""

import subprocess
import json
import sys
import os
import argparse


# ─── Colors ──────────────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"


def hp_bar(pct: int, width: int = 20) -> str:
    """Render an HP bar like [████████████░░░░░░░░] 65%"""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct > 50:
        color = C.GREEN
    elif pct > 25:
        color = C.YELLOW
    else:
        color = C.RED
    bar = color + "█" * filled + C.DIM + "░" * empty + C.RESET
    return f"[{bar}] {pct}%"


# ─── Display ─────────────────────────────────────────────────────────

def print_header(step: int, phase: str, game_state: dict):
    """Print the game state header."""
    battle = game_state.get("battle", {})
    wave = battle.get("wave_index", "?")
    turn = battle.get("turn", "?")
    money = battle.get("money", 0)

    print()
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}  Wave {wave} │ Turn {turn} │ Step {step} │ ${money}{C.RESET}")
    print(f"{C.BOLD}  Phase: {C.CYAN}{phase}{C.RESET}")
    print(f"{C.BOLD}{'═' * 60}{C.RESET}")


def print_field(game_state: dict):
    """Print the battlefield: player vs enemy Pokemon."""
    # Extract active enemies from slot dicts
    enemies = []
    for key in ("enemy_0", "enemy_1"):
        p = game_state.get(key, {})
        if p.get("valid") and p.get("is_on_field"):
            enemies.append(p)
    # Extract active players
    players = []
    for key in ("player_0", "player_1"):
        p = game_state.get(key, {})
        if p.get("valid") and p.get("is_on_field"):
            players.append(p)

    if enemies:
        print(f"\n  {C.RED}Enemy:{C.RESET}")
        for e in enemies:
            name = e.get("species_name", "?")
            level = e.get("level", "?")
            hp_pct = round(e.get("hp_ratio", 0) * 100)
            status = e.get("status_effect", 0)
            status_str = f" [{C.YELLOW}status:{status}{C.RESET}]" if status else ""
            print(f"    {name} Lv{level}  {hp_bar(hp_pct)}{status_str}")

    if players:
        print(f"\n  {C.GREEN}Player:{C.RESET}")
        for p in players:
            name = p.get("species_name", "?")
            level = p.get("level", "?")
            hp = p.get("hp", "?")
            max_hp = p.get("max_hp", "?")
            hp_pct = round(p.get("hp_ratio", 0) * 100)
            status = p.get("status_effect", 0)
            status_str = f" [{C.YELLOW}status:{status}{C.RESET}]" if status else ""
            print(f"    {name} Lv{level}  {hp_bar(hp_pct)} ({hp}/{max_hp}){status_str}")


def print_party(game_state: dict):
    """Print the party summary."""
    party = []
    for key in ("player_0", "player_1", "player_2", "player_3", "player_4", "player_5"):
        p = game_state.get(key, {})
        if p.get("valid"):
            party.append(p)
    if not party:
        return

    print(f"\n  {C.BLUE}Party:{C.RESET}")
    for i, p in enumerate(party):
        name = p.get("species_name", "?")
        level = p.get("level", "?")
        hp_pct = round(p.get("hp_ratio", 0) * 100)
        fainted = p.get("is_fainted", False)
        on_field = p.get("is_on_field", False)
        field_tag = f" {C.GREEN}(active){C.RESET}" if on_field else ""
        if fainted:
            print(f"    [{i}] {C.DIM}{name} Lv{level}  FAINTED{C.RESET}")
        else:
            print(f"    [{i}] {name} Lv{level}  {hp_bar(hp_pct, 10)}{field_tag}")


def print_actions(actions: list) -> dict:
    """Print available actions and return index-to-action mapping."""
    print(f"\n  {C.MAGENTA}Available Actions:{C.RESET}")
    choice_map = {}
    for i, act in enumerate(actions):
        idx = act["index"]
        label = act["label"]
        choice_map[i] = idx
        print(f"    {C.BOLD}{i:>3}{C.RESET}) {label}  {C.DIM}[action={idx}]{C.RESET}")
    return choice_map


def prompt_action(choice_map: dict) -> int:
    """Prompt user to select an action. Returns the action index."""
    while True:
        try:
            raw = input(f"\n  {C.CYAN}Select action [0-{len(choice_map)-1}]: {C.RESET}").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print("Quitting...")
                sys.exit(0)
            choice = int(raw)
            if choice in choice_map:
                return choice_map[choice]
            print(f"  {C.RED}Invalid choice. Enter 0-{len(choice_map)-1}{C.RESET}")
        except ValueError:
            print(f"  {C.RED}Enter a number or 'q' to quit{C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting...")
            sys.exit(0)


# ─── Headless mode ────────────────────────────────────────────────────

def run_headless(args):
    """Spawn the headless RL runner as a subprocess and play via stdio."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli_path = os.path.join(project_root, "dist", "rl", "cli.js")

    if not os.path.exists(cli_path):
        print(f"{C.RED}Error: {cli_path} not found.{C.RESET}")
        print(f"Build first: npx vite build --config vite.headless.config.ts")
        sys.exit(1)

    cmd = ["node", cli_path, "--interactive", f"--seed={args.seed}", f"--waves={args.waves}"]

    print(f"{C.BOLD}=== PokeRogue Headless Player ==={C.RESET}")
    print(f"  Seed:  {args.seed}")
    print(f"  Waves: {args.waves}")
    print(f"  Quit:  type 'q' at any prompt")
    print(f"\n  Booting headless game...")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # game noise goes here
        text=True,
        bufsize=1,  # line-buffered
        cwd=project_root,
    )

    try:
        while True:
            # Read a JSON line from the node process
            line = proc.stdout.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON — might be stray console output that slipped through
                continue

            msg_type = msg.get("type")

            if msg_type == "ready":
                boot_time = msg.get("bootTime", "?")
                print(f"  Booted in {boot_time}ms. Let's play!\n")

            elif msg_type == "state":
                step = msg.get("step", 0)
                phase = msg.get("phase", "unknown")
                game_state = msg.get("gameState", {})
                actions = msg.get("actions", [])

                print_header(step, phase, game_state)
                print_field(game_state)
                print_party(game_state)
                choice_map = print_actions(actions)
                action = prompt_action(choice_map)

                # Send action to the node process
                proc.stdin.write(json.dumps({"action": action}) + "\n")
                proc.stdin.flush()

            elif msg_type == "game_over":
                victory = msg.get("victory", False)
                step = msg.get("step", 0)
                game_state = msg.get("gameState", {})
                print_header(step, "GAME OVER", game_state)
                print_field(game_state)
                if victory:
                    print(f"\n  {C.BG_GREEN}{C.BOLD} VICTORY! {C.RESET}")
                else:
                    print(f"\n  {C.BG_RED}{C.BOLD} DEFEATED {C.RESET}")
                print(f"  Steps: {step}")

            elif msg_type == "done":
                steps = msg.get("steps", 0)
                print(f"\n{C.BOLD}Episode complete. {steps} decisions made.{C.RESET}")
                break

            elif msg_type == "info":
                print(f"  {C.CYAN}{msg.get('message', '')}{C.RESET}")

            elif msg_type == "warning":
                print(f"  {C.YELLOW}Warning: {msg.get('message', '')}{C.RESET}")

            elif msg_type == "error":
                print(f"  {C.RED}Error: {msg.get('message', '')}{C.RESET}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Rendered mode ────────────────────────────────────────────────────

def run_rendered(args):
    """Connect to the browser game via WebSocket and play through the TUI."""
    try:
        import websocket
    except ImportError:
        print(f"{C.RED}Error: websocket-client required for --rendered mode{C.RESET}")
        print(f"Install: pip install websocket-client")
        sys.exit(1)

    import webbrowser

    seed_param = f"&seed={args.seed}" if args.seed else ""
    url = f"http://localhost:{args.port}/?rl=true{seed_param}"
    ws_url = f"ws://localhost:{args.port}/ws/rl"

    print(f"{C.BOLD}=== PokeRogue Rendered Player ==={C.RESET}")
    print(f"  Port:  {args.port}")
    print(f"  Seed:  {args.seed or '(random)'}")
    print(f"  URL:   {url}")
    print(f"  Quit:  type 'q' at any prompt")
    print()
    print(f"  Opening browser...")
    webbrowser.open(url)
    print(f"  Connecting to WebSocket at {ws_url}...")

    try:
        ws = websocket.create_connection(ws_url, timeout=30)
    except Exception as e:
        print(f"{C.RED}Failed to connect: {e}{C.RESET}")
        print(f"Make sure the Vite dev server is running:")
        print(f"  npx vite --config vite.interactive.config.ts")
        sys.exit(1)

    # Disable recv timeout — the user needs time to start the game in the browser
    ws.settimeout(None)

    print(f"  {C.GREEN}Connected!{C.RESET}")
    print(f"  {C.YELLOW}Game will auto-start (skip title/gender/starters).{C.RESET}")
    print(f"  {C.YELLOW}Waiting for game to boot and reach first battle...{C.RESET}")
    print()

    try:
        while True:
            raw = ws.recv()
            if not raw:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "ready":
                print(f"  {C.GREEN}Game ready! Sending start signal...{C.RESET}\n")
                ws.send(json.dumps({"type": "start"}))

            elif msg_type == "state":
                step = msg.get("step", 0)
                phase = msg.get("phase", "unknown")
                game_state = msg.get("gameState", {})
                actions = msg.get("actions", [])

                print_header(step, phase, game_state)
                print_field(game_state)
                print_party(game_state)
                choice_map = print_actions(actions)
                action = prompt_action(choice_map)

                ws.send(json.dumps({"action": action}))

            elif msg_type == "game_over":
                victory = msg.get("victory", False)
                step = msg.get("step", 0)
                game_state = msg.get("gameState", {})
                print_header(step, "GAME OVER", game_state)
                print_field(game_state)
                if victory:
                    print(f"\n  {C.BG_GREEN}{C.BOLD} VICTORY! {C.RESET}")
                else:
                    print(f"\n  {C.BG_RED}{C.BOLD} DEFEATED {C.RESET}")

            elif msg_type == "done":
                steps = msg.get("steps", 0)
                print(f"\n{C.BOLD}Episode complete. {steps} decisions made.{C.RESET}")
                break

            elif msg_type == "info":
                print(f"  {C.CYAN}{msg.get('message', '')}{C.RESET}")

            elif msg_type == "warning":
                print(f"  {C.YELLOW}Warning: {msg.get('message', '')}{C.RESET}")

            elif msg_type == "error":
                print(f"  {C.RED}Error: {msg.get('message', '')}{C.RESET}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except websocket.WebSocketConnectionClosedException:
        print(f"\n{C.YELLOW}WebSocket connection closed.{C.RESET}")
    finally:
        ws.close()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Play PokeRogue via terminal")
    parser.add_argument("--seed", default=None, help="RNG seed (prompts if not provided)")
    parser.add_argument("--waves", type=int, default=50, help="Max waves (default: 50)")
    parser.add_argument("--rendered", action="store_true",
                        help="Connect to browser game via WebSocket (requires Vite dev server)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Vite dev server port for --rendered mode (default: 8000)")
    args = parser.parse_args()

    # Prompt for seed if not provided via CLI
    if args.seed is None:
        try:
            seed = input(f"{C.CYAN}Enter seed (or press Enter for random): {C.RESET}").strip()
            args.seed = seed if seed else f"play-{os.urandom(4).hex()}"
        except (EOFError, KeyboardInterrupt):
            args.seed = f"play-{os.urandom(4).hex()}"
            print()

    if args.rendered:
        run_rendered(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
