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

# Run-config support (src/rl/run_config.py): --config loads a YAML/JSON file
# describing the run (seed, starters, overrides, ...); CLI flags override it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rl.pokerogue_env import PROTOCOL_VERSION  # noqa: E402
from rl.run_config import RunConfig, load_run_config  # noqa: E402


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


# ─── Name Lookup Tables ─────────────────────────────────────────────

# Import from rl.enums if available (canonical source of truth); fall back to
# inline dicts so play.py keeps working standalone without the RL package.
# NOTE: import the PACKAGE module (rl.enums), never sys.path-insert src/rl
# itself — a flat `enums` module alongside an installed `rl.enums` would be
# two distinct module objects (see docs/VERIFICATION.md).
try:
    from rl.enums import (  # noqa: E402
        TYPE_NAMES, STATUS_NAMES, WEATHER_NAMES, TERRAIN_NAMES,
        CATEGORY_NAMES, BATTLE_TYPE_NAMES, TYPE_ABBREV, CATEGORY_ABBREV,
    )
except ImportError:
    TYPE_NAMES = {
        0: "Normal", 1: "Fighting", 2: "Flying", 3: "Poison", 4: "Ground",
        5: "Rock", 6: "Bug", 7: "Ghost", 8: "Steel", 9: "Fire", 10: "Water",
        11: "Grass", 12: "Electric", 13: "Psychic", 14: "Ice", 15: "Dragon",
        16: "Dark", 17: "Fairy", 18: "Stellar", -1: "\u2014",
    }
    STATUS_NAMES = {
        0: "\u2014", 1: "Poison", 2: "Toxic", 3: "Paralysis", 4: "Sleep",
        5: "Freeze", 6: "Burn", 7: "Faint",
    }
    # Fixed: SNOW=5, FOG=6 (matches src/enums/weather-type.ts)
    WEATHER_NAMES = {
        0: "None", 1: "Sunny", 2: "Rain", 3: "Sandstorm", 4: "Hail",
        5: "Snow", 6: "Fog", 7: "Heavy Rain", 8: "Harsh Sun", 9: "Strong Winds",
    }
    TERRAIN_NAMES = {0: "None", 1: "Misty", 2: "Electric", 3: "Grassy", 4: "Psychic"}
    CATEGORY_NAMES = {0: "Physical", 1: "Special", 2: "Status"}
    BATTLE_TYPE_NAMES = {0: "Wild", 1: "Trainer", 2: "Clear", 3: "Mystery"}

    # Short type abbreviations for compact move display
    TYPE_ABBREV = {
        0: "Nor", 1: "Fig", 2: "Fly", 3: "Psn", 4: "Gnd", 5: "Rck", 6: "Bug",
        7: "Gho", 8: "Stl", 9: "Fir", 10: "Wat", 11: "Grs", 12: "Elc", 13: "Psy",
        14: "Ice", 15: "Drg", 16: "Drk", 17: "Fai", 18: "Str", -1: "\u2014",
    }
    CATEGORY_ABBREV = {0: "Phy", 1: "Spe", 2: "Sta"}


def type_name(type_id):
    return TYPE_NAMES.get(type_id, f"?({type_id})")

def status_name(status_id):
    return STATUS_NAMES.get(status_id, f"?({status_id})")

def weather_name(weather_id):
    return WEATHER_NAMES.get(weather_id, f"?({weather_id})")

def terrain_name(terrain_id):
    return TERRAIN_NAMES.get(terrain_id, f"?({terrain_id})")


# ─── HP Bar ──────────────────────────────────────────────────────────

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
    bar = color + "\u2588" * filled + C.DIM + "\u2591" * empty + C.RESET
    return f"[{bar}] {pct}%"


# ─── Display ─────────────────────────────────────────────────────────

def print_header(step: int, phase: str, game_state: dict):
    """Print the game state header."""
    battle = game_state.get("battle", {})
    wave = battle.get("wave_index", "?")
    turn = battle.get("turn", "?")
    money = battle.get("money", 0)

    print()
    print(f"{C.BOLD}{'\u2550' * 60}{C.RESET}")
    print(f"{C.BOLD}  Wave {wave} \u2502 Turn {turn} \u2502 Step {step} \u2502 ${money}{C.RESET}")
    print(f"{C.BOLD}  Phase: {C.CYAN}{phase}{C.RESET}")
    print(f"{C.BOLD}{'\u2550' * 60}{C.RESET}")


def _format_pokemon_line(p: dict, show_hp_abs: bool = False) -> str:
    """Format a single Pokemon's summary line with types, ability, status."""
    name = p.get("species_name", "?")
    level = p.get("level", "?")
    hp_pct = round(p.get("hp_ratio", 0) * 100)

    # Types (array of ints, length 1-2)
    types = p.get("types", [])
    t1 = type_name(types[0]) if len(types) > 0 else "\u2014"
    t2_id = types[1] if len(types) > 1 else -1
    types_str = t1 if t2_id == -1 else f"{t1}/{type_name(t2_id)}"

    # Ability
    ability = p.get("ability_name", "")

    # Status
    status_id = p.get("status_effect", 0)
    status_str = f" {C.YELLOW}[{status_name(status_id)}]{C.RESET}" if status_id else ""

    # HP line
    hp_str = hp_bar(hp_pct)
    if show_hp_abs:
        hp = p.get("hp", "?")
        max_hp = p.get("max_hp", "?")
        hp_str += f" ({hp}/{max_hp})"

    line = f"{name} Lv{level}  {hp_str}{status_str}"
    detail = f"{C.DIM}{types_str}"
    if ability:
        detail += f" \u2502 {ability}"
    detail += C.RESET

    return line, detail


def _format_tags(p: dict) -> str:
    """Format volatile tags for a Pokemon."""
    tags = p.get("volatile_tags", [])
    if not tags:
        return ""
    tag_names = [t.get("tag_type", "?") for t in tags]
    return f"{C.MAGENTA}Tags: {', '.join(tag_names)}{C.RESET}"


def _format_moves_inline(p: dict) -> str:
    """Format a compact inline move summary for a Pokemon."""
    moves = p.get("moves", [])
    if not moves:
        return ""
    parts = []
    for m in moves:
        name = m.get("name", "?")
        m_type = m.get("type", -1)
        cat = m.get("category", -1)
        power = m.get("power", 0)
        t_abbr = TYPE_ABBREV.get(m_type, "?")
        c_abbr = CATEGORY_ABBREV.get(cat, "?")
        power_str = f" {power}p" if power and power > 0 else ""
        parts.append(f"{name}({t_abbr}/{c_abbr}{power_str})")
    return " ".join(parts)


def print_field(game_state: dict):
    """Print the battlefield: player vs enemy Pokemon with types and tags."""
    # Enemies
    enemies = []
    for key in ("enemy_0", "enemy_1"):
        p = game_state.get(key, {})
        if p.get("valid") and p.get("is_on_field"):
            enemies.append(p)

    # Players
    players = []
    for key in ("player_0", "player_1"):
        p = game_state.get(key, {})
        if p.get("valid") and p.get("is_on_field"):
            players.append(p)

    if enemies:
        print(f"\n  {C.RED}Enemy:{C.RESET}")
        for e in enemies:
            line, detail = _format_pokemon_line(e)
            print(f"    {line}")
            print(f"      {detail}")
            moves_str = _format_moves_inline(e)
            if moves_str:
                print(f"      {C.DIM}Moves: {moves_str}{C.RESET}")
            tags = _format_tags(e)
            if tags:
                print(f"      {tags}")

    if players:
        print(f"\n  {C.GREEN}Player:{C.RESET}")
        for p in players:
            line, detail = _format_pokemon_line(p, show_hp_abs=True)
            print(f"    {line}")
            print(f"      {detail}")
            tags = _format_tags(p)
            if tags:
                print(f"      {tags}")

    # Compact field summary
    _print_field_compact(game_state)


def _print_field_compact(game_state: dict):
    """Print a compact 1-line field condition summary."""
    field = game_state.get("field", {})
    if not field:
        return

    parts = []

    # Weather
    w_id = field.get("weather_type", 0)
    if w_id:
        w_turns = field.get("weather_turns_left", 0)
        w_name = weather_name(w_id)
        parts.append(f"{w_name}({w_turns}t)" if w_turns else f"{w_name}(perm)")

    # Terrain
    t_id = field.get("terrain_type", 0)
    if t_id:
        t_turns = field.get("terrain_turns_left", 0)
        t_perm = field.get("terrain_is_permanent", False)
        t_name = terrain_name(t_id)
        if t_perm:
            parts.append(f"{t_name}Terrain(perm)")
        else:
            parts.append(f"{t_name}Terrain({t_turns}t)")

    # Player-side hazards
    p_hazards = _get_side_hazards(field, "player")
    if p_hazards:
        parts.append(f"P:{p_hazards}")

    # Enemy-side hazards
    e_hazards = _get_side_hazards(field, "enemy")
    if e_hazards:
        parts.append(f"E:{e_hazards}")

    # Double battle
    battle = game_state.get("battle", {})
    is_double = battle.get("is_double", False)
    if is_double:
        parts.append("Double")

    if parts:
        print(f"\n  {C.DIM}Field: {' \u2502 '.join(parts)}{C.RESET}")


def _get_side_hazards(field: dict, side: str) -> str:
    """Get hazard summary for a side from arena tags."""
    arena_tags = field.get("arena_tags", [])
    hazards = []
    side_val = 1 if side == "player" else 2  # PLAYER=1, ENEMY=2
    for tag in arena_tags:
        tag_side = tag.get("side", 0)
        # BOTH=0 applies to both sides, or match exact side
        if tag_side == side_val or tag_side == 0:
            tag_type = tag.get("tag_type", "?")
            layers = tag.get("layers", 1)
            if layers > 1:
                hazards.append(f"{tag_type}({layers})")
            else:
                hazards.append(tag_type)
    return " ".join(hazards)


def print_moves_compact(game_state: dict):
    """Print compact move summary for active player Pokemon when in command phase."""
    for key in ("player_0", "player_1"):
        p = game_state.get(key, {})
        if not p.get("valid") or not p.get("is_on_field"):
            continue

        moves = p.get("moves", [])
        if not moves:
            continue

        pname = p.get("species_name", "?")
        parts = []
        for i, m in enumerate(moves):
            name = m.get("name", "?")
            m_type = m.get("type", -1)
            cat_id = m.get("category", -1)
            power = m.get("power", 0)
            t_abbr = TYPE_ABBREV.get(m_type, "?")
            c_abbr = CATEGORY_ABBREV.get(cat_id, "?")
            pp = m.get("pp_remaining", 0)
            pp_max = m.get("pp_max", 0)
            usable = m.get("is_usable", True)
            disabled = "" if usable else f"{C.DIM}X{C.RESET}"

            power_str = f"{power}p" if power and power > 0 else ""
            parts.append(f"[{i}]{disabled}{name}({t_abbr}/{c_abbr} {power_str} {pp}/{pp_max}pp)")

        print(f"\n  {C.BLUE}{pname} Moves: {' '.join(parts)}{C.RESET}")


def print_party(game_state: dict):
    """Print the party summary with actual slot keys."""
    slot_keys = ["player_0", "player_1", "player_2", "player_3", "player_4", "player_5"]
    has_any = False
    for key in slot_keys:
        p = game_state.get(key, {})
        if not p.get("valid"):
            continue
        if not has_any:
            print(f"\n  {C.BLUE}Party:{C.RESET}")
            has_any = True
        name = p.get("species_name", "?")
        level = p.get("level", "?")
        hp_pct = round(p.get("hp_ratio", 0) * 100)
        fainted = p.get("is_fainted", False)
        on_field = p.get("is_on_field", False)
        field_tag = f" {C.GREEN}(active){C.RESET}" if on_field else ""
        slot_label = key.replace("player_", "p")
        if fainted:
            print(f"    [{slot_label}] {C.DIM}{name} Lv{level}  FAINTED{C.RESET}")
        else:
            print(f"    [{slot_label}] {name} Lv{level}  {hp_bar(hp_pct, 10)}{field_tag}")


def print_actions(actions: list) -> set:
    """Print legal actions by their STABLE action id; return the set of legal ids.

    The number shown is the fixed action id (0 = move slot 0 vs enemy, 38 = reroll,
    ...), NOT a per-phase list position — so a given id always means the same thing,
    every phase. See src/rl/README.md for the full action map.
    """
    print(f"\n  {C.MAGENTA}Available Actions (type the id):{C.RESET}")
    valid_ids = set()
    for act in actions:
        idx = act["index"]
        valid_ids.add(idx)
        print(f"    {C.BOLD}{idx:>3}{C.RESET}  {act['label']}")
    return valid_ids


# ─── Detailed Inspection Commands ───────────────────────────────────

def _fmt_arr(arr, labels=None):
    """Format a numeric array with optional labels."""
    if not arr:
        return "[]"
    if labels and len(labels) == len(arr):
        return " ".join(f"{l}:{v}" for l, v in zip(labels, arr))
    return str(arr)


def print_pokemon_detail(poke: dict, label: str):
    """Print EVERY field for a single Pokemon."""
    if not poke.get("valid"):
        print(f"\n  {label}: {C.DIM}(empty slot){C.RESET}")
        return

    name = poke.get("species_name", "?")
    level = poke.get("level", "?")
    hp = poke.get("hp", 0)
    max_hp = poke.get("max_hp", 0)
    hp_pct = round(poke.get("hp_ratio", 0) * 100)
    sid = poke.get("species_id", 0)

    print(f"\n  {C.BOLD}{label}: {name} (#{sid}) Lv{level}{C.RESET}")
    print(f"    HP: {hp}/{max_hp} ({hp_pct}%)  {hp_bar(hp_pct)}")

    # Identity
    types = poke.get("types", [])
    types_str = "/".join(type_name(t) for t in types) if types else "\u2014"
    gender = poke.get("gender", -1)
    gender_str = {-1: "Genderless", 0: "Male", 1: "Female"}.get(gender, f"?({gender})")
    print(f"    Types: {types_str}  \u2502  Gender: {gender_str}  \u2502  Form: {poke.get('form_index', 0)}")
    print(f"    Weight: {poke.get('weight', 0)}kg  \u2502  Catch rate: {poke.get('catch_rate', 0)}  \u2502  "
          f"Base total: {poke.get('base_total', 0)}")
    print(f"    Friendship: {poke.get('friendship', 0)}  \u2502  Nature: {poke.get('nature', 0)}  \u2502  "
          f"Shiny: {poke.get('shiny', False)}  Variant: {poke.get('variant', 0)}")

    # Tera
    tera_type = poke.get("tera_type", -1)
    is_tera = poke.get("is_terastallized", False)
    added_type = poke.get("added_type", -1)
    print(f"    Tera type: {type_name(tera_type)}  \u2502  Terastallized: {is_tera}"
          + (f"  \u2502  Added type: {type_name(added_type)}" if added_type >= 0 else ""))

    # Abilities
    print(f"    Ability: {poke.get('ability_name', '?')} (#{poke.get('ability_id', 0)})  \u2502  "
          f"Revealed: {poke.get('ability_revealed', False)}  \u2502  Suppressed: {poke.get('ability_suppressed', False)}")
    print(f"    Passive: {poke.get('passive_ability_name', '') or '(none)'} (#{poke.get('passive_ability_id', 0)})  \u2502  "
          f"Has passive: {poke.get('has_passive', False)}")

    # Status
    status_id = poke.get("status_effect", 0)
    print(f"    Status: {status_name(status_id)}  \u2502  "
          f"Toxic turns: {poke.get('toxic_turn_count', 0)}  \u2502  "
          f"Sleep turns left: {poke.get('sleep_turns_remaining', 0)}")

    # Stats
    stat6 = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
    stat7 = ["Atk", "Def", "SpA", "SpD", "Spe", "Acc", "Eva"]
    print(f"    Base stats:    {_fmt_arr(poke.get('base_stats', []), stat6)}")
    print(f"    IVs:           {_fmt_arr(poke.get('ivs', []), stat6)}")
    print(f"    Computed stats:{_fmt_arr(poke.get('stats', []), stat6)}")
    print(f"    Stat stages:   {_fmt_arr(poke.get('stat_stages', []), stat7)}")
    print(f"    Nature mults:  {_fmt_arr(poke.get('nature_multipliers', []), stat6[1:])}")

    # Boss
    if poke.get("is_boss"):
        print(f"    {C.RED}BOSS{C.RESET}  segments:{poke.get('boss_segments', 0)}  "
              f"segment_idx:{poke.get('boss_segment_index', 0)}")

    # Field presence
    print(f"    On field: {poke.get('is_on_field', False)}  \u2502  "
          f"Is player: {poke.get('is_player', False)}  \u2502  "
          f"Battler idx: {poke.get('battler_index', -1)}  \u2502  "
          f"Field idx: {poke.get('field_index', -1)}")
    print(f"    Fainted: {poke.get('is_fainted', False)}  \u2502  "
          f"Active: {poke.get('is_active', False)}  \u2502  "
          f"Trapped: {poke.get('is_trapped', False)}  \u2502  "
          f"Grounded: {poke.get('is_grounded', True)}")
    print(f"    AI type: {poke.get('ai_type', 0)}  \u2502  "
          f"Wave turn count: {poke.get('wave_turn_count', 0)}  \u2502  "
          f"Pokeball: {poke.get('pokeball', 0)}")

    # Fusion / Transform / Illusion
    if poke.get("is_fusion"):
        print(f"    Fusion species: #{poke.get('fusion_species_id', None)}")
    if poke.get("transform_species_id") is not None:
        print(f"    Transform species: #{poke.get('transform_species_id')}")
    if poke.get("illusion_species_id") is not None:
        print(f"    Illusion species: #{poke.get('illusion_species_id')}")

    # Extra v5 fields
    print(f"    Exp to next: {poke.get('exp_to_next_level', 0)}  \u2502  "
          f"Luck: {poke.get('luck', 0)}  \u2502  "
          f"Endured this wave: {poke.get('endured_this_wave', False)}")
    stellar = poke.get("stellar_types_boosted", [])
    if stellar:
        print(f"    Stellar types boosted: {[type_name(t) for t in stellar]}")
    berries_last = poke.get("berries_eaten_last", [])
    if berries_last:
        print(f"    Berries eaten last: {berries_last}")

    # Moves
    moves = poke.get("moves", [])
    if moves:
        print(f"    Moves ({len(moves)}):")
        for i, m in enumerate(moves):
            m_name = m.get("name", "?")
            m_type = type_name(m.get("type", -1))
            m_cat = CATEGORY_NAMES.get(m.get("category", -1), "?")
            m_pow = m.get("power", 0)
            m_acc = m.get("accuracy", 0)
            pp_rem = m.get("pp_remaining", 0)
            pp_max = m.get("pp_max", 0)
            pp_used = m.get("pp_used", 0)
            pp_up = m.get("pp_up", 0)
            pri = m.get("priority", 0)
            pow_str = f"{m_pow}pow" if m_pow and m_pow > 0 else "\u2014pow"
            acc_str = f"{m_acc}acc" if m_acc and m_acc > 0 else "\u2014acc"
            pri_str = f"pri:{'+' if pri > 0 else ''}{pri}" if pri != 0 else ""
            print(f"      [{i}] {m_name} (#{m.get('move_id', 0)})  {m_type}/{m_cat}  "
                  f"{pow_str} {acc_str}  {pp_rem}/{pp_max}pp (used:{pp_used} up:{pp_up})  {pri_str}")
            # All flags and secondary effects
            flags = []
            for flag_key in ("makes_contact", "is_sound_based", "is_powder", "is_punching",
                             "is_slicing", "is_biting", "is_ballistic", "is_pulse", "is_dance",
                             "is_protect", "is_sacrifice", "is_ohko", "is_charging",
                             "is_multi_hit", "self_switch", "force_switch", "traps_target",
                             "ignores_protect", "ignores_abilities", "ignores_substitute"):
                if m.get(flag_key):
                    flags.append(flag_key.replace("is_", "").replace("_", "-"))
            eff = m.get("effect_chance", -1)
            if eff > 0:
                flags.append(f"effect:{eff}%")
            se = m.get("status_effect", 0)
            if se:
                flags.append(f"status:{status_name(se)}")
            for ratio_key in ("drain_ratio", "recoil_ratio", "heal_ratio"):
                v = m.get(ratio_key, 0)
                if v:
                    flags.append(f"{ratio_key.replace('_ratio', '')}:{v}")
            if m.get("multi_hit_type", -1) >= 0:
                flags.append(f"multi_hit_type:{m['multi_hit_type']}")
            crit = m.get("crit_stage_boost", 0)
            if crit:
                flags.append(f"crit+{crit}")
            fd = m.get("fixed_damage", 0)
            if fd:
                flags.append(f"fixed_dmg:{fd}")
            tgt = m.get("target", 0)
            flags.append(f"target:{tgt}")
            flags.append(f"usable:{'y' if m.get('is_usable') else 'n'}")
            # Stat changes
            for sc in m.get("stat_changes", []):
                s = sc.get("stages", 0)
                sign = "+" if s > 0 else ""
                tgt_str = "self" if sc.get("self_target") else "foe"
                ch = sc.get("chance", 100)
                sc_str = f"stat{sc.get('stat_id', '?')}:{sign}{s}({tgt_str})"
                if ch < 100:
                    sc_str += f"@{ch}%"
                flags.append(sc_str)
            print(f"        {C.DIM}{' '.join(flags)}{C.RESET}")

    # Move history
    mhist = poke.get("move_history", [])
    if mhist:
        print(f"    Move history ({len(mhist)} entries):")
        for mh in mhist[-6:]:  # show last 6
            print(f"      move_id:{mh.get('move_id', '?')} targets:{mh.get('targets', [])} "
                  f"mode:{mh.get('use_mode', '?')} result:{mh.get('result', '?')}")
        if len(mhist) > 6:
            print(f"      ... ({len(mhist) - 6} more)")

    # Move queue
    mq = poke.get("move_queue", [])
    if mq:
        print(f"    Move queue: {mq}")

    # Volatile tags (full detail — show ALL fields including zero/false)
    tags = poke.get("volatile_tags", [])
    if tags:
        print(f"    Volatile tags ({len(tags)}):")
        for t in tags:
            tag_type = t.get("tag_type", "?")
            extras = {k: v for k, v in t.items() if k != "tag_type" and v is not None}
            extras_str = f" {extras}" if extras else ""
            print(f"      {C.MAGENTA}{tag_type}{C.RESET}{extras_str}")

    # Held items (show ALL fields)
    items = poke.get("held_items", [])
    if items:
        print(f"    Held items ({len(items)}):")
        for it in items:
            it_name = it.get("name", it.get("item_name", "?"))
            stack = it.get("stack_count", 1)
            max_stack = it.get("max_stack_count", "?")
            mod_class = it.get("modifier_class", "")
            mod_id = it.get("modifier_id", "")
            transferable = it.get("is_transferable", None)
            print(f"      {it_name} x{stack}/{max_stack}  class:{mod_class}  id:{mod_id}"
                  + (f"  transferable:{transferable}" if transferable is not None else ""))
            # All remaining optional fields
            shown_keys = {"name", "item_name", "stack_count", "max_stack_count",
                          "modifier_class", "modifier_id", "is_transferable"}
            extras = {k: v for k, v in it.items() if k not in shown_keys and v is not None}
            if extras:
                print(f"        {C.DIM}{extras}{C.RESET}")
    else:
        print(f"    Held items: none")

    # Turn data (full)
    td = poke.get("turn_data", {})
    if td:
        print(f"    Turn data: dmg_taken:{td.get('damage_taken', 0)} "
              f"dmg_dealt:{td.get('total_damage_dealt', 0)} "
              f"hits:{td.get('hit_count', 0)} order:{td.get('order', 0)}")
        print(f"      acted:{td.get('acted', False)} "
              f"switched_in:{td.get('switched_in_this_turn', False)} "
              f"stages_up:{td.get('stat_stages_increased', False)} "
              f"stages_down:{td.get('stat_stages_decreased', False)}")
        td_atk = td.get("attacks_received", [])
        if td_atk:
            print(f"      attacks_received ({len(td_atk)}):")
            for a in td_atk:
                print(f"        {a}")
        td_berries = td.get("berries_eaten", [])
        if td_berries:
            print(f"      berries_eaten: {td_berries}")

    # Battle data
    bd = poke.get("battle_data", {})
    if bd:
        print(f"    Battle data: hit_count:{bd.get('hit_count', 0)} "
              f"eaten_berry:{bd.get('has_eaten_berry', False)} "
              f"berries:{bd.get('berries_eaten', [])}")
        ab_applied = bd.get("abilities_applied", [])
        if ab_applied:
            print(f"      abilities_applied: {ab_applied}")

    # Transform moves (shown when transformed into another species)
    transform_moves = poke.get("transform_moves", [])
    if transform_moves:
        print(f"    Transform moves ({len(transform_moves)}):")
        for i, m in enumerate(transform_moves):
            m_name = m.get("name", "?")
            m_type = type_name(m.get("type", -1))
            m_pow = m.get("power", 0)
            pow_str = f"{m_pow}pow" if m_pow and m_pow > 0 else "\u2014pow"
            print(f"      [{i}] {m_name}  {m_type}  {pow_str}")

    # Attacks received (top-level, convenience copy)
    atk_recv = poke.get("attacks_received", [])
    if atk_recv:
        print(f"    Attacks received ({len(atk_recv)}):")
        for a in atk_recv:
            print(f"      {a}")


def print_moves_detail(game_state: dict):
    """Print full move details for active player Pokemon."""
    for key in ("player_0", "player_1"):
        p = game_state.get(key, {})
        if not p.get("valid") or not p.get("is_on_field"):
            continue

        name = p.get("species_name", "?")
        moves = p.get("moves", [])
        if not moves:
            continue

        print(f"\n  {C.BOLD}Moves for {name}:{C.RESET}")
        for i, m in enumerate(moves):
            m_name = m.get("name", "?")
            m_type = type_name(m.get("type", -1))
            m_cat = CATEGORY_NAMES.get(m.get("category", -1), "?")
            m_pow = m.get("power", 0)
            m_acc = m.get("accuracy", 0)
            pp_cur = m.get("pp_remaining", 0)
            pp_max = m.get("pp_max", 0)
            pri = m.get("priority", 0)
            usable = m.get("is_usable", True)

            pow_str = f"{m_pow}pow" if m_pow and m_pow > 0 else "\u2014pow"
            acc_str = f"{m_acc}acc" if m_acc and m_acc > 0 else "\u2014acc"
            pri_str = f"pri:{'+' if pri > 0 else ''}{pri}"

            usable_str = f"{C.GREEN}yes{C.RESET}" if usable else f"{C.RED}no{C.RESET}"

            print(f"    [{i}] {C.BOLD}{m_name}{C.RESET}  {m_type}/{m_cat}  "
                  f"{pow_str} {acc_str}  {pp_cur}/{pp_max}pp  {pri_str}")

            # Secondary effect details
            details = []
            eff_chance = m.get("effect_chance", -1)
            if eff_chance > 0:
                details.append(f"effect:{eff_chance}%")
            status_eff = m.get("status_effect", 0)
            if status_eff:
                details.append(f"status:{status_name(status_eff)}")
            contact = m.get("makes_contact", False)
            if contact:
                details.append("contact:yes")
            drain = m.get("drain_ratio", 0)
            if drain:
                details.append(f"drain:{drain}")
            recoil = m.get("recoil_ratio", 0)
            if recoil:
                details.append(f"recoil:{recoil}")
            heal = m.get("heal_ratio", 0)
            if heal:
                details.append(f"heal:{heal}")
            if m.get("is_multi_hit"):
                details.append(f"multi_hit:{m.get('multi_hit_type', '?')}")
            if m.get("self_switch"):
                details.append("self_switch")
            if m.get("force_switch"):
                details.append("force_switch")
            if m.get("is_protect"):
                details.append("protect")
            if m.get("is_sacrifice"):
                details.append("sacrifice")
            if m.get("is_ohko"):
                details.append("OHKO")
            if m.get("traps_target"):
                details.append("traps")
            if m.get("is_charging"):
                details.append("charging")
            crit = m.get("crit_stage_boost", 0)
            if crit:
                details.append(f"crit_boost:+{crit}")

            # Stat changes
            stat_changes = m.get("stat_changes", [])
            for sc in stat_changes:
                stat_id = sc.get("stat_id", "?")
                stages = sc.get("stages", 0)
                self_t = sc.get("self_target", False)
                chance = sc.get("chance", 100)
                sign = "+" if stages > 0 else ""
                target = "self" if self_t else "foe"
                sc_str = f"stat{stat_id}:{sign}{stages}({target})"
                if chance < 100:
                    sc_str += f"@{chance}%"
                details.append(sc_str)

            details.append(f"usable:{usable_str}")
            if details:
                print(f"        {C.DIM}{' \u2502 '.join(details)}{C.RESET}")


def print_field_state(game_state: dict):
    """Print EVERY field from the field state."""
    field = game_state.get("field", {})
    if not field:
        print(f"\n  {C.DIM}(no field data){C.RESET}")
        return

    print(f"\n  {C.BOLD}Field State:{C.RESET}")

    # Biome
    print(f"    Biome: {field.get('biome_name', '?')} (#{field.get('biome_id', 0)})  \u2502  "
          f"Double: {field.get('is_double_battle', False)}")

    # Weather
    w_id = field.get("weather_type", 0)
    w_turns = field.get("weather_turns_left", 0)
    w_perm = field.get("weather_is_permanent", False)
    w_supp = field.get("weather_suppressed", False)
    w_str = weather_name(w_id)
    if w_id:
        if w_perm:
            w_str += " (permanent)"
        elif w_turns:
            w_str += f" ({w_turns}t left)"
        if w_supp:
            w_str += " [SUPPRESSED]"
    print(f"    Weather: {w_str}")

    # Terrain
    t_id = field.get("terrain_type", 0)
    t_turns = field.get("terrain_turns_left", 0)
    t_perm = field.get("terrain_is_permanent", False)
    t_str = terrain_name(t_id)
    if t_id:
        if t_perm:
            t_str += " (permanent)"
        else:
            t_str += f" ({t_turns}t left)"
    print(f"    Terrain: {t_str}")

    # Global effects
    print(f"    Trick Room: {field.get('trick_room_active', False)}  \u2502  "
          f"Gravity: {field.get('gravity_active', False)}  \u2502  "
          f"Ignore abilities: {field.get('ignore_abilities', False)}")
    print(f"    Player teras used: {field.get('player_teras_used', 0)}")

    # Hazards (dedicated fields)
    print(f"    Player hazards: Spikes:{field.get('player_spikes_layers', 0)} "
          f"ToxicSpikes:{field.get('player_toxic_spikes_layers', 0)} "
          f"StealthRock:{field.get('player_stealth_rock', False)} "
          f"StickyWeb:{field.get('player_sticky_web', False)}")
    print(f"    Enemy hazards:  Spikes:{field.get('enemy_spikes_layers', 0)} "
          f"ToxicSpikes:{field.get('enemy_toxic_spikes_layers', 0)} "
          f"StealthRock:{field.get('enemy_stealth_rock', False)} "
          f"StickyWeb:{field.get('enemy_sticky_web', False)}")

    # Arena tags (full detail — show ALL fields including zero values)
    arena_tags = field.get("arena_tags", [])
    if arena_tags:
        side_names = {0: "BOTH", 1: "PLAYER", 2: "ENEMY"}
        print(f"    Arena tags ({len(arena_tags)}):")
        for tag in arena_tags:
            tag_type = tag.get("tag_type", "?")
            side = side_names.get(tag.get("side", 0), "?")
            turn_count = tag.get("turn_count", 0)
            layers = tag.get("layers", 1)
            source_id = tag.get("source_id", None)
            tc_str = f"{turn_count}t" if turn_count > 0 else "permanent"
            src_str = f" src:{source_id}" if source_id is not None else ""
            print(f"      {tag_type} [{side}] layers:{layers} {tc_str}{src_str}")

    # Positional tags (formatted)
    pos_tags = field.get("positional_tags", [])
    if pos_tags:
        print(f"    Positional tags ({len(pos_tags)}):")
        for tag in pos_tags:
            pt_type = tag.get("tag_type", "?")
            countdown = tag.get("countdown", "?")
            tgt_idx = tag.get("target_index", "?")
            src_id = tag.get("source_id", None)
            move_id = tag.get("move_id", None)
            heal_hp = tag.get("heal_hp", None)
            extras = []
            if src_id is not None:
                extras.append(f"src:{src_id}")
            if move_id is not None:
                extras.append(f"move:{move_id}")
            if heal_hp is not None:
                extras.append(f"heal_hp:{heal_hp}")
            ext_str = f"  {' '.join(extras)}" if extras else ""
            print(f"      {pt_type} countdown:{countdown} target_idx:{tgt_idx}{ext_str}")


def print_battle_info(game_state: dict):
    """Print EVERY field from battle state."""
    battle = game_state.get("battle", {})
    if not battle:
        print(f"\n  {C.DIM}(no battle data){C.RESET}")
        return

    print(f"\n  {C.BOLD}Battle Info:{C.RESET}")

    bt = battle.get("battle_type", 0)
    bt_name = BATTLE_TYPE_NAMES.get(bt, f"?({bt})")
    field = game_state.get("field", {})
    biome = field.get("biome_name", "?")

    print(f"    Wave: {battle.get('wave_index', '?')}  \u2502  Turn: {battle.get('turn', '?')}  \u2502  "
          f"Type: {bt_name} (spec:{battle.get('battle_spec', 0)})  \u2502  Biome: {biome}")
    print(f"    Double: {battle.get('is_double', False)}  \u2502  "
          f"Game mode: {battle.get('game_mode', 0)}  \u2502  Seed: {battle.get('seed', '?')}")

    # Counts
    print(f"    Money: ${battle.get('money', 0)}  \u2502  Score: {battle.get('score', 0)}")
    print(f"    Player alive: {battle.get('player_alive_count', '?')}  \u2502  "
          f"Enemy alive: {battle.get('enemy_alive_count', '?')}")
    print(f"    Player faints (battle): {battle.get('player_faints_battle', 0)}  \u2502  "
          f"Enemy faints (battle): {battle.get('enemy_faints_battle', 0)}  \u2502  "
          f"Player faints (biome): {battle.get('player_faints_biome', 0)}")
    print(f"    Last move: #{battle.get('last_move_id', None)}  \u2502  "
          f"Escape attempts: {battle.get('escape_attempts', 0)}  \u2502  "
          f"Money scattered: {battle.get('money_scattered', 0)}")

    # Pokeballs
    balls = battle.get("pokeball_counts", {})
    if balls:
        ball_keys = [
            ("pokeball", "Poke"), ("great_ball", "Great"), ("ultra_ball", "Ultra"),
            ("rogue_ball", "Rogue"), ("master_ball", "Master"),
        ]
        ball_parts = [f"{label}:{balls.get(key, 0)}" for key, label in ball_keys]
        print(f"    Balls: {' '.join(ball_parts)}")

    # Actions
    print(f"    Tera: {battle.get('tera_available', False)}  \u2502  "
          f"Run: {battle.get('can_run', False)}  \u2502  "
          f"Catch: {battle.get('can_catch', False)}  \u2502  "
          f"Failed run: {battle.get('failed_run_away', False)}")

    # Style & time
    style_id = battle.get("battle_style", 0)
    style_name = {0: "Switch", 1: "Set"}.get(style_id, f"?({style_id})")
    time_id = battle.get("time_of_day", 0)
    time_name = {0: "Dawn", 1: "Day", 2: "Dusk", 3: "Night"}.get(time_id, f"?({time_id})")
    print(f"    Style: {style_name}  \u2502  Time: {time_name}  \u2502  "
          f"Offset gym: {battle.get('offset_gym', False)}")

    # Shop/modifier meta
    print(f"    Reroll count: {battle.get('reroll_count', 0)}  \u2502  "
          f"Lock tiers: {battle.get('lock_modifier_tiers', False)}  \u2502  "
          f"No shop: {battle.get('has_no_shop', False)}")
    print(f"    Has trainers: {battle.get('has_trainers', False)}  \u2502  "
          f"Spliced only: {battle.get('is_spliced_only', False)}  \u2502  "
          f"Seen enemies: {battle.get('seen_enemy_count', 0)}  \u2502  "
          f"Enemy switch ctr: {battle.get('enemy_switch_counter', 0)}")

    # Trainer (all TrainerInfo fields)
    trainer = battle.get("trainer")
    if trainer:
        t_name = trainer.get("trainer_name", "?")
        t_type = trainer.get("trainer_type", 0)
        t_boss = trainer.get("is_boss", False)
        t_dbl = trainer.get("is_double", False)
        t_size = trainer.get("party_template_size", "?")
        t_spec = trainer.get("specialty_type", None)
        t_tera = trainer.get("tera_mode", None)
        spec_str = f"  specialty:{type_name(t_spec)}" if t_spec is not None else ""
        tera_str = f"  tera_mode:{t_tera}" if t_tera is not None else ""
        print(f"    Trainer: {t_name} (type:{t_type})  boss:{t_boss}  "
              f"double:{t_dbl}  party_size:{t_size}{spec_str}{tera_str}")
    else:
        print(f"    Trainer: none (wild)")

    # Mystery encounter (all MysteryEncounterState + MysteryEncounterOption fields)
    me = battle.get("mystery_encounter")
    if me:
        me_name = me.get("encounter_name", "?")
        me_type = me.get("encounter_type", 0)
        print(f"    Mystery Encounter: {me_name} (type:{me_type})")
        me_opts = me.get("options", [])
        for opt in me_opts:
            avail = "available" if opt.get("is_available") else "unavailable"
            reqs = " (has_reqs)" if opt.get("has_requirements") else ""
            print(f"      [{opt.get('index', '?')}] {opt.get('label', '?')}  {avail}{reqs}")

    # Challenges (all ChallengeInfo fields)
    challenges = battle.get("challenges", [])
    if challenges:
        print(f"    Challenges ({len(challenges)}):")
        for ch in challenges:
            print(f"      {ch.get('challenge_name', '?')} (type:{ch.get('challenge_type', 0)})  "
                  f"value:{ch.get('value', 0)}  severity:{ch.get('severity', 0)}")


def print_shop_detail(game_state: dict):
    """Print ALL shop/reward fields during modifier select phase."""
    shop = game_state.get("shop", {})
    modifiers = game_state.get("modifiers", {})

    if shop:
        print(f"\n  {C.BOLD}Shop:{C.RESET}")

        # ShopState metadata
        can_reroll = shop.get("can_reroll", False)
        reroll_cost = shop.get("reroll_cost", 0)
        shop_money = shop.get("money", 0)
        print(f"    Money: ${shop_money}  |  Reroll: {'yes' if can_reroll else 'no'}"
              f"  Cost: ${reroll_cost}")

        rewards = shop.get("reward_options", [])
        if rewards:
            print(f"    {C.GREEN}Free Rewards ({len(rewards)}):{C.RESET}")
            for r in rewards:
                r_idx = r.get("index", "?")
                r_name = r.get("name", "?")
                r_tier = r.get("tier", "?")
                r_upg = r.get("upgrade_count", 0)
                r_class = r.get("modifier_class", "")
                r_id = r.get("modifier_id", "")
                r_target = r.get("target_kind", "none")
                r_is_poke = r.get("is_pokemon_modifier", False)
                r_type = r.get("type_id", None)
                r_stat = r.get("stat_id", None)
                r_desc = r.get("description", "")
                upg_str = f" +{r_upg}" if r_upg else ""
                type_str = f" type:{type_name(r_type)}" if r_type is not None else ""
                stat_str = f" stat:{r_stat}" if r_stat is not None else ""
                print(f"      [{r_idx}] {r_name} (tier {r_tier}{upg_str})  "
                      f"target:{r_target} poke_mod:{r_is_poke}{type_str}{stat_str}")
                print(f"          {C.DIM}class:{r_class} id:{r_id}{C.RESET}")
                if r_desc:
                    print(f"          {C.DIM}{r_desc}{C.RESET}")

        shop_opts = shop.get("shop_options", [])
        if shop_opts:
            print(f"    {C.YELLOW}Shop Items ({len(shop_opts)}):{C.RESET}")
            for s in shop_opts:
                s_idx = s.get("index", "?")
                s_name = s.get("name", "?")
                s_cost = s.get("cost", 0)
                s_tier = s.get("tier", "?")
                s_class = s.get("modifier_class", "")
                s_id = s.get("modifier_id", "")
                s_target = s.get("target_kind", "none")
                s_affordable = s.get("affordable", False)
                s_type = s.get("type_id", None)
                s_stat = s.get("stat_id", None)
                s_desc = s.get("description", "")
                aff_str = f"{C.GREEN}yes{C.RESET}" if s_affordable else f"{C.RED}no{C.RESET}"
                type_str = f" type:{type_name(s_type)}" if s_type is not None else ""
                stat_str = f" stat:{s_stat}" if s_stat is not None else ""
                print(f"      [{s_idx}] {s_name}  ${s_cost} (tier {s_tier})  "
                      f"affordable:{aff_str}  target:{s_target}{type_str}{stat_str}")
                print(f"          {C.DIM}class:{s_class} id:{s_id}{C.RESET}")
                if s_desc:
                    print(f"          {C.DIM}{s_desc}{C.RESET}")
    elif modifiers:
        print(f"\n  {C.BOLD}Modifiers:{C.RESET}")
        # held_items is a dict keyed by party slot index ("0", "1", ...)
        held = modifiers.get("held_items", {})
        if isinstance(held, dict):
            total_items = sum(len(items) for items in held.values())
            if total_items:
                print(f"    Held items ({total_items} total):")
                for slot, items in held.items():
                    if items:
                        item_strs = [f"{it.get('name', '?')}(x{it.get('stack_count', 1)})"
                                     if it.get('stack_count', 1) > 1 else it.get('name', '?')
                                     for it in items]
                        print(f"      Slot {slot}: {', '.join(item_strs)}")
            else:
                print(f"    Held items: none")
        party_mods = modifiers.get("party_modifiers", [])
        if party_mods:
            for pm in party_mods:
                pm_name = pm.get("name", "?")
                pm_class = pm.get("modifier_class", "")
                print(f"    Party: {pm_name} ({pm_class})")
        lapsing = modifiers.get("lapsing_modifiers", [])
        if lapsing:
            for lm in lapsing:
                lm_name = lm.get("name", "?")
                lm_battles = lm.get("battles_remaining", "?")
                print(f"    Lapsing: {lm_name} ({lm_battles} battles left)")
    else:
        print(f"\n  {C.DIM}(no shop/modifier data){C.RESET}")


def print_tags_summary(game_state: dict):
    """Print volatile tags on all active Pokemon."""
    print(f"\n  {C.BOLD}Volatile Tags Summary:{C.RESET}")
    slots = [
        ("Player 0", "player_0"), ("Player 1", "player_1"),
        ("Enemy 0", "enemy_0"), ("Enemy 1", "enemy_1"),
    ]
    any_tags = False
    for label, key in slots:
        p = game_state.get(key, {})
        if not p.get("valid") or not p.get("is_on_field"):
            continue
        tags = p.get("volatile_tags", [])
        name = p.get("species_name", "?")
        if tags:
            any_tags = True
            print(f"    {label} ({name}):")
            for t in tags:
                tag_type = t.get("tag_type", "?")
                turn_count = t.get("turn_count", -1)
                source = t.get("source_id", None)
                source_move = t.get("source_move", None)
                extras = []
                if turn_count > 0:
                    extras.append(f"{turn_count}t left")
                elif turn_count == 0:
                    extras.append("permanent")
                if source is not None:
                    extras.append(f"src_id:{source}")
                if source_move is not None:
                    extras.append(f"src_move:{source_move}")
                # All known special tag fields (v1-v5)
                for special_key in ("substitute_hp", "stockpile_count", "encore_move_id",
                                    "disabled_move_id", "type_boost_type", "type_boost_value",
                                    "crit_boost_stages", "gorilla_tactics_move_id",
                                    "highest_stat_boost_stat", "highest_stat_boost_multiplier",
                                    "supreme_overlord_faint_count", "autotomize_count"):
                    val = t.get(special_key)
                    if val is not None:
                        extras.append(f"{special_key}:{val}")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                print(f"      {C.MAGENTA}{tag_type}{C.RESET}{extra_str}")
        else:
            print(f"    {label} ({name}): {C.DIM}none{C.RESET}")

    if not any_tags:
        print(f"    {C.DIM}No active tags on any Pokemon{C.RESET}")


def print_obs_layout(game_state: dict):
    """Print observation vector segment summary (layout from observation.py)."""
    from rl.enums import POKEMON_SLOT_KEYS
    from rl.observation import (
        BATTLE_META_DIM,
        DERIVED_FIELDS_DIM,
        FIELD_STATE_DIM,
        MODIFIER_INVENTORY_DIM,
        MODIFIER_PHASE_DIM,
        OBSERVATION_DIM,
        PHASE_INDICATOR_DIM,
        POKEMON_BLOCK_DIM,
    )

    print(f"\n  {C.BOLD}Observation Layout ({OBSERVATION_DIM} float32):{C.RESET}")

    offset = 0
    for key in POKEMON_SLOT_KEYS:
        p = game_state.get(key, {})
        valid = p.get("valid", False)
        name = p.get("species_name", "(empty)") if valid else "(empty)"
        hp_pct = round(p.get("hp_ratio", 0) * 100) if valid else 0
        level = p.get("level", 0) if valid else 0
        v_str = f"valid:{'1' if valid else '0'}"
        end = offset + POKEMON_BLOCK_DIM - 1
        if valid:
            print(f"    [{offset:>4}-{end:>4}] {key:<10}  {name} Lv{level}  HP:{hp_pct}%  {v_str}")
        else:
            print(f"    [{offset:>4}-{end:>4}] {key:<10}  {C.DIM}{name}{C.RESET}  {v_str}")
        offset += POKEMON_BLOCK_DIM

    field = game_state.get("field", {})
    battle = game_state.get("battle", {})
    w_name = weather_name(field.get("weather_type", 0))
    t_name = terrain_name(field.get("terrain_type", 0))
    blocks = [
        ("field", FIELD_STATE_DIM, f"Weather:{w_name} Terrain:{t_name}"),
        ("battle", BATTLE_META_DIM, f"Wave:{battle.get('wave_index', '?')} Turn:{battle.get('turn', '?')} Money:${battle.get('money', 0)}"),
        ("modphase", MODIFIER_PHASE_DIM, "(reward/shop options)"),
        ("inventory", MODIFIER_INVENTORY_DIM, "(held/party/lapsing/enemy modifiers)"),
        ("derived", DERIVED_FIELDS_DIM, "(type eff, STAB, speed ranks)"),
        ("phase", PHASE_INDICATOR_DIM, str((game_state.get("phase") or {}).get("current_phase", "?"))),
    ]
    for label, size, info in blocks:
        end = offset + size - 1
        print(f"    [{offset:>4}-{end:>4}] {label:<10}  {info}")
        offset += size

    print(f"    {C.DIM}Total: {offset} floats{C.RESET}")


def print_phase_info(game_state: dict):
    """Print EVERY field from the phase section."""
    phase = game_state.get("phase", {})
    if not phase:
        print(f"\n  {C.DIM}(no phase data){C.RESET}")
        return

    print(f"\n  {C.BOLD}Phase Info:{C.RESET}")
    print(f"    Current phase: {phase.get('current_phase', '?')}")
    print(f"    Command field index: {phase.get('command_field_index', None)}")
    print(f"    Command pokemon: {phase.get('command_pokemon_species', None)}")

    # Action mask
    mask = phase.get("action_mask", [])
    if mask:
        valid = [i for i, v in enumerate(mask) if v]
        print(f"    Action mask: {len(valid)}/{len(mask)} valid: {valid}")

    valid_actions = phase.get("valid_actions", [])
    if valid_actions:
        print(f"    Valid actions: {valid_actions}")

    # Learn move
    lm_id = phase.get("learn_move_id")
    if lm_id is not None:
        print(f"    Learn move: #{lm_id} ({phase.get('learn_move_name', '?')})")
        print(f"      Stats: {phase.get('learn_move_stats', None)}")
        print(f"      Current moves: {phase.get('learn_move_current', None)}")

    # Biome options
    biome_opts = phase.get("biome_options")
    if biome_opts is not None:
        print(f"    Biome options: {biome_opts}")

    # Mystery
    mystery = phase.get("mystery_option_count")
    if mystery is not None:
        print(f"    Mystery option count: {mystery}")

    # Game over
    if phase.get("is_game_over") is not None:
        print(f"    Game over: {phase.get('is_game_over')}  Victory: {phase.get('is_victory')}")

    # Action labels
    action_labels = game_state.get("action_labels", [])
    if action_labels:
        print(f"    Action labels ({len(action_labels)}):")
        for al in action_labels:
            print(f"      [{al.get('index', '?')}] {al.get('label', '?')}")

    # Metadata
    step = game_state.get("step")
    ts = game_state.get("timestamp")
    if step is not None:
        print(f"    Step: {step}  Timestamp: {ts}")


def print_modifiers_full(game_state: dict):
    """Print EVERY field from the modifiers section."""
    modifiers = game_state.get("modifiers", {})
    if not modifiers:
        print(f"\n  {C.DIM}(no modifier data){C.RESET}")
        return

    print(f"\n  {C.BOLD}Modifiers (full):{C.RESET}")

    # Held items (show ALL fields)
    held = modifiers.get("held_items", {})
    if isinstance(held, dict):
        total = sum(len(items) for items in held.values())
        print(f"    Held items ({total} total across {len(held)} slots):")
        for slot, items in sorted(held.items()):
            if items:
                for it in items:
                    it_name = it.get("name", "?")
                    stack = it.get("stack_count", 1)
                    max_stack = it.get("max_stack_count", "?")
                    mod_class = it.get("modifier_class", "")
                    mod_id = it.get("modifier_id", "")
                    transferable = it.get("is_transferable", None)
                    print(f"      Slot {slot}: {it_name} x{stack}/{max_stack}  "
                          f"class:{mod_class}  id:{mod_id}  transferable:{transferable}")
                    # All remaining optional fields (include zero/false)
                    shown = {"name", "stack_count", "max_stack_count", "modifier_class",
                             "modifier_id", "is_transferable"}
                    extras = {k: v for k, v in it.items() if k not in shown and v is not None}
                    if extras:
                        print(f"        {C.DIM}{extras}{C.RESET}")
            else:
                print(f"      Slot {slot}: (empty)")

    # Party modifiers (show all fields explicitly)
    party = modifiers.get("party_modifiers", [])
    if party:
        print(f"    Party modifiers ({len(party)}):")
        for pm in party:
            pm_name = pm.get("name", "?")
            pm_class = pm.get("modifier_class", "")
            pm_id = pm.get("modifier_id", "")
            pm_stack = pm.get("stack_count", 1)
            pm_max = pm.get("max_stack_count", "?")
            extras = []
            if pm.get("type_id") is not None:
                extras.append(f"type:{type_name(pm['type_id'])}")
            if pm.get("stat_id") is not None:
                extras.append(f"stat:{pm['stat_id']}")
            if pm.get("status_effect") is not None:
                extras.append(f"status:{status_name(pm['status_effect'])}")
            ext_str = f"  {' '.join(extras)}" if extras else ""
            print(f"      {pm_name} x{pm_stack}/{pm_max}  class:{pm_class}  id:{pm_id}{ext_str}")
    else:
        print(f"    Party modifiers: none")

    # Lapsing (use correct field name: battles_remaining)
    lapsing = modifiers.get("lapsing_modifiers", [])
    if lapsing:
        print(f"    Lapsing modifiers ({len(lapsing)}):")
        for lm in lapsing:
            lm_name = lm.get("name", "?")
            lm_class = lm.get("modifier_class", "")
            lm_id = lm.get("modifier_id", "")
            lm_stack = lm.get("stack_count", 1)
            lm_battles = lm.get("battles_remaining", "?")
            extras = []
            if lm.get("stat_id") is not None:
                extras.append(f"stat:{lm['stat_id']}")
            if lm.get("boost") is not None:
                extras.append(f"boost:{lm['boost']}")
            ext_str = f"  {' '.join(extras)}" if extras else ""
            print(f"      {lm_name} x{lm_stack}  {lm_battles} battles left  "
                  f"class:{lm_class}  id:{lm_id}{ext_str}")

    # Enemy
    enemy = modifiers.get("enemy_modifiers", [])
    if enemy:
        print(f"    Enemy modifiers ({len(enemy)}):")
        for em in enemy:
            em_name = em.get("name", "?")
            em_class = em.get("modifier_class", "")
            em_stack = em.get("stack_count", 1)
            extras = {k: v for k, v in em.items()
                      if k not in ("name", "modifier_class", "stack_count") and v is not None}
            ext_str = f"  {extras}" if extras else ""
            print(f"      {em_name} x{em_stack}  class:{em_class}{ext_str}")


def dump_state(game_state: dict, step: int):
    """Dump full gameState JSON to a file."""
    filename = f"state_dump_{step}.json"
    with open(filename, "w") as f:
        json.dump(game_state, f, indent=2)
    print(f"\n  {C.GREEN}State dumped to {filename} ({os.path.getsize(filename)} bytes){C.RESET}")


def print_inspect_help():
    """Print help for inspection commands."""
    print(f"\n  {C.BOLD}Inspection Commands:{C.RESET}")
    print(f"    {C.CYAN}i{C.RESET} / info     \u2014 Print ENTIRE state (all 12 pokemon + field + battle + phase + mods)")
    print(f"    {C.CYAN}m{C.RESET} / moves    \u2014 Print move details for active player Pokemon")
    print(f"    {C.CYAN}f{C.RESET} / field    \u2014 Print all field state (weather, terrain, hazards, arena tags)")
    print(f"    {C.CYAN}b{C.RESET} / battle   \u2014 Print all battle metadata")
    print(f"    {C.CYAN}p{C.RESET} <slot>     \u2014 Print full Pokemon detail (e.g. 'p 0', 'p e0', 'p player_2')")
    print(f"    {C.CYAN}t{C.RESET} / tags     \u2014 Print volatile tags on all active Pokemon")
    print(f"    {C.CYAN}s{C.RESET} / shop     \u2014 Print shop/reward details")
    print(f"    {C.CYAN}r{C.RESET} / phase    \u2014 Print phase info (action mask, valid actions, etc.)")
    print(f"    {C.CYAN}x{C.RESET} / mods     \u2014 Print full modifier inventory (held, party, lapsing, enemy)")
    print(f"    {C.CYAN}o{C.RESET} / obs      \u2014 Print observation vector layout summary")
    print(f"    {C.CYAN}d{C.RESET} / dump     \u2014 Dump full raw gameState to JSON file")
    print(f"    {C.CYAN}h{C.RESET} / ?        \u2014 Show this help")
    print(f"    {C.CYAN}q{C.RESET} / quit     \u2014 Quit the game")
    print()


def handle_inspect_command(cmd: str, game_state: dict, step: int) -> bool:
    """Handle an inspection command. Returns True if command was handled."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("i", "info"):
        # Print everything: all 12 pokemon slots, field, battle, phase, modifiers
        all_slots = [
            "player_0", "player_1", "player_2", "player_3", "player_4", "player_5",
            "enemy_0", "enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5",
        ]
        for slot in all_slots:
            poke = game_state.get(slot, {})
            if poke.get("valid"):
                print_pokemon_detail(poke, slot)
        print_field_state(game_state)
        print_battle_info(game_state)
        print_phase_info(game_state)
        print_modifiers_full(game_state)
        print_shop_detail(game_state)
        return True

    elif command in ("m", "moves"):
        print_moves_detail(game_state)
        return True

    elif command in ("f", "field"):
        print_field_state(game_state)
        return True

    elif command in ("b", "battle"):
        print_battle_info(game_state)
        return True

    elif command == "p":
        slot = arg.strip() if arg else "player_0"
        # Allow shorthand: "p 0" -> "player_0", "p e1" -> "enemy_1"
        if slot.isdigit():
            slot = f"player_{slot}"
        elif slot.startswith("e") and len(slot) == 2 and slot[1].isdigit():
            slot = f"enemy_{slot[1]}"
        elif slot.startswith("p") and len(slot) == 2 and slot[1].isdigit():
            slot = f"player_{slot[1]}"
        poke = game_state.get(slot, {})
        print_pokemon_detail(poke, slot)
        return True

    elif command in ("t", "tags"):
        print_tags_summary(game_state)
        return True

    elif command in ("s", "shop"):
        print_shop_detail(game_state)
        return True

    elif command in ("r", "phase"):
        print_phase_info(game_state)
        return True

    elif command in ("x", "mods"):
        print_modifiers_full(game_state)
        return True

    elif command in ("o", "obs"):
        print_obs_layout(game_state)
        return True

    elif command in ("d", "dump"):
        dump_state(game_state, step)
        return True

    elif command in ("h", "?", "help"):
        print_inspect_help()
        return True

    return False


def prompt_action(valid_ids: set, game_state: dict = None, step: int = 0) -> int:
    """Prompt for a STABLE action id (one of valid_ids). Returns the chosen id.

    Also handles inspection commands (i, m, f, b, p, t, s, o, d, h).
    """
    ids_hint = ",".join(str(i) for i in sorted(valid_ids))
    while True:
        try:
            raw = input(f"\n  {C.CYAN}Action id [{ids_hint}] or cmd (h=help): {C.RESET}").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print("Quitting...")
                sys.exit(0)

            # Try as a stable action id first
            try:
                choice = int(raw)
                if choice in valid_ids:
                    return choice
                print(f"  {C.RED}Not a legal action id. Legal: {ids_hint}{C.RESET}")
                continue
            except ValueError:
                pass

            # Try as inspection command
            if game_state and raw:
                if handle_inspect_command(raw, game_state, step):
                    continue

            print(f"  {C.RED}Unknown command '{raw}'. Enter an action id or 'h' for help.{C.RESET}")

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

    cmd = ["node", cli_path, "--interactive", *args.run_config.to_cli_args()]

    print(f"{C.BOLD}=== PokeRogue Headless Player ==={C.RESET}")
    print(f"  Seed:  {args.seed}")
    print(f"  Waves: {args.waves}")
    if args.run_config.source:
        print(f"  Config: {args.run_config.source}")
    print(f"  Quit:  type 'q' at any prompt")
    print(f"  Help:  type 'h' at any prompt for inspection commands")
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
                print_moves_compact(game_state)
                print_party(game_state)
                valid_ids = print_actions(actions)
                action = prompt_action(valid_ids, game_state, step)

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

    url = f"http://localhost:{args.port}/?rl=true{args.run_config.to_url_query()}"
    ws_url = f"ws://localhost:{args.port}/ws/rl"

    print(f"{C.BOLD}=== PokeRogue Rendered Player ==={C.RESET}")
    print(f"  Port:  {args.port}")
    print(f"  Seed:  {args.seed or '(random)'}")
    print(f"  URL:   {url}")
    print(f"  Quit:  type 'q' at any prompt")
    print(f"  Help:  type 'h' at any prompt for inspection commands")
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

    started = False
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
                proto = msg.get("protocolVersion")
                if proto is not None and proto != PROTOCOL_VERSION:
                    raise SystemExit(
                        f"browser bundle speaks protocol {proto}, this client needs {PROTOCOL_VERSION} — "
                        "hard-reload the browser tab (Ctrl+Shift+R) and rerun"
                    )
                if started:
                    print(f"\n  {C.RED}Browser session restarted (page reload?) — "
                          f"starting a FRESH episode.{C.RESET}\n")
                print(f"  {C.GREEN}Game ready! Sending start signal...{C.RESET}\n")
                started = True
                ws.send(json.dumps({"type": "start"}))

            elif msg_type == "state":
                step = msg.get("step", 0)
                phase = msg.get("phase", "unknown")
                game_state = msg.get("gameState", {})
                actions = msg.get("actions", [])

                print_header(step, phase, game_state)
                print_field(game_state)
                print_moves_compact(game_state)
                print_party(game_state)
                valid_ids = print_actions(actions)
                action = prompt_action(valid_ids, game_state, step)

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
    parser.add_argument("--config", default=None,
                        help="run-config YAML/JSON (see src/rl/run_config.py); CLI flags override it")
    parser.add_argument("--seed", default=None, help="RNG seed (prompts if not provided)")
    parser.add_argument("--waves", type=int, default=None, help="Max waves (default: 50)")
    parser.add_argument("--starters", default=None,
                        help="comma-separated SpeciesId names, e.g. MEWTWO,LUGIA,RAYQUAZA")
    parser.add_argument("--rendered", action="store_true",
                        help="Connect to browser game via WebSocket (requires Vite dev server)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Vite dev server port for --rendered mode (default: 8000)")
    args = parser.parse_args()

    # Config file first, CLI flags override its values.
    cfg = load_run_config(args.config) if args.config else RunConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.waves is not None:
        cfg.waves = args.waves
    if args.starters is not None:
        cfg.starters = args.starters

    # Prompt for seed if not provided anywhere
    if cfg.seed is None:
        try:
            seed = input(f"{C.CYAN}Enter seed (or press Enter for random): {C.RESET}").strip()
            cfg.seed = seed if seed else f"play-{os.urandom(4).hex()}"
        except (EOFError, KeyboardInterrupt):
            cfg.seed = f"play-{os.urandom(4).hex()}"
            print()
    if cfg.waves is None:
        cfg.waves = 50

    args.run_config = cfg
    args.seed = cfg.seed
    args.waves = cfg.waves

    if args.rendered:
        run_rendered(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
