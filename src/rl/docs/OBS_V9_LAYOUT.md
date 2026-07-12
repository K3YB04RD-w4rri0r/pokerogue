# PokeRogue RL — Observation Protocol v9 Layout Design (protocol 4 → 5)

**Status:** IMPLEMENTED and shipped (protocol 5 live on both encoders);
originally design-complete; numbers verified against the rl-framework
branch and `.rl-audit/redundancy-ledger.json` (3,769-decision corpus;
6,242/10,403 dims dead, 804 exact-dup, 128 low-info).
**Wire authority:** `src/rl/spaces.ts`; `src/rl/observation.py` is the
bitwise mirror.
**Headline:** `OBSERVATION_DIM` **10,403 → 6,991** (−3,412, −32.8%).
Action space untouched (58). Protocol version 4→5 (`src/rl/cli.ts`
protocolVersion; `src/rl/pokerogue_env.py` PROTOCOL_VERSION).

Savings ledger (nets to −3,412): move-block cut −3,648 (48 move slots ×
−76) · volatile-tag cut −84 (12 × −7) · new per-pokemon dims +108 (12 × 9)
· field +8 · shop tail +138 (6 × 23) · learn_move block +66.

User decisions baked in: fog-of-war ships as a toggle DEFAULTING to
full-info (layout is fog-ready); move vector cut to the evidence
keep-list; ONE fixed layout (no layout toggles — the escape hatch for
exotic features is a gym ObservationWrapper over `info["game_state"]`,
which continues to serialize everything).

---

## 1. Compact move vector — `MOVE_BLOCK_DIM = 60` (was 136)

Aliveness metric = varies across corpus AND not an exact byte-duplicate
(ledger `dead` + `duplicate_groups`). Keep line = 30%.

Exact order (offsets within the 60-dim vector):

| off | field | w | notes |
|---|---|---|---|
| 0 | valid | 1 | structural |
| 1–19 | type_onehot | 19 | categorical stays one-hot |
| 20–22 | category_onehot | 3 | 94% alive |
| 23 | power | 1 | /250, v8 convention |
| 24 | accuracy | 1 | /100 (≤0 ⇒ 1.0) |
| 25 | pp_ratio | 1 | |
| 26 | priority | 1 | /7 clamp −1..1 |
| 27 | effect_chance | 1 | /100 |
| 28 | drain_ratio | 1 | 46% |
| 29 | heal_ratio | 1 | 50% |
| 30 | **multi_hit_count** | 1 | **NEW** — replaces `is_multi_hit` |
| 31 | force_switch | 1 | 31% (Roar family) |
| 32 | is_protect | 1 | 67% |
| 33 | traps_target | 1 | 50% |
| 34 | makes_contact | 1 | 75% |
| 35 | is_usable | 1 | 42%, mask-adjacent |
| 36 | status_effect | 1 | /7, 92% |
| 37 | stat_change_self_sum | 1 | /12, 94% |
| 38 | stat_change_target_sum | 1 | /12, 98% |
| 39 | recoil_ratio | 1 | 54% |
| 40 | crit_stage_boost | 1 | /3, 56% |
| 41–43 | target_class_onehot | 3 | 90% |
| 44 | ignores_protect | 1 | 65% |
| 45 | is_sound_based | 1 | 83% |
| 46 | can_flinch | 1 | 77% |
| 47 | can_confuse | 1 | 77% |
| 48 | has_variable_power | 1 | 90% |
| 49 | weather_change | 1 | /9, 31% |
| 50 | sets_arena_tag | 1 | 40% |
| 51 | applies_battler_tag | 1 | 77% |
| 52 | applies_move_restriction | 1 | 50% |
| 53 | is_wind_move | 1 | 56% |
| 54 | is_reckless_move | 1 | 33% |
| 55 | is_reflectable | 1 | 88% |
| 56 | is_triage_move | 1 | 50% |
| 57 | steals_item | 1 | 50% |
| 58 | hits_semi_invulnerable | 1 | 38% |
| 59 | **has_other_effect** | 1 | **NEW** catch-all |

Sum: 46 base + 13 kept tail flags + 1 catch-all = **60**.

**multi_hit_count encoding** (from serialized `multi_hit_type`, enum
`src/enums/multi-hit-type.ts`): `count/5` where count = 0 if not
multi-hit (`multi_hit_type = -1`); TWO(0)→2; TWO_TO_FIVE(1)→5; THREE(2)→3;
TEN(3)→clamp→5; BEAT_UP(4)→clamp→5. Values ∈ {0, 0.4, 0.6, 1.0}; 0 = not
multi-hit preserves the old boolean signal losslessly.

**has_other_effect (exact OR-set, 77 flags):** 1.0 iff any of the
following v8 booleans would have been 1 —
base-section cuts (4): self_switch, is_ohko, is_charging, is_sacrifice;
v6 tail cuts (25): is_recharge, is_frenzy, is_typeless,
creates_substitute, suppresses_ability, has_variable_type,
has_variable_category, bypass_burn_penalty, ignores_stat_stages,
terrain_change, removes_arena_tags, sets_hazard, sets_screen,
arena_tag_self_side, applies_continuous_damage, is_user_hp_damage,
is_target_half_hp, is_counter_damage, is_level_damage, is_delayed_attack,
post_victory_stat_boost, hides_user, hides_target, check_all_hits,
affected_by_gravity;
v7 tail cuts (45): steals_berry, removes_item, copies_stats,
inverts_stats, resets_stats, swaps_stat_stages, steals_stat_boosts,
averages_stats, swaps_single_stat, shifts_own_stat, splits_hp, reduces_pp,
revives_ally, copies_last_move, calls_random_move, calls_moveset_move,
copies_move_temp, copies_move_perm, copies_ability, swaps_abilities,
changes_ability, gives_ability, suppresses_if_acted, bypass_redirect,
forces_target_next, forces_target_last, has_conditional_priority,
cures_party_status, transfers_status, heals_status, removes_battler_tag,
removes_substitutes, transforms_into_target, is_curse, is_wish,
is_destiny_bond, swaps_arena_tags, clears_weather, clears_terrain,
has_variable_target, resists_last_type, has_variable_accuracy,
uses_alt_stat, overrides_type_chart, scatters_money;
v8 tail cuts (3): survives_at_1hp, matches_user_hp, hp_cost_stat_boost.
(4+25+45+3 = 77; 136 − 77 + 1 = 60 ✓.)

Notes: weather_change and terrain_change were scalars in v8;
weather_change is KEPT as a scalar; terrain_change (2% alive) folds into
has_other_effect as `terrain_change != 0`. Borderline cuts documented:
`self_switch` (27%) is the top v10 re-add candidate if pivot play emerges;
is_charging/is_recharge/is_frenzy cuts are cushioned because the
corresponding STATES stay observable via kept volatile tags
CHARGING/RECHARGING/FRENZY.

---

## 2. Pokemon block — `POKEMON_BLOCK_DIM = 513` (was 815) = 273 non-move + 4×60

Uniform across all 12 slots. Exact non-move order:

| off | field | w | change vs v8 |
|---|---|---|---|
| 0 | valid | 1 | |
| 1 | hp_ratio | 1 | |
| 2 | level | 1 | /100 |
| 3–8 | base_stats | 6 | /255 |
| 9–15 | stat_stages | 7 | /6 |
| 16–34 | type1_onehot | 19 | |
| 35–53 | type2_onehot | 19 | keep (42%, bench-confounded) |
| 54–61 | status_onehot | 8 | |
| 62–66 | nature_mults | 5 | fog-maskable (enemy) |
| 67–106 | ability_features | 40 | keep (confounded) |
| 107–146 | passive_features | 40 | keep |
| 147 | is_terastallized | 1 | |
| 148–166 | tera_type_onehot | 19 | keep (see tera decision) |
| 167–235 | volatile_tags | **69** | −7 turn-transient tags |
| 236 | other_tag_count | 1 | /10 |
| 237–258 | is_boss, boss_shield_ratio, is_trapped, is_grounded, weight, catch_rate, is_fainted, wave_turn_count, damage_taken, acted, toxic_turn_count, sleep_turns_remaining, held_item_count, species_id, gender, friendship, move_queue_len, hit_count, ability_suppressed, is_mega, is_max, move_effectiveness | 22 | unchanged v8 order |
| 259–263 | computed_stats | 5 | /500; fog-maskable (enemy) |
| 264–266 | **ai_type_onehot** | 3 | **NEW**: RANDOM/SMART_RANDOM/SMART; all-zero on player slots |
| 267–270 | **move_known[0..3]** | 4 | **NEW** revealed-indicator |
| 271 | **ability_known** | 1 | **NEW** |
| 272 | **was_seen** | 1 | **NEW**; players 0, enemies 1 under full obs |
| 273–512 | moves[0..3] | 240 | 4 × 60 |

Check: 271 − 7 + 3 + 6 = 273; 273 + 240 = **513**; 12 × 513 = **6,156**.

**Volatile tag curation.** Cut exactly 7 tags — TURN_END-removed every
turn, structurally unobservable at decision boundaries, 0/12 alive:
**FLINCHED, PROTECTED, ENDURING, HELPING_HAND, MAGIC_COAT, POWDER,
CENTER_OF_ATTENTION**. Keep the remaining 69 in v8 relative order.
Notable keeps despite 0-corpus-aliveness (persistent, decision-visible):
SEEDED, INGRAIN, AQUA_RING, PERISH_SONG, CURSED, SALT_CURED, OCTOLOCK,
DROWSY, MINIMIZED, DESTINY_BOND, RECHARGING, FLYING, UNDERGROUND,
UNDERWATER, HIDDEN, ALWAYS_CRIT, ALWAYS_GET_HIT, GRUDGE.

**Tera decision: keep is_terastallized(1) + full tera_type_onehot(19) on
all 12 slots.** The 85-dim byte-duplication with type1 is semantic
correlation (most mons tera into their own type), not dead structure; a
differs-flag would destroy exact type identity precisely when tera
matters. Duplication is cheap for the network; ambiguity is not.

**Slot-mapping change (zero dims):** in **singles**, the ally slot doubles
as first bench on both sides: `party[1]→player_1/enemy_1`,
`party[2..5]→slots 2..5`; doubles unchanged. New semantics: *slot 1 =
second active in doubles, first bench member in singles*. Makes party[5]
encodable for the first time. Encoder slot-assignment only; action space,
switch mapping and mask building untouched.

---

## 3. Field / battle / modifier / derived / phase — changes only

- **field: 94 → 102.** Dims 0–93 identical to v8. Append (from
  `field.positional_tags`, side via target_index 0–1 player / 2–3 enemy):
  94 player_wish_active · 95 player_wish_turns (countdown/8) ·
  96 player_future_sight_active · 97 player_future_sight_turns (/8) ·
  98–101 the same four for the enemy side. Wish = PENDING_HEAL entries;
  Future Sight/Doom Desire = DELAYED_ATTACK entries. Multiple pending on a
  side: active=1, turns = min countdown.
- **battle_meta: 40, unchanged.**
- **modifier_phase: 225 → 363** = header 3 + rewards 3×28 + **shop 12×23**.
  MAX_SHOP_OPTIONS_ENCODED 6→12; slot layout unchanged; shop slots 7–12
  now visible to actions 40–51.
- **modifier_inventory: 220, unchanged.** **derived: 28, unchanged.**
  **phase_onehot: 16, unchanged.**

---

## 4. Top-level layout — `OBSERVATION_DIM = 6,991`

| block | dims | base | end (excl) |
|---|---|---|---|
| player_0 | 513 | 0 | 513 |
| player_1 | 513 | 513 | 1026 |
| enemy_0 | 513 | 1026 | 1539 |
| enemy_1 | 513 | 1539 | 2052 |
| player_2 | 513 | 2052 | 2565 |
| player_3 | 513 | 2565 | 3078 |
| player_4 | 513 | 3078 | 3591 |
| player_5 | 513 | 3591 | 4104 |
| enemy_2 | 513 | 4104 | 4617 |
| enemy_3 | 513 | 4617 | 5130 |
| enemy_4 | 513 | 5130 | 5643 |
| enemy_5 | 513 | 5643 | 6156 |
| field | 102 | 6156 | 6258 |
| battle_meta | 40 | 6258 | 6298 |
| modifier_phase | 363 | 6298 | 6661 |
| modifier_inventory | 220 | 6661 | 6881 |
| derived | 28 | 6881 | 6909 |
| **learn_move (NEW)** | **66** | **6909** | **6975** |
| phase_onehot | 16 | 6975 | **6991** |

**learn_move block (66 = 60 + 6):** offsets 0–59 = the OFFERED move as a
compact move vector (its `valid` dim doubles as offer-active; all 66 dims
zero outside learn_move phase); 60–65 = learner party-index one-hot.
Sourced from `phase.learn_move_stats` + new `learn_move_party_index`.
Placement after derived keeps battle-stable blocks contiguous and
phase_onehot as the conventional tail. Implementation prerequisite:
phase-router must WRITE `metadata.learnMoveStats` +
`metadata.learnMovePartyIndex` (currently learn_move_stats is null).

---

## 5. Fog-of-war toggle (ships default-FULL; layout is fog-READY)

Plumbing: `encodeObservation(gameState, opts?: { fogOfWar?: boolean })` ←
cli.ts/browser-bridge run option ← `pokerogue_env.py` kwarg
`fog_of_war: bool = False` ← run_config field. Python:
`encode_observation(state, fog_of_war=False)`. State-builder additions:
per-pokemon `was_seen`, `move_known[4]`, `ability_known` (derived from
`battle.seenEnemyPartyMemberIds`, `move_history`, `ability_revealed`).

Masking rules (fog mode; player slots and non-pokemon blocks never
masked):

| dim group (enemy slots) | full obs | fog |
|---|---|---|
| move blocks 273–512 | encoded | move j zeroed unless in move_history; move_known[j]=0/1 |
| ability/passive features 67–146 | encoded | zeroed unless ability_revealed; ability_known=0/1 |
| nature_mults 62–66 | encoded | zeroed (IV/nature-derived) |
| computed_stats 259–263 | encoded | zeroed (IV/nature-derived) |
| whole bench block if !was_seen | encoded, was_seen=1 | all 513 dims zero (incl. valid; was_seen=0) |
| ai_type 264–266 | encoded | encoded (engine meta, behaviorally inferable) |
| indicators 267–272 | constant (1 enemies / 0 players) | actual revealed status |
| tera_type (unterastallized enemy) | encoded | encoded — documented leak |
| derived speed_ordering | encoded | encoded — documented approximation |

---

## 6. Acceptance criteria

1. Bitwise TS↔Python parity on regenerated goldens, fog off AND on.
2. Layout canary green against §2/§4 offsets.
3. Ledger re-run on the same corpus: <10% dead after exclusions (fog
   indicators under full obs + player-slot ai_type; kept
   persistent-but-unexercised volatile tags; never-drawn one-hot lanes;
   mode-constant battle flags; structurally-off is_mega/is_max +
   player-side boss dims; learn_move block if corpus lacks learn_move
   decisions).
4. All RL tests pass; smoke episode bit-identical (74 steps, 50.74 — the
   cross-version anchor; auto-mode reads mask/gameState, not the obs).
