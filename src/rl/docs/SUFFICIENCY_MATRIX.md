# Obs-v9 Sufficiency Matrix (Phase 2) — WORKING DRAFT

> **Resolved by v9 (2026-07):** the ❌ rows below described the v8 layout and
> drove the v9 design — learn-move features, multi_hit, ai_type, 12 shop
> slots and the singles bench remap are all encoded now (see
> OBS_V9_LAYOUT.md). Rows are kept as the design rationale, not as the
> current state.

For each decision type: the inputs the GAME's own computation uses, traced
to observation dims. Status: ✅ encoded · ⚠️ partial/approximated · ❌ missing.
Every row cites the game-code source. Rows marked TODO(verify) still need
code-grounding; do not design v9 off them.

## Command phase — move choice

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Move base power/type/category/accuracy/PP/priority | move data | `*/moves[i]/*` | ✅ |
| True type effectiveness vs target | `getAttackTypeEffectiveness` (Freeze-Dry, Strong Winds, Tera Shell, inverse challenge) | `derived/type_eff*` uses a STATIC chart | ⚠️ ignores move-specific overrides, abilities (Levitate/Wonder Guard), Strong Winds, inverse battles |
| Effective ATK/DEF/SPATK/SPDEF (items, abilities, Ruin, paralysis…) | `getEffectiveStat` (pokemon.ts:1477): StatBoosterModifier items, FieldMultiplyStat (Ruin), StatMultiplierAbAttr, ally abattrs, … | raw `stats[1..5]` + stat stages + ability feature vecs | ⚠️ item/ability multipliers not folded; agent must learn them from ability features |
| Speed order incl. paralysis, Tailwind, Trick Room, Quick Claw | TurnStartPhase ordering | `derived/speed_rank` (raw stat × stage only) | ⚠️ ignores paralysis/Tailwind/Trick Room (TR is elsewhere as a flag) |
| Secondary effect chances (flinch/status/stat) | move attrs | `effect_chance`, `can_flinch`, `status_effect`, `stat_changes` | ✅ |
| Crit chance factors | crit stages, items, abilities | `crit_stage_boost`, item features, CRIT_BOOST tag | ✅/⚠️ (no combined chance) |
| Target's remaining HP (exact) | live | `hp_ratio` (+ exact hp via max_hp) | ✅ (omniscient; see fog-of-war) |
| Boss shield segments | `bossSegments/SegmentIndex` | `is_boss`, `boss_shield_ratio` | ✅ |
| Own move usability reasons (Disable/Encore/Taunt) | `isUsable` | `is_usable` + volatile tags | ✅ |

## Command phase — catch decision

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Catch-rate formula inputs (maxHP, curHP, catchRate, ball mult, status) | attempt-capture-phase.ts (logged formula) | `catch_rate`, `hp_ratio`, `status_onehot`, ball counts | ✅ |
| Ball legality (boss shields, END biome, 1-target) | `checkCanUseBall`/`handleBallCommand` | mask (17-21) + `battle/can_catch` | ✅ (fixed 2026-07-07) |
| Party-full consequence (auto-release flow) | AttemptCapturePhase | not encoded | ❌ minor |

## Command phase — flee decision

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Speed ratio, boss flag, escape attempts | AttemptRunPhase formula | speeds ⚠ (see above), `is_boss`, `escape_attempts` | ⚠️ |

## Switch decision (voluntary + forced)

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Bench member stats/types/moves | party | `player_2..5/*` | ⚠️ 6th member invisible in singles (structural, v9) |
| Incoming hazard damage (Spikes/SR effectiveness vs member) | arena tags + type chart | hazard layers ✅; per-member SR damage not precomputed | ⚠️ learnable |
| Trapped legality | `isTrapped` | mask + `is_trapped` | ✅ |

## Shop / reward decision

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Reward option identity/tier | typeOptions | 3 × 28 dims (tier, features) | ✅ |
| Shop item identity/cost/affordability | shop options | 6 × 23 dims, natural order | ⚠️ slots 7-12 buyable but unencoded; items 13-14 (wave 171+) have no actions |
| True prices (Black Sludge) | HealShopCostModifier | fixed 2026-07-07 | ✅ |
| Reroll cost/lock state | `getRerollCost`, lockModifierTiers | header dims + `lock_modifier_tiers` | ✅ |
| Money horizon (upcoming waves' income) | money formula | `money` only | ⚠️ learnable |
| Current held-item stacks (diminishing returns) | inventory | 2 items/slot encoded of up to N | ⚠️ top-2 only |

## Learn-move decision

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| OFFERED move's features | LearnMovePhase.moveId | **not encoded** (only in dict: learn_move_id/name) | ❌ CRITICAL (v9) |
| Current moveset features | moveset | `*/moves` | ✅ |

## Structural / cross-cutting

| Item | Status |
|---|---|
| 6th party member (both sides) invisible in singles | ❌ v9 slot redesign |
| Positional tags (Wish/Future Sight) captured in dict, not encoded | ❌ v9 |
| Lapsing modifier durations (top-1 only) | ⚠️ |
| Enemy info omniscience (moves/IVs/nature/ability from wave start) | fog-of-war design (plan §3) |
| `ability_revealed`, `seenEnemyPartyMemberIds` tracked but unused | inputs for fog-of-war |

## Accuracy pipeline (verified 2026-07-07, agent-audited with citations)

Hit roll: `moveAccuracy * accuracyMultiplier` (move-effect-phase.ts:453);
`calculateBattleAccuracy` (move.ts:1033-1068) + `getAccuracyMultiplier`
(pokemon.ts:3475-3546).

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| User ACC / target EVA stat stages | pokemon.ts:3481-3482 | stat_stages[5]/[6] (spaces.ts:743) | ✅ |
| Gravity ×1.67 | move.ts:1063 | field gravity + GRAVITY arena tag | ✅ |
| Fog ×0.9 | move.ts:1055 | weather one-hot(10) | ✅ |
| Lock-On/Telekinesis/Exposed tags | mep.ts:441-444; pokemon.ts:3497 | curated volatile tags | ✅ |
| Wide Lens | move.ts:1052 | generic item feature, top-2-items channel only | ⚠️ |
| X Accuracy (temp booster) | pokemon.ts:3492 | lapsing_modifiers top-1 only | ⚠️ |
| ACC/EVA abilities (Compound Eyes, Sand Veil, Hustle, Victory Star) | pokemon.ts:3509-3536 | 40-dim ability features, no dedicated acc/eva dim | ⚠️ |

## Weather / terrain damage (verified)

Weather ×1.5/×0.5 fire-water (weather.ts:79-102); terrain ×1.3 grounded
(terrain.ts:47-67). Weather type+turns, terrain type+turns all encoded
(spaces.ts:917-928). ✅ complete.

## Multi-hit & items (verified)

| Required input | Game source | Obs dims | Status |
|---|---|---|---|
| Move is multi-hit | MultiHitAttr | `is_multi_hit` flag | ✅ |
| Hit count/type (2-5) | state-builder.ts:920 serializes `multi_hit_type` | **dropped by encoder** | ❌ v9 |
| Multi Lens stacks (max 2, dmg split) | modifier.ts:2697-2776 | MULTI_LENS feature via top-2-items channel | ⚠️ |

## Enemy AI type (verified) — NEW GAP

`AiType` RANDOM/SMART_RANDOM/SMART (enums/ai-type.ts) drives enemy move
selection (pokemon.ts:6664-6812); default SMART for boss/trainer.
state-builder serializes `ai_type` (state-builder.ts:1174) but **spaces.ts
never encodes it** — the agent cannot distinguish a uniform-random wild mon
from a score-maximizing boss. ❌ v9 (3-dim one-hot per enemy).

## PP economics (verified)

Ether row 0 every shop wave; Elixir/Max Ether wave ≥ 51; Max Elixir ≥ 111
(modifier-type.ts:2629-2668). Offered ones encode via shop_options ✅; PP
dims per move ✅. Complete for the encoded 6 slots (see shop-tail gap).
