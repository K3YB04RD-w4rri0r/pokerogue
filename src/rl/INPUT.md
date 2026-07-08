# PokéRogue RL Agent — Complete Game State & Decision Reference

This document contains EVERYTHING an RL agent needs to understand, observe, and act
in a PokéRogue game. It covers all game state accessible from `globalScene` and
`BattleScene`, every decision point with its action space, and the mechanics that
determine battle outcomes.

Generated: 2026-02-09 by research team (8 parallel agents).

---

## Table of Contents

1. [Observation Space Overview](#1-observation-space-overview)
2. [Pokemon State](#2-pokemon-state)
3. [Field & Arena State](#3-field--arena-state)
4. [Battle & Run State](#4-battle--run-state)
5. [Modifier / Item State](#5-modifier--item-state)
6. [Move Mechanics](#6-move-mechanics)
7. [Ability Mechanics](#7-ability-mechanics)
8. [Decision Points & Action Space](#8-decision-points--action-space)
9. [Difficulty Scaling & Formulas](#9-difficulty-scaling--formulas)
10. [PokéRogue-Specific Mechanics](#10-pokérogue-specific-mechanics)
11. [Current Encoding Gaps](#11-current-encoding-gaps)

---

## 1. Observation Space Overview

Current encoding: **6,991 float32** values (**v9**, protocol 5 — verified
TS↔Python bit-identical; full design + audit evidence in
`docs/OBS_V9_LAYOUT.md`, `docs/AUDIT_FINDINGS_P1.md`).

| Block | Dims | Base | Content |
|-------|------|------|---------|
| Pokemon ×12 | 6,156 | 0 | 12 slots × 513 dims (273 non-move + 4 moves × 60; slot order: player_0, player_1, enemy_0, enemy_1, player bench ×4, enemy bench ×4) |
| Field State | 102 | 6,156 | Weather/terrain one-hots + turns, per-side arena tag banks (28×2), hazard layers, key-tag turn counters, teras used, **Wish/Future-Sight per side (v9)** |
| Battle Meta | 40 | 6,258 | Wave, turn, money, pokeballs, alive/faint counts, biome, game-mode flags |
| Modifier Phase | 363 | 6,298 | Header (3) + 3 reward options × 28 + **12 shop options × 23 (v9: all buy actions observable)** |
| Modifier Inventory | 220 | 6,661 | Held items (4 active slots × 45), party mods, lapsing, enemy aggregates |
| Derived | 28 | 6,881 | Type effectiveness (2×4×2), STAB (2×4), speed ranks (4) |
| **Learn-Move (v9)** | 66 | 6,909 | The OFFERED move as a compact move vector + learner party-index one-hot; zero outside the learn_move phase |
| Phase Indicator | 16 | 6,975 | One-hot decision phase |

**v9 changes** (protocol 4 → 5; evidence-based redesign):
- Per-move vector 136 → **60**: core scalars/one-hots + the ~15 effect
  flags the audit found alive; ~77 near-dead flags fold into ONE
  `has_other_effect` catch-all. NEW `multi_hit_count` (was a boolean).
- Volatile-tag bank 76 → **69**: 7 TURN_END-transient tags cut (they
  lapse before every decision and were structurally unobservable).
- NEW per-Pokemon: `ai_type_onehot(3)` (RANDOM/SMART_RANDOM/SMART — the
  enemy's move-selection intelligence), `move_known(4)`,
  `ability_known(1)`, `was_seen(1)` revealed-indicators.
- **Singles slot mapping**: slot 1 = second active in doubles, FIRST
  BENCH member in singles — the 6th party member (both sides) is now
  observable (it was silently dropped before v9).
- **Fog of war** (`fog_of_war=True` env kwarg / `--fog-of-war` /
  `&fog=1`): enemy unseen moves, unrevealed abilities, IV/nature-derived
  values and never-seen bench members are zeroed; the indicators carry
  what is known. Default OFF (full information).

Cut dims are NOT lost: `state-builder.ts` still serializes everything into
`info["game_state"]` — a Python `ObservationWrapper` can append any field
without touching the protocol (see README "Bring your own features").

Action space: **58 discrete actions** with validity mask (unchanged).
Python-side dim names for every index: `src/rl/feature_names.py`
(`dim_to_name(i)` / `name_to_dim(name)`).

---

## 2. Pokemon State

### 2.1 Identity & Species

```typescript
pokemon.species.speciesId    // SpeciesId enum (0-1025+)
pokemon.formIndex            // 0+ (Mega, Alolan, etc.)
pokemon.level                // 1-100
pokemon.gender               // Gender: MALE=0, FEMALE=1, GENDERLESS=2
pokemon.friendship           // 0-255 (affects Return/Frustration)
pokemon.pokeball             // PokeballType (0-5)
```

Species-level data (from `pokemon.getSpeciesForm()`):
```typescript
speciesForm.baseStats        // number[6]: [HP, ATK, DEF, SPATK, SPDEF, SPD]
speciesForm.type1             // PokemonType (primary)
speciesForm.type2             // PokemonType | null (secondary)
speciesForm.ability1          // AbilityId
speciesForm.ability2          // AbilityId
speciesForm.abilityHidden     // AbilityId
speciesForm.baseTotal         // number (BST)
speciesForm.catchRate         // number
speciesForm.weight            // kg (affects Low Kick, Heavy Slam)
```

### 2.2 Stats

**IVs** (0-31 each, no EVs in PokéRogue):
```typescript
pokemon.ivs[Stat.HP]         // 0-31
pokemon.ivs[Stat.ATK]        // ...through Stat.SPD
```

**Calculated stats** (stored):
```typescript
pokemon.stats[Stat.HP]       // Permanent calculated stat
pokemon.getStat(Stat.ATK)    // True stat (ignores Transform)
pokemon.getStat(Stat.ATK, false)  // In-battle stat (uses Transform overrides)
```

**Stat formula**:
```
HP:    floor((2 * base + IV) * level / 100) + level + 10
Other: floor(floor((2 * base + IV) * level / 100) + 5) * natureMultiplier)
```

Base stats modified by: Vitamins (`BaseStatModifier`), Old Gateau, Shuckle Juice,
fusion averaging, Challenge mode (FLIP_STAT).

**Nature** (25 natures, 0-24):
```typescript
pokemon.nature               // Original nature
pokemon.getNature()          // Effective nature (mint override)
getNatureStatMultiplier(nature, stat)  // 1.0, 1.1, or 0.9
```

**Effective in-battle stats** (full calculation):
```typescript
pokemon.getEffectiveStat(Stat.ATK, opponent?, move?)
// Includes: stat stages, held items, field abilities, own abilities,
// ally abilities, weather bonuses, paralysis penalty, Slow Start,
// Unburden, Tailwind, Protocosm/Protosynthesis
```

### 2.3 Stat Stages

```typescript
pokemon.getStatStage(Stat.ATK)   // -6 to +6
pokemon.getStatStages()          // number[7]: [ATK, DEF, SPATK, SPDEF, SPD, ACC, EVA]
// Stored in pokemon.summonData.statStages (indexed by stat-1, since HP=0 has no stage)
```

Stage multiplier: `max(2, 2+stage) / max(2, 2-stage)`, capped at 4.0

### 2.4 HP & Health

```typescript
pokemon.hp                   // Current HP
pokemon.getMaxHp()           // = pokemon.getStat(Stat.HP)
pokemon.getHpRatio()         // hp/maxHp (rounded to 0.01)
pokemon.isFullHp()           // hp >= maxHp
pokemon.isFainted()          // hp <= 0
pokemon.isActive()           // Not fainted AND allowed by challenges
```

### 2.5 Status Conditions (Non-Volatile)

```typescript
pokemon.status?.effect       // StatusEffect: NONE=0, POISON=1, TOXIC=2,
                             //   PARALYSIS=3, SLEEP=4, FREEZE=5, BURN=6, FAINT=7
pokemon.status?.toxicTurnCount      // Toxic damage multiplier
pokemon.status?.sleepTurnsRemaining // Turns until wake
```

### 2.6 Volatile Status (Battler Tags)

Stored in `pokemon.summonData.tags` (BattlerTag[]). Reset on switch-out.

```typescript
pokemon.getTag(BattlerTagType.X)    // BattlerTag | undefined
pokemon.findTags(predicate)         // BattlerTag[]
```

**96 distinct BattlerTagType values.** Key ones for RL:

| Tag | Effect | RL Impact |
|-----|--------|-----------|
| SUBSTITUTE | Blocks damage (has `.hp`) | Absorbs hits |
| CONFUSED | May hit self (33%) | Risky to attack |
| FLINCHED | Can't move this turn | Wasted turn |
| TRAPPED / BIND / WRAP / FIRE_SPIN | Can't switch | Limits options |
| ENCORE | Forced to use same move | Predictable |
| DISABLED | Specific move disabled | Reduced moveset |
| TAUNT | Can't use status moves | Setup denied |
| TORMENT | Can't repeat moves | Forces variation |
| CURSED | Loses 25% HP/turn | Ticking clock |
| PERISH_SONG | Faints after countdown | Must switch |
| LEECH_SEED | HP drain each turn | Slow death |
| SLOW_START | -50% ATK/SPD 5 turns | Huge debuff |
| UNBURDEN | 2x SPD | Speed doubled |
| FLYING/UNDERGROUND/UNDERWATER | Semi-invulnerable | Can't be hit normally |
| CHARGED | Next Electric 2x | Setup payoff |
| CRIT_BOOST / FOCUS_ENERGY | +crit stage | Higher damage |
| PROTECT | Blocks moves this turn | Full protection |
| TAR_SHOT | Fire moves 2x | Type weakness added |
| SALT_CURED | DOT each turn | Persistent damage |
| AQUA_RING | Heals each turn | Sustain |
| INGRAIN | Heals + can't switch | Trade-off |
| ENDURING | Survive at 1 HP | Safety net |

Each tag has: `tagType`, `turnCount`, `sourceMove`, `sourceId`, `isBatonPassable`

### 2.7 Types

```typescript
pokemon.getTypes()                    // PokemonType[] (1-3 elements)
pokemon.getTypes(true)                // Includes tera type if active
pokemon.isOfType(PokemonType.FIRE)    // Boolean check
pokemon.teraType                      // Assigned tera type
pokemon.isTerastallized               // Boolean
pokemon.stellarTypesBoosted           // PokemonType[] (Stellar boost tracker)
pokemon.summonData.addedType          // From Forest's Curse / Trick-or-Treat
```

**PokemonType enum**: 19 valid (0=NORMAL through 18=STELLAR), UNKNOWN=-1

Type effectiveness: `pokemon.getAttackTypeEffectiveness(moveType, { source })` handles
multi-type, Freeze-Dry, Strong Winds, Tera Shell, etc.

### 2.8 Abilities

```typescript
pokemon.getAbility().id              // Primary AbilityId (0-623)
pokemon.getPassiveAbility().id       // Passive AbilityId
pokemon.hasPassive()                 // Boolean (boss/candy)
pokemon.canApplyAbility()            // Not suppressed
pokemon.canApplyAbility(true)        // Passive not suppressed
pokemon.hasAbility(AbilityId.X)      // Checks both, accounts for suppression
pokemon.summonData.abilitySuppressed // Gastro Acid active
```

Suppression: Gastro Acid, Neutralizing Gas arena tag, Mold Breaker (temporary).

### 2.9 Moves

```typescript
const moveset = pokemon.getMoveset(); // PokemonMove[] (up to 4)
for (const pm of moveset) {
  pm.moveId           // MoveId enum
  pm.ppUsed           // PP consumed
  pm.ppUp             // 0-3 (PP Ups applied)
  pm.getMovePp()      // Max PP
  pm.getPpRatio()     // Remaining PP ratio
  pm.isUsable(pokemon) // [boolean, reason]

  const move = pm.getMove(); // Move object
  move.type            // PokemonType
  move.category        // PHYSICAL=0, SPECIAL=1, STATUS=2
  move.power           // Base power (0 for status)
  move.accuracy        // Percent (-1 = always hits)
  move.pp              // Base PP
  move.priority        // -7 to +5
  move.target          // MoveTarget enum
  move.flags           // MoveFlags bitfield
}
```

Move history: `pokemon.summonData.moveHistory` (TurnMove[])

### 2.10 Fusion

PokéRogue-specific mechanic:
```typescript
pokemon.isFusion()              // Boolean
pokemon.fusionSpecies            // PokemonSpecies | null
// Fusion averages base stats: ceil((base + fusionBase) / 2)
// Types merge via complex logic
// Ability can come from fusion species
```

### 2.11 Positioning & Field Access

```typescript
pokemon.isOnField()             // Active on field
pokemon.isPlayer()              // Player's pokemon
pokemon.getBattlerIndex()       // PLAYER=0, PLAYER_2=1, ENEMY=2, ENEMY_2=3
pokemon.getFieldIndex()         // 0 or 1 (position in doubles)

globalScene.getPlayerField()    // PlayerPokemon[] (active, 1-2)
globalScene.getEnemyField()     // EnemyPokemon[] (active, 1-2)
globalScene.getPlayerParty()    // Full party (up to 6)
globalScene.getEnemyParty()     // Full enemy party
```

**CRITICAL**: `getEnemyField().filter()` loses slot position! Index 0 = ENEMY,
index 1 = ENEMY_2. Filtering shifts indices. Always access by slot directly.

### 2.12 Boss Properties (Enemy Only)

```typescript
enemy.isBoss()                  // Has boss segments
enemy.bossSegments              // Total segments (0, 2-5+)
enemy.bossSegmentIndex          // Current segment (0 = no shields left)
enemy.aiType                    // RANDOM=0, SMART_RANDOM=1, SMART=2
```

### 2.13 Data Lifecycle

| Data Object | Scope | Resets When |
|-------------|-------|-------------|
| `summonData` | Per-summon | Switch out, battle end |
| `summonData.statStages` | Per-summon | Switch out |
| `summonData.tags` | Per-summon | Switch out |
| `tempSummonData` | Volatile | Reload, NOT saved |
| `battleData` | Per-battle | New battle |
| `waveData` | Per-wave | New wave/battle |
| `turnData` | Per-turn | Each turn AND switch |
| `customPokemonData` | Permanent | Never (mystery encounter overrides) |

### 2.14 Visibility / Information Hiding

**Always visible**: Species, level, HP bar, status icon, types, gender, held items
(partial), stat stages, tera type (when active), boss segments.

**Hidden from player**: IVs, exact enemy HP, nature, exact stats, ability (until
revealed), passive ability (until revealed), enemy moveset (until used).

---

## 3. Field & Arena State

### 3.1 Arena Properties

```typescript
globalScene.arena.biomeId         // BiomeId enum
globalScene.arena.weather          // Weather | null
globalScene.arena.terrain          // Terrain | null
globalScene.arena.tags             // ArenaTag[]
globalScene.arena.weatherType      // WeatherType getter (NONE if null)
globalScene.arena.terrainType      // TerrainType getter (NONE if null)
globalScene.arena.playerTerasUsed  // Tera uses in current biome
globalScene.arena.playerFaints     // Player faints in current biome
globalScene.arena.ignoreAbilities  // Mold Breaker active
```

### 3.2 Weather

**WeatherType enum** (10 values, 0-9):
```
NONE=0, SUNNY=1, RAIN=2, SANDSTORM=3, HAIL=4,
SNOW=5, FOG=6, HEAVY_RAIN=7, HARSH_SUN=8, STRONG_WINDS=9
```

Weather object: `{ weatherType, turnsLeft, maxDuration }`
- `turnsLeft = 0` means permanent (biome weather or legendary weather)
- Default duration: 5 turns when set by Pokemon

| Weather | Fire | Water | Other |
|---------|------|-------|-------|
| SUNNY / HARSH_SUN | 1.5x | 0.5x | |
| RAIN / HEAVY_RAIN | 0.5x | 1.5x | |
| SANDSTORM | | | 1/16 HP DOT (non-Ground/Rock/Steel) |
| HAIL | | | 1/16 HP DOT (non-Ice) |
| SNOW | | | 1.5x DEF for Ice types |
| HARSH_SUN | | Cancels Water attacks entirely | |
| HEAVY_RAIN | Cancels Fire attacks entirely | | |
| STRONG_WINDS | | | Halves Flying weaknesses |

Suppressed by Cloud Nine / Air Lock (`isEffectSuppressed()`).

### 3.3 Terrain

**TerrainType enum** (5 values, 0-4):
```
NONE=0, MISTY=1, ELECTRIC=2, GRASSY=3, PSYCHIC=4
```

Terrain object: `{ terrainType, turnsLeft, maxDuration }` — Duration: 5 turns.

| Terrain | Damage Boost | Special Effect |
|---------|-------------|----------------|
| ELECTRIC | Electric 1.3x | Prevents Sleep (grounded) |
| GRASSY | Grass 1.3x | Heals 1/16 HP/turn (grounded) |
| PSYCHIC | Psychic 1.3x | Blocks priority moves vs grounded |
| MISTY | | Blocks status (grounded), halves Dragon damage |

Only affects **grounded** Pokemon.

### 3.4 Arena Tags

**ArenaTagSide**: `BOTH=0, PLAYER=1, ENEMY=2`

Each tag has: `tagType`, `turnCount`, `side`, `sourceMove`, `sourceId`

#### Entry Hazards

| Tag | Max Layers | Grounded? | Effect |
|-----|-----------|-----------|--------|
| SPIKES | 3 | Yes | 1/8, 1/6, 1/4 max HP |
| STEALTH_ROCK | 1 | No | 1/8 * Rock effectiveness |
| TOXIC_SPIKES | 2 | Yes | 1 layer=Poison, 2=Toxic. Poison types absorb. |
| STICKY_WEB | 1 | Yes | -1 Speed stage on switch-in |

Duration: Infinite (persist until removed by Rapid Spin, Defog, etc.)

#### Screens

| Tag | Reduces | Duration | Effect |
|-----|---------|----------|--------|
| REFLECT | Physical | ~5 turns | 0.5x singles, 0.67x doubles |
| LIGHT_SCREEN | Special | ~5 turns | 0.5x singles, 0.67x doubles |
| AURORA_VEIL | Both | ~5 turns | 0.5x singles, 0.67x doubles |

Bypassed by Infiltrator ability.

#### Field-Wide Effects

| Tag | Effect | Duration |
|-----|--------|----------|
| TRICK_ROOM | Reverses speed order | ~5 turns |
| GRAVITY | Grounds all, removes Flying immunity | ~5 turns |
| TAILWIND | 2x Speed for one side | 4 turns |
| SAFEGUARD | Blocks status for one side | ~5 turns |
| MIST | Prevents stat lowering from opponents | ~5 turns |
| NO_CRIT | Prevents crits for one side | ~5 turns |
| FAIRY_LOCK | Prevents switching (non-Ghost) | ~2 turns |
| IMPRISON | Blocks shared moves on opponents | While source active |

#### Single-Turn Protection

| Tag | Blocks |
|-----|--------|
| QUICK_GUARD | Priority moves |
| WIDE_GUARD | Spread moves |
| MAT_BLOCK | Physical + Special attacks |
| CRAFTY_SHIELD | Status moves |

#### Pledge Combinations

| Tag | Effect | Duration |
|-----|--------|----------|
| FIRE_GRASS_PLEDGE | 1/8 HP DOT to non-Fire | 4 turns |
| WATER_FIRE_PLEDGE | 2x secondary effect chance | 4 turns |
| GRASS_WATER_PLEDGE | 1/4 Speed | 4 turns |

#### Other

| Tag | Effect |
|-----|--------|
| NEUTRALIZING_GAS | Suppresses all abilities |
| MUD_SPORT | Electric 0.33x |
| WATER_SPORT | Fire 0.33x |
| ION_DELUGE | Normal → Electric (1 turn) |

### 3.5 Positional Tags

```typescript
globalScene.arena.positionalTagManager.tags  // PositionalTag[]
```

| Tag | Effect |
|-----|--------|
| DELAYED_ATTACK | Future Sight / Doom Desire — hits after countdown |
| WISH | Heals target after countdown |

### 3.6 Turn-End Order

1. Pokemon battler tags lapse
2. Turn heal modifiers apply
3. Grassy Terrain heal
4. Post-turn ability effects
5. Status/held item transfer
6. Arena tags lapse
7. Weather lapse
8. Terrain lapse

---

## 4. Battle & Run State

### 4.1 Battle Object (Per-Wave)

```typescript
const battle = globalScene.currentBattle;
battle.waveIndex           // 1-200 (Classic), 1-∞ (Endless)
battle.battleType           // WILD=0, TRAINER=1, CLEAR=2, MYSTERY_ENCOUNTER=3
battle.battleSpec           // DEFAULT=0, FINAL_BOSS=1
battle.double               // Boolean (doubles battle)
battle.turn                 // 0+ (incremented each turn)
battle.escapeAttempts       // Failed flee attempts (adds bonus)
battle.enemyFaints          // Enemy KO count this wave
battle.lastMove             // Last move used
battle.trainer              // Trainer | null
battle.mysteryEncounter     // MysteryEncounter | undefined
```

### 4.2 Run-Level State (BattleScene)

```typescript
globalScene.money              // Currency (starts at 1000)
globalScene.score              // Accumulated score
globalScene.seed               // Run seed
globalScene.pokeballCounts     // { [PokeballType]: number }
globalScene.lockModifierTiers  // Boolean (Lock Capsule)
globalScene.modifiers          // PersistentModifier[] (player items)
```

Starting pokeball inventory: 5 Pokeballs, 0 of everything else.

### 4.3 Game Mode

```typescript
globalScene.gameMode.modeId    // CLASSIC=0, ENDLESS=1, SPLICED_ENDLESS=2, DAILY=3, CHALLENGE=4
globalScene.gameMode.isClassic
globalScene.gameMode.isEndless
globalScene.gameMode.hasTrainers
globalScene.gameMode.hasNoShop  // Daily
globalScene.gameMode.hasMysteryEncounters  // Classic/Challenge only
```

Key methods:
- `isBoss(waveIndex)`: Every wave divisible by 10
- `isWaveFinal(waveIndex)`: Classic=200, Daily=50, Endless=every 250
- `isEndlessBoss(waveIndex)`: Every 50 in Endless (Paradox)
- `isEndlessMajorBoss(waveIndex)`: Every 1000 (Eternamax)

### 4.4 Boss Mechanics

**Segment calculation** (`getEncounterBossSegments`):
```
base = 2
+1 if level >= 100
+1 if BST >= 670
+floor(waveIndex / 250) additional
Daily final boss = 5 segments always
```

Shield mechanics:
- Shield prevents HP dropping below segment threshold in one hit
- On shield break: +1 random stat boost
- If ≥3 total segments, last shield break: +2 boost
- If ≥5 total segments, second-to-last break: +2 boost
- Shields block stat drops and status effects until cleared

Boss AI: `AiType.SMART` (evaluates all moves, picks best)
Wild non-boss: `AiType.SMART_RANDOM` (weighted random)

### 4.5 Trainer Info

```typescript
const trainer = globalScene.currentBattle.trainer;
trainer?.config.trainerType    // TrainerType enum
trainer?.isDouble()            // Double battle
trainer?.getPartyTemplate()    // Party size/strength
```

Trainer Pokemon IVs: `randSeedIntRange(floor(waveIndex/10), 31)` — higher waves = better IVs.

### 4.6 Mystery Encounters

```typescript
globalScene.mysteryEncounterSaveData.encounteredEvents    // All MEs this run
globalScene.mysteryEncounterSaveData.encounterSpawnChance  // Current spawn weight
```

Spawn: weight starts at 3/256, increases by 3 per miss. Targets ~12 MEs per run.

---

## 5. Modifier / Item State

### 5.1 Storage & Access

```typescript
globalScene.modifiers                              // PersistentModifier[] (player)
globalScene.getModifiers(ModifierClass, player=true) // Find by type
globalScene.findModifier(predicate, player=true)    // Find single
globalScene.findModifiers(predicate, player=true)   // Find multiple
globalScene.enemyModifiers                          // Enemy items (private)
```

### 5.2 Modifier Hierarchy

```
PersistentModifier (stackable, persists)
├── LapsingPersistentModifier (has battleCount)
├── PokemonHeldItemModifier (per-pokemon, has pokemonId)
│   ├── BerryModifier, TurnHealModifier, AttackTypeBoosterModifier, ...
│   └── 26 subclasses total
├── Party-wide modifiers (MoneyMultiplier, ExpBooster, etc.)
└── EnemyPersistentModifier (enemy buffs, invisible)
ConsumableModifier (one-time, not stored)
├── PokemonHpRestoreModifier, TmModifier, EvolutionItemModifier, ...
```

### 5.3 Key Held Items (Per-Pokemon)

| Modifier | Max Stack | Effect |
|----------|-----------|--------|
| TurnHealModifier (Leftovers) | 4 | 1/16 HP/turn per stack |
| HitHealModifier (Shell Bell) | 4 | 1/8 damage dealt healed |
| SurviveDamageModifier (Focus Band) | 5 | Chance to survive at 1HP |
| BypassSpeedChanceModifier (Quick Claw) | 3 | Chance to move first |
| FlinchChanceModifier (King's Rock) | 3 | Flinch chance on hit |
| CritBoosterModifier (Scope Lens) | 1 | +1 crit stage |
| AttackTypeBoosterModifier (Charcoal etc.) | 99 | +20% type damage |
| PokemonMultiHitModifier (Multi Lens) | 2 | Extra hit per stack |
| BerryModifier | 2-3 | Various (heal, status cure, stat boost) |
| PokemonInstantReviveModifier (Reviver Seed) | 1 | Auto-revive once |
| ResetNegativeStatStageModifier (White Herb) | 2 | Clear stat drops |
| SwitchEffectTransferModifier (Baton) | 1 | Baton Pass on switch |
| PokemonExpBoosterModifier (Lucky Egg) | 99 | +40%/100% EXP |

### 5.4 Key Party-Wide Modifiers

| Modifier | Max | Effect |
|----------|-----|--------|
| MoneyMultiplierModifier (Amulet Coin) | 5 | +20% money per stack |
| ExpShareModifier | 5 | Share EXP to non-participants |
| PreserveBerryModifier (Berry Pouch) | 3 | 30% preserve berry per stack |
| HealingBoosterModifier (Healing Charm) | 5 | 1.1x healing |
| ExtraModifierModifier (Golden Pokeball) | 3 | +1 reward option per stack |
| LockModifierTiersModifier (Lock Capsule) | 1 | Lock reward tiers |
| MapModifier | 1 | Choose biome at fork |
| TerastallizeAccessModifier | 1 | Enables tera |
| IvScannerModifier | 1 | Scan enemy IVs on encounter |

### 5.5 Modifier Tiers

```
COMMON=0 (75.0%), GREAT=1 (19.0%), ULTRA=2 (4.7%),
ROGUE=3 (1.2%), MASTER=4 (0.1%), LUXURY=5
```

Luck upgrades: `upgradeOdds = floor(128 / ((partyLuckValue + 4) / 4))`

### 5.6 Shop Economy

**Reroll cost**: `ceil(waveIndex/10) * baseValue * 2^rerollCount * multiplier`
- Base: 250 (or sum of tier values with Lock Capsule)
- Black Sludge: 2.5x multiplier
- Doubles with each reroll

**Shop tiers** (unlock by wave): Tier 0 always, Tier 1 wave 20+, ... Tier 6 wave 170+.

### 5.7 Berry Types

```
SITRUS=0 (heal at low HP), LUM=1 (cure status), ENIGMA=2 (heal on SE hit),
LIECHI=3 (+ATK), GANLON=4 (+DEF), PETAYA=5 (+SPATK), APICOT=6 (+SPDEF),
SALAC=7 (+SPD), LANSAT=8 (+crit), STARF=9 (+random), LEPPA=10 (restore PP)
```

### 5.8 Enemy Modifiers (Invisible)

```typescript
globalScene.getModifiers(EnemyPersistentModifier, false)
```

| Type | Max | Effect |
|------|-----|--------|
| EnemyDamageBoosterModifier | 999 | 1.05x per stack |
| EnemyDamageReducerModifier | 999 | 0.975x per stack |
| EnemyTurnHealModifier | 10 | 2% HP per stack |
| EnemyAttackStatusEffectChanceModifier | 10 | Status on hit |
| EnemyEndureChanceModifier | 10 | Endure chance |
| EnemyFusionChanceModifier | 10 | Fusion chance |

---

## 6. Move Mechanics

### 6.1 Move Properties

```typescript
move.id             // MoveId
move.type           // PokemonType
move.category       // PHYSICAL=0, SPECIAL=1, STATUS=2
move.power          // Base power (0 for status)
move.accuracy       // Percent (-1 = always hits)
move.pp             // Base PP
move.priority       // -7 to +5
move.target          // MoveTarget enum
move.flags          // MoveFlags bitfield
```

### 6.2 Damage Formula

Core calculation in `pokemon.getBaseDamage()` and `MoveEffectPhase`:

```
damage = floor(
  floor(floor(2 * level / 5 + 2) * power * A/D) / 50 + 2
  * targets * parentalBond * weather * terrain * critical
  * random * STAB * typeEffectiveness * burn
  * abilityMultipliers * itemMultipliers
)
```

Where:
- `A` = effective ATK stat (physical) or SPATK (special)
- `D` = effective DEF stat or SPDEF
- `targets` = 0.75 if hitting multiple targets in doubles
- `parentalBond` = 0.25 for Parental Bond second hit
- `weather` = 1.5 or 0.5 (see weather table)
- `terrain` = 1.3 (matching type on grounded)
- `critical` = 1.5 base (modified by Sniper to 2.25)
- `random` = 0.85 to 1.00
- `STAB` = 1.5 (2.0 with Adaptability)
- `typeEffectiveness` = product of type chart values
- `burn` = 0.5 for physical if burned (unless Guts)

### 6.3 Type Chart

Standard Pokemon type chart with 19 types. Key interactions:
- NORMAL immune to GHOST (and vice versa)
- GROUND immune to ELECTRIC
- FAIRY immune to DRAGON
- STEEL resists 10 types
- ELECTRIC has only 1 weakness (Ground)

Special cases in PokéRogue:
- Freeze-Dry: always SE vs Water regardless of type chart
- Strong Winds: halves Flying weaknesses
- Tera Shell: all hits NVE at full HP
- Stellar tera: 2x power STAB (once per type)

### 6.4 Priority System

| Priority | Example Moves |
|----------|--------------|
| +5 | Helping Hand |
| +4 | Protect, Detect, King's Shield |
| +3 | Fake Out, Follow Me (+3 from Triage for heals) |
| +2 | Extreme Speed, Grassy Glide (in terrain) |
| +1 | Quick Attack, Aqua Jet, Shadow Sneak, Sucker Punch |
| 0 | Most moves |
| -1 | Vital Throw |
| -3 | Focus Punch |
| -5 | Teleport |
| -6 | Roar, Whirlwind, Circle Throw |
| -7 | Trick Room |

Ability modifiers: Prankster (+1 status), Gale Wings (+1 Flying at full HP), Triage (+3 heals).
Trick Room reverses order within same priority bracket.

### 6.5 Critical Hit Mechanics

Base crit rate stages:
```
Stage 0: 1/24 (4.17%)
Stage 1: 1/8  (12.5%)
Stage 2: 1/2  (50%)
Stage 3+: 1/1 (100%)
```

Crit stage sources: Focus Energy (+2), Scope Lens (+1), Super Luck (+1), Lansat Berry (+1), high-crit moves (+1).

Crits ignore: negative ATK/SPATK stages on attacker, positive DEF/SPDEF stages on defender, Reflect/Light Screen/Aurora Veil.

### 6.6 Accuracy Formula

```
finalAccuracy = moveAccuracy * (accStage / evaStage) * abilityMods * itemMods
```

- `accStage`: `max(2, 2+stage) / max(2, 2-stage)` for attacker's ACC
- `evaStage`: same for defender's EVA
- Always-hit moves (accuracy = -1): bypass all accuracy checks
- No Guard: both sides always hit

### 6.7 Move Targets

```
NEAR_ENEMY        — Single adjacent enemy (needs target in doubles)
NEAR_OTHER         — Single adjacent non-self (ally or enemy)
OTHER              — Any single non-self
ALL_NEAR_ENEMIES  — All adjacent enemies (multi-target, no explicit target)
ALL_NEAR_OTHERS   — All adjacent non-self (multi-target)
ALL_ENEMIES        — All enemies
USER               — Self only
USER_AND_ALLIES   — Self + allies
PARTY              — Entire party
ALL                — Everything on field
RANDOM_NEAR_ENEMY — Random enemy (game picks)
```

**CRITICAL for RL**: Multi-target moves (ALL_NEAR_ENEMIES etc.) must NOT pass explicit
targets to `handleCommand` — the game computes them via `getMoveTargets()`. Only
NEAR_ENEMY/NEAR_OTHER/OTHER need explicit target selection.

### 6.8 Move Flags

```
CONTACT          — Triggers Rough Skin, Static, etc.
SOUND            — Bypasses Substitute, blocked by Soundproof
PULSE            — Boosted by Mega Launcher
BITE             — Boosted by Strong Jaw
PUNCH            — Boosted by Iron Fist
WIND             — Triggers Wind Rider, Wind Power
SLICING          — Boosted by Sharpness
BALLBOMB         — Blocked by Bulletproof
POWDER           — Blocked by Overcoat, doesn't affect Grass
DANCE            — Copied by Dancer
RECHARGE         — Must recharge next turn (Hyper Beam etc.)
```

### 6.9 Multi-Hit Moves

- Fixed 2 hits: Double Kick, Dual Wingbeat
- Fixed 3 hits: Triple Axel, Triple Kick
- Fixed 5 hits: Population Bomb
- Variable 2-5: Bullet Seed, Rock Blast (distribution: 35/35/15/15%)
- Skill Link: always max hits
- Multi Lens: +1 hit per stack
- Parental Bond: +1 hit at 25% power

### 6.10 Two-Turn Moves & Semi-Invulnerability

| Move | Turn 1 State | Vulnerable To |
|------|-------------|---------------|
| Fly | FLYING | Thunder, Hurricane, etc. |
| Dig | UNDERGROUND | Earthquake, Magnitude |
| Dive | UNDERWATER | Surf, Whirlpool |
| Phantom Force | PHANTOM_FORCE | Nothing (full invuln) |
| Solar Beam | Charging (skip in Sun) | Not invulnerable |

---

## 7. Ability Mechanics

### 7.1 Architecture

- **624 AbilityId values** (0=NONE through 623)
- **~120 AbAttr classes** in `ab-attrs.ts` (6166 lines)
- Each ability composed of multiple AbAttr attributes
- Every Pokemon can have **primary + passive** ability (boss/candy)

### 7.2 Checking Abilities

```typescript
pokemon.getAbility().id              // Primary
pokemon.getPassiveAbility().id       // Passive
pokemon.hasPassive()                 // Has passive?
pokemon.canApplyAbility(passive?)    // Not suppressed?
pokemon.hasAbility(AbilityId.X)      // Either ability
```

### 7.3 Weather-Setting Abilities (On Entry)

| Ability | Weather |
|---------|---------|
| DRIZZLE | RAIN |
| DROUGHT | SUNNY |
| SAND_STREAM | SANDSTORM |
| SNOW_WARNING | SNOW |
| PRIMORDIAL_SEA | HEAVY_RAIN (clears on leave) |
| DESOLATE_LAND | HARSH_SUN (clears on leave) |
| DELTA_STREAM | STRONG_WINDS (clears on leave) |

### 7.4 Speed-Doubling Abilities

| Ability | Condition |
|---------|-----------|
| SWIFT_SWIM | Rain |
| CHLOROPHYLL | Sun |
| SAND_RUSH | Sandstorm |
| SLUSH_RUSH | Hail/Snow |
| SURGE_SURFER | Electric Terrain |
| UNBURDEN | After losing item (tag) |

### 7.5 Offensive Ability Highlights

| Category | Examples | Effect |
|----------|---------|--------|
| Stat 2x | HUGE_POWER, PURE_POWER | 2x ATK |
| Type power | TRANSISTOR (Electric 1.3x), DRAGONS_MAW (Dragon 1.5x) |
| Move class | IRON_FIST (+20% punch), TOUGH_CLAWS (+30% contact), STRONG_JAW (+50% bite) |
| Conditional | TECHNICIAN (≤60bp → 1.5x), ANALYTIC (last → 1.3x), STAKEOUT (switch-in → 2x) |
| STAB boost | ADAPTABILITY (1.5x → 2.0x STAB) |
| Type change | PIXILATE/REFRIGERATE/AERILATE/GALVANIZE (Normal → type + 1.2x) |
| Multi-hit | PARENTAL_BOND (+1 hit at 25%), SKILL_LINK (max hits) |
| Crit | SUPER_LUCK (+1 stage), SNIPER (1.5x crit mult), MERCILESS (auto-crit poisoned) |
| Field-wide | DARK_AURA (1.33x Dark), FAIRY_AURA (1.33x Fairy) |
| Ruin | TABLETS_OF_RUIN (opp -25% ATK), SWORD_OF_RUIN (opp -25% DEF), etc. |

### 7.6 Defensive Ability Highlights

| Category | Examples | Effect |
|----------|---------|--------|
| Damage reduce | MULTISCALE (0.5x at full HP), FUR_COAT (0.5x phys), ICE_SCALES (0.5x spec) |
| Type immune + heal | VOLT_ABSORB, WATER_ABSORB, EARTH_EATER | Immune + 25% heal |
| Type immune + boost | LIGHTNING_ROD (+SpAtk), MOTOR_DRIVE (+Spd), SAP_SIPPER (+Atk) |
| Full immune | WONDER_GUARD (only SE hits), LEVITATE (Ground immune) |
| Endure | STURDY (survive at 1HP from full), DISGUISE (block first hit) |
| Non-direct block | MAGIC_GUARD (no weather/status/recoil damage) |
| Status immune | LIMBER (no para), IMMUNITY (no poison), COMATOSE (no NV status) |
| Bounce | MAGIC_BOUNCE (reflects status moves) |

### 7.7 Stat-Modifying Abilities

| Trigger | Examples |
|---------|---------|
| On entry | INTIMIDATE (-1 ATK all opponents), DOWNLOAD (+1 ATK or SPATK) |
| On KO | MOXIE (+1 ATK), BEAST_BOOST (+1 highest stat) |
| On being hit | STAMINA (+1 DEF), WEAK_ARMOR (-1 DEF +2 SPD) |
| On stat drop | DEFIANT (+2 ATK), COMPETITIVE (+2 SPATK) |
| Multiplier | SIMPLE (2x all changes), CONTRARY (inverts all changes) |

### 7.8 Trapping Abilities

| Ability | Condition |
|---------|-----------|
| SHADOW_TAG | Always (unless opponent also has it) |
| ARENA_TRAP | Target grounded |
| MAGNET_PULL | Target Steel type |

Bypassed by Ghost types and Run Away.

### 7.9 Mold Breaker Abilities

MOLD_BREAKER, TURBOBLAZE, TERAVOLT, MYCELIUM_MIGHT (status only).
Temporarily ignores target's **ignorable** abilities during the move.

---

## 8. Decision Points & Action Space

### 8.1 Action Layout (58 Discrete)

```
0-3    FIGHT_ENEMY     Move 0-3 → enemy slot 0 (or multi/self target)
4-7    FIGHT_ENEMY2    Move 0-3 → enemy slot 1 (doubles, single-target)
8-11   FIGHT_ALLY      Move 0-3 → ally (doubles, ally-targeting)
12-16  SWITCH          Switch to party slot 1-5
17-21  BALL            Pokeball/Great/Ultra/Rogue/Master
22     RUN             Flee battle
23-26  TERA_ENEMY      Tera + move 0-3 → enemy 0
27-30  TERA_ENEMY2     Tera + move 0-3 → enemy 1
31-34  TERA_ALLY       Tera + move 0-3 → ally
35-37  SELECT_REWARD   Pick reward option 0-2
38     REROLL          Reroll modifiers
39     SKIP            Skip/decline
40-51  BUY_SHOP        Buy shop item 0-11
52-57  PARTY_TARGET    Apply modifier to party slot 0-5
```

### 8.2 CommandPhase — Battle Actions

**When**: Every turn, for each active player Pokemon.
**UI Mode**: `UiMode.COMMAND` → `UiMode.FIGHT`

Valid actions:
- **FIGHT (0-11)**: Move must pass `isUsable()` (PP > 0, not disabled, etc.)
  - Single-target: needs valid alive target
  - Multi-target: always valid if ANY enemy alive, game computes targets
  - Self-target: always valid, goes in FIGHT_ENEMY range
- **SWITCH (12-16)**: Non-fainted, non-active, not trapped
- **BALL (17-21)**: Wild battles only, not doubles, ball count > 0, boss needs Master Ball if shielded
- **RUN (22)**: Not trainer battle, not trapped, not END biome
- **TERA (23-34)**: No other party member already terastallized on field

Skip conditions: Move queue non-empty (Encore, two-turn), ally used BALL/RUN in doubles, Commander ability.

### 8.3 SelectTargetPhase — Target Selection

**When**: After move selection in doubles when multiple targets valid.
**UI Mode**: `UiMode.TARGET_SELECT`

Maps `BattlerIndex.ENEMY` → action 0, `ENEMY_2` → action 4, `PLAYER` → action 8, `PLAYER_2` → action 9.

### 8.4 SelectModifierPhase — Rewards & Shop

**When**: After winning a wave.
**UI Mode**: `UiMode.MODIFIER_SELECT`

- **SELECT_REWARD (35-37)**: Pick free reward (up to 3+)
- **REROLL (38)**: Re-generate options (costs money)
- **SKIP (39)**: Proceed to next wave
- **BUY_SHOP (40-51)**: Purchase from shop (costs money)

Two-step targeting: If item is `PokemonModifierType`, enters MODIFIER_TARGET sub-phase
→ actions 52-57 for eligible party members.

Shop purchases keep phase active (can continue shopping).

### 8.5 SwitchPhase — Forced Switch

**When**: After faint.
**UI Mode**: `UiMode.PARTY`

Actions: SWITCH (12-16), only non-fainted, non-active slots.

### 8.6 CheckSwitchPhase — Optional Switch

**When**: After KO in Switch battle style.
**UI Mode**: `UiMode.CONFIRM`

Binary: Action 0 = accept switch, Action 39 = decline.
Auto-skips in SET battle style.

### 8.7 LearnMovePhase — Move Replacement

**When**: Level-up with 4 moves known + new move available.
**UI Mode**: `UiMode.CONFIRM` → `UiMode.SUMMARY`

Actions: 0-3 = replace slot, 39 = don't learn.

### 8.8 EvolutionPhase — Evolution Choice

**When**: Evolution conditions met.
Auto-handled (always proceed).

### 8.9 SelectBiomePhase — Biome Fork

**When**: Biome changes AND player has MapModifier AND multiple linked biomes.
Actions: 0-3 for biome options. Otherwise auto-resolved.

### 8.10 MysteryEncounterPhase

**When**: Random events during exploration.
**UI Mode**: `UiMode.MYSTERY_ENCOUNTER`

Actions: 0-3 for encounter options (2-4 choices).

### 8.11 RevivalBlessingPhase

**When**: Revival Blessing move used.
Actions: PARTY_TARGET (52-57) for fainted members only.

### 8.12 GameOverPhase

**When**: All player Pokemon faint or victory.
Actions: 0 = retry, 1 = quit. For RL training: always quit.

### 8.13 ScanIvsPhase (Auto-Skipped)

IV Scanner modifier triggers on encounter. Phase router auto-declines and logs all
enemy IVs that exceed player's dex baseline. No agent decision.

---

## 9. Difficulty Scaling & Formulas

### 9.1 Enemy Level Formula

```
levelWaveIndex = gameMode.getWaveForDifficulty(waveIndex)
baseLevel = 1 + levelWaveIndex/2 + (levelWaveIndex/25)^2
Boss: floor(baseLevel * 1.2) + random offset
Non-boss: baseLevel + gaussian offset
```

Daily: `getWaveForDifficulty(w) = w + 30 + floor(w/5)` (steeper curve).

### 9.2 Flee Formula

```
speedRatio = playerSpeed / enemySpeed
speedCap = 6 (boss) or 4 (non-boss)
minChance = 5%
maxChance = 25% (boss) or 95% (non-boss)
escapeBonus = 2 (boss) or 10 (non-boss) per attempt
escapeChance = clamp(slope * speedRatio + min + bonus * attempts, min, max)
```

### 9.3 Catch Rate Formula (Gen 6 Style)

```
modifiedCatchRate = (((3*maxHP - 2*currentHP) * catchRate * ballMult) / (3*maxHP)) * statusMult
shakeProbability = round(65536 / (255 / modifiedCatchRate)^0.1875)
```

Ball multipliers: Poke=1.0, Great=1.5, Ultra=2.0, Rogue=3.0, Master=guaranteed.

### 9.4 Money Formula

```
waveSetIndex = ceil(waveIndex/10) - 1
moneyValue = (waveSetIndex + 1 + (0.75 + ((waveIndex-1) % 10 + 1) / 10)) * 100)^(1 + 0.005*waveSetIndex)
```

### 9.5 Trainer IV Scaling

Trainer Pokemon IVs: `randSeedIntRange(floor(waveIndex/10), 31)`.
Wave 100 → min IV 10, Wave 200 → min IV 20.

---

## 10. PokéRogue-Specific Mechanics

### 10.1 Fusion System

Pokemon can be fused via DNA Splicers (modifier):
- Stats averaged: `ceil((base + fusionBase) / 2)`
- Types merged via complex logic
- Abilities can come from either species
- Visual: combined sprites

### 10.2 Passive Abilities

Every boss Pokemon + starter-candy Pokemon have TWO abilities:
- Primary ability (normal)
- Passive ability (always active alongside primary)
Both checked by `hasAbility()` and `canApplyAbility()`.

### 10.3 No EVs

PokéRogue has NO traditional EVs. Instead:
- Vitamins (HP Up etc.) → `BaseStatModifier` (directly modifies base stat)
- Old Gateau → `PokemonBaseStatFlatModifier`
- Shuckle Juice → `PokemonBaseStatTotalModifier`

### 10.4 Wave Milestones (Classic Mode)

| Wave | Event |
|------|-------|
| Every 10 | Boss encounter + full party heal |
| 20/40/60/80/100/120/140/160 | Gym leaders (maybe offset by 10) |
| 180 | Elite Four |
| 190 | Champion |
| 200 | Final Boss (Eternatus → Eternamax) |
| 10-180 | Mystery encounters possible |

### 10.5 Biome System

New biome every 10 waves (Classic) or 5 waves (Endless).
Biome determines: encounter tables, wild Pokemon species, weather patterns, music.
With MapModifier: player chooses at forks. Without: random or deterministic.

### 10.6 Starter Cost System

Each species has a cost (1-10). Total points for team selection determined by
game mode. Higher-cost starters have better base stats / abilities.

---

## 11. Current Encoding Gaps

Issues identified in the existing 2,951-dim observation space (`src/rl/spaces.ts`):

1. **Screen remaining turns** — Currently binary (present/absent); duration matters
2. **Tailwind remaining turns** — Same issue
3. **Positional tags (Wish, Future Sight)** — Not encoded at all
4. **Weather suppression** — Cloud Nine/Air Lock active not encoded
5. **Permanent vs temporary weather** — turnsLeft=0 maps to 0 (same as no weather)
6. **Neutralizing Gas source count** — Matters for when it wears off
7. **Stealth Rock effectiveness** — Could pre-compute per team member type
8. **Individual enemy held items** — Not fully encoded
9. **Lapsing modifier battle counts** — Duration remaining not tracked
10. **Berry consumed state** — Whether berry was already triggered
11. **Move secondary effects** — Not encoded (flinch chance, status chance, etc.)
12. **Ability interactions** — No encoding of ability matchup implications
13. **Boss shield HP thresholds** — Exact HP at which shields break
14. **Mystery encounter options** — Not in observation space
15. **Fusion species data** — Partially encoded

### 11.1 Verification Findings (87 Missing Properties)

A code-level audit of every property in `pokemon.ts`, `battle-scene.ts`, `battle.ts`,
`arena.ts`, and `modifier.ts` identified the following items absent from the current
2,951-dim Float32 encoding AND from the original version of this document:

#### Pokemon Properties (22 missing)

| Property | Source | RL Relevance |
|----------|--------|-------------|
| `moveQueue` | `summonData.moveQueue` | Queued moves (Encore, two-turn, Outrage) |
| `waveTurnCount` | `waveData.abilityRevealed` (calc) | Turns active this wave |
| `isTrapped` | `getTag(TRAPPED\|BIND\|...)` | Can't switch out |
| `isGrounded` | `isGrounded()` | Terrain/hazard interaction |
| `transformSpeciesId` | `summonData.speciesForm` | Transform disguise |
| `transformMoves` | `summonData.moveset` | Transformed moveset |
| `illusionSpeciesId` | `summonData.illusion` | Illusion/Zoroark active |
| `attacksReceived` | `turnData.attacksReceived` | Recent damage taken |
| `weight` | `getSpeciesForm().weight` | Low Kick, Heavy Slam calc |
| `catchRate` | `getSpeciesForm().catchRate` | Catch probability |
| `baseTotal` | `getSpeciesForm().baseTotal` | Species power metric |
| `stellarTypesBoosted` | `stellarTypesBoosted` | Stellar type boost tracker |
| `abilityRevealed` | `summonData.abilityRevealed` | Info hiding |
| `isFainted` | `isFainted()` | Shorthand for hp<=0 |
| `isActive` | `isActive()` | On field and not fainted |
| `turnData.damageTaken` | `turnData.damageTaken` | Per-turn damage |
| `turnData.order` | `turnData.order` | Turn order index |
| `turnData.hitCount` | `turnData.hitCount` | Multi-hit tracking |
| `battleData.hitsLanded` | `battleData.hitsLanded` | Battle-long stat |
| `battleData.abilitiesApplied` | `battleData.abilitiesApplied` | Triggered abilities |
| `shiny` | `pokemon.shiny` | Cosmetic but tracked |
| `variant` | `pokemon.variant` | Shiny variant |

#### Battle/Arena Properties (31 missing)

| Property | Source | RL Relevance |
|----------|--------|-------------|
| `battleStyle` | Settings | SET vs SWITCH (CheckSwitchPhase) |
| `challenges` | `gameData.challenges` | Active challenge modifiers |
| `failedRunAway` | `battle.failedRunAway` (implied) | Escape failed flag |
| `moneyScattered` | `globalScene.moneyScattered` | Lost money tracking |
| `timeOfDay` | `arena.timeOfDay` | Affects some encounters |
| `trainerType` | `battle.trainer?.config.trainerType` | Trainer identity |
| `trainerIsDouble` | `battle.trainer?.isDouble()` | Trainer forces doubles |
| `mysteryEncounterType` | `battle.mysteryEncounter?.encounterType` | ME type |
| `mysteryEncounterOptions` | ME option labels/availability | ME decisions |
| `playerFaintsTotal` | Cumulative across run | Run health metric |
| `playerFaintsBiome` | `arena.playerFaints` | Biome performance |
| `lockModifierTiers` | `globalScene.lockModifierTiers` | Reroll tier lock |
| `rerollCount` | SelectModifierPhase state | Reroll cost tracking |
| `weatherIsPermanent` | `weather.turnsLeft === 0` | Permanent vs temp |
| `weatherSuppressed` | Cloud Nine / Air Lock | Weather inactive |
| `positionalTags` | `arena.positionalTagManager` | Future Sight, Wish |
| `arenaTagSourceId` | Per-tag source tracking | Tag origin |
| `playerStealthRock` | Convenience hazard flag | Quick access |
| `playerStickyWeb` | Convenience hazard flag | Quick access |
| `enemyStealthRock` | Convenience hazard flag | Quick access |
| `enemyStickyWeb` | Convenience hazard flag | Quick access |
| `trickRoomActive` | Redundant convenience flag | Speed reversal |
| `gravityActive` | Redundant convenience flag | Grounding effect |
| `ignoreAbilities` | `arena.ignoreAbilities` | Mold Breaker |
| `biomeName` | `Biome[biomeId]` | Display name |
| `battleSpec` | `battle.battleSpec` | Final boss flag |
| `playerAliveBattle` | Battle-scope faints | Wave health |
| `enemyAliveBattle` | Battle-scope faints | Wave progress |
| `lastMoveId` | `battle.lastMove` | Last move used |
| `gameMode` | `globalScene.gameMode.modeId` | Run mode |
| `seed` | `globalScene.seed` | Determinism |

#### Modifier Properties (34 missing)

**Held Items (17 not tracked)**:
ContactHealModifier, GripClawModifier, PokemonFormChangeItemModifier,
SwitchEffectTransferModifier (partial), SpeciesStatBoosterModifier,
AllMovePpRestoreModifier, PokemonStatusProtectModifier,
BypassSpeedChanceModifier (partial), TurnStatusEffectModifier,
ContactPoisonProtectModifier, ContactStatStageChangeChanceModifier,
ResetNegativeStatStageModifier (partial), PokemonInstantReviveModifier (partial),
PokemonIncrementingStatModifier, EnemyFusionChanceModifier,
EnemyEndureChanceModifier, EnemyAttackStatusEffectChanceModifier.

**Party-Wide (13 not tracked)**:
MoneyInterestModifier, HiddenAbilityRateBooster, ShinyRateBooster,
FusionCostReducerModifier, GigantamaxAccessModifier,
MegaEvolutionAccessModifier, PokemonIncrementingStatModifier,
HealingBoosterModifier (partial), ExpShareModifier (partial),
PreserveBerryModifier (partial), MapModifier, SuperExpModifier,
IvScannerModifier.

**Lapsing (3 not tracked)**:
DoubleBattleChanceBoosterModifier, TempExtraModifierModifier,
TempCritBoostModifier.

**Consumables (~20 types not modeled for shop decisions)**:
RememberMoveModifier, PokemonPpRestoreModifier, PokemonAllMovePpRestoreModifier,
PokemonPpUpModifier, PokemonNatureChangeModifier, PokemonLevelIncrementModifier,
TmModifier, EvolutionItemModifier, PokemonHpRestoreModifier,
PokemonStatusHealModifier, PokemonFormChangeModifier, MoneyRewardModifier,
PokemonHealModifier, RememberMoveAccessModifier, FusedConsumableModifier, and more.

---

## 12. Python State Schema

The complete Python TypedDict schema for the RL state dictionary is defined in
`src/rl/state_schema.py`. It captures ALL properties listed in this document
(sections 2-10) plus the verification findings (section 11.1) as typed Python
dictionaries suitable for JSON serialization.

Key design points:
- **31 TypedDict classes** composing the top-level `GameState`
- **62 constants** mirroring TypeScript enum counts
- **6 helper factory functions** for zero-initialized empty slots
- Shop state (`ShopState`) is `Optional` — only present during `SelectModifierPhase`
- Wire protocol messages (`StateMessage`, `ActionMessage`, `GameOverMessage`, etc.)
  define the JSON-lines format between Node.js runner and Python agent
