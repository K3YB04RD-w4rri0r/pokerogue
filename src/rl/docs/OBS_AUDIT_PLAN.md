# Observation v9 — Audit & Redesign Plan

Status: PROPOSED (not started). The current 10,403-dim observation (v8,
protocol 4) predates several framework fixes and has never had a
systematic audit. Before any v9 layout is designed, three questions need
evidence-grade answers, in this order:

1. **Redundancy** — which of the 10,403 dims carry no information
   (never vary, duplicate another dim, are derivable from others)?
2. **Sufficiency** — for each decision type, is every piece of
   information an optimal policy needs actually present and correct?
3. **Observability** — which dims leak information a player cannot see,
   and how should fog-of-war masking + revealed-indicators work?

Only after 1-3 produce ledgers does the v9 layout get designed (phase 4).
This is a multi-session effort; each phase has a standalone deliverable.

---

## Phase 1 — Redundancy analysis (data-driven)

Build on existing tooling: `feature_names.py` (complete dim→name map),
`--dump-obs` JSONL dumps, `gen_coverage_corpus.py` (forces rare states),
`check_dim_exercise.py` + `coverage-manifests/dim-exercise-ledger.json`
(which dims ever vary).

Work:
- Generate a LARGE decision corpus: many seeds × deep waves × forced
  scenarios (doubles, weather, boss, trainer, shop-heavy, status-heavy).
- Per-dim statistics over the corpus:
  - dead dims (never vary anywhere — extend the existing ledger),
  - constant-within-episode dims (game-mode flags: 9+ dims that are
    byte-identical every step of a classic run),
  - exact duplicates / r≈1.0 pairs (e.g. `is_fainted` vs `hp_ratio==0`,
    one-hot complements, `base_total` vs sum of `base_stats`,
    `stack_ratio` vs `features[17]` — the latter is already documented
    as "kept for compatibility"),
  - block-level overlap (species_id + base_stats + computed_stats +
    nature_mults encode overlapping information three ways).
- Deliverable: `redundancy-ledger.json` with a keep / merge / drop
  verdict + justification per dim, and the measured evidence.

## Phase 2 — Sufficiency audit (code-driven)

Cross-reference the GAME's decision-relevant formulas against the
encoding. For each decision type, enumerate the inputs of the underlying
game computation and trace each to an obs dim (or a blindspot):

- **Move choice**: damage formula inputs (`pokemon.ts` getBaseDamage /
  MoveEffectPhase): effective stats (incl. paralysis, items, abilities),
  true type effectiveness (`getAttackTypeEffectiveness` — the derived
  block currently uses a STATIC chart that ignores Freeze-Dry, Levitate,
  Strong Winds, inverse battles), accuracy pipeline, priority brackets,
  speed order (current derived rank ignores paralysis/Tailwind/Trick
  Room), secondary effects.
- **Catch decision**: catch-rate formula inputs (status, HP, ball mult,
  boss shields) — vs encoded catch_rate/can_catch.
- **Switch decision**: matchup preview for BENCH members (bench moves
  are encoded; is anything missing for evaluating an incoming matchup?).
- **Shop/reward decision**: full costs, tiers, item semantics
  (20-dim feature table coverage vs the actual modifier pool), money
  horizon. Known gaps: shop slots 7-12 unencoded; items 13-14
  unreachable (action space); Lock Capsule / luck interactions.
- **Learn-move decision**: the OFFERED move's features are entirely
  absent (agent chooses a replacement blind) — confirmed blindspot.
- **Structural gaps**: the 6th party member invisible in singles
  (player AND enemy side); positional tags (Wish/Future Sight) captured
  in the dict but not encoded; lapsing-modifier duration only top-1.
- Deliverable: `sufficiency-matrix.md` — decision type × required info ×
  {encoded, partial, missing, wrong-approximation}, each row citing the
  game-code source and the dim name(s).

## Phase 3 — Observability & fog-of-war spec

Current state: the observation is omniscient (enemy movesets, IVs, exact
stats, nature, abilities visible from wave start; `ability_revealed` is
captured but unused). Design, per dim group:
- the masking rule (e.g. enemy move slot hidden until seen in
  `moveHistory`; ability until `waveData.abilityRevealed`; enemy bench
  until in `seenEnemyPartyMemberIds`; IVs/nature never visible —
  replaced by visible proxies),
- paired **revealed-indicator dims** so "unknown" is distinguishable
  from "absent/zero",
- an env/RunConfig toggle (`observability: full | fog`) — full-info
  stays available for oracle/distillation training; fog is the
  human-comparable benchmark mode.

## Phase 4 — v9 layout + migration

Only after 1-3: design the new layout (drop/merge per the redundancy
ledger, add per the sufficiency matrix, masking + indicators per the
observability spec), bump protocolVersion, lockstep TS/Python encoders,
regenerate goldens + canary + dim-exercise manifests, rewrite INPUT.md.
Old checkpoints are invalidated by design — one retraining event.

## Ground rules

- Audit before design; design before code.
- Every drop/add/mask decision must cite ledger or game-code evidence.
- The bitwise TS↔Python parity harness stays the acceptance gate at
  every step.
