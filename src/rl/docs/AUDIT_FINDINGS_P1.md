# Obs Audit — Phase 1 Findings (Redundancy Ledger v2)

Corpus: 3,769 decisions / 26 episodes, interactive seeded-random driving
(shops, balls, switches exercised) + forced-scenario coverage corpus +
scenario diversity (6-member parties, doubles, waves 25/55/95/140, weather,
status pressure, rich economy). Ledger: `.rl-audit/redundancy-ledger.json`
(regenerate: `bash tools/verify/gen_audit_corpus.sh && python3
tools/verify/audit_obs_redundancy.py '.rl-audit/*.jsonl'`).

## Headline numbers (of 10,403 dims)

| Category | Dims | % |
|---|---|---|
| Dead (never varied anywhere in corpus) | 6,242 | 60.0% |
| Exact duplicates (byte-identical to another dim) | 804 | 7.7% |
| Near-duplicates (306 pairs, r≥0.999) | ~300 | ~3% |
| Rare binary flips (<0.1% of steps) | 128 | 1.2% |
| **Carrying independent signal** | **~3,000** | **~29%** |

v1 (auto-mode corpus) measured 85% dead; interactive driving revived
~2,600 dims — the remaining 60% is structural, not under-exercise
(though rare mechanics will revive a few hundred more; treat per-dim
verdicts as "candidate", not final, until the v9 design pass).

## Where the waste concentrates

1. **Move feature blocks: 63% of the entire observation** (12 slots ×
   4 moves × 136 dims = 6,528). Every block, including the ACTIVE enemy's,
   is majority-dead (enemy_0/moves: 269/544 dead+dup; bench blocks:
   350-508/544). The 136-dim per-move vector (18-type one-hot + category +
   target-class + ~30 effect features) is massively over-provisioned for
   how much move diversity a run actually sees. v9 lever: shrink the
   per-move vector (effect features → compact scalars), keep full fidelity
   only for on-field mons.
2. **tera_type one-hots ≈ type1 one-hots**: 85 duplicate dims. Tera rarely
   changes the answer; encode tera as (used?, tera-type-index-if-different)
   instead of a full 18-one-hot per slot.
3. **Padding-locked features**: `valid == ability_feat[18] == ...` groups —
   ability/passive feature dims that never deviate from slot validity.
   The 40-dim ability vector earns ~a dozen live dims in practice.
4. **shop/shop block 124/138 dead** even with driven purchases — the
   6×23 shop encoding rarely differentiates (wave-gated rows + narrow item
   pool); pairs with the sufficiency finding that slots 7-12 are missing.
5. **inventory/held 112/180 dead+dup** — top-2-items channel wastes most
   of its width (stack_ratio duplicates stack_count, feature tails dead).

## Cross-check with Phase 2 (sufficiency)

The observation is simultaneously TOO BIG and MISSING things: see
SUFFICIENCY_MATRIX.md — ai_type and multi_hit_type serialized but never
encoded; learn-move offer invisible; 6th member invisible; shop tail
missing. v9 should fund those additions with the cuts above.

## Recommended v9 sizing direction (for discussion, not final)

Dropping dead+dup structural waste and compacting move/ability/tera/item
blocks points to a ~3,500-4,500 dim observation with MORE decision-relevant
content than v8. Exact layout belongs to Phase 4 after the fog-of-war
masking spec (Phase 3) fixes which dims need revealed-indicators.

---

# v9 Acceptance Re-run (2026-07-08, post-implementation)

Same generator, 3,755 decisions / 28 episodes (now incl. trainer-battle
scenarios), v9 encoding (6,991 dims). Ledger:
`.rl-audit/redundancy-ledger-v9.json` (the audit tool now also reports the
lane-collapsed metric).

**Why lanes, not per-slot dims:** v9 keeps UNIFORM pokemon/move/shop/item
slots (per the approved design — parity maintenance beats micro-sizing).
A per-slot dim registers dead whenever that slot happened to be empty or
held a non-matching item class; the layout question is whether the LANE
(the semantic position, collapsed across slots) ever carries signal.

| Metric | v8 | v9 |
|---|---|---|
| Raw per-slot deadness | 60.0% | 44.7% |
| Fully-dead MOVE-vector lanes | ~60 of 136 | **0 of 60** |
| Fully-dead lanes overall | — | 470/921 (21.8% of dims), decomposed below |

Dead-lane decomposition (470):
- **395 lane-sparsity** — one-hot / modifier-feature / volatile-tag
  positions that only matching entities can light up (structurally
  partial by nature; each lane is justified by its class, not its
  corpus frequency).
- **16 mode-constant battle flags** (documented: classic-mode run
  metadata).
- **15 learn-move lanes** — the corpus produced no learn_move decisions;
  the block is separately verified live (nonzero at learn_move phases).
- **44 rare-mechanic lanes (~1.5% of dims)** — gravity, Trick Room,
  screens/tailwind turn counters, hazard layers, Wish/Future Sight,
  teras-used, enemy-boost aggregate stacks. All deliberate keeps; their
  exercise needs targeted coverage scenarios (backlog: a hazards/screens/
  trick-room forced scenario in gen_audit_corpus.sh).

**Verdict: v9 accepted.** The redesigned move vector carries zero dead
lanes; every remaining dead dim is lane sparsity, a documented constant,
or an enumerated rare mechanic. Duplicates dropped 804 → 649, dominated
by the deliberate tera≡type1 semantic correlation and small-corpus
cross-slot coincidences.
