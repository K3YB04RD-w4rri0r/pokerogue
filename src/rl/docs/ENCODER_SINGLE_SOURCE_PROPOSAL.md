# Proposal: single-sourcing the dual encoders (data-first codegen)

Status: **Phase 1 IMPLEMENTED (2026-07-14, owner-approved).** Phase 2 not started.
Scope: engineering only — no observation-design changes, no dim/layout changes, bitwise-identical output.

Implementation notes (deltas from the plan below):
- The emitter is `test/rl/encoder-data-sync.test.ts` (the repo's golden-file idiom:
  `UPDATE_RL_ENCODER_DATA=1` regenerates `src/rl/generated/encoder-data.json`; the same test IS
  the staleness gate and runs inside the ordinary vitest stage). No new tooling was added —
  tsx/vite-node don't exist in this repo, vitest resolves the TS aliases for free.
- Python loads the JSON via `src/rl/encoder_data.py` (no generated .py module needed).
- REFINEMENT: `enums.py`'s public IntEnum classes and string-vocab lists stay hand-written —
  regenerating them risks silent public-API renames for zero drift benefit. Instead
  `tools/verify/check_generated_sync.py` (rl-verify stage V-2) asserts every hand member exists
  TS-side with the same value (one documented Python-only sentinel: `MultiHitType.NA`).
- Pre-refactor drift audit: ALL ~30 tables matched TS exactly (the mirror was healthy; the
  refactor changed provenance, not values). Stale doc copies fixed: `observation.py` v7
  docstring, `state_schema.py` (now imports the generated dims), flag-count comments.

## Problem

The observation encoder exists twice: `spaces.ts` (wire authority) and `observation.py` (hand-written
mirror), plus satellite mirrors `enums.py`, `feature_names.py`, `state_schema.py`. Today only
discipline and parity tests keep them bitwise-equal. Measured surface of the hand-mirror
(2026-07-14 inventory):

- **~30 mirrored tables/constants**, including `_ABILITY_FEATURES` (311×40 ≈ 12,400 floats,
  hand-transcribed from a TS file that is itself generated), `_MODIFIER_FEATURES` (109×20),
  the 19×19 type chart, 69 curated volatile tags, 28 arena-tag order, 75 other-effect flags,
  22 hand-copied IntEnums, and 6 string→id tables.
- **Dimension scalars duplicated in three Python places** (`observation.py`, `enums.py DIMS`,
  `state_schema.py`) — the third is stale right now (`NUM_CURATED_TAGS = 48` vs 69; v7 block
  table in the `observation.py` module docstring; "9,875 dims" in `state_schema.py`).
- **Every protocol bump requires a lockstep re-mirror** of 5+ files (v8: `bdefeea2af3`, v9:
  `947e42088ed` — the v9 commit itself fixed a stale v7 constant left by the previous bump).
- **Real drift incidents**: 22-dim lapsing-modifier divergence (`f6f15c834a5` item d); the
  "parity-invisible" class where BOTH encoders agreed on the wrong answer (`enemy_command`
  all-zero phase one-hot, enemy-view side swaps — 74 dims); a stale corpus dump breaking parity
  after v8.

## What the inventories established

- `encodeObservation` is a **pure function** of the gameState JSON — fresh buffer, no globals,
  no I/O. All impurity lives upstream in `state-builder.ts`.
- The mirror's content splits cleanly:
  - **DATA (~80% of drift surface, 100% mechanically mirrorable)**: index maps, one-hot vocab,
    feature tables, enum values, dims, flag lists, slot orders, divisors.
  - **LOGIC (hand-written in both, guarded by the parity suite)**: the block encoders, the
    write-cursor layout, fog-of-war gating, four data-dependent reorderings (held-item priority
    sort, shop filter-reindex, lapsing argmax, speed ranking), one cross-block patch
    (command_field_index), and float64→float32 rounding-sensitive arithmetic shapes.
- The parity machinery (fixtures + goldens bitwise, episode-dump strict parity, invariants,
  layout canary) is strong on logic but cannot catch table drift that shifts BOTH sides — and
  cannot catch stale third copies at all (nothing imports `state_schema.py`).

## Options considered

- **(A) Full codegen of `observation.py` from a declarative spec** — requires a DSL expressing
  cursor layout, fog gates, and the reordering closures. High effort, high risk, replaces
  battle-tested code; the parity suite already guards exactly this logic. **Rejected.**
- **(B) Data/logic split: generate the DATA from the TS source; logic stays hand-written.**
  TS remains the single authority (already the stated contract). **Recommended.**
- **(C) Neutral shared spec both sides generate from** — inverts authority and churns the
  proven TS side for no additional drift protection over (B). **Rejected.**

## Recommended plan

### Phase 1 — single-source the data (kills the dominant drift class)

1. **Emitter** (`scripts/rl-emit-encoder-data.ts`, run via tsx/vite-node): imports the live TS
   modules (`spaces.ts` exports, `ability-features.ts`, `modifier-features.ts`, the `src/enums/*`
   used at build time) and serializes every mirrored table/constant to
   `src/rl/generated/encoder-data.json` (sorted keys, stable formatting) **and** emits
   `src/rl/_generated_tables.py` (checked in, header: GENERATED — DO NOT EDIT, `# fmt: off`).
   No AST parsing — the emitter serializes the same runtime values the TS encoder uses.
2. **Refactor the Python mirror to import the generated module** for: ability/modifier feature
   tables, type chart, stage multipliers, other-effect flags, curated tags, arena order,
   party/enemy mod ids + divisors, priority keys, slot keys, all dims, all IntEnums and
   STR_TO_ID tables. `observation.py`'s encode logic is untouched; `enums.py` becomes a thin
   re-export shim (public API unchanged — zero churn for the env/tools/examples importers).
3. **CI gate** (new rl-verify stage + rl-framework CI): regenerate → `git diff --exit-code` on
   the generated files, so the checked-in copy can never go stale; then the existing parity
   suite proves bitwise equality end-to-end.
4. **Sweep the stale copies while at it**: fix the v7 module docstring, `state_schema.py`'s
   three stale constants, the "77 flags" comment (actual: 75), and drop the dead
   `ARENA_TAG_INDEX_MAP` in `spaces.ts`.

Acceptance: all existing gates green (goldens unchanged — this is a refactor, not a re-encode);
`fixture_parity` + `check_parity --strict` + full `rl-verify`; a mutation test (perturb one
table entry in TS → regeneration changes Python → parity gate fails loudly).

### Phase 2 — layout manifest (optional, after Phase 1 settles)

`feature_names.py` is a third hand-mirror of the LAYOUT (names/offsets in parallel with the
encoder). Emit a block/offset/dim manifest from the TS side (the layout-canary test already
probes real offsets) and generate `feature_names.py` from manifest + tables. Same gate pattern.

### Explicitly deferred

Logic codegen (option A) — revisit only if the encode logic itself starts churning; today it
changes an order of magnitude less often than the tables, and the parity suite covers it.

## Risks / mitigations

- **Emitter environment**: the imported TS modules must stay jsdom/Phaser-free — they are today
  (spaces.ts imports only enums); the emitter fails loudly at import if that regresses.
- **Generated-file churn in diffs**: stable ordering + one file; reviewers treat it like goldens.
- **Behavioral change risk in the refactor**: none tolerated — bitwise gates are the acceptance
  bar; the refactor moves value definitions, not values.
- **Python import cost**: a generated module of ~15k literals imports in milliseconds (same data
  already lives in `observation.py` today).

## Effort estimate

Phase 1: emitter ~150-line TS + ~200-line Python refactor + CI stage + stale-copy sweep;
validated by the existing verification pipeline in one run. Phase 2: about half that.
