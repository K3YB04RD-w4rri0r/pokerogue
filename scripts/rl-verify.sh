#!/usr/bin/env bash
# Full verification suite for the RL environment.
# Usage: bash scripts/rl-verify.sh [quick]
#   quick: 1 smoke seed, fewer waves, skips corpus/bench/soak
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-full}"
ART=".rl-verify"
# Same interpreter contract as rl-setup.sh: a conda/venv user who installed
# deps into $PYTHON must not silently fall back to the system python3.
PY="${PYTHON:-python3}"
# Start each run from a clean slate. The parity/dim-exercise checks glob
# $ART/{smoke,parity,corpus}/*.jsonl, so a stale dump from a prior run (a
# removed scenario, or an older OBSERVATION_DIM) would shape-mismatch and fail
# the suite. Wipe before regenerating.
rm -rf "$ART"
mkdir -p "$ART/smoke" "$ART/parity" "$ART/corpus"

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

step "V-1 python lint (ruff)"
# check-only by design: `ruff format` would split observation.py's deliberate
# one-dim-per-line encoder idiom; the lint rules (pyflakes/bugbear/import
# order) are the bug-catching part.
"$PY" -m ruff check src/rl tools examples/rl

step "V-2 generated-data sync (hand-written enums vs encoder-data.json)"
# The generated JSON's own staleness vs the TS source is gated inside V11
# (test/rl/encoder-data-sync.test.ts); this stage covers the residual
# hand-written Python mirrors. See docs/ENCODER_SINGLE_SOURCE_PROPOSAL.md.
"$PY" tools/verify/check_generated_sync.py

step "V-2b layout name probes (feature_names vs actual encoder order)"
(cd tools/verify && "$PY" check_layout_names.py)

step "V1 build"
pnpm rl:build

step "V11 vitest (rewards, encoding, semantic)"
pnpm exec vitest run test/rl/

step "V0 parser completeness (static)"
"$PY" tools/verify/check_parser_completeness.py

step "V13a observation coverage (registry diffs + manifests)"
"$PY" tools/verify/check_obs_coverage.py

step "V5a fixture parity (golden cross-language)"
"$PY" tools/verify/fixture_parity.py

step "V2 auto-mode smoke"
node dist/rl/cli.js --seed=verify-auto --waves=3 --log >"$ART/auto-smoke.log" 2>&1
tail -12 "$ART/auto-smoke.log"

if [ "$MODE" = "quick" ]; then
  step "V3/V6 smoke + mask property (quick: 1 seed)"
  "$PY" tools/verify/run_episodes.py --seed-list verify-q1 --waves 6 \
    --probe-invalid 0.03 --dump-dir "$ART/smoke"
else
  step "V3/V6 smoke + mask property (5 seeds, probes on)"
  "$PY" tools/verify/run_episodes.py --seeds 5 --seed-prefix verify-s --waves 10 \
    --probe-invalid 0.02 --dump-dir "$ART/smoke"

  step "V5b parity dumps (3 auto + 4 random interactive)"
  for i in 1 2 3; do
    # log files (not /dev/null): under set -e a crashing dump otherwise dies
    # with nothing but the step banner
    node dist/rl/cli.js --seed="parity-a$i" --waves=8 --dump-obs="$ART/parity/auto-$i.jsonl" \
      >"$ART/parity/auto-$i.log" 2>&1 || { tail -20 "$ART/parity/auto-$i.log"; exit 1; }
  done
  # Fog-of-war parity: the fog encoder (TS AND Python) is otherwise UNEXERCISED
  # — no other corpus sets --fog-of-war, so a drift between the two fog paths
  # would pass silently. This dump carries fogOfWar=true so check_parity
  # re-encodes with fog and cross-checks both branches.
  node dist/rl/cli.js --seed="parity-fog" --waves=8 --fog-of-war --dump-obs="$ART/parity/fog.jsonl" \
    >"$ART/parity/fog.log" 2>&1 || { tail -20 "$ART/parity/fog.log"; exit 1; }
  "$PY" tools/verify/run_episodes.py --seed-list parity-r1,parity-r2,parity-r3,parity-r4 \
    --waves 12 --dump-dir "$ART/parity"

  step "V14 deep-coverage corpus (scenarios assert their target situations)"
  "$PY" tools/verify/gen_coverage_corpus.py --out-dir "$ART/corpus"
fi

step "V5 TS<->Python parity + invariants"
"$PY" tools/verify/check_parity.py "$ART/smoke/*.jsonl" "$ART/parity/*.jsonl" "$ART/corpus/*.jsonl" --atol 1e-6

if [ "$MODE" != "quick" ]; then
  step "V13b dim exercise (never-varied dims classified; suspected-bug gate)"
  "$PY" tools/verify/check_dim_exercise.py "$ART/smoke/*.jsonl" "$ART/parity/*.jsonl" "$ART/corpus/*.jsonl" --gate
fi

if [ "$MODE" != "quick" ]; then
  step "V15 mask gating (shielded-boss balls, Struggle availability)"
  "$PY" tools/verify/check_mask_gating.py
fi

if [ "${RL_VERIFY_RENDERED:-}" = "1" ]; then
  step "V16 rendered E2E (headless Chromium; needs playwright, slow on software rendering)"
  "$PY" tools/verify/check_rendered.py
fi

step "V7 determinism (auto)"
"$PY" tools/verify/check_determinism.py --seed verify-det --waves 6 --mode auto

step "V7 determinism (interactive, seeded actions)"
"$PY" tools/verify/check_determinism.py --seed verify-det --waves 6 --mode interactive --action-seed 42

step "V7b determinism (in-process reset == fresh process, bitwise)"
"$PY" tools/verify/check_determinism.py --seed verify-det --waves 6 --mode inprocess --action-seed 42

# Grunt-wave depth: masked-random policies die around wave 6-10, so the
# default matrix never reached the evil-grunt trainer waves (35+) whose
# gender-variant roll was a seeded-determinism fix (eb73e2a7f6e) — it was
# only ever validated manually. STARTING_WAVE_OVERRIDE pins the episode
# into a verified grunt battle (seed verify-det-grunt @ wave 35 = Magma
# Grunt), making the trainer-name/variant path part of the bitwise gate.
step "V7c determinism at grunt-wave depth (seeded gender roll)"
"$PY" tools/verify/check_determinism.py --seed verify-det-grunt --waves 40 --mode inprocess --action-seed 42 \
  --cli-arg=--override=STARTING_WAVE_OVERRIDE=35

if [ "$MODE" != "quick" ]; then
  step "V12 throughput bench (per-process)"
  "$PY" tools/verify/bench_throughput.py --episodes 3 --waves 8 --out "$ART/bench.json"

  # 300 eps @ waves 12 (was 100 @ 4): a real training run died at ~episode
  # 500 / waves 20 — a soak an order of magnitude below the operating point
  # cannot catch lifecycle bugs. Node stderr is captured for post-mortems
  # (the original death was undiagnosable: stderr had gone to DEVNULL).
  step "V12b in-process soak (wrapper policy: recycle every 50)"
  "$PY" tools/verify/bench_throughput.py --inprocess --soak 300 --waves 12 --recycle-every 50 \
    --stderr-log "$ART/soak-node-stderr.log" \
    --out "$ART/soak-inprocess.json"
fi

step "ALL CHECKS PASSED"
