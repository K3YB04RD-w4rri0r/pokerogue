#!/usr/bin/env bash
# Full verification suite for the RL environment.
# Usage: bash scripts/rl-verify.sh [quick]
#   quick: 1 smoke seed, fewer waves, skips corpus/bench/soak
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-full}"
ART=".rl-verify"
# Start each run from a clean slate. The parity/dim-exercise checks glob
# $ART/{smoke,parity,corpus}/*.jsonl, so a stale dump from a prior run (a
# removed scenario, or an older OBSERVATION_DIM) would shape-mismatch and fail
# the suite. Wipe before regenerating.
rm -rf "$ART"
mkdir -p "$ART/smoke" "$ART/parity" "$ART/corpus"

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

step "V1 build"
pnpm rl:build

step "V11 vitest (rewards, encoding, semantic)"
pnpm exec vitest run test/rl/

step "V0 parser completeness (static)"
python3 tools/verify/check_parser_completeness.py

step "V13a observation coverage (registry diffs + manifests)"
python3 tools/verify/check_obs_coverage.py

step "V5a fixture parity (golden cross-language)"
python3 tools/verify/fixture_parity.py

step "V2 auto-mode smoke"
node dist/rl/cli.js --seed=verify-auto --waves=3 --log >"$ART/auto-smoke.log" 2>&1
tail -12 "$ART/auto-smoke.log"

if [ "$MODE" = "quick" ]; then
  step "V3/V6 smoke + mask property (quick: 1 seed)"
  python3 tools/verify/run_episodes.py --seed-list verify-q1 --waves 6 \
    --probe-invalid 0.03 --dump-dir "$ART/smoke"
else
  step "V3/V6 smoke + mask property (5 seeds, probes on)"
  python3 tools/verify/run_episodes.py --seeds 5 --seed-prefix verify-s --waves 10 \
    --probe-invalid 0.02 --dump-dir "$ART/smoke"

  step "V5b parity dumps (3 auto + 4 random interactive)"
  for i in 1 2 3; do
    node dist/rl/cli.js --seed="parity-a$i" --waves=8 --dump-obs="$ART/parity/auto-$i.jsonl" \
      >/dev/null 2>&1
  done
  python3 tools/verify/run_episodes.py --seed-list parity-r1,parity-r2,parity-r3,parity-r4 \
    --waves 12 --dump-dir "$ART/parity"

  step "V14 deep-coverage corpus (scenarios assert their target situations)"
  python3 tools/verify/gen_coverage_corpus.py --out-dir "$ART/corpus"
fi

step "V5 TS<->Python parity + invariants"
python3 tools/verify/check_parity.py "$ART/smoke/*.jsonl" "$ART/parity/*.jsonl" "$ART/corpus/*.jsonl" --atol 1e-6

if [ "$MODE" != "quick" ]; then
  step "V13b dim exercise (never-varied dims classified; suspected-bug gate)"
  python3 tools/verify/check_dim_exercise.py "$ART/smoke/*.jsonl" "$ART/parity/*.jsonl" "$ART/corpus/*.jsonl" --gate
fi

step "V7 determinism (auto)"
python3 tools/verify/check_determinism.py --seed verify-det --waves 6 --mode auto

step "V7 determinism (interactive, seeded actions)"
python3 tools/verify/check_determinism.py --seed verify-det --waves 6 --mode interactive --action-seed 42

step "V7b determinism (in-process reset == fresh process, bitwise)"
python3 tools/verify/check_determinism.py --seed verify-det --waves 6 --mode inprocess --action-seed 42

if [ "$MODE" != "quick" ]; then
  step "V12 throughput bench (per-process)"
  python3 tools/verify/bench_throughput.py --episodes 3 --waves 8 --out "$ART/bench.json"

  step "V12b in-process soak (wrapper policy: recycle every 50)"
  python3 tools/verify/bench_throughput.py --inprocess --soak 100 --waves 4 --recycle-every 50 \
    --out "$ART/soak-inprocess.json"
fi

step "ALL CHECKS PASSED"
