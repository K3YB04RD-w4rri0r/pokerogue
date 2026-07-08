#!/usr/bin/env bash
# Evidence corpus for the obs-v9 redundancy audit (Phase 1).
# Diversity axes: seeds x starting waves x party compositions x battle
# styles x forced scenarios. Every run writes --dump-obs JSONL that
# audit_obs_redundancy.py consumes.
#
# Usage: bash tools/verify/gen_audit_corpus.sh [out-dir]   (default .rl-audit)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-.rl-audit}"
mkdir -p "$OUT"

# Interactive seeded-RANDOM driving (exercises shops, balls, switches, teras
# — auto mode always fights and skips every shop, which under-exercises ~2k
# dims; ledger v1 measured 85% dead largely because of it).
run() { # name waves extra-cli-args...
  local name="$1" waves="$2"; shift 2
  echo "=== $name ==="
  local extra=()
  for a in "$@"; do extra+=("--cli-arg=$a"); done
  python3 tools/verify/run_episodes.py --seed-list "audit-$name" --waves "$waves" \
    --dump-dir "$OUT" ${extra[@]+"${extra[@]}"} >/dev/null 2>&1 \
    || echo "  (run $name reported failures — dump kept)"
  mv "$OUT/audit-$name.jsonl" "$OUT/$name.jsonl" 2>/dev/null || true
}

# Baseline diversity: several seeds, default party, early-to-mid waves
for i in 1 2 3 4; do
  run "base-$i" 12
done

# Full 6-member party (fills player_2..5 + exercises 6th-member paths)
run "party6-1" 10 --starters=MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS
run "party6-2" 10 --starters=CATERPIE,PIDGEY,RATTATA,PIKACHU,EEVEE,MAGIKARP

# Deep-wave starts (bosses, trainers, richer shops, higher tiers)
run "wave25" 6  --override=STARTING_WAVE_OVERRIDE=25 --override=STARTING_LEVEL_OVERRIDE=30
run "wave55" 6  --override=STARTING_WAVE_OVERRIDE=55 --override=STARTING_LEVEL_OVERRIDE=60
run "wave95" 6  --override=STARTING_WAVE_OVERRIDE=95 --override=STARTING_LEVEL_OVERRIDE=80
run "wave140" 6 --override=STARTING_WAVE_OVERRIDE=140 --override=STARTING_LEVEL_OVERRIDE=95

# Doubles (fills player_1/enemy_1, ally-targeting actions, spread moves)
run "double-1" 10 --override=BATTLE_STYLE_OVERRIDE='"double"' \
  --starters=MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS
run "double-2" 10 --override=BATTLE_STYLE_OVERRIDE='"double"'

# Weather / terrain / status pressure
run "weather-sand" 8 --override=WEATHER_OVERRIDE=3
run "weather-rain" 8 --override=WEATHER_OVERRIDE=2
run "status-toxic" 8 --override=OPP_MOVESET_OVERRIDE='[92,79]'   # Toxic, Sleep Powder

# Rich economy (large money exercises shop affordability dims)
run "rich" 10 --override=STARTING_MONEY_OVERRIDE=99999

# Trainer battles (fills enemy bench slots 2-5: multi-mon enemy parties;
# without these the enemy_3..5 blocks register corpus-dead)
run "trainer-1" 8 --override=BATTLE_TYPE_OVERRIDE='"trainer"' --override=STARTING_WAVE_OVERRIDE=35 --override=STARTING_LEVEL_OVERRIDE=40
run "trainer-2" 8 --override=BATTLE_TYPE_OVERRIDE='"trainer"' --override=STARTING_WAVE_OVERRIDE=95 --override=STARTING_LEVEL_OVERRIDE=80 --starters=MEWTWO,LUGIA,RAYQUAZA,DIALGA,GIRATINA,ARCEUS

echo
echo "corpus in $OUT:"
wc -l "$OUT"/*.jsonl | tail -1
echo "next: python3 tools/verify/audit_obs_redundancy.py '$OUT/*.jsonl' --out $OUT/redundancy-ledger.json"

# Forced-scenario coverage corpus (each scenario asserts its target situation)
echo "=== coverage-corpus scenarios ==="
python3 tools/verify/gen_coverage_corpus.py --out-dir "$OUT" >/dev/null 2>&1 \
  || echo "  (coverage corpus reported failures — dumps kept)"

wc -l "$OUT"/*.jsonl | tail -1
