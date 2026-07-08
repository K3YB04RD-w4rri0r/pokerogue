#!/usr/bin/env bash
# One-command setup for the PokeRogue RL environment.
#
#   bash scripts/rl-setup.sh            # env + headless game + smoke check
#   bash scripts/rl-setup.sh --train    # also install training extras (sb3 + torch)
#
# Idempotent: safe to re-run; each step no-ops when already satisfied.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN=0
for arg in "$@"; do
  [ "$arg" = "--train" ] && TRAIN=1
done

step() { printf '\n\033[1;36m== %s\033[0m\n' "$1"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

step "Toolchain checks"
command -v node >/dev/null || die "node not found — install Node >= 24.9 (https://nodejs.org, nvm, or a tarball)"
NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
[ "$NODE_MAJOR" -ge 20 ] || die "node $(node -v) too old — the toolchain needs >= 24.9 (>=20 may work for the runtime, but install 24.9+ to match CI)"
if ! command -v pnpm >/dev/null; then
  echo "pnpm not found — enabling via corepack"
  corepack enable pnpm || die "corepack enable failed — install pnpm manually (npm i -g pnpm)"
fi
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null || die "python3 not found (set PYTHON=/path/to/python to override)"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "python >= 3.10 required"
echo "node $(node -v) | pnpm $(pnpm -v) | $($PY --version)"

step "Git submodules (game data: assets + locales)"
git submodule update --init assets locales

step "Node dependencies"
pnpm install --frozen-lockfile 2>/dev/null || pnpm install

step "Build the headless game bundle (dist/rl/cli.js)"
pnpm rl:build

step "Python dependencies"
"$PY" -m pip install -q -r requirements-rl.txt
if [ "$TRAIN" = 1 ]; then
  echo "installing training extras (sb3 + torch — pass a torch index-url via PIP_INDEX_URL for CPU/CUDA control)"
  "$PY" -m pip install -q stable-baselines3 sb3-contrib torch
fi

step "Smoke check: one 3-wave headless episode"
node dist/rl/cli.js --seed=setup-smoke --waves=3 --lean >/dev/null || die "headless episode failed"
echo "headless episode: OK"

step "Smoke check: Python env round-trip"
"$PY" - <<'EOF'
import sys
sys.path.insert(0, "src")
from rl.pokerogue_env import PokeRogueEnv, PROTOCOL_VERSION
env = PokeRogueEnv(waves=2, seed="setup-smoke", lean=True)
obs, info = env.reset()
mask = env.action_masks()
obs, r, term, trunc, info = env.step(int(mask.argmax()))
env.close()
print(f"gym round-trip: OK (protocol {PROTOCOL_VERSION}, obs {obs.shape[0]} dims)")
EOF

printf '\n\033[1;32mSetup complete.\033[0m Next steps:\n'
echo "  bash scripts/rl-verify.sh quick     # the full correctness gate (run before any PR)"
echo "  python3 tools/run_policy.py --policy maxdamage --waves 10   # watch a bot play (headless)"
echo "  python3 examples/rl/train_maskable_ppo.py --config examples/rl/first_train.yaml   # train (needs --train extras)"
echo "  docs: src/rl/README.md · src/rl/INPUT.md · src/rl/docs/"
