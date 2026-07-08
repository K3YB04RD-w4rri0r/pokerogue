# Training on a big box (reference: Ryzen 9 5950X + 2× RTX 3090 + 128GB)

The game is the bottleneck and it's pure CPU (one node process per
worker); the GPUs serve the learner only. On a 5950X (16c/32t):

| Setup | Expected throughput |
|---|---|
| 1 worker | ~50-80 steps/s |
| 24 workers (`--num-envs 24`) | ~1,500-2,000 steps/s ≈ **130-170M steps/day** |

RAM: ~250-450MB per worker + learner → ~12-14GB at 24 workers. 128GB is
never the constraint.

## One-time setup

```bash
git clone <repo> && cd pokerogue
git checkout rl-framework
git submodule update --init assets locales   # tests + i18n
# Node >= 20 (nvm/tarball) + pnpm via corepack
pnpm install && pnpm rl:build
pip install -r requirements-rl.txt stable-baselines3 sb3-contrib torch  # CUDA wheel for the 3090s
bash scripts/rl-verify.sh quick               # must be fully green
```

## Recommended first serious run

```bash
python3 examples/rl/train_maskable_ppo.py \
  --config examples/rl/first_train.yaml \
  --num-envs 24 \
  --n-steps 512 \
  --net-arch 1024,512 \
  --device cuda:0 \
  --timesteps 20000000 \
  --checkpoint-every 1000000 \
  --tensorboard runs/v9 \
  --save models/ppo_v9_20m.zip
```

- `--num-envs 24`: leave ~8 threads for the learner + OS.
- `--n-steps 512`: longer rollouts amortize the SubprocVecEnv lockstep
  (every vec-step waits for the slowest worker) and the update pause.
- `--net-arch 1024,512` + `--device cuda:0`: a right-sized policy for the
  6,991-dim observation; this is where a 3090 starts earning its keep.
  (sb3's default 64×64 CPU net is fine for smoke tests only.)
- Watch curves: `tensorboard --logdir runs/`.

## The second 3090

PPO's learner won't saturate one 3090 for a long time. The second GPU's
best use is a PARALLEL EXPERIMENT, not data-parallelism: e.g. full-info
vs `fog_of_war: true` trained side by side, or two reward configs.

## Strategy that beats raw compute

1. **Curriculum**: train short horizons first, then warm-start longer
   ones (`MaskablePPO.load(path, env=new_env)`). See the next section
   for the RIGHT way to build the stages.
2. **Eval discipline**: `tools/eval_policy.py --policies random maxdamage
   sb3:models/<ckpt>.zip --episodes 20 --seed-prefix eval-holdout` after
   every checkpoint; training reward alone lies.
3. Behavior-cloning warm start from `maxdamage` demonstrations is the
   next efficiency lever if curves plateau (generate with
   `tools/run_policy.py --policy maxdamage` + a dump flag).

## Curriculum design — the reset question, answered

**What happens at the wave cap?** The episode ends as `truncated=True`
(the env reserves `terminated` for real game-overs) and the next episode
starts fresh. sb3 handles this correctly: on `TimeLimit.truncated` it
BOOTSTRAPS the value of the final state instead of treating it as death,
so the agent does not learn end-of-world artifacts (e.g. "dump all money
before wave 20 because nothing exists after"). This is exactly why the
truncated/terminated distinction matters — and why the cap is safe to
train against.

**Prefer horizon extension over start-offset.** Always start at wave 1
and raise the CAP per stage (20 → 50 → 100 → 200), warm-starting each
stage from the previous model. Every state the agent sees is then
self-generated and self-consistent — levels, items, money, party are
whatever it actually earned by that wave. There is no plausibility
problem by construction. The cost (replaying early waves each episode)
shrinks as the policy gets competent, and early waves are the cheapest
steps in the game.

**Start-offset runs are OFF-DISTRIBUTION and only for targeted drills**
(e.g. boss-wave practice). A run that starts at wave 40 with fresh
starters has none of the ~40 waves of accumulated levels, items, EXP
Share economics or money a real run would have — training on such states
teaches a game that does not exist. If you must, approximate
plausibility with the config overrides:

- **Level**: the game's own enemy scaling is
  `enemyLevel ≈ 1 + wave/2 + (wave/25)²` (battle.ts getLevelForWave).
  Set `starting_level` to about that value (players typically run at or
  slightly above wild levels):
  wave 20 → ~12 · wave 40 → ~24 · wave 60 → ~37 · wave 100 → ~67 ·
  wave 140 → ~102 · wave 180 → ~143.
- **Money**: roughly proportional to waves cleared; `starting_money` of
  ~120 × wave is in the right region.
- **Items**: `starting_modifiers` (e.g. `[{name: EXP_SHARE, count: 2},
  {name: GOLDEN_EXP_CHARM, count: 1}, ...]`) — a hand-picked loadout is
  a crude approximation of 40 waves of shop decisions; keep drills short
  so the approximation error matters less.

**The gold standard (roadmap, not built): snapshot resets.** The game
natively serializes sessions; an env option to reset FROM a bank of
mid-run saves (recorded from real policy runs at waves 20/40/80/...)
would give exactly-on-distribution curriculum starts. Medium effort:
load a session blob during headless boot instead of a fresh run. Worth
building the day horizon-extension stops being enough.
