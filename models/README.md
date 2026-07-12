# Trained model checkpoints

| File | What it is | How it was produced |
|------|------------|---------------------|
| `ppo_v9_first.zip` | First real MaskablePPO checkpoint on the v9 observation layout (200k steps, ~10.7 MB). Eval: ~300.7 mean reward, mean wave 23.7 (see `src/rl/CHANGELOG.md`). | `python examples/rl/train_maskable_ppo.py --config examples/rl/first_train.yaml` |

Evaluate a checkpoint:

```bash
python tools/eval_policy.py --policies random sb3:models/ppo_v9_first.zip --episodes 20 --waves 20
```

## Why binaries live in git

Checkpoints are committed directly (no LFS) so a fresh clone can run the
worked examples in `tools/eval_policy.py` and the docs without a download
step. Policy: keep AT MOST the single current reference checkpoint here —
git never forgets a committed blob, so every additional/obsolete checkpoint
permanently bloats clones. Superseded or experimental checkpoints belong in
release assets or external storage. (`ppo_v9_smoke.zip`, an unreferenced
10.5 MB smoke-test artifact, was removed for exactly this reason — it
remains reachable in git history.)
