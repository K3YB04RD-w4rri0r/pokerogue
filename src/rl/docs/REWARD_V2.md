# Reward v2 — exploit-hardened reward design (2026-07)

The v1 reward function shipped with the framework was audited during the July 2026 verification
campaign (see `VERIFICATION_CAMPAIGN_2026-07.md` and `rl-substance-findings-2026-07.json`). Several
findings were fixed in place (flee-farming RD1, slot-indexed HP deltas RD3, terminal-shaping drop,
pick-cancel farming); the remaining reward-design findings (RD2 stall-beats-losing, RD4 money
non-stationarity, RD5 dead terminal signals, RD6 chip-heal farming, RD7 non-potential shaping,
RD10 reroll fishing) required a redesign — this document. The v2 design itself went through a
five-lens adversarial review (exploit search, RL theory, game mechanics, numeric orderings,
implementation risk) before implementation; the review overturned three pieces of the first draft
(a discount-eroded lump stall penalty, an ungated wave-cap bonus, per-delta log money) — the
mechanisms below are the corrected versions.

Protocol note: reward v2 shipped as **protocolVersion 6**. The observation layout is unchanged
(obs v9, 6,991 dims); the bump exists because the WIRE SEMANTICS of `done` payloads changed
(livelock/wave_cap are terminal and their `reward` embeds the payments below). Version skew is not
graceful — an old client would bootstrap V(s_final) on top of an embedded penalty — so the
handshake refuses instead.

## The four mechanisms

### 1. Stalling is priced at the stalling steps (RD2)

v1: truncation carried no penalty, so a near-death agent's return-optimal move was any
repeated-observation loop (−0.4 for a 40-step livelock vs −50 for losing). The trained v9 agent
found the shop instance of this.

A naive fix — a −100 lump at livelock detection — fails under discounting: paid 40 steps late it
is worth −100·0.99⁴⁰ ≈ −67 at the decision point, still better than a worst-case fighting loss
(≈ −86). v2 therefore charges stalling **where it happens**:

- `stallStepPenalty` (−2.5) per decision once the consecutive already-seen-observation counter
  exceeds `STALL_GRACE_STEPS` (5). Legitimate play produces novel observations every step (HP, PP,
  turn counters move), so the grace window is never touched by honest play.
- `stallPenalty` (−50 = runLost: stalling IS losing) as the terminal lump when the counter reaches
  `NO_PROGRESS_LIMIT` (40) and the episode ends with reason `livelock`.
- Livelock is reported **terminated**, not truncated: the final observation IS the repeated
  observation, and bootstrapping V(s_final) under truncation drives V(stall) toward
  stallPenalty/(1−γ) ≈ −5,000 through self-reference. Termination keeps the target bounded at
  ≈ −50, exactly like a loss.

Detected-stall present value: −2.5·Σ₆⁴⁰ γᵗ − 50·γ⁴⁰ ≈ **−104**, strictly below the worst fighting
loss (≈ −85 PV). Detector-evading stalls (obs kept novel by oscillating HP/PP) have no positive
channel left to farm (see mechanisms 2 and 4) and drift at turnPenalty to an unpenalized step_cap.

The guard runs engine-side in BOTH transports (headless CLI and rendered bridge, same evaluation
point: after obs encode, before the cap check). The Python env keeps its own copy as defense in
depth; a v6 CLI always trips first, and the backstop mirrors the CLI's semantics (terminated +
stallPenalty read from the `ready` handshake's resolved reward config).

### 2. Money pays on a cumulative log scale (RD4, RD10)

v1: +0.01/unit linear against a money curve that grows super-linearly with wave — a wave-50 Relic
Gold paid +77, three times a boss clear; modifier selection was hijacked toward money items.

Per-delta log (the first draft) fails differently: ln is concave, so n small gains pay far more
than one lump (5×100 → +4.6 vs 500 → +1.24), and Golden Punch (money on every hit, ×Amulet Coin)
becomes a +0.5–1.5/step perpetual farm whose observations never repeat (money is an obs feature),
evading the livelock hash.

v2: `moneyGainedLog` (0.2) pays `w·[ln(1+G_post) − ln(1+G_pre)]` where **G is cumulative money
gained this episode**. The sum telescopes to w·ln(1+total) ≈ +2.6 max regardless of how income is
chunked — splitting is exactly neutral, and the marginal reward of farmed income decays to zero.

`moneySpentLog` (−0.05) prices spending **per delta** (deliberately asymmetric: splitting a spend
only raises the charge). Against the real reroll cost curve (250·⌈wave/10⌉·2ⁿ) a reroll costs
−0.28 (wave 1) to −0.39 (wave 50, second), versus a fishing upside bounded by the tier-bonus delta
(≤ +0.4, since `modifierSelected` pays regardless of tier) — reroll-fishing EV is negative from
the first reroll at every depth.

### 3. The wave cap is a surrogate win, scaled by honesty (RD5)

v1: under the standard training cap (waves: 50) `runWon` could never fire (classic victory is wave
200) — run outcome was effectively absent from the objective, and wave-cap truncation bootstrapped
V(s') off the end of the value function's training distribution.

v2: reaching the cap pays `waveCapReached` (+100) and the episode is **terminated** (the bonus
stands in for continuation value; bootstrapping on top would double-count). Two guards make this
sound:

- **Clean-ratio scaling**: fleeing also advances waveIndex, and an unscaled bonus would pay +98
  for flee-rushing the endgame (flee is legal on most wild waves, including wave-50 bosses). The
  bonus is multiplied by cleanWaveAdvances/totalWaveAdvances — pure flee-rush earns ~0, each fled
  wave forfeits its share (≈ −2 at cap 50, on top of −2 ranAway and the foregone +10 waveCleared).
  A late survival-flee by a doomed team keeps most of the bonus: retreating to reach the milestone
  alive is priced (−14/wave vs ~+15 for fighting it out), not forbidden — that is a design intent,
  not a leak.
- **step_cap stays a plain truncation** (no penalty, bootstrap V(s_final)): the step budget is a
  compute artifact invisible to the observation; penalizing it would put honest-slow play below a
  genuine wipe on an unlearnable signal. The profitable farms that used to make burning to the
  step cap attractive are closed at the source by mechanisms 1–2 and 4.

`runWon`/`runLost` are unchanged; `runWon` is documented as wave-200-only. Eval tooling reports
cap% (surrogate wins) alongside win%.

### 4. Damage pays once, shaping is potential-based (RD6, RD7)

- `hpDamageDealt` pays only **new lows** in each enemy's HP ratio (per-id, per-episode min-map):
  an enemy healing (Recover, Leftovers, boss regen) no longer resets the payable pool, capping
  total damage reward at ~1.0 per enemy for its life — mirroring the one-shot-per-faint enemyKo.
  A maxHp change (form change) drops the ratio without damage; the tracker rebases without paying.
- `statBoostReward`/`statusInflictionReward` (still 0 by default) are now **signed** deltas:
  boost → switch-out (stages reset) → re-boost nets ~0 instead of farming +w per cycle. Fainted
  mons no longer count as "statused" (FAINT is technically a StatusEffect), and both potentials
  are charged back (−w·Φ) at every terminal so ending an episode holding boosts is not free
  credit. The γ<1 residual (a hold bias of w·(1−γ)·Φ per step) is documented: keep
  w < |turnPenalty| / ((1−γ)·Φ_max) if you enable these.

## Config migration

`moneyGained` was **removed** (not re-semanticized): a stale config must fail loudly rather than
silently train at a different scale. Unknown or removed keys in the `reward:` section are hard
errors in `run_config.py` AND in the CLI's `--reward-config` parser; the rendered bridge drops
them with a console.error. New keys: `moneyGainedLog`, `moneySpentLog`, `stallStepPenalty`,
`stallPenalty`, `waveCapReached`.

v1 and v2 reward magnitudes are **not comparable** — re-baseline any stored eval numbers (the
campaign's v1 table: random −17.76, maxdamage 88.94, ppo_v9_first 272.46 at waves 20).

## Episode-end reference

| ending | wire | flags | payment |
|---|---|---|---|
| game over | `game_over` | terminated | runWon / runLost + Φ chargeback |
| wave cap | `done reason:"wave_cap"` | terminated | waveCapReached × clean-ratio + Φ chargeback |
| livelock | `done reason:"livelock"` | terminated | stallPenalty + Φ chargeback (steps already charged) |
| step cap | `done reason:"step_cap"` | truncated | none (bootstrap V(s_final)) |
| timeout / protocol error | — | truncated | none (env fault ≠ agent fault) |
