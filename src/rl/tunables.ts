/**
 * Robustness tunables for the RL environment, centralized.
 *
 * Every value here directly affects when an episode truncates or a guard
 * fires — they were previously scattered literals across phase-router,
 * browser-bridge and cli. Values are UNCHANGED from their original sites;
 * each entry documents what moving it does.
 */

/**
 * Consecutive executed modifier decisions with an unchanged progress
 * signature (money | remaining rewards | party HP) before the shop mask
 * collapses to skip-only. Lower = earlier forced exit from a livelocked
 * shop, at the cost of cutting short a legitimately slow shopper.
 */
export const MODIFIER_LIVELOCK_LIMIT = 4;

/**
 * Default progress-aware watchdog for advanceToNextDecision (ms). The
 * deadline SLIDES on genuine phase/tween progress, so this only fires on a
 * true stall. CLI --timeout overrides per run.
 */
export const DECISION_TIMEOUT_MS = 30_000;

/**
 * Tween/timer durations at or above this are treated as "infinite"
 * (decorative loops) and EXCLUDED from the progress-aware watchdog's
 * pending-work scan, so a looping background animation can't keep a stalled
 * episode alive forever.
 */
export const INFINITE_TWEEN_SENTINEL_MS = 600_000;

/**
 * Hard step budget per episode as a multiple of the wave cap — a safety
 * backstop behind the real wave-cap truncation (a wave rarely needs more
 * than ~50 decisions).
 */
export const MAX_STEPS_PER_WAVE = 50;

/**
 * Consecutive already-seen-observation decisions before a transport (headless
 * CLI or rendered bridge) declares a livelock and ends the episode as
 * TERMINATED with the stallPenalty (reward v2). The Python env keeps its own
 * copy of this limit as a defense-in-depth backstop; a v6 CLI always trips
 * first (it checks before sending the state).
 */
export const NO_PROGRESS_LIMIT = 40;

/**
 * Repeated-observation decisions tolerated before stallStepPenalty starts
 * being charged per decision (reward v2 [RD2]). The per-step charge exists
 * because a lump penalty 40 steps out is discounted to ~0.67x at gamma=0.99,
 * which left stalling preferable to a worst-case loss. Legitimate play
 * produces novel observations (HP/PP/turn counters move), so exceeding even
 * a few consecutive repeats means a no-op loop.
 */
export const STALL_GRACE_STEPS = 5;

/** Rendered transport: time-scale applied to cinematic sequences (evolution,
 *  egg hatching) so watching an agent doesn't stall on animations. */
export const CINEMATIC_TIMESCALE = 50;

/** Rendered transport: how long to wait for evolution assets before giving
 *  up and fast-forwarding anyway (ms). */
export const EVOLUTION_ASSET_RACE_MS = 15_000;
