/**
 * Environment-agnostic game-override applier, shared by the headless runner
 * (standalone-setup.ts, from `--override=KEY=VALUE` CLI flags) and the
 * rendered browser bridge (browser-bridge.ts, from `&override=KEY=VALUE`
 * URL params) so both modes accept the same config surface.
 *
 * Deliberately contains NO Node-only imports (standalone-setup pulls in
 * node:fs and must never be imported from the browser). Enum imports are fine.
 */

import { BattleType } from "#enums/battle-type";

/**
 * Install getters for specific override keys on the live overrides module.
 * Unknown keys (typos) are reported via console.warn and skipped — a
 * silently unapplied override would invalidate a scenario without trace.
 *
 * Mystery Encounters are PERMANENTLY disabled in the RL environment (their
 * option phases fall outside the 58-action interface), and this is the single
 * choke point both transports funnel through — so the hard-disable is enforced
 * here and cannot be re-enabled by any override. Two independent re-enable
 * vectors are closed: MYSTERY_ENCOUNTER_RATE_OVERRIDE (forced to 0 last, so a
 * user value can't win) and BATTLE_TYPE_OVERRIDE=MYSTERY_ENCOUNTER (which forces
 * an ME wave directly, bypassing the rate gate — refused).
 *
 * Values persist for the lifetime of the module graph (headless in-process
 * resets keep them; the browser keeps them until a reload).
 */
export async function applyOverrideValues(
  overrides: Record<string, unknown>,
  /** The overrides module AS SEEN BY THE CALLER's import graph. Under the
   *  Vite dev server a dynamic import here can resolve a DIFFERENT module
   *  instance than the one game code reads (observed in rendered mode:
   *  overrides "applied" on one instance, game reading pristine defaults
   *  from another — the reason the committed MYSTERY_ENCOUNTER_RATE_OVERRIDE
   *  belt-and-suspenders exists). Callers inside the game graph (the browser
   *  bridge) MUST pass their statically-imported module. */
  targetModule?: { default?: Record<string, unknown>; defaultOverrides: Record<string, unknown> },
): Promise<void> {
  const overridesModule = targetModule ?? (await import("#app/overrides"));
  const { defaultOverrides } = overridesModule;
  const target = ((overridesModule as any).default ?? overridesModule) as Record<string, unknown>;

  const install = (key: string, value: unknown): void => {
    Object.defineProperty(target, key, {
      get: () => value,
      configurable: true,
      enumerable: true,
    });
  };

  for (const [key, value] of Object.entries(overrides)) {
    if (!(key in defaultOverrides)) {
      console.warn(`[rl] Unknown override key "${key}" — not a property of DefaultOverrides, skipping`);
      continue;
    }
    // BATTLE_TYPE_OVERRIDE=MYSTERY_ENCOUNTER forces an ME wave directly
    // (battle-scene.ts:1306-1309) and short-circuits the rate gate — refuse it
    // while letting every other battle type through.
    if (key === "BATTLE_TYPE_OVERRIDE" && Number(value) === BattleType.MYSTERY_ENCOUNTER) {
      console.warn("[rl] Mystery Encounters are disabled — ignoring BATTLE_TYPE_OVERRIDE=MYSTERY_ENCOUNTER");
      continue;
    }
    install(key, value);
  }

  // Force ME spawn rate off LAST so no caller override can re-enable it
  // (rate 0 ⇒ isWaveMysteryEncounter always returns false, battle-scene.ts:3559).
  install("MYSTERY_ENCOUNTER_RATE_OVERRIDE", 0);
}

/**
 * Derive deterministic 16-bit trainerId/secretId from the episode seed.
 *
 * `new GameData()` (constructed on every scene.reset(clearData) and on every
 * fresh browser session) draws both ids from Math.random. They XOR into `E`
 * of the shiny formula (`(E ^ F) < threshold`, pokemon.ts) — so with unseeded
 * ids, the shiny verdict of every generated mon is nondeterministic across
 * same-seed runs (~1/2048 per mon), a shiny consumes an extra seeded RNG draw
 * (shifting the whole downstream stream), and party luck feeds shop tiers.
 *
 * FNV-1a over the seed string: no game-RNG stream consumption, identical in
 * both transports, stable across processes.
 */
export function deriveTrainerIds(seed: string): { trainerId: number; secretId: number } {
  let h = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const trainerId = h & 0xffff;
  // Second round with a domain separator so the two ids are independent.
  h ^= 0x5f;
  h = Math.imul(h, 0x01000193) >>> 0;
  const secretId = h & 0xffff;
  return { trainerId, secretId };
}
