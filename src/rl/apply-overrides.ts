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
export async function applyOverrideValues(overrides: Record<string, unknown>): Promise<void> {
  const overridesModule = await import("#app/overrides");
  const { defaultOverrides } = overridesModule;
  const target = (overridesModule as any).default ?? overridesModule;

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
