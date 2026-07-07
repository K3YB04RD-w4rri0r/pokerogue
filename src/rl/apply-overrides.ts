/**
 * Environment-agnostic game-override applier, shared by the headless runner
 * (standalone-setup.ts, from `--override=KEY=VALUE` CLI flags) and the
 * rendered browser bridge (browser-bridge.ts, from `&override=KEY=VALUE`
 * URL params) so both modes accept the same config surface.
 *
 * Deliberately contains NO Node-only imports (standalone-setup pulls in
 * node:fs and must never be imported from the browser).
 */

/**
 * Install getters for specific override keys on the live overrides module.
 * Unknown keys (typos) are reported via console.warn and skipped — a
 * silently unapplied override would invalidate a scenario without trace.
 *
 * Values persist for the lifetime of the module graph (headless in-process
 * resets keep them; the browser keeps them until a reload).
 */
export async function applyOverrideValues(overrides: Record<string, unknown>): Promise<void> {
  const overridesModule = await import("#app/overrides");
  const { defaultOverrides } = overridesModule;
  const target = (overridesModule as any).default ?? overridesModule;

  for (const [key, value] of Object.entries(overrides)) {
    if (!(key in defaultOverrides)) {
      console.warn(`[rl] Unknown override key "${key}" — not a property of DefaultOverrides, skipping`);
      continue;
    }
    Object.defineProperty(target, key, {
      get: () => value,
      configurable: true,
      enumerable: true,
    });
  }
}
