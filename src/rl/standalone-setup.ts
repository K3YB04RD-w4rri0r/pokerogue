/**
 * Standalone initialization for the headless game runner.
 * Replaces `test/setup/vitest.setup.ts` without requiring Vitest.
 *
 * Handles:
 * - Global stubs (localStorage, Canvas, etc.) via test-file-initialization
 * - i18n locale loading via filesystem-backed fetch
 * - Overrides reset to defaults
 */

// IMPORTANT: No static imports of game modules here. Game code (overrides,
// test-file-initialization, and everything they transitively pull in, e.g.
// trainer-config.ts) calls i18next.t() at module-evaluation time. A static
// import would evaluate that whole graph when THIS module loads — before
// initStandalone() has initialized i18next — leaving every module-init-time
// translation undefined (symptom: crash in TrainerConfig.getTitle when a
// rival/gendered trainer is encountered). All game imports must be dynamic,
// inside initStandalone(), after `await import("#plugins/i18n")`.
import fs from "node:fs";
import path from "node:path";

/**
 * Reset the overrides default export so all override properties return their defaults.
 *
 * In Vitest this is done via `vi.mock(import("#app/overrides"))` which replaces the module.
 * Here we use `Object.defineProperty` to install getters on the live module export
 * that return the corresponding `defaultOverrides` values.
 */
async function resetOverrides(): Promise<void> {
  // Dynamic import to get the live module default export
  const overridesModule = await import("#app/overrides");
  const { defaultOverrides } = overridesModule;
  const overrides = (overridesModule as any).default ?? overridesModule;

  for (const key of Object.keys(defaultOverrides)) {
    Object.defineProperty(overrides, key, {
      get: () => (defaultOverrides as any)[key],
      configurable: true,
      enumerable: true,
    });
  }
}

/**
 * Install a fetch override that serves locale JSON files from the local filesystem.
 *
 * The i18n plugin uses `i18next-http-backend` which calls `fetch()` for locale files.
 * In Vitest this is handled by MSW (Mock Service Worker). For standalone operation
 * we intercept fetch calls to `/locales/` and read from disk instead.
 */
function setupLocaleFetch(): void {
  const baseFetch = global.fetch;

  global.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;

    if (url.includes("/locales/")) {
      // Strip query params and extract relative locale path
      const cleanUrl = url.split("?")[0];
      const localeMatch = cleanUrl.match(/locales\/(.+)$/);
      if (localeMatch) {
        const localePath = path.join(process.cwd(), "locales", localeMatch[1]);
        try {
          const data = fs.readFileSync(localePath, "utf-8");
          return new Response(data, {
            status: 200,
            headers: new Headers({ "Content-Type": "application/json" }),
          });
        } catch {
          return new Response("{}", {
            status: 200,
            headers: new Headers({ "Content-Type": "application/json" }),
          });
        }
      }
    }

    // For Google Fonts requests (triggered by i18n font loading), return empty
    if (url.includes("fonts.googleapis.com")) {
      return new Response("", { status: 200 });
    }

    // Delegate everything else to the existing fetch (likely MockFetch)
    if (baseFetch) {
      return baseFetch(input as any, init);
    }
    return new Response("{}", { status: 404 });
  }) as typeof fetch;
}

/**
 * Initialize the standalone headless environment.
 * Must be called once before creating any GameManager instances.
 */
export async function initStandalone(overrides?: Record<string, unknown>): Promise<void> {
  // Install locale fetch before i18n plugin loads
  setupLocaleFetch();

  // Import i18n plugin (triggers top-level await for i18next.init).
  // This MUST complete before any game module is imported — game modules
  // (e.g. trainer-config.ts) call i18next.t() at module-evaluation time.
  await import("#plugins/i18n");

  // Reset overrides to defaults (replaces vi.mock for overrides module)
  await resetOverrides();

  // Mystery Encounters are REMOVED from the RL environment by decision:
  // their option phases are not part of the 58-action interface (each ME
  // would burn a router timeout), so the env's game scope is classic mode
  // without MEs. An explicit MYSTERY_ENCOUNTER_RATE_OVERRIDE in the caller's
  // overrides wins (deliberate re-enable for manual play/experiments).
  const effectiveOverrides: Record<string, unknown> = {
    MYSTERY_ENCOUNTER_RATE_OVERRIDE: 0,
    ...(overrides ?? {}),
  };

  // Apply override values on top of the defaults (e.g. BATTLE_STYLE_OVERRIDE /
  // STARTING_WAVE_OVERRIDE from the CLI's --override flags, used by the
  // coverage-corpus generator to force rare situations).
  await applyOverrides(effectiveOverrides);

  // Run the standard test stubs (localStorage, Canvas, matchMedia, etc.)
  // and initialize game data (abilities, species, moves, etc.).
  // Imported dynamically so the game-data module graph evaluates only now.
  const { initTests } = await import("#test/test-utils/test-file-initialization");
  initTests();
}

/**
 * Install getters for specific override keys on the live overrides module.
 * Unknown keys (typos) are reported to stderr and skipped — a silently
 * unapplied override would invalidate a corpus scenario without trace.
 */
async function applyOverrides(overrides: Record<string, unknown>): Promise<void> {
  const overridesModule = await import("#app/overrides");
  const { defaultOverrides } = overridesModule;
  const target = (overridesModule as any).default ?? overridesModule;

  for (const [key, value] of Object.entries(overrides)) {
    if (!(key in defaultOverrides)) {
      process.stderr.write(
        `[standalone-setup] Unknown override key "${key}" — not a property of DefaultOverrides, skipping\n`,
      );
      continue;
    }
    Object.defineProperty(target, key, {
      get: () => value,
      configurable: true,
      enumerable: true,
    });
  }
}
