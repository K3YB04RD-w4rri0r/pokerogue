/**
 * Standalone initialization for the headless game runner.
 * Replaces `test/setup/vitest.setup.ts` without requiring Vitest.
 *
 * Handles:
 * - Global stubs (localStorage, Canvas, etc.) via test-file-initialization
 * - i18n locale loading via filesystem-backed fetch
 * - Overrides reset to defaults
 */

import { defaultOverrides } from "#app/overrides";
import { initTests } from "#test/test-utils/test-file-initialization";
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
export async function initStandalone(): Promise<void> {
  // Install locale fetch before i18n plugin loads
  setupLocaleFetch();

  // Import i18n plugin (triggers top-level await for i18next.init)
  await import("#plugins/i18n");

  // Reset overrides to defaults (replaces vi.mock for overrides module)
  resetOverrides();

  // Run the standard test stubs (localStorage, Canvas, matchMedia, etc.)
  // and initialize game data (abilities, species, moves, etc.)
  initTests();
}
