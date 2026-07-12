/**
 * Headless bootstrap for the RL framework.
 *
 * Boots the PokéRogue game in headless mode (no rendering, no browser, pure game
 * logic) suitable for RL training. Combines setup from three existing sources:
 *
 * 1. `src/rl/interactive-boot.ts` -- jsdom globals, canvas mock
 * 2. `src/rl/standalone-setup.ts` -- locale fetch, overrides reset, initTests()
 * 3. `test/test-utils/game-wrapper.ts` -- Phaser mock injection via injectMandatory()
 *
 * The initialization sequence is:
 * 1. Install jsdom browser globals (FontFace, localStorage, matchMedia, canvas, etc.)
 * 2. Install filesystem-backed fetch for i18n locale files
 * 3. Import i18n plugin (triggers i18next initialization)
 * 4. Reset overrides to defaults
 * 5. Run setupStubs + initializeGame() via initTests()
 * 6. Create Phaser.Game with type: Phaser.HEADLESS
 * 7. Create BattleScene, apply GameWrapper.injectMandatory() mocks
 * 8. Set globalScene.moveAnimations = false and other speed settings
 * 9. Install MockFetch for API calls
 *
 * IMPORTANT: Does NOT import from 'vitest' anywhere.
 */

import type { BattleScene } from "#app/battle-scene";

// ---------------------------------------------------------------------------
// Configuration interface
// ---------------------------------------------------------------------------

export interface HeadlessConfig {
  /** RNG seed for deterministic replay. Defaults to "rl-headless". */
  seed?: string;
  /** Whether to bypass the login phase. Defaults to true. */
  bypassLogin?: boolean;
  /** Whether to suppress console noise via MockConsole. Defaults to true (uses test infra). */
  quietConsole?: boolean;
  /**
   * Game override values applied once at boot (keys of the Overrides object,
   * e.g. BATTLE_STYLE_OVERRIDE, STARTING_WAVE_OVERRIDE). Used by the
   * coverage-corpus generator to force rare game situations. Persist for the
   * process lifetime — in-process resets keep them.
   */
  overrides?: Record<string, unknown> | undefined;
}

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

/** The Phaser.Game instance (created once, reused across episodes). */
let phaserGame: InstanceType<typeof import("phaser").Game> | null = null;

/** Whether the jsdom globals have already been installed. */
let globalsInstalled = false;

/** Whether initStandalone() has been called (one-time init). */
let standaloneInitialized = false;

// ---------------------------------------------------------------------------
// Phase 1: Install jsdom browser globals
// ---------------------------------------------------------------------------

/**
 * Install browser globals required by Phaser even in HEADLESS mode.
 *
 * This mirrors the setup in `interactive-boot.ts:14-76` but is idempotent
 * (safe to call multiple times). We use jsdom to provide a realistic DOM
 * environment that Phaser's module-load-time checks expect.
 *
 * WHY DUPLICATED: interactive-boot.ts immediately runs `await import(...)` at
 * module scope after setting globals, which is incompatible with our two-phase
 * init (install globals first, then call initHeadless() later). We extract
 * only the global installation logic here.
 */
async function installJsdomGlobals(): Promise<void> {
  if (globalsInstalled) {
    return;
  }

  // Dynamic import so jsdom is only loaded when this function is called
  const { JSDOM } = await import("jsdom");

  const dom = new JSDOM("<!DOCTYPE html><html><head></head><body></body></html>", {
    url: "http://localhost",
    pretendToBeVisual: true,
  });

  const win = dom.window as Record<string, unknown>;

  const globals: Record<string, unknown> = {
    window: win,
    document: win.document,
    navigator: win.navigator,
    HTMLCanvasElement: win.HTMLCanvasElement,
    HTMLElement: win.HTMLElement,
    HTMLVideoElement: win.HTMLVideoElement,
    HTMLDivElement: win.HTMLDivElement,
    Element: win.Element,
    screen: win.screen ?? { width: 1920, height: 1080, availWidth: 1920, availHeight: 1080 },
    localStorage: win.localStorage,
    Image: win.Image,
    XMLHttpRequest: win.XMLHttpRequest,
    DOMParser: win.DOMParser,
    Blob: win.Blob,
    requestAnimationFrame:
      typeof win.requestAnimationFrame === "function"
        ? (win.requestAnimationFrame as (cb: FrameRequestCallback) => number).bind(win)
        : (cb: () => void) => setTimeout(cb, 16),
    cancelAnimationFrame:
      typeof win.cancelAnimationFrame === "function"
        ? (win.cancelAnimationFrame as (handle: number) => void).bind(win)
        : clearTimeout,
    getComputedStyle:
      typeof win.getComputedStyle === "function"
        ? (win.getComputedStyle as (el: Element) => CSSStyleDeclaration).bind(win)
        : () => ({}),
    matchMedia: () => ({
      matches: false,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
    FontFace: class FontFace {
      family: string;
      source: string;
      descriptors: unknown;
      constructor(family: string, source: string, descriptors?: unknown) {
        this.family = family;
        this.source = source;
        this.descriptors = descriptors;
      }
      load(): Promise<FontFace> {
        return Promise.resolve(this);
      }
    },
  };

  // `self` is used by spector.js (bundled with Phaser CJS build)
  globals.self = win;

  for (const [key, value] of Object.entries(globals)) {
    try {
      Object.defineProperty(globalThis, key, {
        value,
        writable: true,
        configurable: true,
      });
    } catch {
      // Some globals may resist override on certain Node.js versions; skip them.
    }
  }

  // Canvas context stub -- Phaser needs this even in HEADLESS mode for TextureManager.
  // Mirrors interactive-boot.ts:79-110.
  const mockContext: Record<string, unknown> = {
    font: "",
    measureText: () => ({ width: 0 }),
    save: () => {},
    scale: () => {},
    clearRect: () => {},
    fillRect: () => {},
    fillText: () => {},
    getImageData: (_x: number, _y: number, w: number, h: number) => ({
      data: new Uint8ClampedArray(w * h * 4),
    }),
    putImageData: () => {},
    createImageData: () => ({ data: new Uint8ClampedArray(0) }),
    setTransform: () => {},
    drawImage: () => {},
    restore: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    closePath: () => {},
    stroke: () => {},
    translate: () => {},
    rotate: () => {},
    arc: () => {},
    fill: () => {},
    transform: () => {},
    rect: () => {},
    clip: () => {},
    createLinearGradient: () => ({ addColorStop: () => {} }),
    createRadialGradient: () => ({ addColorStop: () => {} }),
    canvas: null,
  };

  const mockWebGLContext: Record<string, unknown> = {
    getExtension: () => null,
    getParameter: () => null,
    getShaderPrecisionFormat: () => ({ precision: 23, rangeMin: 127, rangeMax: 127 }),
    createShader: () => ({}),
    createProgram: () => ({}),
    createBuffer: () => ({}),
    createTexture: () => ({}),
    createFramebuffer: () => ({}),
    createRenderbuffer: () => ({}),
    bindBuffer: () => {},
    bindTexture: () => {},
    bindFramebuffer: () => {},
    bindRenderbuffer: () => {},
    enable: () => {},
    disable: () => {},
    blendFunc: () => {},
    viewport: () => {},
    clear: () => {},
    clearColor: () => {},
    shaderSource: () => {},
    compileShader: () => {},
    attachShader: () => {},
    linkProgram: () => {},
    useProgram: () => {},
    getAttribLocation: () => 0,
    getUniformLocation: () => ({}),
    uniform1i: () => {},
    uniform1f: () => {},
    uniform2f: () => {},
    uniform3f: () => {},
    uniform4f: () => {},
    uniformMatrix4fv: () => {},
    enableVertexAttribArray: () => {},
    vertexAttribPointer: () => {},
    drawArrays: () => {},
    drawElements: () => {},
    bufferData: () => {},
    texImage2D: () => {},
    texParameteri: () => {},
    pixelStorei: () => {},
    activeTexture: () => {},
    deleteTexture: () => {},
    deleteBuffer: () => {},
    deleteShader: () => {},
    deleteProgram: () => {},
    deleteFramebuffer: () => {},
    deleteRenderbuffer: () => {},
    getProgramParameter: () => true,
    getShaderParameter: () => true,
    getProgramInfoLog: () => "",
    getShaderInfoLog: () => "",
    framebufferTexture2D: () => {},
    framebufferRenderbuffer: () => {},
    renderbufferStorage: () => {},
    checkFramebufferStatus: () => 36053,
    scissor: () => {},
    colorMask: () => {},
    stencilFunc: () => {},
    stencilOp: () => {},
    stencilMask: () => {},
    depthFunc: () => {},
    depthMask: () => {},
    blendEquation: () => {},
    blendFuncSeparate: () => {},
    generateMipmap: () => {},
    canvas: null,
    drawingBufferWidth: 800,
    drawingBufferHeight: 600,
  };

  // Override getContext on the jsdom HTMLCanvasElement prototype
  const htmlCanvas = win.HTMLCanvasElement as { prototype: { getContext: (type: string) => unknown } };
  htmlCanvas.prototype.getContext = function (type: string) {
    if (type === "2d") {
      return { ...mockContext, canvas: this };
    }
    if (type === "webgl" || type === "webgl2" || type === "experimental-webgl") {
      return { ...mockWebGLContext, canvas: this };
    }
    return null;
  };

  // document.fonts stub (i18n plugin uses document.fonts.add)
  const winDoc = win.document as Record<string, unknown>;
  Object.defineProperty(winDoc, "fonts", {
    writable: true,
    value: { add: () => {} },
  });

  // Gamepad stub
  const nav = win.navigator as Record<string, unknown>;
  (nav as { getGamepads: () => never[] }).getGamepads = () => [];

  // matchMedia on the jsdom window object (some code references window.matchMedia directly)
  if (typeof win.matchMedia !== "function") {
    (win as Record<string, unknown>).matchMedia = () => ({
      matches: false,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    });
  }

  // Install Phaser as a global BEFORE any game modules load.
  // phaser3-rex-plugins (bundled, not external) references `Phaser` as a global
  // variable at module evaluation time (e.g., `var Utils = Phaser.Renderer.WebGL.Utils`).
  // This must happen after jsdom globals are set (Phaser needs window/document)
  // but before any import chain that pulls in rex-plugins code.
  const Phaser = (await import("phaser")).default;
  (globalThis as Record<string, unknown>).Phaser = Phaser;
  (win as Record<string, unknown>).Phaser = Phaser;

  globalsInstalled = true;
}

// ---------------------------------------------------------------------------
// Phase 2: One-time static data initialization
// ---------------------------------------------------------------------------

/**
 * Run the one-time standalone setup: locale fetch, i18n, overrides, stubs,
 * and initializeGame().
 *
 * Delegates to `standalone-setup.ts:initStandalone()` which handles:
 * - Installing filesystem-backed fetch for locale files
 * - Importing the i18n plugin
 * - Resetting overrides to defaults
 * - Running test stubs (localStorage mock, Canvas mock, etc.) and initializeGame()
 */
async function runOneTimeInit(overrides?: Record<string, unknown>): Promise<void> {
  if (standaloneInitialized) {
    return;
  }
  const { initStandalone } = await import("#app/rl/standalone-setup");
  await initStandalone(overrides);
  standaloneInitialized = true;
}

// ---------------------------------------------------------------------------
// Phase 3: Create Phaser game + BattleScene with mock injection
// ---------------------------------------------------------------------------

/**
 * Create a Phaser.Game instance in HEADLESS mode.
 * The game instance is reused across episodes (only one is ever created).
 */
async function getOrCreatePhaserGame(seed: string): Promise<InstanceType<typeof import("phaser").Game>> {
  if (phaserGame) {
    return phaserGame;
  }

  const Phaser = (await import("phaser")).default;

  phaserGame = new Phaser.Game({
    type: Phaser.HEADLESS,
    // Minimal dimensions -- no actual rendering occurs
    width: 1920,
    height: 1080,
    // Disable audio at the Phaser level
    audio: {
      noAudio: true,
    },
  });

  // Seed the RNG after Game creation (Phaser.Math.RND is null before Game instantiation)
  Phaser.Math.RND.sow([seed]);

  return phaserGame;
}

/**
 * Create a BattleScene, apply all mock injections, and run scene.create().
 *
 * This follows the same flow as `GameManager` constructor + `GameWrapper.setScene()`:
 * 1. Create BattleScene (which calls initGlobalScene internally)
 * 2. Create GameWrapper (applies prototype stubs)
 * 3. Call gameWrapper.setScene(scene) which runs injectMandatory() + scene.create()
 * 4. Apply RL-specific speed settings
 * 5. Install MockFetch for API calls
 *
 * We import GameWrapper and related classes dynamically to avoid loading Phaser
 * before jsdom globals are in place.
 */
// Top-level children of these scene containers present at boot time are
// scene-lifetime (arena bases, trainer back-sprite, overlays, HUD texts);
// anything appearing later is per-episode debris. Captured once after the
// first scene creation, used by purgeEphemeralDisplayChildren on every reset.
const PURGED_CONTAINERS = ["field", "fieldUI"] as const;
let displayBaseline: WeakSet<object> | null = null;

function captureDisplayBaseline(scene: BattleScene): void {
  displayBaseline = new WeakSet<object>();
  const sceneAny = scene as any;
  for (const key of PURGED_CONTAINERS) {
    for (const child of sceneAny[key]?.list ?? []) {
      if (child && typeof child === "object") {
        displayBaseline.add(child);
      }
    }
  }
}

function purgeEphemeralDisplayChildren(scene: BattleScene): void {
  if (!displayBaseline) {
    return;
  }
  try {
    const sceneAny = scene as any;
    for (const key of PURGED_CONTAINERS) {
      const container = sceneAny[key];
      for (const child of [...(container?.list ?? [])]) {
        if (child && typeof child === "object" && !displayBaseline.has(child)) {
          container.remove(child, true);
        }
      }
    }
  } catch {
    /* cleanup is best-effort */
  }
}

/**
 * Handler-aware episode cleanup. Several UI handlers destroy their per-use
 * children only inside tween onComplete callbacks (e.g. the shop's
 * ModifierOption objects in ModifierSelectUiHandler.clear()); headless mock
 * tweens never fire those, so the LAST use of each handler in an episode
 * leaks its children until reset. Mid-episode uses are already cleaned by the
 * phase-router's idempotent show() patch — this covers the final use.
 * Handler-specific (no blind tree purging) to avoid destroying lazily-created
 * persistent members that handlers still reference.
 */
async function purgeHandlerEphemera(scene: BattleScene): Promise<void> {
  try {
    const { UiMode } = await import("#enums/ui-mode");
    const handlers = (scene as any).ui?.handlers;
    if (!handlers) {
      return;
    }
    const modifierHandler = handlers[UiMode.MODIFIER_SELECT] as
      | {
          modifierContainer?: { removeAll(destroy?: boolean): unknown };
          options?: unknown[];
          shopOptionsRows?: unknown[][];
        }
      | undefined;
    if (modifierHandler) {
      modifierHandler.modifierContainer?.removeAll(true);
      modifierHandler.options?.splice(0, modifierHandler.options.length);
      modifierHandler.shopOptionsRows?.splice(0, modifierHandler.shopOptionsRows.length);
    }

    // Option-select handler family (CONFIRM, OPTION_SELECT, ...): each show
    // creates a new BBCodeText; the destroy of the previous one lives in a
    // path the RL mode-transitions can skip, so stale texts accumulate in
    // optionSelectTextContainer. Keep only the handler's CURRENT text.
    for (const h of handlers as Record<string, any>[]) {
      const container = h?.optionSelectTextContainer;
      if (!container?.list) {
        continue;
      }
      const current = h.optionSelectText;
      for (const child of [...container.list]) {
        if (child !== current && child?.name === "text-option-select") {
          container.remove(child, true);
        }
      }
    }
  } catch {
    /* cleanup is best-effort */
  }
}

/**
 * Sweep destroyed-but-still-listed children out of mock display lists.
 *
 * Mock containers don't track parentage (and setting parentContainer on real
 * Phaser children drags their destroy through display-list internals that
 * need a real scene), so a destroyed child stays in its container's `list`:
 * one cursor image per menu open, one BBCodeText per option dialog, ... —
 * the dominant per-episode UI growth. Detection:
 *   - mock objects mark themselves `__rlDestroyed` in destroy()
 *   - real Phaser objects (e.g. rex BBCodeText) null their `scene` in destroy
 */
function sweepDestroyedDisplayChildren(scene: BattleScene): void {
  const isDestroyed = (c: any): boolean =>
    c?.__rlDestroyed === true || (typeof c?.type === "string" && "scene" in c && c.scene == null);

  const sweep = (node: any, depth = 0): void => {
    if (!node || !Array.isArray(node.list) || depth > 10) {
      return;
    }
    const kept = node.list.filter((c: any) => !isDestroyed(c));
    if (kept.length !== node.list.length) {
      node.list = kept;
    }
    for (const child of kept) {
      sweep(child, depth + 1);
    }
  };

  try {
    const sceneAny = scene as any;
    sweep(sceneAny.ui);
    sweep(sceneAny.field);
    sweep(sceneAny.fieldUI);
  } catch {
    /* cleanup is best-effort */
  }
}

async function createScene(
  game: InstanceType<typeof import("phaser").Game>,
  config: HeadlessConfig,
): Promise<BattleScene> {
  // Dynamic imports to ensure they run AFTER globals are installed
  const { GameWrapper } = await import("#test/test-utils/game-wrapper");
  const { BattleScene: BattleSceneClass } = await import("#app/battle-scene");
  const { globalScene } = await import("#app/global-scene");
  const { MockFetch } = await import("#test/test-utils/mocks/mock-fetch");
  const { mockFn } = await import("#app/rl/mocks/spy");
  const { ExpGainsSpeed } = await import("#enums/exp-gains-speed");
  const { ExpNotification } = await import("#enums/exp-notification");
  const { PlayerGender } = await import("#enums/player-gender");

  // In the headless Vite build, VITE_BYPASS_LOGIN is already compiled to "1" at build time,
  // so the app-constants module exports `bypassLogin = true`. The spy system cannot redefine
  // this (ESM module exports are non-configurable), and we don't need to -- it's already true.
  // Pass false to GameWrapper so it skips the spy attempt.
  const bypassLogin = false;

  // Clear localStorage between sessions (same as GameManager constructor)
  localStorage.clear();

  // Deterministic clock mode: MockClock skips its real 1ms interval; the
  // phase-router pumps timers at decision boundaries instead (drainMockTimers).
  (globalThis as { __rlDeterministicClock?: boolean }).__rlDeterministicClock = true;
  // Create GameWrapper (applies prototype stubs: MoveAnim, Pokemon, BattleScene)
  const gameWrapper = new GameWrapper(game, bypassLogin);

  let scene: BattleScene;

  if (globalScene) {
    // Reuse existing scene (same pattern as GameManager.resetScene)
    scene = globalScene;
    gameWrapper.scene = scene;
    // Re-apply mandatory mocks (they may have been restored between episodes)
    gameWrapper.injectMandatory();
    // Reset the scene state for a new episode
    scene.reset(false, true);
    // Purge ephemeral display children left by previous episodes (pokeballs,
    // anim sprites, damage numbers, battle-info boxes, end cards). In the
    // browser these are destroyed by tween onComplete callbacks; headless
    // mock tweens often never fire those, so children accumulate across
    // in-process resets (~500 nodes/episode observed -> linear slowdown +
    // retained-heap growth). The baseline is captured at the FIRST reset —
    // by then every scene-lifetime child provably exists; the one episode of
    // debris it includes is a harmless constant.
    if (displayBaseline) {
      purgeEphemeralDisplayChildren(scene);
    } else {
      captureDisplayBaseline(scene);
    }
    await purgeHandlerEphemera(scene);
    sweepDestroyedDisplayChildren(scene);
    // Clear starter preferences to avoid stale state across episodes
    // (mirrors GameManager.resetScene behavior)
    try {
      const { UiMode } = await import("#enums/ui-mode");
      const starterHandler = scene.ui?.handlers?.[UiMode.STARTER_SELECT];
      if (
        starterHandler
        && typeof (starterHandler as unknown as Record<string, unknown>).clearStarterPreferences === "function"
      ) {
        (starterHandler as unknown as { clearStarterPreferences: () => void }).clearStarterPreferences();
      }
    } catch {
      // UI handlers may not be initialized on first run; safe to skip
    }
    scene.phaseManager.toTitleScreen(true);
    scene.phaseManager.shiftPhase();
  } else {
    // First-time: create a new BattleScene
    scene = new BattleSceneClass();
    // Create TextInterceptor before setScene(), because setScene() calls scene.create()
    // which starts LoginPhase, which triggers UI.showText(). MockText patches
    // UI.prototype.showText to forward to scene.messageWrapper, so it must exist first.
    const { TextInterceptor } = await import("#test/test-utils/text-interceptor");
    new TextInterceptor(scene);
    // setScene() calls injectMandatory() + scene.create()
    gameWrapper.setScene(scene);
  }

  // Patch MessageUiHandler.prototype.showText to handle null/undefined text.
  // In headless mode, some UI text paths may receive undefined text (e.g., when
  // i18n keys aren't fully resolved or when UI handlers are called in unexpected
  // sequences). The original showTextInternal calls text.split() which crashes on undefined.
  // Idempotent: createScene runs on every in-process episode reset — without the
  // marker each reset would wrap the previous wrapper (stacked patches leak).
  const { MessageUiHandler } = await import("#ui/message-ui-handler");
  if (!(MessageUiHandler.prototype.showText as { __rlNullSafe?: boolean }).__rlNullSafe) {
    const origShowText = MessageUiHandler.prototype.showText;
    MessageUiHandler.prototype.showText = function (
      text: string,
      delay?: number | null,
      callback?: (() => void) | null,
      callbackDelay?: number | null,
      prompt?: boolean | null,
      promptDelay?: number | null,
    ) {
      origShowText.call(this, text ?? "", delay, callback, callbackDelay, prompt, promptDelay);
    };
    (MessageUiHandler.prototype.showText as { __rlNullSafe?: boolean }).__rlNullSafe = true;
  }

  // --- RL-specific speed settings ---
  // These match GameManager.runToTitle() settings for fast headless execution.
  scene.gameSpeed = 5;
  scene.moveAnimations = false;
  scene.showLevelUpStats = false;
  scene.expGainsSpeed = ExpGainsSpeed.SKIP;
  scene.expParty = ExpNotification.SKIP;
  scene.hpBarSpeed = 3;
  scene.enableTutorials = false;
  scene.gameData.gender = PlayerGender.MALE;
  scene.fieldVolume = 0;

  // Install MockFetch for API calls (account/info, savedata, etc.)
  global.fetch = mockFn(MockFetch) as typeof fetch;

  // --- Apply deterministic seed ---
  // BattleScene.reset() calls setSeed(randomString(24)) which uses Math.random() (NOT
  // Phaser's seeded RNG), so the --seed CLI arg has no effect on the game seed.
  // Fix: override the game seed with the config seed, then re-derive wave seeds.
  if (config.seed) {
    const Phaser = (await import("phaser")).default;
    Phaser.Math.RND.sow([config.seed]);
    scene.setSeed(config.seed);
    scene.resetSeed();
    // GameData's trainerId/secretId come from Math.random and gate every
    // shiny roll (and thus luck -> shop tiers) — derive them from the seed
    // so same-seed episodes are bitwise-reproducible (see deriveTrainerIds).
    const { deriveTrainerIds } = await import("#rl/apply-overrides");
    const ids = deriveTrainerIds(config.seed);
    scene.gameData.trainerId = ids.trainerId;
    scene.gameData.secretId = ids.secretId;
  }

  return scene;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialize the game in headless mode.
 *
 * This is the main entry point for the RL framework. It performs all necessary
 * setup (jsdom globals, i18n, static game data, Phaser, mocks) and returns a
 * ready-to-use BattleScene.
 *
 * The first call performs one-time initialization (heavy). Subsequent calls
 * reuse the Phaser.Game instance and reset the scene for a new episode.
 *
 * @param config - Optional configuration for the headless session.
 * @returns A BattleScene ready for phase execution.
 *
 * @example
 * ```ts
 * const scene = await initHeadless({ seed: "my-seed" });
 * // scene is now ready for battle -- use phase router or GameManager to drive it
 * ```
 */
export async function initHeadless(config?: HeadlessConfig): Promise<BattleScene> {
  const resolvedConfig: HeadlessConfig = {
    seed: "rl-headless",
    bypassLogin: true,
    quietConsole: true,
    ...config,
  };

  // Phase 1: Install browser globals (idempotent)
  await installJsdomGlobals();

  // Phase 2: One-time static data init (idempotent)
  await runOneTimeInit(resolvedConfig.overrides);

  // Phase 3: Create/reuse Phaser.Game
  const game = await getOrCreatePhaserGame(resolvedConfig.seed!);

  // Phase 4: Create/reset BattleScene with mock injection
  const scene = await createScene(game, resolvedConfig);

  return scene;
}

/**
 * Tear down the headless environment.
 *
 * Cleans up:
 * - The MockClock's setInterval timer
 * - Restores all spied-on mocks
 * - Destroys the Phaser.Game instance
 * - Resets module-level state so initHeadless() can be called again
 *
 * Call this when you're done with the RL session (e.g., at the end of training).
 */
export async function destroyHeadless(): Promise<void> {
  const { restoreAllMocks } = await import("#app/rl/mocks/spy");
  restoreAllMocks();

  // Destroy the Phaser game instance
  if (phaserGame) {
    try {
      phaserGame.destroy(true);
    } catch {
      // Phaser.destroy() may throw in headless/Node.js context; safe to ignore.
    }
    phaserGame = null;
  }

  // Reset init flags so a fresh session can be started
  standaloneInitialized = false;
}

/**
 * Reset the scene for a new episode without full teardown.
 *
 * This is faster than destroyHeadless() + initHeadless() because it reuses
 * the existing Phaser.Game and re-initializes only the BattleScene state.
 * Use this between RL episodes for maximum throughput.
 *
 * @param config - Optional configuration for the new episode.
 * @returns A fresh BattleScene ready for the next episode.
 */
export async function resetHeadless(config?: HeadlessConfig): Promise<BattleScene> {
  const resolvedConfig: HeadlessConfig = {
    seed: "rl-headless",
    bypassLogin: true,
    quietConsole: true,
    ...config,
  };

  if (!phaserGame) {
    // If no game exists, fall back to full init
    return initHeadless(resolvedConfig);
  }

  const { restoreAllMocks } = await import("#app/rl/mocks/spy");
  restoreAllMocks();

  // Re-seed RNG for the new episode
  const Phaser = (await import("phaser")).default;
  Phaser.Math.RND.sow([resolvedConfig.seed ?? "rl-headless"]);

  // Create/reset scene
  const scene = await createScene(phaserGame, resolvedConfig);

  return scene;
}
