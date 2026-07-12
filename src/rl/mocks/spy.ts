/**
 * Lightweight standalone spy/mock utilities that replace Vitest's `vi.fn()` and `vi.spyOn()`.
 *
 * These use the same `Object.defineProperty` mechanism as Vitest under the hood,
 * allowing the test harness to run without a Vitest dependency.
 *
 * Vitest interop contract (load-bearing — the shared test harness hands these
 * spies to ordinary vitest tests):
 * - Spies are stamped with `_isMockFunction`, `getMockName()` and a live `.mock`
 *   object, which is exactly what `@vitest/expect`'s call matchers read, so
 *   `expect(spy).toHaveBeenCalledWith(...)` works on shim spies.
 * - Because of the stamp, a later `vi.spyOn(obj, prop)` on a shim-spied member
 *   returns the shim spy itself (vitest early-returns recognized mocks) — so the
 *   full `MockInstance` method surface used across the suite must exist here,
 *   or tests die with "x is not a function" at runtime with no compile error.
 * - Getter spies install the (stamped, recording) spy function itself as the
 *   getter. Installing a raw closure instead would poison the property for the
 *   whole worker when mixed with `vi.spyOn(obj, prop, "get")` + `restoreMocks`.
 */

/** Tracked spy entry for restoration */
interface SpyEntry {
  obj: any;
  prop: string;
  descriptor: PropertyDescriptor | undefined;
}

/** Global registry of active spies for bulk restoration */
const activeSpies: SpyEntry[] = [];

/** One recorded call result, matching vitest's `MockResult` shape. */
interface MockResult {
  type: "return" | "throw";
  value: any;
}

/** The vitest-compatible `.mock` metadata object. */
export interface MockContext<T extends (...args: any[]) => any = (...args: any[]) => any> {
  calls: Parameters<T>[];
  results: MockResult[];
  instances: any[];
  invocationCallOrder: number[];
  lastCall: Parameters<T> | undefined;
  contexts: any[];
  settledResults: any[];
}

/**
 * A mock function instance compatible with the subset of Vitest's `MockInstance` the
 * shared test harness (and any test re-acquiring a shim spy via `vi.spyOn`) uses.
 */
export interface MockInstance<T extends (...args: any[]) => any = (...args: any[]) => any> {
  (...args: Parameters<T>): ReturnType<T>;
  mockReturnValue(val: ReturnType<T>): this;
  mockReturnValueOnce(val: ReturnType<T>): this;
  mockReturnThis(): this;
  mockResolvedValue(val: Awaited<ReturnType<T>>): this;
  mockResolvedValueOnce(val: Awaited<ReturnType<T>>): this;
  mockRejectedValue(err: unknown): this;
  mockRejectedValueOnce(err: unknown): this;
  mockImplementation(fn: (...args: Parameters<T>) => ReturnType<T>): this;
  mockImplementationOnce(fn: (...args: Parameters<T>) => ReturnType<T>): this;
  mockClear(): this;
  mockReset(): this;
  mockRestore(): void;
  calls: Parameters<T>[];
  mock: MockContext<T>;
  _isMockFunction: true;
  getMockName(): string;
}

/** Global invocation counter for ordering spy calls */
let invocationCounter = 0;

/**
 * Stamp Vitest-compatible metadata onto a spy function so that
 * `expect(spy).toHaveBeenCalledWith(...)` and similar matchers work.
 */
function stampVitestCompat(fn: any, calls: any[][], mockName = "spy"): void {
  fn._isMockFunction = true;
  fn.getMockName = () => mockName;

  // The `.mock` object must reflect the same `calls` array
  fn.mock = {
    calls,
    results: [] as MockResult[],
    instances: [] as any[],
    invocationCallOrder: [] as number[],
    lastCall: undefined as any[] | undefined,
    contexts: [] as any[],
    settledResults: [] as any[],
  };
}

/**
 * Record a call result in the Vitest-compatible `.mock` metadata.
 */
function recordResult(fn: any, result: MockResult, thisArg: any): void {
  fn.mock.results.push(result);
  fn.mock.instances.push(thisArg);
  fn.mock.contexts.push(thisArg);
  fn.mock.invocationCallOrder.push(++invocationCounter);
  fn.mock.lastCall = fn.mock.calls.at(-1);
}

/** Mutable behavior state shared by the call body and the mock* methods. */
interface SpyState {
  /** The implementation restored by mockReset() (the original fn/method). */
  baseImpl: ((...args: any[]) => any) | undefined;
  currentImpl: ((...args: any[]) => any) | undefined;
  onceQueue: ((...args: any[]) => any)[];
  calls: any[][];
}

/** The shared call body: consume once-queue, invoke impl, record result/throw. */
function invokeSpy(spy: any, state: SpyState, thisArg: any, args: any[]): any {
  state.calls.push(args);
  const impl = state.onceQueue.length > 0 ? state.onceQueue.shift() : state.currentImpl;
  let result: any;
  try {
    result = impl ? impl.apply(thisArg, args) : undefined;
  } catch (err) {
    recordResult(spy, { type: "throw", value: err }, thisArg);
    throw err;
  }
  recordResult(spy, { type: "return", value: result }, thisArg);
  return result;
}

/**
 * Attach the full mock-method surface to a spy.
 *
 * @param onBehaviorChange - called whenever a mock* method changes behavior;
 *   method spies use this to (re-)install themselves on the target object.
 */
function attachMockMethods(spy: any, state: SpyState, onBehaviorChange: () => void = () => {}): void {
  spy.mockReturnValue = (val: any) => {
    state.currentImpl = () => val;
    onBehaviorChange();
    return spy;
  };
  spy.mockReturnValueOnce = (val: any) => {
    state.onceQueue.push(() => val);
    onBehaviorChange();
    return spy;
  };
  spy.mockReturnThis = () => {
    state.currentImpl = function (this: any) {
      return this;
    };
    onBehaviorChange();
    return spy;
  };
  spy.mockResolvedValue = (val: any) => {
    state.currentImpl = () => Promise.resolve(val);
    onBehaviorChange();
    return spy;
  };
  spy.mockResolvedValueOnce = (val: any) => {
    state.onceQueue.push(() => Promise.resolve(val));
    onBehaviorChange();
    return spy;
  };
  spy.mockRejectedValue = (err: unknown) => {
    state.currentImpl = () => Promise.reject(err);
    onBehaviorChange();
    return spy;
  };
  spy.mockRejectedValueOnce = (err: unknown) => {
    state.onceQueue.push(() => Promise.reject(err));
    onBehaviorChange();
    return spy;
  };
  spy.mockImplementation = (fn: (...args: any[]) => any) => {
    state.currentImpl = fn;
    onBehaviorChange();
    return spy;
  };
  spy.mockImplementationOnce = (fn: (...args: any[]) => any) => {
    state.onceQueue.push(fn);
    onBehaviorChange();
    return spy;
  };
  spy.mockClear = () => {
    state.calls.length = 0;
    spy.mock.results.length = 0;
    spy.mock.instances.length = 0;
    spy.mock.contexts.length = 0;
    spy.mock.invocationCallOrder.length = 0;
    spy.mock.lastCall = undefined;
    return spy;
  };
  spy.mockReset = () => {
    spy.mockClear();
    state.onceQueue.length = 0;
    state.currentImpl = state.baseImpl;
    return spy;
  };
}

/**
 * Create a standalone mock function, optionally wrapping an existing implementation.
 * Replaces `vi.fn()` and `vi.fn(impl)`.
 */
export function mockFn<T extends (...args: any[]) => any>(impl?: T): MockInstance<T> {
  const state: SpyState = { baseImpl: impl, currentImpl: impl, onceQueue: [], calls: [] };

  const fn = function (this: any, ...args: any[]): any {
    return invokeSpy(fn, state, this, args);
  } as unknown as MockInstance<T>;

  fn.calls = state.calls as Parameters<T>[];
  stampVitestCompat(fn, state.calls);
  attachMockMethods(fn, state);

  fn.mockRestore = () => {
    (fn as any).mockReset();
  };

  return fn;
}

/**
 * Spy on an object's property or method, replacing it with a mock.
 * Replaces `vi.spyOn(obj, prop)` and `vi.spyOn(obj, prop, "get")`.
 *
 * @param obj - The target object
 * @param prop - The property/method name
 * @param accessType - If `"get"`, spy on the getter; otherwise spy on the method
 */
export function spyOn<T extends object>(obj: T, prop: string & keyof T, accessType?: "get" | "set"): MockInstance {
  // Save the original descriptor for restoration
  const originalDescriptor = Object.getOwnPropertyDescriptor(obj, prop);
  activeSpies.push({ obj, prop, descriptor: originalDescriptor });

  if (accessType === "get") {
    return createGetterSpy(obj, prop, originalDescriptor);
  }

  return createMethodSpy(obj, prop);
}

/**
 * Create a spy that intercepts a getter property.
 *
 * The spy function ITSELF is installed as the getter (stamped + recording):
 * reads land in `.mock.calls`, and a later `vi.spyOn(obj, prop, "get")` sees a
 * recognized mock instead of capturing an anonymous closure as the "original"
 * getter (which would survive restoration and poison the worker-global default).
 */
function createGetterSpy(obj: any, prop: string, originalDescriptor: PropertyDescriptor | undefined): MockInstance {
  const originalGet = originalDescriptor?.get ?? (() => originalDescriptor?.value);
  const state: SpyState = { baseImpl: originalGet, currentImpl: originalGet, onceQueue: [], calls: [] };

  const spy = function (this: any, ...args: any[]): any {
    return invokeSpy(spy, state, this, args);
  } as MockInstance;

  spy.calls = state.calls;
  stampVitestCompat(spy, state.calls, prop);

  const install = () => {
    const current = Object.getOwnPropertyDescriptor(obj, prop);
    if (current?.get !== spy) {
      Object.defineProperty(obj, prop, { get: spy, configurable: true });
    }
  };
  attachMockMethods(spy, state, install);

  spy.mockRestore = () => {
    if (originalDescriptor) {
      Object.defineProperty(obj, prop, originalDescriptor);
    } else {
      delete obj[prop];
    }
    (spy as any).mockReset();
  };

  return spy;
}

/** Create a spy that intercepts a method call */
function createMethodSpy(obj: any, prop: string): MockInstance {
  const originalMethod = obj[prop];
  const baseImpl = typeof originalMethod === "function" ? originalMethod : undefined;
  const state: SpyState = { baseImpl, currentImpl: baseImpl, onceQueue: [], calls: [] };

  const spy = function (this: any, ...args: any[]): any {
    return invokeSpy(spy, state, this, args);
  } as MockInstance;

  spy.calls = state.calls;
  stampVitestCompat(spy, state.calls, prop);

  const install = () => {
    if (obj[prop] !== spy) {
      obj[prop] = spy;
    }
  };
  attachMockMethods(spy, state, install);

  spy.mockRestore = () => {
    obj[prop] = originalMethod;
    (spy as any).mockReset();
  };

  // Install the spy on the object
  install();

  return spy;
}

/**
 * Restore all active spies to their original state.
 * Should be called between test runs or game sessions.
 */
export function restoreAllMocks(): void {
  while (activeSpies.length > 0) {
    const entry = activeSpies.pop()!;
    if (entry.descriptor) {
      Object.defineProperty(entry.obj, entry.prop, entry.descriptor);
    } else {
      delete entry.obj[entry.prop];
    }
  }
}

/**
 * Polling-based async wait until a condition is met.
 * Replaces `vi.waitUntil()`.
 */
export async function waitUntil(
  conditionFn: () => boolean,
  opts: { timeout?: number; interval?: number } = {},
): Promise<void> {
  const { timeout = 5000, interval = 50 } = opts;
  const start = Date.now();
  while (!conditionFn()) {
    if (Date.now() - start > timeout) {
      throw new Error(`waitUntil timed out after ${timeout}ms`);
    }
    await new Promise(r => setTimeout(r, interval));
  }
}
