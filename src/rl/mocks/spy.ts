/**
 * Lightweight standalone spy/mock utilities that replace Vitest's `vi.fn()` and `vi.spyOn()`.
 *
 * These use the same `Object.defineProperty` mechanism as Vitest under the hood,
 * allowing the test harness to run without a Vitest dependency.
 */

/** Tracked spy entry for restoration */
interface SpyEntry {
  obj: any;
  prop: string;
  descriptor: PropertyDescriptor | undefined;
}

/** Global registry of active spies for bulk restoration */
const activeSpies: SpyEntry[] = [];

/**
 * A mock function instance compatible with the subset of Vitest's `MockInstance` that the
 * test harness actually uses: `.mockReturnValue()`, `.mockReturnValueOnce()`,
 * `.mockImplementation()`, `.mockRestore()`, and `.calls`.
 */
export interface MockInstance<T extends (...args: any[]) => any = (...args: any[]) => any> {
  (...args: Parameters<T>): ReturnType<T>;
  mockReturnValue(val: ReturnType<T>): this;
  mockReturnValueOnce(val: ReturnType<T>): this;
  mockImplementation(fn: (...args: Parameters<T>) => ReturnType<T>): this;
  mockRestore(): void;
  calls: Parameters<T>[];
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
    results: [] as any[],
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
function recordResult(fn: any, result: any, thisArg: any): void {
  fn.mock.results.push({ type: "return", value: result });
  fn.mock.instances.push(thisArg);
  fn.mock.invocationCallOrder.push(++invocationCounter);
  fn.mock.lastCall = fn.mock.calls.at(-1);
}

/**
 * Create a standalone mock function, optionally wrapping an existing implementation.
 * Replaces `vi.fn()` and `vi.fn(impl)`.
 */
export function mockFn<T extends (...args: any[]) => any>(impl?: T): MockInstance<T> {
  let currentImpl: ((...args: any[]) => any) | undefined = impl;
  const onceQueue: any[] = [];
  const calls: any[][] = [];

  const fn = function (this: any, ...args: any[]): any {
    calls.push(args);
    let result: any;
    if (onceQueue.length > 0) {
      result = onceQueue.shift();
    } else if (currentImpl) {
      result = currentImpl.apply(this, args);
    }
    recordResult(fn, result, this);
    return result;
  } as MockInstance<T>;

  fn.calls = calls;
  stampVitestCompat(fn, calls);

  fn.mockReturnValue = (val: any) => {
    currentImpl = () => val;
    return fn;
  };

  fn.mockReturnValueOnce = (val: any) => {
    onceQueue.push(val);
    return fn;
  };

  fn.mockImplementation = (newImpl: (...args: any[]) => any) => {
    currentImpl = newImpl;
    return fn;
  };

  fn.mockRestore = () => {
    currentImpl = impl;
    onceQueue.length = 0;
    calls.length = 0;
    fn.mock.results.length = 0;
    fn.mock.instances.length = 0;
    fn.mock.invocationCallOrder.length = 0;
    fn.mock.lastCall = undefined;
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

/** Create a spy that intercepts a getter property */
function createGetterSpy(obj: any, prop: string, originalDescriptor: PropertyDescriptor | undefined): MockInstance {
  const spy = mockFn();

  spy.mockReturnValue = (val: any) => {
    Object.defineProperty(obj, prop, {
      get: () => val,
      configurable: true,
    });
    return spy;
  };

  spy.mockReturnValueOnce = (val: any) => {
    const previousGetter =
      Object.getOwnPropertyDescriptor(obj, prop)?.get ?? originalDescriptor?.get ?? (() => originalDescriptor?.value);

    let consumed = false;
    Object.defineProperty(obj, prop, {
      get: () => {
        if (!consumed) {
          consumed = true;
          // Restore previous getter after one read
          Object.defineProperty(obj, prop, {
            get: previousGetter,
            configurable: true,
          });
          return val;
        }
        return previousGetter();
      },
      configurable: true,
    });
    return spy;
  };

  spy.mockRestore = () => {
    if (originalDescriptor) {
      Object.defineProperty(obj, prop, originalDescriptor);
    } else {
      delete obj[prop];
    }
  };

  return spy;
}

/** Create a spy that intercepts a method call */
function createMethodSpy(obj: any, prop: string): MockInstance {
  const originalMethod = obj[prop];
  let currentImpl: ((...args: any[]) => any) | undefined =
    typeof originalMethod === "function" ? originalMethod : undefined;
  const onceQueue: { type: "value" | "impl"; data: any }[] = [];
  const calls: any[][] = [];

  const spy = function (this: any, ...args: any[]): any {
    calls.push(args);
    let result: any;
    if (onceQueue.length > 0) {
      const entry = onceQueue.shift()!;
      if (entry.type === "impl") {
        result = entry.data.apply(this, args);
      } else {
        result = entry.data;
      }
    } else if (currentImpl) {
      result = currentImpl.apply(this, args);
    }
    recordResult(spy, result, this);
    return result;
  } as MockInstance;

  spy.calls = calls;
  stampVitestCompat(spy, calls, prop);

  spy.mockReturnValue = (val: any) => {
    currentImpl = () => val;
    obj[prop] = spy;
    return spy;
  };

  spy.mockReturnValueOnce = (val: any) => {
    onceQueue.push({ type: "value", data: val });
    obj[prop] = spy;
    return spy;
  };

  spy.mockImplementation = (fn: (...args: any[]) => any) => {
    currentImpl = fn;
    obj[prop] = spy;
    return spy;
  };

  spy.mockRestore = () => {
    obj[prop] = originalMethod;
    onceQueue.length = 0;
    calls.length = 0;
    spy.mock.results.length = 0;
    spy.mock.instances.length = 0;
    spy.mock.invocationCallOrder.length = 0;
    spy.mock.lastCall = undefined;
    currentImpl = typeof originalMethod === "function" ? originalMethod : undefined;
  };

  // Install the spy on the object
  obj[prop] = spy;

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
