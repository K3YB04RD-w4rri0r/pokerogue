import { mockFn, restoreAllMocks, spyOn } from "#app/rl/mocks/spy";
import { describe, expect, it, vi } from "vitest";

/**
 * Contract tests for the standalone mock shim (`src/rl/mocks/spy.ts`).
 *
 * The shared test harness (`test/test-utils`) hands shim spies to ordinary
 * vitest tests, so the shim must interoperate with vitest three ways:
 * 1. vitest matchers must accept shim spies (`_isMockFunction` + `.mock` shape);
 * 2. `vi.spyOn` on a shim-spied member must return a spy exposing the full
 *    method surface tests use (`mockClear`, `mockResolvedValue`, ...);
 * 3. getter spies must install a RECOGNIZED mock as the getter — a raw closure
 *    would be captured by a later `vi.spyOn(obj, prop, "get")` as the
 *    "original" and, after restoreMocks, poison the property worker-wide.
 */
describe("RL mock shim vitest contract", () => {
  it("method spy records calls and satisfies vitest matchers", () => {
    const obj = { greet: (name: string) => `hi ${name}` };
    const spy = spyOn(obj, "greet");

    expect(obj.greet("ash")).toBe("hi ash");
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith("ash");
    expect(spy.mock.lastCall).toEqual(["ash"]);

    spy.mockRestore();
  });

  it("implements the full MockInstance surface tests rely on", async () => {
    const fn = mockFn<(...args: unknown[]) => unknown>();

    fn.mockReturnValueOnce(1).mockReturnValue(2);
    expect(fn()).toBe(1);
    expect(fn()).toBe(2);

    fn.mockImplementationOnce(() => 3);
    expect(fn()).toBe(3);
    expect(fn()).toBe(2);

    fn.mockResolvedValueOnce("first");
    fn.mockResolvedValue("later");
    await expect(fn()).resolves.toBe("first");
    await expect(fn()).resolves.toBe("later");

    fn.mockRejectedValueOnce(new Error("boom"));
    await expect(fn()).rejects.toThrow("boom");

    fn.mockClear();
    expect(fn.mock.calls).toHaveLength(0);
    await expect(fn()).resolves.toBe("later"); // mockClear keeps behavior

    fn.mockReset();
    expect(fn()).toBeUndefined(); // mockReset restores base impl (none)
  });

  it("records throw results without desyncing calls/results", () => {
    const fn = mockFn(() => {
      throw new Error("nope");
    });
    expect(() => fn()).toThrow("nope");
    expect(fn.mock.calls).toHaveLength(1);
    expect(fn.mock.results).toHaveLength(1);
    expect(fn.mock.results[0].type).toBe("throw");
  });

  it("vi.spyOn on a shim-spied method returns a spy with the full surface", async () => {
    const obj = { load: () => Promise.resolve("real") };
    const shimSpy = spyOn(obj, "load"); // what a shared helper does

    // What a test does afterwards: vitest recognizes the stamped shim spy
    const reacquired = vi.spyOn(obj, "load");
    expect(vi.isMockFunction(reacquired)).toBe(true);

    reacquired.mockResolvedValue("mocked"); // must not throw
    await expect(obj.load()).resolves.toBe("mocked");
    reacquired.mockClear(); // must not throw

    shimSpy.mockRestore();
    await expect(obj.load()).resolves.toBe("real");
  });

  it("getter spy records reads and installs a recognized mock (no descriptor poisoning)", () => {
    const target = { flag: false };
    Object.defineProperty(target, "flag", {
      get: () => false,
      configurable: true,
    });

    const spy = spyOn(target as object, "flag" as never, "get");
    spy.mockReturnValue(true);

    expect(target.flag).toBe(true);
    expect(spy).toHaveBeenCalled(); // reads are recorded

    // The installed getter is the stamped spy itself — a later
    // vi.spyOn(target, "flag", "get") must see a mock, not a raw closure.
    const installed = Object.getOwnPropertyDescriptor(target, "flag")?.get;
    expect(vi.isMockFunction(installed)).toBe(true);

    // Shim restoration returns the true original getter.
    restoreAllMocks();
    expect(target.flag).toBe(false);
  });

  it("restoreAllMocks unwinds nested spies LIFO to the true original", () => {
    const obj = { value: () => "original" };
    spyOn(obj, "value").mockReturnValue("outer");
    spyOn(obj, "value").mockReturnValue("inner");
    expect(obj.value()).toBe("inner");

    restoreAllMocks();
    expect(obj.value()).toBe("original");
  });
});
