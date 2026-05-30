/**
 * Lightweight assertion utilities replacing `expect` from Vitest.
 *
 * These throw `Error` on failure instead of using Vitest's matcher infrastructure.
 * Covers only the subset of `expect` actually used in the test-utils harness files.
 */

/**
 * Assertion helper that mimics `expect(received)` from Vitest.
 * Supports `.toBe()`, `.toBeDefined()`, `.toBeGreaterThan()`,
 * `.toBeLessThan()`, `.toBeLessThanOrEqual()`.
 */
export function expectValue(received: any) {
  return {
    toBe(expected: any) {
      if (received !== expected) {
        throw new Error(`Expected ${JSON.stringify(received)} to be ${JSON.stringify(expected)}`);
      }
    },
    toBeDefined() {
      if (received === undefined) {
        throw new Error("Expected value to be defined, but received undefined");
      }
    },
    toBeGreaterThan(n: number) {
      if (!(received > n)) {
        throw new Error(`Expected ${received} to be greater than ${n}`);
      }
    },
    toBeLessThan(n: number) {
      if (!(received < n)) {
        throw new Error(`Expected ${received} to be less than ${n}`);
      }
    },
    toBeLessThanOrEqual(n: number) {
      if (!(received <= n)) {
        throw new Error(`Expected ${received} to be less than or equal to ${n}`);
      }
    },
  };
}

/**
 * Standalone `expect` object providing `.fail()`.
 */
export const standaloneExpect = {
  fail(message: string): never {
    throw new Error(message);
  },
};
