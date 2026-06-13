/**
 * Golden-fixture tests for the TS observation encoder.
 *
 * Each fixture in test/rl/fixtures/*.state.json is encoded and compared
 * byte-for-byte against its golden (<name>.golden.b64). The Python encoder is
 * checked against the SAME goldens by tools/verify/fixture_parity.py — the
 * golden files are what tie the two languages together.
 *
 * Regenerate goldens ONLY after an intentional layout/encoding change:
 *   UPDATE_RL_GOLDEN=1 pnpm exec vitest run test/rl/spaces-encoding.test.ts
 */
import { ACTION_SPACE_SIZE, encodeObservation, extractActionMask, OBSERVATION_DIM } from "#rl/spaces";
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { BATTLE, battleDim, FIELD, fieldDim, PKMN, pokemonDim } from "./semantic/obs-layout";

const FIXTURE_DIR = path.join(__dirname, "fixtures");
const FIXTURES = ["full", "minimal", "edge"] as const;

function loadFixture(name: string): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(path.join(FIXTURE_DIR, `${name}.state.json`), "utf8"));
}

function goldenPath(name: string): string {
  return path.join(FIXTURE_DIR, `${name}.golden.b64`);
}

function toBase64(obs: Float32Array): string {
  return Buffer.from(obs.buffer, obs.byteOffset, obs.byteLength).toString("base64");
}

describe("RL Encoding - Golden Fixtures", () => {
  for (const name of FIXTURES) {
    it(`encodes ${name}.state.json to a finite ${OBSERVATION_DIM}-dim vector matching its golden`, () => {
      const fixture = loadFixture(name);
      const obs = encodeObservation(fixture);

      expect(obs.length).toBe(OBSERVATION_DIM);
      for (let i = 0; i < obs.length; i++) {
        if (!Number.isFinite(obs[i])) {
          throw new Error(`non-finite value ${obs[i]} at dim ${i} in fixture ${name}`);
        }
      }

      if (process.env.UPDATE_RL_GOLDEN) {
        fs.writeFileSync(goldenPath(name), toBase64(obs));
        return;
      }

      expect(fs.existsSync(goldenPath(name)), "golden missing — run with UPDATE_RL_GOLDEN=1 to create").toBe(true);
      const golden = fs.readFileSync(goldenPath(name), "utf8").trim();
      expect(toBase64(obs)).toBe(golden);
    });
  }

  it("probes known dims of the full fixture against the fixture values", () => {
    const fixture = loadFixture("full") as any;
    const obs = encodeObservation(fixture);

    // Pokemon block probes
    expect(obs[pokemonDim("player_0", PKMN.VALID)]).toBe(fixture.player_0?.valid ? 1 : 0);
    if (fixture.player_0?.valid) {
      expect(obs[pokemonDim("player_0", PKMN.HP_RATIO)]).toBeCloseTo(fixture.player_0.hp_ratio, 5);
      expect(obs[pokemonDim("player_0", PKMN.LEVEL)]).toBeCloseTo(Math.min(fixture.player_0.level / 100, 1), 5);
    }

    // Battle block probes
    expect(obs[battleDim(BATTLE.WAVE)]).toBeCloseTo(Math.min(fixture.battle.wave_index / 200, 1), 5);
    expect(obs[battleDim(BATTLE.IS_CLASSIC)]).toBe(fixture.battle.is_classic ? 1 : 0);

    // Field block probe: weather one-hot
    const weather = fixture.field?.weather_type ?? 0;
    expect(obs[fieldDim(FIELD.WEATHER_OH + weather)]).toBe(1);
  });

  it("returns an all-false mask for malformed mask lengths (TS-side semantics)", () => {
    // KNOWN ASYMMETRY: the TS extractor returns all-false unless the mask is
    // exactly ACTION_SPACE_SIZE long; the Python extractor pads/truncates.
    // Only reachable with malformed states — documented here, harmonization
    // tracked in docs/LIMITATIONS.md.
    const malformed = { phase: { current_phase: "command", action_mask: new Array(57).fill(true) } };
    const mask = extractActionMask(malformed);
    expect(mask.length).toBe(ACTION_SPACE_SIZE);
    expect(mask.every(v => v === false)).toBe(true);
  });
});
