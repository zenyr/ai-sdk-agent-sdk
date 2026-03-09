import { describe, expect, test } from "bun:test";
import type { ManifestModel } from "../src/domain/types";
import { buildModelVariants } from "../src/domain/variants";

const model: ManifestModel = {
  id: "claude-sonnet-4-6",
  name: "Claude Sonnet 4.6",
  family: "claude-sonnet",
  releaseDate: "2025-11-01",
  attachment: true,
  reasoning: true,
  toolCall: true,
  temperature: true,
  limit: {
    context: 200000,
    output: 8000,
  },
};

const opusModel: ManifestModel = {
  ...model,
  id: "claude-opus-4-6",
  name: "Claude Opus 4.6",
};

const haikuModel: ManifestModel = {
  ...model,
  id: "claude-haiku-4-5",
  name: "Claude Haiku 4.5",
};

describe("variants", () => {
  test("uses adaptive thinking with effort for sonnet 4.6", () => {
    const variants = buildModelVariants(model);

    expect(variants).toBeDefined();
    if (variants === undefined) {
      return;
    }

    expect(variants.low).toEqual({
      thinking: {
        type: "adaptive",
      },
      effort: "low",
    });
    expect(variants.medium).toEqual({
      thinking: {
        type: "adaptive",
      },
      effort: "medium",
    });
    expect(variants.high).toEqual({
      thinking: {
        type: "adaptive",
      },
      effort: "high",
    });
    expect(variants.max).toBeUndefined();
  });

  test("adds max effort for opus 4.6", () => {
    const variants = buildModelVariants(opusModel);

    expect(variants?.max).toEqual({
      thinking: {
        type: "adaptive",
      },
      effort: "max",
    });
  });

  test("clamps thinking budgets to model output on non-adaptive models", () => {
    const variants = buildModelVariants(haikuModel);

    expect(variants).toBeDefined();
    if (variants === undefined) {
      return;
    }

    expect(variants.low?.thinking).toEqual({
      type: "enabled",
      budgetTokens: 4000,
    });
    expect(variants.medium?.thinking).toEqual({
      type: "enabled",
      budgetTokens: 7999,
    });
    expect(variants.high?.thinking).toEqual({
      type: "enabled",
      budgetTokens: 7999,
    });
    expect(variants.max?.thinking).toEqual({
      type: "enabled",
      budgetTokens: 7999,
    });
  });

  test("adds none variant when requested", () => {
    const variants = buildModelVariants(model, true);

    expect(variants?.none).toEqual({
      thinking: {
        type: "disabled",
      },
    });
  });
});
