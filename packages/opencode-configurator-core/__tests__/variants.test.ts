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

describe("variants", () => {
  test("clamps thinking budgets to model output", () => {
    const variants = buildModelVariants(model);

    expect(variants).toBeDefined();
    if (variants === undefined) {
      return;
    }

    expect(variants.low?.thinking.budgetTokens).toBe(4000);
    expect(variants.medium?.thinking.budgetTokens).toBe(7999);
    expect(variants.high?.thinking.budgetTokens).toBe(7999);
    expect(variants.max?.thinking.budgetTokens).toBe(7999);
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
