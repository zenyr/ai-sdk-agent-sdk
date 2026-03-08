import type { ManifestModel, ThinkingVariant } from "./types";

const baseBudgets = {
  low: 4000,
  medium: 12000,
  high: 16000,
};

export const buildModelVariants = (model: ManifestModel, includeNoneVariant = false) => {
  if (!model.reasoning) {
    return undefined;
  }

  const maxBudget = Math.min(31999, model.limit.output - 1);
  if (maxBudget < 1) {
    return undefined;
  }

  const variants: Record<string, ThinkingVariant> = {
    low: {
      thinking: {
        type: "enabled",
        budgetTokens: Math.min(baseBudgets.low, maxBudget),
      },
    },
    medium: {
      thinking: {
        type: "enabled",
        budgetTokens: Math.min(baseBudgets.medium, maxBudget),
      },
    },
    high: {
      thinking: {
        type: "enabled",
        budgetTokens: Math.min(baseBudgets.high, maxBudget),
      },
    },
    max: {
      thinking: {
        type: "enabled",
        budgetTokens: maxBudget,
      },
    },
  };

  if (includeNoneVariant) {
    variants.none = {
      thinking: {
        type: "disabled",
      },
    };
  }

  return variants;
};
