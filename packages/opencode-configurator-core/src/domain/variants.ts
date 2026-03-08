import type { ManifestModel, ThinkingVariant } from "./types";

const baseBudgets = {
  low: 4000,
  medium: 12000,
  high: 16000,
};

const supportsAdaptiveThinking = (model: ManifestModel) => {
  return model.id === "claude-sonnet-4-6" || model.id === "claude-opus-4-6";
};

const supportsMaxEffort = (model: ManifestModel) => {
  return model.id === "claude-opus-4-6";
};

export const buildModelVariants = (model: ManifestModel, includeNoneVariant = false) => {
  if (!model.reasoning) {
    return undefined;
  }

  if (supportsAdaptiveThinking(model)) {
    const variants: Record<string, ThinkingVariant> = {
      low: {
        thinking: {
          type: "adaptive",
        },
        effort: "low",
      },
      medium: {
        thinking: {
          type: "adaptive",
        },
        effort: "medium",
      },
      high: {
        thinking: {
          type: "adaptive",
        },
        effort: "high",
      },
    };

    if (supportsMaxEffort(model)) {
      variants.max = {
        thinking: {
          type: "adaptive",
        },
        effort: "max",
      };
    }

    if (includeNoneVariant) {
      variants.none = {
        thinking: {
          type: "disabled",
        },
      };
    }

    return variants;
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
