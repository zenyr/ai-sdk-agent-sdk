import type {
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
  LanguageModelV3ProviderTool,
  SharedV3Warning,
} from "@ai-sdk/provider";
import { collectAnthropicProviderOptionWarnings } from "../internal/anthropic-option-warnings";
import { isRecord, readNumber, readRecord, readString } from "../shared/type-readers";
import type { CompletionMode } from "./completion-mode";

export {
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../../tool-routing-core/index.ts";

type ParsedAnthropicProviderOptions = {
  effort?: "low" | "medium" | "high" | "max";
  thinking?:
    | { type: "adaptive" }
    | { type: "disabled" }
    | { type: "enabled"; budgetTokens?: number };
};

export const parseAnthropicProviderOptions = (
  options: LanguageModelV3CallOptions,
): ParsedAnthropicProviderOptions => {
  const providerOptions = options.providerOptions;
  if (!isRecord(providerOptions)) {
    return {};
  }

  const anthropicOptions = readRecord(providerOptions, "anthropic");
  if (anthropicOptions === undefined) {
    return {};
  }

  const parsed: ParsedAnthropicProviderOptions = {};

  const effort = readString(anthropicOptions, "effort");
  if (effort === "low" || effort === "medium" || effort === "high" || effort === "max") {
    parsed.effort = effort;
  }

  const thinkingRecord = readRecord(anthropicOptions, "thinking");
  if (thinkingRecord !== undefined) {
    const thinkingType = readString(thinkingRecord, "type");

    if (thinkingType === "adaptive") {
      parsed.thinking = { type: "adaptive" };
    }

    if (thinkingType === "disabled") {
      parsed.thinking = { type: "disabled" };
    }

    if (thinkingType === "enabled") {
      const budgetTokens = readNumber(thinkingRecord, "budgetTokens");

      if (typeof budgetTokens === "number") {
        parsed.thinking = {
          type: "enabled",
          budgetTokens,
        };
      }

      if (typeof budgetTokens !== "number") {
        parsed.thinking = {
          type: "enabled",
        };
      }
    }
  }

  return parsed;
};

export const collectWarnings = (
  options: LanguageModelV3CallOptions,
  mode: CompletionMode,
): SharedV3Warning[] => {
  const warnings: SharedV3Warning[] = collectAnthropicProviderOptionWarnings(
    options.providerOptions,
  );

  if (typeof options.temperature === "number") {
    warnings.push({
      type: "unsupported",
      feature: "temperature",
      details: "claude-agent-sdk backend does not expose direct temperature control.",
    });
  }

  if (typeof options.topP === "number") {
    warnings.push({
      type: "unsupported",
      feature: "topP",
      details: "claude-agent-sdk backend does not expose direct topP control.",
    });
  }

  if (typeof options.topK === "number") {
    warnings.push({
      type: "unsupported",
      feature: "topK",
      details: "claude-agent-sdk backend does not expose direct topK control.",
    });
  }

  if (typeof options.presencePenalty === "number") {
    warnings.push({
      type: "unsupported",
      feature: "presencePenalty",
      details: "claude-agent-sdk backend does not expose direct presence penalty control.",
    });
  }

  if (typeof options.frequencyPenalty === "number") {
    warnings.push({
      type: "unsupported",
      feature: "frequencyPenalty",
      details: "claude-agent-sdk backend does not expose direct frequency penalty control.",
    });
  }

  if (typeof options.seed === "number") {
    warnings.push({
      type: "unsupported",
      feature: "seed",
      details: "claude-agent-sdk backend does not expose deterministic seed control.",
    });
  }

  if (typeof options.maxOutputTokens === "number") {
    warnings.push({
      type: "compatibility",
      feature: "maxOutputTokens",
      details:
        "maxOutputTokens is best-effort only because claude-agent-sdk controls decoding internally.",
    });
  }

  if (
    mode.type === "tools" &&
    options.tools !== undefined &&
    options.tools.some(
      (toolDefinition: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool) => {
        return toolDefinition.type === "provider";
      },
    )
  ) {
    warnings.push({
      type: "unsupported",
      feature: "provider-defined tools",
      details:
        "provider-defined tools are ignored when using claude-agent-sdk compatibility backend.",
    });
  }

  return warnings;
};
