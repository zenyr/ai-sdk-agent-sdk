import type {
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FunctionTool,
  LanguageModelV3ProviderTool,
  SharedV3Warning,
} from "@ai-sdk/provider";
import { collectAnthropicProviderOptionWarnings } from "../internal/anthropic-option-warnings";
import {
  isRecord,
  readNumber,
  readRecord,
  readString,
  safeJsonStringify,
} from "../shared/type-readers";
import type { CompletionMode } from "./completion-mode";

type ParsedAnthropicProviderOptions = {
  effort?: "low" | "medium" | "high" | "max";
  thinking?:
    | { type: "adaptive" }
    | { type: "disabled" }
    | { type: "enabled"; budgetTokens?: number };
};

type StructuredToolCall = {
  toolName: string;
  input: unknown;
};

type StructuredToolEnvelope = {
  type: "tool-calls";
  calls: StructuredToolCall[];
};

type StructuredTextEnvelope = {
  type: "text";
  text: string;
};

const readLegacyToolCall = (value: unknown): StructuredToolCall | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const toolName = readString(value, "tool") ?? readString(value, "toolName");
  if (typeof toolName !== "string") {
    return undefined;
  }

  if ("input" in value) {
    return {
      toolName,
      input: value.input,
    };
  }

  if ("parameters" in value) {
    return {
      toolName,
      input: value.parameters,
    };
  }

  if ("arguments" in value) {
    return {
      toolName,
      input: value.arguments,
    };
  }

  return undefined;
};

const readLegacyToolCallsEnvelope = (value: unknown): StructuredToolEnvelope | undefined => {
  const singleCall = readLegacyToolCall(value);
  if (singleCall !== undefined) {
    return {
      type: "tool-calls",
      calls: [singleCall],
    };
  }

  if (!isRecord(value)) {
    return undefined;
  }

  const toolCalls = value.tool_calls;
  if (!Array.isArray(toolCalls)) {
    return undefined;
  }

  const parsedCalls = toolCalls
    .map(readLegacyToolCall)
    .filter((call): call is StructuredToolCall => {
      return call !== undefined;
    });

  if (parsedCalls.length === 0) {
    return undefined;
  }

  return {
    type: "tool-calls",
    calls: parsedCalls,
  };
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
export const isStructuredTextEnvelope = (value: unknown): value is StructuredTextEnvelope => {
  if (!isRecord(value)) {
    return false;
  }

  return value.type === "text" && typeof value.text === "string";
};

export const isStructuredToolEnvelope = (value: unknown): value is StructuredToolEnvelope => {
  if (!isRecord(value)) {
    return false;
  }

  if (value.type !== "tool-calls" || !Array.isArray(value.calls)) {
    return false;
  }

  return value.calls.every((call) => {
    if (!isRecord(call)) {
      return false;
    }

    return typeof call.toolName === "string" && "input" in call;
  });
};

export const parseStructuredEnvelopeFromUnknown = (
  value: unknown,
): StructuredToolEnvelope | StructuredTextEnvelope | undefined => {
  if (isStructuredToolEnvelope(value)) {
    return value;
  }

  if (isStructuredTextEnvelope(value)) {
    return value;
  }

  return readLegacyToolCallsEnvelope(value);
};

export const parseStructuredEnvelopeFromText = (
  value: string,
): StructuredToolEnvelope | StructuredTextEnvelope | undefined => {
  const trimmedValue = value.trim();
  if (trimmedValue.length === 0) {
    return undefined;
  }

  try {
    const parsedValue: unknown = JSON.parse(trimmedValue);

    return parseStructuredEnvelopeFromUnknown(parsedValue);
  } catch {
    return undefined;
  }
};

export const mapStructuredToolCallsToContent = (
  calls: StructuredToolCall[],
  idGenerator: () => string,
): LanguageModelV3Content[] => {
  return calls.map((call) => {
    return {
      type: "tool-call",
      toolCallId: idGenerator(),
      toolName: call.toolName,
      input: safeJsonStringify(call.input),
      providerExecuted: false,
    };
  });
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
