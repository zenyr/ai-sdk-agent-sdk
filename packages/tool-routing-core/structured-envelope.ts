import type { LanguageModelV3Content } from "@ai-sdk/provider";

import { isRecord, readString, safeJsonStringify } from "./shared/type-readers";

export type StructuredToolCall = {
  toolName: string;
  input: unknown;
};

export type StructuredToolEnvelope = {
  type: "tool-calls";
  calls: StructuredToolCall[];
};

export type StructuredTextEnvelope = {
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
