import type { LanguageModelV3FunctionTool, LanguageModelV3ProviderTool } from "@ai-sdk/provider";

export const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

export const readRecord = (
  record: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined => {
  const value = record[key];
  if (!isRecord(value)) {
    return undefined;
  }

  return value;
};

export const readString = (record: Record<string, unknown>, key: string): string | undefined => {
  const value = record[key];
  if (typeof value !== "string") {
    return undefined;
  }

  return value;
};

export const readNumber = (record: Record<string, unknown>, key: string): number | undefined => {
  const value = record[key];
  if (typeof value !== "number") {
    return undefined;
  }

  return value;
};

export const readArray = (record: Record<string, unknown>, key: string): unknown[] | undefined => {
  const value = record[key];
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value;
};

export const safeJsonStringify = (value: unknown): string => {
  try {
    return JSON.stringify(value);
  } catch {
    return "null";
  }
};

export const isFunctionTool = (
  tool: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool,
): tool is LanguageModelV3FunctionTool => {
  return tool.type === "function";
};
