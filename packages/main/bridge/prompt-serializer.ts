import type { LanguageModelV3Message } from "@ai-sdk/provider";

import { isRecord, readString, safeJsonStringify } from "../shared/type-readers";

export const contentPartToText = (part: unknown): string => {
  if (!isRecord(part)) {
    return safeJsonStringify(part);
  }

  const type = readString(part, "type");
  if (type === "text") {
    const text = readString(part, "text");
    return typeof text === "string" ? text : "";
  }

  if (type === "file") {
    const mediaType = readString(part, "mediaType") ?? "application/octet-stream";
    return `[file:${mediaType}]`;
  }

  if (type === "reasoning") {
    const text = readString(part, "text");
    return typeof text === "string" ? `[reasoning]\n${text}` : "[reasoning]";
  }

  if (type === "tool-call") {
    const toolName = readString(part, "toolName") ?? "unknown_tool";
    const input = part.input;
    return `[tool-call:${toolName}] ${safeJsonStringify(input)}`;
  }

  if (type === "tool-result") {
    const toolName = readString(part, "toolName") ?? "unknown_tool";
    const output = part.output;
    return `[tool-result:${toolName}] ${safeJsonStringify(output)}`;
  }

  return safeJsonStringify(part);
};

export const serializeMessage = (message: LanguageModelV3Message): string => {
  if (message.role === "system") {
    return `[system]\n${message.content}`;
  }

  const serializedContent = message.content.map(contentPartToText).join("\n");
  return `[${message.role}]\n${serializedContent}`;
};

export const serializePrompt = (prompt: LanguageModelV3Message[]): string => {
  return prompt.map(serializeMessage).join("\n\n");
};
