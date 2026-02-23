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
    return "";
  }

  if (type === "tool-call") {
    const toolName = readString(part, "toolName") ?? "unknown_tool";
    const toolCallId = readString(part, "toolCallId");
    const toolCallSuffix =
      typeof toolCallId === "string" && toolCallId.length > 0 ? `#${toolCallId}` : "";
    const input = part.input;
    return `[tool-call:${toolName}${toolCallSuffix}] ${safeJsonStringify(input)}`;
  }

  if (type === "tool-result") {
    const toolName = readString(part, "toolName") ?? "unknown_tool";
    const toolCallId = readString(part, "toolCallId");
    const toolCallSuffix =
      typeof toolCallId === "string" && toolCallId.length > 0 ? `#${toolCallId}` : "";
    const output = part.output;
    return `[tool-result:${toolName}${toolCallSuffix}] ${safeJsonStringify(output)}`;
  }

  return safeJsonStringify(part);
};

export const serializeMessage = (message: LanguageModelV3Message): string => {
  if (message.role === "system") {
    return `[system]\n${message.content}`;
  }

  const serializedContent = message.content
    .map(contentPartToText)
    .filter((part) => {
      return part.length > 0;
    })
    .join("\n");
  return `[${message.role}]\n${serializedContent}`;
};

export const serializePromptMessages = (prompt: LanguageModelV3Message[]): string[] => {
  return prompt.map(serializeMessage);
};

export const serializePromptMessagesWithoutSystem = (
  prompt: LanguageModelV3Message[],
): string[] => {
  return prompt
    .filter((message) => {
      return message.role !== "system";
    })
    .map(serializeMessage);
};

export const extractSystemPromptFromMessages = (
  prompt: LanguageModelV3Message[],
): string | undefined => {
  const systemParts = prompt
    .filter((message) => {
      return message.role === "system";
    })
    .map((message) => {
      return message.content.trim();
    })
    .filter((content) => {
      return content.length > 0;
    });

  if (systemParts.length === 0) {
    return undefined;
  }

  return systemParts.join("\n\n");
};

export const joinSerializedPromptMessages = (serializedMessages: string[]): string => {
  return serializedMessages.join("\n\n");
};

export const serializePrompt = (prompt: LanguageModelV3Message[]): string => {
  return joinSerializedPromptMessages(serializePromptMessages(prompt));
};
