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

  if (type === "image") {
    const mediaType = readString(part, "mediaType") ?? "image/*";
    return `[image:${mediaType}]`;
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

  if (message.role === "user") {
    return serializedContent;
  }

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

const serializeMessageForResumeQuery = (message: LanguageModelV3Message): string | undefined => {
  if (message.role === "system") {
    return undefined;
  }

  if (message.role !== "assistant") {
    return serializeMessage(message);
  }

  const serializedToolCalls = message.content
    .filter((part) => {
      if (!isRecord(part)) {
        return false;
      }

      return readString(part, "type") === "tool-call";
    })
    .map(contentPartToText)
    .filter((part) => {
      return part.length > 0;
    })
    .join("\n");

  if (serializedToolCalls.length === 0) {
    return undefined;
  }

  return `[assistant]\n${serializedToolCalls}`;
};

export const serializePromptMessagesForResumeQuery = (
  prompt: LanguageModelV3Message[],
): string[] => {
  return prompt
    .map(serializeMessageForResumeQuery)
    .filter((serializedMessage): serializedMessage is string => {
      return typeof serializedMessage === "string" && serializedMessage.length > 0;
    });
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
