import type { LanguageModelV3Content } from "@ai-sdk/provider";
import type { SDKAssistantMessage } from "@anthropic-ai/claude-agent-sdk";

import {
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseStructuredEnvelopeFromText,
} from "../../bridge/parse-utils";
import { isRecord, readString, safeJsonStringify } from "../../shared/type-readers";

const extractAssistantText = (assistantMessage: SDKAssistantMessage | undefined): string => {
  if (assistantMessage === undefined) {
    return "";
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return "";
  }

  const text = contentBlocks
    .map((block) => {
      if (!isRecord(block)) {
        return "";
      }

      if (block.type !== "text") {
        return "";
      }

      const textPart = readString(block, "text");
      return typeof textPart === "string" ? textPart : "";
    })
    .join("");

  return text;
};

export const hasToolModePrimaryContent = (content: LanguageModelV3Content[]): boolean => {
  return content.some((part) => {
    if (part.type === "tool-call") {
      return true;
    }

    if (part.type === "text") {
      return part.text.trim().length > 0;
    }

    return false;
  });
};

export const recoverToolModeToolCallsFromAssistant = (args: {
  assistantMessage: SDKAssistantMessage | undefined;
  idGenerator: () => string;
  mapToolName: (toolName: string) => string;
}): LanguageModelV3Content[] => {
  if (args.assistantMessage === undefined) {
    return [];
  }

  const contentBlocks = args.assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return [];
  }

  const toolCalls: LanguageModelV3Content[] = [];

  for (const block of contentBlocks) {
    if (!isRecord(block)) {
      continue;
    }

    const blockType = readString(block, "type");
    if (
      blockType !== "tool_use" &&
      blockType !== "mcp_tool_use" &&
      blockType !== "server_tool_use"
    ) {
      continue;
    }

    const rawToolName = readString(block, "name");
    if (rawToolName === undefined) {
      continue;
    }

    const toolCallId = readString(block, "id") ?? args.idGenerator();
    const inputValue = "input" in block ? block.input : {};

    toolCalls.push({
      type: "tool-call",
      toolCallId,
      toolName: args.mapToolName(rawToolName),
      input: safeJsonStringify(inputValue),
      providerExecuted: false,
    });
  }

  return toolCalls;
};

export const recoverToolModeContentFromAssistantText = (args: {
  assistantMessage: SDKAssistantMessage | undefined;
  idGenerator: () => string;
}): LanguageModelV3Content[] => {
  const assistantText = extractAssistantText(args.assistantMessage);
  if (assistantText.length === 0) {
    return [];
  }

  const parsedEnvelope = parseStructuredEnvelopeFromText(assistantText);

  if (isStructuredToolEnvelope(parsedEnvelope)) {
    const toolCalls = mapStructuredToolCallsToContent(parsedEnvelope.calls, args.idGenerator);

    if (toolCalls.length > 0) {
      return toolCalls;
    }
  }

  if (isStructuredTextEnvelope(parsedEnvelope)) {
    return [{ type: "text", text: parsedEnvelope.text }];
  }

  return [{ type: "text", text: assistantText }];
};
