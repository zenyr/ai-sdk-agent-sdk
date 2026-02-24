import type {
  SDKAssistantMessage,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";

import { isRecord, readString } from "../../shared/type-readers";

export const EMPTY_TOOL_ROUTING_OUTPUT_ERROR = "empty-tool-routing-output";
export const EMPTY_TOOL_ROUTING_OUTPUT_TEXT =
  "Tool routing produced no tool call or text response.";

export const isAssistantMessage = (message: SDKMessage): message is SDKAssistantMessage => {
  return message.type === "assistant";
};

export const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === "result";
};

export const isPartialAssistantMessage = (
  message: SDKMessage,
): message is SDKPartialAssistantMessage => {
  return message.type === "stream_event";
};

export const isStructuredOutputRetryExhausted = (resultMessage: SDKResultMessage): boolean => {
  return resultMessage.subtype === "error_max_structured_output_retries";
};

export const extractAssistantText = (assistantMessage: SDKAssistantMessage | undefined): string => {
  if (assistantMessage === undefined) {
    return "";
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return "";
  }

  return contentBlocks
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
};
