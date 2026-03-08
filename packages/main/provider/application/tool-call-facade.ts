import type { LanguageModelV3Content, LanguageModelV3StreamPart } from "@ai-sdk/provider";

import { fromBridgeToolName, isBridgeToolName, normalizeToolInputJson } from "../domain/tool-bridge-config";
import {
  appendPendingBridgeToolInputDelta,
  finishPendingBridgeToolInput,
  type PendingBridgeToolInputs,
  startPendingBridgeToolInput,
} from "./bridge-tool-input-buffer";

type ToolCallContent = Extract<LanguageModelV3Content, { type: "tool-call" }>;
type ToolInputStartPart = Extract<LanguageModelV3StreamPart, { type: "tool-input-start" }>;
type ToolInputDeltaPart = Extract<LanguageModelV3StreamPart, { type: "tool-input-delta" }>;
type ToolInputEndPart = Extract<LanguageModelV3StreamPart, { type: "tool-input-end" }>;

export const createBridgeToolCall = (args: {
  toolCallId: string;
  toolName: string;
  rawInput: string;
  providerExecuted: boolean;
}): ToolCallContent => {
  return {
    type: "tool-call",
    toolCallId: args.toolCallId,
    toolName: fromBridgeToolName(args.toolName),
    input: normalizeToolInputJson(args.rawInput),
    providerExecuted: args.providerExecuted,
  };
};

export const startBridgeToolCallCapture = (args: {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  part: ToolInputStartPart;
}): boolean => {
  if (!isBridgeToolName(args.part.toolName)) {
    return false;
  }

  startPendingBridgeToolInput({
    pendingBridgeToolInputs: args.pendingBridgeToolInputs,
    id: args.part.id,
    toolName: args.part.toolName,
  });
  return true;
};

export const appendBridgeToolCallCapture = (args: {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  part: ToolInputDeltaPart;
}): boolean => {
  return appendPendingBridgeToolInputDelta({
    pendingBridgeToolInputs: args.pendingBridgeToolInputs,
    id: args.part.id,
    delta: args.part.delta,
  });
};

export const finishBridgeToolCallCapture = (args: {
  pendingBridgeToolInputs: PendingBridgeToolInputs;
  part: ToolInputEndPart;
  providerExecuted: boolean;
}): ToolCallContent | undefined => {
  const finishedBridgeToolInput = finishPendingBridgeToolInput({
    pendingBridgeToolInputs: args.pendingBridgeToolInputs,
    id: args.part.id,
  });

  if (finishedBridgeToolInput === undefined) {
    return undefined;
  }

  return createBridgeToolCall({
    toolCallId: args.part.id,
    toolName: finishedBridgeToolInput.toolName,
    rawInput: finishedBridgeToolInput.rawInput,
    providerExecuted: args.providerExecuted,
  });
};
