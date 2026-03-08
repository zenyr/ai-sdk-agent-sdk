import { describe, expect, test } from "bun:test";

import type { PendingBridgeToolInputs } from "../application/bridge-tool-input-buffer";
import {
  appendBridgeToolCallCapture,
  createBridgeToolCall,
  finishBridgeToolCallCapture,
  startBridgeToolCallCapture,
} from "../application/tool-call-facade";

const createPendingInputs = (): PendingBridgeToolInputs => {
  return new Map();
};

describe("tool-call-facade", () => {
  test("ignores non-bridge tool names", () => {
    const pendingBridgeToolInputs = createPendingInputs();

    const started = startBridgeToolCallCapture({
      pendingBridgeToolInputs,
      part: {
        type: "tool-input-start",
        id: "toolu_1",
        toolName: "lookup_weather",
        providerExecuted: false,
        dynamic: false,
      },
    });

    expect(started).toBeFalse();
    expect(pendingBridgeToolInputs.size).toBe(0);
  });

  test("captures bridge tool input and returns normalized tool-call", () => {
    const pendingBridgeToolInputs = createPendingInputs();

    const started = startBridgeToolCallCapture({
      pendingBridgeToolInputs,
      part: {
        type: "tool-input-start",
        id: "toolu_1",
        toolName: "mcp__ai_sdk_tool_bridge__bash",
        providerExecuted: true,
        dynamic: false,
      },
    });
    const appended = appendBridgeToolCallCapture({
      pendingBridgeToolInputs,
      part: {
        type: "tool-input-delta",
        id: "toolu_1",
        delta: '{ "command": "bun test" }',
      },
    });
    const toolCall = finishBridgeToolCallCapture({
      pendingBridgeToolInputs,
      part: {
        type: "tool-input-end",
        id: "toolu_1",
      },
      providerExecuted: true,
    });

    expect(started).toBeTrue();
    expect(appended).toBeTrue();
    expect(toolCall).toEqual({
      type: "tool-call",
      toolCallId: "toolu_1",
      toolName: "bash",
      input: '{"command":"bun test"}',
      providerExecuted: true,
    });
    expect(pendingBridgeToolInputs.size).toBe(0);
  });

  test("creates normalized bridge tool call directly", () => {
    const toolCall = createBridgeToolCall({
      toolCallId: "toolu_2",
      toolName: "mcp__ai_sdk_tool_bridge__lookup_weather",
      rawInput: '{"city":"seoul"}',
      providerExecuted: false,
    });

    expect(toolCall).toEqual({
      type: "tool-call",
      toolCallId: "toolu_2",
      toolName: "lookup_weather",
      input: '{"city":"seoul"}',
      providerExecuted: false,
    });
  });
});
