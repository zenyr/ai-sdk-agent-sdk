import { describe, expect, test } from "bun:test";

import {
  appendPendingBridgeToolInputDelta,
  finishPendingBridgeToolInput,
  startPendingBridgeToolInput,
} from "../application/bridge-tool-input-buffer";

describe("bridge-tool-input-buffer", () => {
  test("collects deltas and returns normalized raw input on finish", () => {
    const pendingBridgeToolInputs = new Map<string, { toolName: string; deltas: string[] }>();

    startPendingBridgeToolInput({
      pendingBridgeToolInputs,
      id: "tool-call-1",
      toolName: "mcp__ai_sdk_tool_bridge__bash",
    });

    expect(
      appendPendingBridgeToolInputDelta({
        pendingBridgeToolInputs,
        id: "tool-call-1",
        delta: '{"command":"',
      }),
    ).toBeTrue();

    expect(
      appendPendingBridgeToolInputDelta({
        pendingBridgeToolInputs,
        id: "tool-call-1",
        delta: 'ls"',
      }),
    ).toBeTrue();

    const finishedBridgeToolInput = finishPendingBridgeToolInput({
      pendingBridgeToolInputs,
      id: "tool-call-1",
    });

    expect(finishedBridgeToolInput).toBeDefined();
    if (finishedBridgeToolInput === undefined) {
      return;
    }

    expect(finishedBridgeToolInput.toolName).toBe("mcp__ai_sdk_tool_bridge__bash");
    expect(finishedBridgeToolInput.rawInput).toBe('{"command":"ls"');
    expect(pendingBridgeToolInputs.size).toBe(0);
  });

  test("returns false or undefined for unknown tool-input ids", () => {
    const pendingBridgeToolInputs = new Map<string, { toolName: string; deltas: string[] }>();

    const appended = appendPendingBridgeToolInputDelta({
      pendingBridgeToolInputs,
      id: "missing-id",
      delta: "{}",
    });
    expect(appended).toBeFalse();

    const finishedBridgeToolInput = finishPendingBridgeToolInput({
      pendingBridgeToolInputs,
      id: "missing-id",
    });
    expect(finishedBridgeToolInput).toBeUndefined();
  });
});
