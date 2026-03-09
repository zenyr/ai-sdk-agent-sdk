import { describe, expect, test } from "bun:test";

import {
  resolveToolModeEnvelopeFromText,
  resolveToolModeEnvelopeFromUnknown,
} from "../application/tool-mode-envelope-facade";

describe("tool-mode-envelope-facade", () => {
  test("maps structured tool envelope from unknown", () => {
    const resolved = resolveToolModeEnvelopeFromUnknown({
      value: {
        type: "tool-calls",
        calls: [
          {
            toolName: "lookup_weather",
            input: {
              city: "seoul",
            },
          },
        ],
      },
      idGenerator: () => "call-1",
    });

    expect(resolved.toolCalls).toEqual([
      {
        type: "tool-call",
        toolCallId: "call-1",
        toolName: "lookup_weather",
        input: '{"city":"seoul"}',
        providerExecuted: false,
      },
    ]);
    expect(resolved.text).toBeUndefined();
  });

  test("maps structured text envelope from text", () => {
    const resolved = resolveToolModeEnvelopeFromText({
      text: '{"type":"text","text":"done"}',
      idGenerator: () => "call-1",
    });

    expect(resolved.toolCalls).toEqual([]);
    expect(resolved.text).toBe("done");
  });

  test("returns empty resolution for non-envelope input", () => {
    const resolved = resolveToolModeEnvelopeFromUnknown({
      value: {
        message: "plain",
      },
      idGenerator: () => "call-1",
    });

    expect(resolved.toolCalls).toEqual([]);
    expect(resolved.text).toBeUndefined();
  });
});
