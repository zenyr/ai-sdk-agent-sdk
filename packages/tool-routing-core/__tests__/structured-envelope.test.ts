import { describe, expect, test } from "bun:test";

import {
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../index";

describe("tool-routing-core", () => {
  test("parses explicit tool-calls envelope from text", () => {
    const parsed = parseStructuredEnvelopeFromText(
      '{"type":"tool-calls","calls":[{"toolName":"weather","input":{"city":"seoul"}}]}',
    );

    expect(isStructuredToolEnvelope(parsed)).toBeTrue();
    if (!isStructuredToolEnvelope(parsed)) {
      return;
    }

    expect(parsed.calls).toEqual([
      {
        toolName: "weather",
        input: {
          city: "seoul",
        },
      },
    ]);
  });

  test("parses legacy single-call envelope", () => {
    const parsed = parseStructuredEnvelopeFromUnknown({
      tool: "bash",
      parameters: {
        command: "bun test",
      },
    });

    expect(isStructuredToolEnvelope(parsed)).toBeTrue();
    if (!isStructuredToolEnvelope(parsed)) {
      return;
    }

    expect(parsed.calls).toEqual([
      {
        toolName: "bash",
        input: {
          command: "bun test",
        },
      },
    ]);
  });

  test("maps structured tool calls to language model tool-call content", () => {
    const content = mapStructuredToolCallsToContent(
      [
        {
          toolName: "weather",
          input: {
            city: "seoul",
          },
        },
      ],
      () => "tool-call-1",
    );

    expect(content).toEqual([
      {
        type: "tool-call",
        toolCallId: "tool-call-1",
        toolName: "weather",
        input: '{"city":"seoul"}',
        providerExecuted: false,
      },
    ]);
  });
});
