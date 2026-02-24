import { describe, expect, test } from "bun:test";
import type { LanguageModelV3Content } from "@ai-sdk/provider";
import type { SDKAssistantMessage } from "@anthropic-ai/claude-agent-sdk";

import {
  hasToolModePrimaryContent,
  recoverToolModeContentFromAssistantText,
  recoverToolModeToolCallsFromAssistant,
} from "../domain/tool-recovery";

const createAssistantMessage = (content: unknown[]): SDKAssistantMessage => {
  return {
    type: "assistant",
    uuid: "00000000-0000-0000-0000-000000000001",
    parent_tool_use_id: null,
    session_id: "sess-1",
    message: {
      id: "m1",
      role: "assistant",
      model: "claude",
      stop_reason: "end_turn",
      stop_sequence: null,
      type: "message",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        service_tier: "standard",
      },
      content,
    },
  };
};

describe("tool-recovery", () => {
  test("hasToolModePrimaryContent ignores whitespace-only text", () => {
    const content: LanguageModelV3Content[] = [{ type: "text", text: "   " }];

    expect(hasToolModePrimaryContent(content)).toBe(false);
  });

  test("recoverToolModeToolCallsFromAssistant maps tool names and input", () => {
    const recovered = recoverToolModeToolCallsFromAssistant({
      assistantMessage: createAssistantMessage([
        {
          type: "tool_use",
          id: "tool-1",
          name: "mcp__ai_sdk_tool_bridge__weather",
          input: { city: "Seoul" },
        },
      ]),
      idGenerator: () => "generated-id",
      mapToolName: (toolName) => {
        return toolName.replace("mcp__ai_sdk_tool_bridge__", "");
      },
    });

    expect(recovered).toEqual([
      {
        type: "tool-call",
        toolCallId: "tool-1",
        toolName: "weather",
        input: '{"city":"Seoul"}',
        providerExecuted: false,
      },
    ]);
  });

  test("recoverToolModeContentFromAssistantText parses structured text envelope", () => {
    const recovered = recoverToolModeContentFromAssistantText({
      assistantMessage: createAssistantMessage([
        {
          type: "text",
          text: '{"type":"text","text":"plain-result"}',
        },
      ]),
      idGenerator: () => "generated-id",
    });

    expect(recovered).toEqual([{ type: "text", text: "plain-result" }]);
  });
});
