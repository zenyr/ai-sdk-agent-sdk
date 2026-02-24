import { describe, expect, test } from "bun:test";

import {
  buildIncomingSessionState,
  buildPromptQueryInputWithIncomingSession,
  readSessionIdFromQueryMessages,
} from "../domain/incoming-session-state";

const user = (text: string) => {
  return {
    role: "user" as const,
    content: [{ type: "text" as const, text }],
  };
};

describe("incoming-session-state", () => {
  test("resumes from incoming session when prompt prefix matches", () => {
    const previousPrompt = [user("hello")];
    const incomingSessionState = buildIncomingSessionState({
      incomingSessionKey: "conv-1",
      sessionId: "sess-1",
      promptMessages: previousPrompt,
    });

    const nextPrompt = [user("hello"), user("next")];

    const promptQueryInput = buildPromptQueryInputWithIncomingSession({
      promptMessages: nextPrompt,
      incomingSessionKey: "conv-1",
      previousSessionStates: [],
      previousIncomingSessionStates: [incomingSessionState],
    });

    expect(promptQueryInput.resumeSessionId).toBe("sess-1");
    expect(promptQueryInput.prompt).toBe("next");
  });

  test("reads session id from assistant message when result is missing", () => {
    const sessionId = readSessionIdFromQueryMessages({
      resultMessage: undefined,
      assistantMessage: {
        type: "assistant",
        uuid: "00000000-0000-0000-0000-000000000001",
        parent_tool_use_id: null,
        session_id: "assistant-session-id",
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
          content: [{ type: "text", text: "ok" }],
        },
      },
    });

    expect(sessionId).toBe("assistant-session-id");
  });
});
