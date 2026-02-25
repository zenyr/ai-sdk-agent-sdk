import { describe, expect, test } from "bun:test";
import type { LanguageModelV3Message } from "@ai-sdk/provider";
import { persistQuerySessionState } from "../application/session-persistence";
import type { PromptSessionState } from "../domain/prompt-session-state";
import type { IncomingSessionState } from "../incoming-session-store";

const userMessage = (text: string): LanguageModelV3Message => {
  return {
    role: "user",
    content: [{ type: "text", text }],
  };
};

describe("session-persistence", () => {
  test("does nothing when session id is missing", async () => {
    let setPromptSessionStateCalled = false;
    let persistedIncomingSessionState: IncomingSessionState | undefined;

    await persistQuerySessionState({
      resultMessage: undefined,
      assistantMessage: undefined,
      initSystemMessage: undefined,
      incomingSessionKey: "conversation-1",
      serializedPromptMessages: ["user:hello"],
      promptMessages: [userMessage("hello")],
      previousSessionStates: () => [],
      setPromptSessionStates: () => {
        setPromptSessionStateCalled = true;
      },
      persistIncomingSessionState: async (incomingSessionState) => {
        persistedIncomingSessionState = incomingSessionState;
      },
    });

    expect(setPromptSessionStateCalled).toBeFalse();
    expect(persistedIncomingSessionState).toBeUndefined();
  });

  test("updates prompt session states when result has session id", async () => {
    const capturedPromptSessionStates: PromptSessionState[][] = [];

    await persistQuerySessionState({
      resultMessage: {
        type: "result",
        subtype: "success",
        stop_reason: "end_turn",
        result: "ok",
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
          service_tier: "standard",
        },
        session_id: "result-session-1",
      },
      assistantMessage: undefined,
      initSystemMessage: undefined,
      incomingSessionKey: undefined,
      serializedPromptMessages: ["user:hello"],
      promptMessages: [userMessage("hello")],
      previousSessionStates: () => [],
      setPromptSessionStates: (sessionStates) => {
        capturedPromptSessionStates.push(sessionStates);
      },
      persistIncomingSessionState: async () => {},
    });

    expect(capturedPromptSessionStates).toHaveLength(1);

    const sessionStates = capturedPromptSessionStates[0];
    expect(sessionStates).toBeDefined();
    if (sessionStates === undefined) {
      return;
    }

    const firstSessionState = sessionStates[0];
    expect(firstSessionState).toBeDefined();
    if (firstSessionState === undefined) {
      return;
    }

    expect(firstSessionState.sessionId).toBe("result-session-1");
    expect(firstSessionState.serializedPromptMessages).toEqual(["user:hello"]);
  });

  test("persists incoming session state using init system session id fallback", async () => {
    let persistedIncomingSessionState: IncomingSessionState | undefined;

    await persistQuerySessionState({
      resultMessage: undefined,
      assistantMessage: undefined,
      initSystemMessage: {
        type: "system",
        subtype: "init",
        agents: [],
        apiKeySource: "env",
        tools: ["tool-1"],
        mcp_servers: [],
        model: "claude-3-5-sonnet-latest",
        permissionMode: "dontAsk",
        slash_commands: [],
        output_style: "default",
        skills: [],
        plugins: [],
        uuid: "system-uuid",
        cwd: "/workspace",
        betas: [],
        claude_code_version: "1.0.0",
        session_id: "init-session-1",
      },
      incomingSessionKey: "conversation-2",
      serializedPromptMessages: undefined,
      promptMessages: [userMessage("second")],
      previousSessionStates: () => [],
      setPromptSessionStates: () => {},
      persistIncomingSessionState: async (incomingSessionState) => {
        persistedIncomingSessionState = incomingSessionState;
      },
    });

    expect(persistedIncomingSessionState).toBeDefined();
    if (persistedIncomingSessionState === undefined) {
      return;
    }

    expect(persistedIncomingSessionState.incomingSessionKey).toBe("conversation-2");
    expect(persistedIncomingSessionState.sessionId).toBe("init-session-1");
    expect(persistedIncomingSessionState.promptMessageCount).toBe(1);
  });
});
