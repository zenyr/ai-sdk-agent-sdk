import { describe, expect, test } from "bun:test";

import { buildPromptQueryInput, mergePromptSessionState, readIncomingSessionKey } from "../index";

describe("session-join-core", () => {
  test("buildPromptQueryInput resumes when previous prompt is prefix", () => {
    const promptQueryInput = buildPromptQueryInput({
      promptMessages: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "next" }],
        },
      ],
      previousSessionStates: [
        {
          sessionId: "session-1",
          serializedPromptMessages: ["hello"],
        },
      ],
    });

    expect(promptQueryInput.resumeSessionId).toBe("session-1");
    expect(promptQueryInput.prompt).toBe("next");
  });

  test("mergePromptSessionState keeps latest and dedupes identical prompts", () => {
    const merged = mergePromptSessionState({
      previousSessionStates: [
        {
          sessionId: "session-old",
          serializedPromptMessages: ["same"],
        },
      ],
      nextSessionState: {
        sessionId: "session-new",
        serializedPromptMessages: ["same"],
      },
    });

    expect(merged).toEqual([
      {
        sessionId: "session-new",
        serializedPromptMessages: ["same"],
      },
    ]);
  });

  test("readIncomingSessionKey prioritizes conversation header", () => {
    const incomingSessionKey = readIncomingSessionKey({
      prompt: [],
      headers: {
        "x-conversation-id": "conversation-1",
      },
      providerOptions: {
        agentSdk: {
          conversationId: "provider-1",
        },
      },
    });

    expect(incomingSessionKey).toBe("conversation-1");
  });
});
