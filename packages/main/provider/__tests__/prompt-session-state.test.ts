import { describe, expect, test } from "bun:test";
import type { LanguageModelV3Message } from "@ai-sdk/provider";

import { buildPromptQueryInput, mergePromptSessionState } from "../domain/prompt-session-state";

const userMessage = (text: string): LanguageModelV3Message => {
  return {
    role: "user",
    content: [{ type: "text", text }],
  };
};

describe("prompt-session-state", () => {
  test("buildPromptQueryInput resumes with appended prompt when prefix matches", () => {
    const promptQueryInput = buildPromptQueryInput({
      promptMessages: [userMessage("hello"), userMessage("next")],
      previousSessionStates: [
        {
          sessionId: "sess-1",
          serializedPromptMessages: ["hello"],
        },
      ],
    });

    expect(promptQueryInput.resumeSessionId).toBe("sess-1");
    expect(promptQueryInput.prompt).toBe("next");
  });

  test("buildPromptQueryInput falls back to full prompt when prefix mismatches", () => {
    const promptQueryInput = buildPromptQueryInput({
      promptMessages: [userMessage("hello"), userMessage("next")],
      previousSessionStates: [
        {
          sessionId: "sess-1",
          serializedPromptMessages: ["different"],
        },
      ],
    });

    expect(promptQueryInput.resumeSessionId).toBeUndefined();
    expect(promptQueryInput.prompt).toBe("hello\n\nnext");
  });

  test("mergePromptSessionState dedupes and keeps newest first", () => {
    const mergedStates = mergePromptSessionState({
      previousSessionStates: [
        {
          sessionId: "sess-1",
          serializedPromptMessages: ["hello"],
        },
        {
          sessionId: "sess-2",
          serializedPromptMessages: ["shared"],
        },
      ],
      nextSessionState: {
        sessionId: "sess-3",
        serializedPromptMessages: ["shared"],
      },
    });

    expect(mergedStates).toEqual([
      {
        sessionId: "sess-3",
        serializedPromptMessages: ["shared"],
      },
      {
        sessionId: "sess-1",
        serializedPromptMessages: ["hello"],
      },
    ]);
  });
});
