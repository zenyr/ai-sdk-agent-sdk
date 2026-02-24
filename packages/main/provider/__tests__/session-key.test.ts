import { describe, expect, test } from "bun:test";

import { readIncomingSessionKey } from "../domain/session-key";

describe("readIncomingSessionKey", () => {
  test("prefers headers over telemetry and providerOptions", () => {
    const options = {
      prompt: [],
      headers: {
        "x-conversation-id": "header-1",
      },
      experimental_telemetry: {
        metadata: {
          conversationId: "telemetry-1",
        },
      },
      providerOptions: {
        agentSdk: {
          conversationId: "provider-1",
        },
      },
    };

    const sessionKey = readIncomingSessionKey(options);

    expect(sessionKey).toBe("header-1");
  });

  test("reads from canonical and fallback provider option namespaces", () => {
    const canonicalOptions = {
      prompt: [],
      providerOptions: {
        agent_sdk: {
          sessionId: "canonical-1",
        },
      },
    };

    const canonical = readIncomingSessionKey(canonicalOptions);

    expect(canonical).toBe("canonical-1");

    const fallbackOptions = {
      prompt: [],
      providerOptions: {
        unknownNamespace: {
          conversation_id: "fallback-1",
        },
      },
    };

    const fallback = readIncomingSessionKey(fallbackOptions);

    expect(fallback).toBe("fallback-1");
  });
});
