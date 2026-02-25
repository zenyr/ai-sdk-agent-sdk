import { afterEach, describe, expect, mock, test } from "bun:test";
import type { LanguageModelV3Message } from "@ai-sdk/provider";
import type { IncomingSessionState } from "../incoming-session-store";

const originalConsoleWarn = console.warn;

afterEach(() => {
  mock.restore();
  console.warn = originalConsoleWarn;
});

const userMessage = (text: string): LanguageModelV3Message => {
  return {
    role: "user",
    content: [{ type: "text", text }],
  };
};

const buildMockResultUsage = () => {
  return {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const readOptionsFromQueryCall = (
  queryCalls: unknown[],
  index: number,
): Record<string, unknown> | undefined => {
  const queryCall = queryCalls[index];
  if (!isRecord(queryCall)) {
    return undefined;
  }

  const options = queryCall.options;
  if (!isRecord(options)) {
    return undefined;
  }

  return options;
};

const readResumeFromQueryCall = (queryCalls: unknown[], index: number): string | undefined => {
  const options = readOptionsFromQueryCall(queryCalls, index);
  if (options === undefined) {
    return undefined;
  }

  const resume = options.resume;
  if (typeof resume !== "string" || resume.length === 0) {
    return undefined;
  }

  return resume;
};

const importLanguageModelWithMockedRuntime = async (args: {
  queryCalls: unknown[];
  getStore: (incomingSessionKey: string) => Promise<unknown>;
  setStore: (incomingSessionKey: string, state: unknown) => Promise<void>;
}) => {
  let callCount = 0;

  mock.module("@anthropic-ai/claude-agent-sdk", () => {
    return {
      query: async function* (request: unknown) {
        args.queryCalls.push(request);
        callCount += 1;

        yield {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-${callCount}`,
        };
      },
    };
  });

  const moduleId = `../agent-sdk-language-model.ts?session-store-${Date.now()}-${Math.random()}`;
  const { AgentSdkAnthropicLanguageModel } = await import(moduleId);

  return new AgentSdkAnthropicLanguageModel({
    modelId: "claude-3-5-haiku-latest",
    provider: "anthropic.messages",
    settings: {},
    idGenerator: () => `id-${Date.now()}-${Math.random()}`,
    sessionStore: {
      get: async ({ incomingSessionKey }: { modelId: string; incomingSessionKey: string }) => {
        const state = await args.getStore(incomingSessionKey);
        if (!isRecord(state)) {
          return undefined;
        }

        const key = state.incomingSessionKey;
        const sessionId = state.sessionId;
        const promptMessageCount = state.promptMessageCount;

        if (
          typeof key !== "string" ||
          typeof sessionId !== "string" ||
          typeof promptMessageCount !== "number"
        ) {
          return undefined;
        }

        return {
          incomingSessionKey: key,
          sessionId,
          promptMessageCount,
          firstPromptMessageSignature:
            typeof state.firstPromptMessageSignature === "string"
              ? state.firstPromptMessageSignature
              : undefined,
          lastPromptMessageSignature:
            typeof state.lastPromptMessageSignature === "string"
              ? state.lastPromptMessageSignature
              : undefined,
        };
      },
      set: async ({
        incomingSessionKey,
        state,
      }: {
        modelId: string;
        incomingSessionKey: string;
        state: IncomingSessionState;
      }) => {
        await args.setStore(incomingSessionKey, state);
      },
    },
  });
};

describe("agent-sdk-language-model", () => {
  test("logs session-store get failures and still returns a result", async () => {
    const queryCalls: unknown[] = [];
    const warnings: string[] = [];
    console.warn = (...args) => {
      warnings.push(args.map((value) => String(value)).join(" "));
    };

    const model = await importLanguageModelWithMockedRuntime({
      queryCalls,
      getStore: async () => {
        throw new Error("get failed");
      },
      setStore: async () => {},
    });

    const result = await model.doGenerate({
      prompt: [userMessage("hello")],
      headers: {
        "x-conversation-id": "conversation-get-fail",
      },
    });

    expect(result.finishReason.unified).toBe("stop");
    expect(queryCalls).toHaveLength(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.includes("session store get failed")).toBeTrue();
    expect(warnings[0]?.includes("conversation-get-fail")).toBeTrue();
  });

  test("logs session-store set failures once and keeps in-memory resume", async () => {
    const queryCalls: unknown[] = [];
    const warnings: string[] = [];
    console.warn = (...args) => {
      warnings.push(args.map((value) => String(value)).join(" "));
    };

    const model = await importLanguageModelWithMockedRuntime({
      queryCalls,
      getStore: async () => {
        return undefined;
      },
      setStore: async () => {
        throw new Error("set failed");
      },
    });

    const firstTurnOptions = {
      prompt: [userMessage("first")],
      headers: {
        "x-conversation-id": "conversation-set-fail",
      },
    };
    await model.doGenerate(firstTurnOptions);

    const secondTurnOptions = {
      prompt: [userMessage("second")],
      headers: {
        "x-conversation-id": "conversation-set-fail",
      },
    };
    await model.doGenerate(secondTurnOptions);

    expect(queryCalls).toHaveLength(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("session-1");
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.includes("session store set failed")).toBeTrue();
    expect(warnings[0]?.includes("conversation-set-fail")).toBeTrue();
  });
});
