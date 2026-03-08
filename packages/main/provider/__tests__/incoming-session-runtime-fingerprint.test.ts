import { describe, expect, mock, test } from "bun:test";
import type { LanguageModelV3Message } from "@ai-sdk/provider";
import type { IncomingSessionState } from "../incoming-session-store";

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

const readOptionsFromQueryCall = (queryCalls: unknown[], index: number): Record<string, unknown> | undefined => {
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

const readRuntimeFingerprint = (sessionStoreArgs: unknown[], index: number): string | undefined => {
  const call = sessionStoreArgs[index];
  if (!isRecord(call)) {
    return undefined;
  }

  const runtimeFingerprint = call.runtimeFingerprint;
  if (typeof runtimeFingerprint !== "string" || runtimeFingerprint.length === 0) {
    return undefined;
  }

  return runtimeFingerprint;
};

const importLanguageModelWithSessionStore = async (args: {
  queryCalls: unknown[];
  storeGetCalls?: unknown[];
  storeSetCalls?: unknown[];
  settings: Record<string, unknown>;
  getStore: (incomingSessionKey: string, runtimeFingerprint: string) => Promise<unknown>;
  setStore: (incomingSessionKey: string, runtimeFingerprint: string, state: unknown) => Promise<void>;
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

  const moduleId = `../agent-sdk-language-model.ts?runtime-fingerprint-${Date.now()}-${Math.random()}`;
  const { AgentSdkAnthropicLanguageModel } = await import(moduleId);

  return new AgentSdkAnthropicLanguageModel({
    modelId: "claude-3-5-haiku-latest",
    provider: "anthropic.messages",
    settings: args.settings,
    idGenerator: () => `id-${Date.now()}-${Math.random()}`,
    sessionStore: {
      get: async ({
        incomingSessionKey,
        runtimeFingerprint,
      }: {
        modelId: string;
        runtimeFingerprint: string;
        incomingSessionKey: string;
      }) => {
        args.storeGetCalls?.push({ incomingSessionKey, runtimeFingerprint });
        const state = await args.getStore(incomingSessionKey, runtimeFingerprint);
        if (!isRecord(state)) {
          return undefined;
        }

        const key = state.incomingSessionKey;
        const sessionId = state.sessionId;
        const promptMessageCount = state.promptMessageCount;

        if (typeof key !== "string" || typeof sessionId !== "string" || typeof promptMessageCount !== "number") {
          return undefined;
        }

        return {
          incomingSessionKey: key,
          sessionId,
          promptMessageCount,
          firstPromptMessageSignature:
            typeof state.firstPromptMessageSignature === "string" ? state.firstPromptMessageSignature : undefined,
          lastPromptMessageSignature:
            typeof state.lastPromptMessageSignature === "string" ? state.lastPromptMessageSignature : undefined,
        };
      },
      set: async ({
        runtimeFingerprint,
        incomingSessionKey,
        state,
      }: {
        modelId: string;
        runtimeFingerprint: string;
        incomingSessionKey: string;
        state: IncomingSessionState;
      }) => {
        args.storeSetCalls?.push({ incomingSessionKey, runtimeFingerprint, state });
        await args.setStore(incomingSessionKey, runtimeFingerprint, state);
      },
    },
  });
};

describe("incoming-session runtime fingerprint", () => {
  test("isolates stored session ids by runtime fingerprint", async () => {
    const leftQueryCalls: unknown[] = [];
    const rightQueryCalls: unknown[] = [];
    const leftStoreSetCalls: unknown[] = [];
    const rightStoreGetCalls: unknown[] = [];
    const persistedStates = new Map<string, unknown>();

    const leftModel = await importLanguageModelWithSessionStore({
      queryCalls: leftQueryCalls,
      storeSetCalls: leftStoreSetCalls,
      settings: {
        baseURL: "http://localhost:5470/claude",
        apiKey: "api-key-left",
      },
      getStore: async (incomingSessionKey, runtimeFingerprint) => {
        return persistedStates.get(`${runtimeFingerprint}:${incomingSessionKey}`);
      },
      setStore: async (incomingSessionKey, runtimeFingerprint, state) => {
        persistedStates.set(`${runtimeFingerprint}:${incomingSessionKey}`, state);
      },
    });

    await leftModel.doGenerate({
      prompt: [userMessage("first")],
      headers: {
        "x-conversation-id": "conversation-shared",
      },
    });

    const rightModel = await importLanguageModelWithSessionStore({
      queryCalls: rightQueryCalls,
      storeGetCalls: rightStoreGetCalls,
      settings: {
        baseURL: "http://localhost:5470/claude",
        apiKey: "api-key-right",
      },
      getStore: async (incomingSessionKey, runtimeFingerprint) => {
        return persistedStates.get(`${runtimeFingerprint}:${incomingSessionKey}`);
      },
      setStore: async (incomingSessionKey, runtimeFingerprint, state) => {
        persistedStates.set(`${runtimeFingerprint}:${incomingSessionKey}`, state);
      },
    });

    await rightModel.doGenerate({
      prompt: [userMessage("second")],
      headers: {
        "x-conversation-id": "conversation-shared",
      },
    });

    expect(leftStoreSetCalls).toHaveLength(1);
    expect(rightStoreGetCalls).toHaveLength(1);
    expect(readRuntimeFingerprint(leftStoreSetCalls, 0)).toBeDefined();
    expect(readRuntimeFingerprint(rightStoreGetCalls, 0)).toBeDefined();
    expect(readRuntimeFingerprint(leftStoreSetCalls, 0)).not.toBe(readRuntimeFingerprint(rightStoreGetCalls, 0));
    expect(readResumeFromQueryCall(rightQueryCalls, 0)).toBeUndefined();
  });
});
