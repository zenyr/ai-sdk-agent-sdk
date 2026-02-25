import { describe, expect, test } from "bun:test";
import type { SharedV3Warning } from "@ai-sdk/provider";

import { createAbortBridge, prepareQueryContext } from "../application/query-context";
import type { ToolBridgeConfig } from "../domain/tool-bridge-config";
import type { IncomingSessionState } from "../incoming-session-store";

const userMessage = (text: string) => {
  return {
    role: "user" as const,
    content: [{ type: "text" as const, text }],
  };
};

const weatherTool = {
  type: "function" as const,
  name: "lookup_weather",
  description: "Lookup weather",
  inputSchema: {
    type: "object",
    additionalProperties: false,
    required: ["city"],
    properties: {
      city: {
        type: "string",
      },
    },
  },
};

const isAsyncIterable = (value: unknown): value is AsyncIterable<unknown> => {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  if (!(Symbol.asyncIterator in value)) {
    return false;
  }

  return typeof Reflect.get(value, Symbol.asyncIterator) === "function";
};

const partialExecutorWarning: SharedV3Warning = {
  type: "compatibility",
  feature: "toolExecutors.partial",
  details: "missing executors",
};

describe("query-context", () => {
  test("hydrates incoming session key and reuses session id for single user turns", async () => {
    const hydratedKeys: string[] = [];
    const previousIncomingSessionState: IncomingSessionState = {
      incomingSessionKey: "conversation-1",
      sessionId: "session-1",
      promptMessageCount: 1,
    };

    const queryContext = await prepareQueryContext({
      options: {
        prompt: [userMessage("hello")],
        headers: {
          "x-conversation-id": "conversation-1",
        },
      },
      providerSettingWarnings: [],
      previousSessionStates: () => [],
      previousIncomingSessionStates: () => [previousIncomingSessionState],
      hydrateIncomingSessionState: async (incomingSessionKey) => {
        hydratedKeys.push(incomingSessionKey);
      },
      buildToolBridgeConfig: () => {
        return undefined;
      },
      buildPartialToolExecutorWarning: () => partialExecutorWarning,
    });

    expect(hydratedKeys).toEqual(["conversation-1"]);
    expect(queryContext.incomingSessionKey).toBe("conversation-1");
    expect(queryContext.promptQueryInput.resumeSessionId).toBe("session-1");
    expect(queryContext.prompt).toBe("hello");
  });

  test("sets json output format and prompt preamble in json mode", async () => {
    const queryContext = await prepareQueryContext({
      options: {
        prompt: [userMessage("return payload")],
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            required: ["ok"],
            properties: {
              ok: {
                type: "boolean",
              },
            },
          },
        },
      },
      providerSettingWarnings: [],
      previousSessionStates: () => [],
      previousIncomingSessionStates: () => [],
      hydrateIncomingSessionState: async () => {},
      buildToolBridgeConfig: () => {
        return undefined;
      },
      buildPartialToolExecutorWarning: () => partialExecutorWarning,
    });

    expect(queryContext.completionMode.type).toBe("json");
    expect(typeof queryContext.queryPrompt).toBe("string");
    expect(
      queryContext.prompt.startsWith("Return only JSON that matches the required schema."),
    ).toBeTrue();
    expect(queryContext.outputFormat).toEqual({
      type: "json_schema",
      schema: {
        type: "object",
        required: ["ok"],
        properties: {
          ok: {
            type: "boolean",
          },
        },
      },
    });
  });

  test("adds partial executor warning when tools are only partially bridged", async () => {
    let partialWarningCalls = 0;

    const queryContext = await prepareQueryContext({
      options: {
        prompt: [userMessage("call weather")],
        tools: [weatherTool],
        toolChoice: { type: "required" },
      },
      providerSettingWarnings: [],
      previousSessionStates: () => [],
      previousIncomingSessionStates: () => [],
      hydrateIncomingSessionState: async () => {},
      buildToolBridgeConfig: () => {
        return {
          allowedTools: ["mcp__ai_sdk_tool_bridge__lookup_weather"],
          mcpServers: {},
          hasAnyExecutor: true,
          allToolsHaveExecutors: false,
          missingExecutorToolNames: ["lookup_weather"],
        };
      },
      buildPartialToolExecutorWarning: (missingExecutorToolNames) => {
        partialWarningCalls += 1;
        expect(missingExecutorToolNames).toEqual(["lookup_weather"]);
        return partialExecutorWarning;
      },
    });

    expect(partialWarningCalls).toBe(1);
    expect(queryContext.useNativeToolExecution).toBeFalse();
    expect(queryContext.completionMode.type).toBe("tools");
    expect(queryContext.warnings).toContainEqual(partialExecutorWarning);
  });

  test("enables native tool execution when all bridged tools have executors", async () => {
    let partialWarningCalled = false;

    const queryContext = await prepareQueryContext({
      options: {
        prompt: [userMessage("call weather")],
        tools: [weatherTool],
        toolChoice: { type: "required" },
      },
      providerSettingWarnings: [],
      previousSessionStates: () => [],
      previousIncomingSessionStates: () => [],
      hydrateIncomingSessionState: async () => {},
      buildToolBridgeConfig: (): ToolBridgeConfig => {
        return {
          allowedTools: ["mcp__ai_sdk_tool_bridge__lookup_weather"],
          mcpServers: {},
          hasAnyExecutor: true,
          allToolsHaveExecutors: true,
          missingExecutorToolNames: [],
        };
      },
      buildPartialToolExecutorWarning: () => {
        partialWarningCalled = true;
        return partialExecutorWarning;
      },
    });

    expect(queryContext.useNativeToolExecution).toBeTrue();
    expect(partialWarningCalled).toBeFalse();
    expect(
      queryContext.warnings.some((warning) => {
        return warning.type === "compatibility" && warning.feature === "toolExecutors.partial";
      }),
    ).toBeFalse();
  });

  test("switches to async query prompt for multimodal input", async () => {
    const queryContext = await prepareQueryContext({
      options: {
        prompt: [
          {
            role: "user",
            content: [
              { type: "text", text: "describe image" },
              {
                type: "file",
                mediaType: "image/png",
                data: "data:image/png;base64,Zm9v",
              },
            ],
          },
        ],
      },
      providerSettingWarnings: [],
      previousSessionStates: () => [],
      previousIncomingSessionStates: () => [],
      hydrateIncomingSessionState: async () => {},
      buildToolBridgeConfig: () => {
        return undefined;
      },
      buildPartialToolExecutorWarning: () => partialExecutorWarning,
    });

    expect(isAsyncIterable(queryContext.queryPrompt)).toBeTrue();
  });

  test("createAbortBridge mirrors external abort signal", () => {
    const externalAbortController = new AbortController();
    const { abortController, cleanupAbortListener } = createAbortBridge(
      externalAbortController.signal,
    );

    expect(abortController.signal.aborted).toBeFalse();
    externalAbortController.abort();
    expect(abortController.signal.aborted).toBeTrue();

    cleanupAbortListener();
  });
});
