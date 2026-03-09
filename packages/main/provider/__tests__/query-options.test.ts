import { describe, expect, test } from "bun:test";

import { buildAgentQueryOptions } from "../application/query-options";

describe("query-options", () => {
  const onStderr = () => {};

  test("forces maxTurns to 1 without tool mode", () => {
    const queryOptions = buildAgentQueryOptions({
      modelId: "claude-3-5-haiku-latest",
      settings: {
        apiKey: "api-key",
        baseURL: "https://proxy.example/v1/",
      },
      allowedTools: [],
      mcpServers: undefined,
      resumeSessionId: "session-1",
      systemPrompt: "system",
      maxTurns: 7,
      useNativeToolExecution: false,
      abortController: new AbortController(),
      outputFormat: undefined,
      effort: undefined,
      thinking: undefined,
      includePartialMessages: false,
      onStderr,
    });

    expect(queryOptions.maxTurns).toBe(1);
    expect(queryOptions.model).toBe("claude-3-5-haiku-latest");
    expect(queryOptions.resume).toBe("session-1");
    expect(queryOptions.systemPrompt).toBe("system");
    expect(queryOptions.cwd).toBe(process.cwd());

    const env = queryOptions.env;
    expect(env).toBeDefined();
    if (env === undefined) {
      return;
    }

    expect(env.ANTHROPIC_API_KEY).toBe("api-key");
    expect(env.ANTHROPIC_BASE_URL).toBe("https://proxy.example/v1");
    expect(queryOptions.includePartialMessages).toBeFalse();
  });

  test("keeps configured maxTurns with native tool execution", () => {
    const queryOptions = buildAgentQueryOptions({
      modelId: "claude-3-5-haiku-latest",
      settings: {},
      allowedTools: [],
      mcpServers: {},
      resumeSessionId: undefined,
      systemPrompt: undefined,
      maxTurns: 4,
      useNativeToolExecution: true,
      abortController: new AbortController(),
      outputFormat: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: {
            ok: {
              type: "boolean",
            },
          },
        },
      },
      effort: "medium",
      thinking: {
        type: "enabled",
        budgetTokens: 256,
      },
      includePartialMessages: true,
      onStderr,
    });

    expect(queryOptions.maxTurns).toBe(4);
    expect(queryOptions.allowedTools).toEqual([]);
    expect(queryOptions.mcpServers).toEqual({});
    expect(queryOptions.includePartialMessages).toBeTrue();
    expect(queryOptions.permissionMode).toBe("dontAsk");
  });

  test("keeps configured maxTurns in tool mode without native execution", () => {
    const queryOptions = buildAgentQueryOptions({
      modelId: "claude-3-5-haiku-latest",
      settings: {},
      allowedTools: ["tool-1"],
      mcpServers: {},
      resumeSessionId: "session-1",
      systemPrompt: "system",
      maxTurns: 9,
      useNativeToolExecution: false,
      abortController: new AbortController(),
      outputFormat: undefined,
      effort: undefined,
      thinking: undefined,
      includePartialMessages: false,
      onStderr,
    });

    expect(queryOptions.maxTurns).toBe(9);
  });

  test("forwards experimental_agentSdk options while preserving adapter-owned fields", () => {
    const queryOptions = buildAgentQueryOptions({
      modelId: "claude-3-5-haiku-latest",
      settings: {
        experimental_agentSdk: {
          cwd: "/tmp/agent-sdk-e2e",
          permissionMode: "plan",
          settingSources: ["project"],
          debug: true,
        },
      },
      allowedTools: ["tool-1"],
      mcpServers: {},
      resumeSessionId: "session-1",
      systemPrompt: "system",
      maxTurns: 9,
      useNativeToolExecution: false,
      abortController: new AbortController(),
      outputFormat: undefined,
      effort: undefined,
      thinking: undefined,
      includePartialMessages: false,
      onStderr,
    });

    expect(queryOptions.cwd).toBe("/tmp/agent-sdk-e2e");
    expect(queryOptions.permissionMode).toBe("dontAsk");
    expect(queryOptions.settingSources).toEqual([]);
    expect(queryOptions.debug).toBeTrue();
    expect(queryOptions.maxTurns).toBe(9);
  });
});
