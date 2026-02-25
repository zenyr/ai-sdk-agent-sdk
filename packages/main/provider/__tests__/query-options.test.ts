import { describe, expect, test } from "bun:test";

import { buildAgentQueryOptions } from "../application/query-options";

describe("query-options", () => {
  test("forces maxTurns to 1 without native tool execution", () => {
    const queryOptions = buildAgentQueryOptions({
      modelId: "claude-3-5-haiku-latest",
      settings: {
        apiKey: "api-key",
        baseURL: "https://proxy.example/v1/",
      },
      allowedTools: ["tool-1"],
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
    });

    expect(queryOptions.maxTurns).toBe(4);
    expect(queryOptions.allowedTools).toEqual([]);
    expect(queryOptions.mcpServers).toEqual({});
    expect(queryOptions.includePartialMessages).toBeTrue();
    expect(queryOptions.permissionMode).toBe("dontAsk");
  });
});
