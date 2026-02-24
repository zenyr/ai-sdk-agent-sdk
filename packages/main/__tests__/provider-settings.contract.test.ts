import { afterEach, describe, expect, mock, test } from "bun:test";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

afterEach(() => {
  mock.restore();
});

const buildMockResultUsage = () => {
  return {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
};

const buildSuccessfulResult = () => {
  return {
    type: "result",
    subtype: "success",
    stop_reason: "end_turn",
    result: "ok",
    usage: buildMockResultUsage(),
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const readQueryCall = (
  queryCalls: unknown[],
  index: number,
): Record<string, unknown> | undefined => {
  const queryCall = queryCalls[index];
  if (!isRecord(queryCall)) {
    return undefined;
  }

  return queryCall;
};

const readOptionsFromQueryCall = (
  queryCalls: unknown[],
  index: number,
): Record<string, unknown> | undefined => {
  const queryCall = readQueryCall(queryCalls, index);
  if (queryCall === undefined) {
    return undefined;
  }

  const options = queryCall.options;
  if (!isRecord(options)) {
    return undefined;
  }

  return options;
};

const readPromptFromQueryCall = (queryCalls: unknown[], index: number): string | undefined => {
  const queryCall = readQueryCall(queryCalls, index);
  if (queryCall === undefined) {
    return undefined;
  }

  const prompt = queryCall.prompt;
  if (typeof prompt !== "string") {
    return undefined;
  }

  return prompt;
};

const isAsyncIterable = (value: unknown): value is AsyncIterable<unknown> => {
  if (!isRecord(value)) {
    return false;
  }

  if (!(Symbol.asyncIterator in value)) {
    return false;
  }

  const asyncIterator = Reflect.get(value, Symbol.asyncIterator);
  return typeof asyncIterator === "function";
};

const readFirstPromptStreamMessageFromQueryCall = async (
  queryCalls: unknown[],
  index: number,
): Promise<Record<string, unknown> | undefined> => {
  const queryCall = readQueryCall(queryCalls, index);
  if (queryCall === undefined) {
    return undefined;
  }

  const prompt = queryCall.prompt;
  if (!isAsyncIterable(prompt)) {
    return undefined;
  }

  for await (const promptMessage of prompt) {
    if (!isRecord(promptMessage)) {
      return undefined;
    }

    return promptMessage;
  }

  return undefined;
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

const readEnvFromFirstQueryCall = (queryCalls: unknown[]): Record<string, unknown> | undefined => {
  const options = readOptionsFromQueryCall(queryCalls, 0);
  if (!isRecord(options)) {
    return undefined;
  }

  const env = options.env;
  if (!isRecord(env)) {
    return undefined;
  }

  return env;
};

const readPromptFromFirstQueryCall = (queryCalls: unknown[]): string | undefined => {
  return readPromptFromQueryCall(queryCalls, 0);
};

const readOutputFormatFromFirstQueryCall = (
  queryCalls: unknown[],
): Record<string, unknown> | undefined => {
  const options = readOptionsFromQueryCall(queryCalls, 0);
  if (!isRecord(options)) {
    return undefined;
  }

  const outputFormat = options.outputFormat;
  if (!isRecord(outputFormat)) {
    return undefined;
  }

  return outputFormat;
};

const readOptionsFromFirstQueryCall = (
  queryCalls: unknown[],
): Record<string, unknown> | undefined => {
  return readOptionsFromQueryCall(queryCalls, 0);
};

const readSystemPromptFromFirstQueryCall = (queryCalls: unknown[]): string | undefined => {
  const options = readOptionsFromFirstQueryCall(queryCalls);
  if (options === undefined) {
    return undefined;
  }

  const systemPrompt = options.systemPrompt;
  if (typeof systemPrompt !== "string" || systemPrompt.length === 0) {
    return undefined;
  }

  return systemPrompt;
};

const createUniqueCacheKey = (prefix: string): string => {
  return `${prefix}-${Date.now()}-${Math.random()}`;
};

const importIndexWithMockedQuery = async (args: {
  queryCalls: unknown[];
  resultFactory?: () => unknown;
  messagesFactory?: () => unknown[];
}) => {
  mock.module("@anthropic-ai/claude-agent-sdk", () => {
    return {
      createSdkMcpServer: (options: { name: string }) => {
        return {
          type: "sdk",
          name: options.name,
          instance: {
            tools: "tools" in options && Array.isArray(options.tools) ? options.tools : [],
          },
        };
      },
      query: async function* (request: unknown) {
        args.queryCalls.push(request);

        const messages = args.messagesFactory?.();
        if (Array.isArray(messages)) {
          for (const message of messages) {
            yield message;
          }

          return;
        }

        yield args.resultFactory?.() ?? buildSuccessfulResult();
      },
    };
  });

  const moduleId = `../index.ts?provider-settings-${Date.now()}-${Math.random()}`;
  return import(moduleId);
};

describe("provider settings contract", () => {
  test("forwards baseURL and apiKey into claude-agent-sdk env", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      apiKey: "api-key-test",
      baseURL: "https://proxy.example/v1/",
    });

    const model = provider("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Say hello." }],
        },
      ],
    });

    const env = readEnvFromFirstQueryCall(queryCalls);
    expect(env).toBeDefined();

    if (env === undefined) {
      return;
    }

    expect(env.ANTHROPIC_API_KEY).toBe("api-key-test");
    expect(env.ANTHROPIC_BASE_URL).toBe("https://proxy.example/v1");
  });

  test("forwards authToken into claude-agent-sdk env", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      authToken: "auth-token-test",
      baseURL: "https://auth-proxy.example/v1/",
    });

    const model = provider("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Say hello." }],
        },
      ],
    });

    const env = readEnvFromFirstQueryCall(queryCalls);
    expect(env).toBeDefined();

    if (env === undefined) {
      return;
    }

    expect(env.ANTHROPIC_AUTH_TOKEN).toBe("auth-token-test");
    expect(env.ANTHROPIC_BASE_URL).toBe("https://auth-proxy.example/v1");
  });

  test("preserves custom provider name", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      name: "anthropic.proxy",
    });

    const model = provider("claude-3-5-haiku-latest");
    expect(model.provider).toBe("anthropic.proxy");
  });

  test("uses custom generateId for tool-call id generation", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: "result",
          subtype: "success",
          stop_reason: "tool_use",
          result: "",
          structured_output: {
            type: "tool-calls",
            calls: [
              {
                toolName: "lookup_weather",
                input: {
                  city: "seoul",
                },
              },
            ],
          },
          usage: buildMockResultUsage(),
        };
      },
    });

    const provider = createAnthropic({
      generateId: () => "fixed-tool-call-id",
    });

    const model = provider("claude-3-5-haiku-latest");
    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Call tool." }],
        },
      ],
      tools: [
        {
          type: "function",
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
        },
      ],
      toolChoice: { type: "required" },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolCallId).toBe("fixed-tool-call-id");
  });

  test("tool mode uses in-process MCP bridge instead of output schema prompting", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Call tool when needed." }],
        },
      ],
      tools: [
        {
          type: "function",
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
        },
      ],
      toolChoice: { type: "required" },
    });

    const prompt = readPromptFromFirstQueryCall(queryCalls);
    expect(prompt).toBeDefined();

    if (prompt === undefined) {
      return;
    }

    expect(prompt.includes("Call tool when needed.")).toBeTrue();
    expect(prompt.includes("You are in tool routing mode.")).toBeFalse();

    const outputFormat = readOutputFormatFromFirstQueryCall(queryCalls);
    expect(outputFormat).toBeUndefined();

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    if (Array.isArray(options.allowedTools) && options.allowedTools.length > 0) {
      expect(options.allowedTools).toContain("mcp__ai_sdk_tool_bridge__lookup_weather");

      const mcpServers = options.mcpServers;
      expect(isRecord(mcpServers)).toBeTrue();

      if (!isRecord(mcpServers)) {
        return;
      }

      const bridgeServer = mcpServers.ai_sdk_tool_bridge;
      expect(isRecord(bridgeServer)).toBeTrue();

      if (!isRecord(bridgeServer)) {
        return;
      }

      expect(bridgeServer.type).toBe("sdk");
      expect(bridgeServer.name).toBe("ai_sdk_tool_bridge");
    }
  });

  test("tool mode preserves configured thinking", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Call tool when needed." }],
        },
      ],
      tools: [
        {
          type: "function",
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
        },
      ],
      toolChoice: { type: "required" },
      providerOptions: {
        anthropic: {
          thinking: {
            type: "adaptive",
          },
        },
      },
    });

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    expect(isRecord(options.thinking)).toBeTrue();

    if (!isRecord(options.thinking)) {
      return;
    }

    expect(options.thinking.type).toBe("adaptive");
  });

  test("tool mode returns explicit error for empty successful output", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "",
          usage: buildMockResultUsage(),
        };
      },
    });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Call tool when needed." }],
        },
      ],
      tools: [
        {
          type: "function",
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
        },
      ],
      toolChoice: { type: "required" },
    });

    expect(result.finishReason.unified).toBe("error");
    expect(result.finishReason.raw).toBe("empty-tool-routing-output");

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("text");

    if (firstContentPart === undefined || firstContentPart.type !== "text") {
      return;
    }

    expect(firstContentPart.text).toContain("Tool routing produced no tool call");
  });

  test("tool mode recovers native MCP tool-use from error_max_turns", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: "assistant",
            message: {
              content: [
                {
                  type: "tool_use",
                  id: "toolu_native_1",
                  name: "mcp__ai_sdk_tool_bridge__bash",
                  input: {
                    command: 'bun -e "console.log(Math.random())"',
                    description: "Run Math.random once",
                  },
                },
              ],
            },
          },
          {
            type: "result",
            subtype: "error_max_turns",
            stop_reason: null,
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: [],
            uuid: "uuid-native-tool-1",
            session_id: "session-native-tool-1",
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "run bash" }],
        },
      ],
      tools: [
        {
          type: "function",
          name: "bash",
          description: "Run shell command",
          inputSchema: {
            type: "object",
            additionalProperties: false,
            required: ["command", "description"],
            properties: {
              command: {
                type: "string",
              },
              description: {
                type: "string",
              },
            },
          },
        },
      ],
      toolChoice: { type: "required" },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolCallId).toBe("toolu_native_1");
    expect(firstContentPart.toolName).toBe("bash");
    expect(firstContentPart.input).toContain("Math.random");
    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.finishReason.raw).toBe("tool_use");
  });

  test("tool mode recovers native MCP tool-use when query returns no result message", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-no-result-tool",
                model: "mock-model",
              },
            },
          },
          {
            type: "stream_event",
            event: {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "tool_use",
                id: "toolu_no_result_1",
                name: "mcp__ai_sdk_tool_bridge__bash",
              },
            },
          },
          {
            type: "stream_event",
            event: {
              type: "content_block_delta",
              index: 0,
              delta: {
                type: "input_json_delta",
                partial_json:
                  '{"command":"bun -e \\"console.log(Math.random())\\"","description":"Run Math.random once"}',
              },
            },
          },
          {
            type: "stream_event",
            event: {
              type: "content_block_stop",
              index: 0,
            },
          },
          {
            type: "stream_event",
            event: {
              type: "message_delta",
              delta: {
                stop_reason: "tool_use",
              },
              usage: {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "run bash" }],
        },
      ],
      tools: [
        {
          type: "function",
          name: "bash",
          description: "Run shell command",
          inputSchema: {
            type: "object",
            additionalProperties: false,
            required: ["command", "description"],
            properties: {
              command: {
                type: "string",
              },
              description: {
                type: "string",
              },
            },
          },
        },
      ],
      toolChoice: { type: "required" },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolCallId).toBe("toolu_no_result_1");
    expect(firstContentPart.toolName).toBe("bash");
    expect(firstContentPart.input).toContain("Math.random");
    expect(result.finishReason.unified).toBe("tool-calls");
    expect(result.finishReason.raw).toBe("tool_use");
  });

  test("tool mode recovers from structured output retry exhaustion in doGenerate", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: "assistant",
            message: {
              content: [{ type: "text", text: '{"type":"text","text":"안녕하세요"}' }],
            },
          },
          {
            type: "result",
            subtype: "error_max_structured_output_retries",
            stop_reason: "end_turn",
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: ['[{"expected":"string","code":"invalid_type","path":["reason"]}]'],
            uuid: "uuid-1",
            session_id: "session-1",
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
      tools: [
        {
          type: "function",
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
        },
      ],
      toolChoice: { type: "required" },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("text");

    if (firstContentPart === undefined || firstContentPart.type !== "text") {
      return;
    }

    expect(firstContentPart.text).toBe("안녕하세요");
    expect(result.finishReason.unified).toBe("stop");
  });

  test("tool mode recovers legacy single tool-call object from assistant text", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: "assistant",
            message: {
              content: [
                {
                  type: "text",
                  text: '{"tool":"bash","parameters":{"command":"bun -e \\"console.log(Math.random())\\"","description":"Run Math.random once"}}',
                },
              ],
            },
          },
          {
            type: "result",
            subtype: "error_max_structured_output_retries",
            stop_reason: "end_turn",
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: ['[{"expected":"string","code":"invalid_type","path":["reason"]}]'],
            uuid: "uuid-legacy-tool-1",
            session_id: "session-legacy-tool-1",
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "run bash" }],
        },
      ],
      tools: [
        {
          type: "function",
          name: "bash",
          description: "Run shell command",
          inputSchema: {
            type: "object",
            additionalProperties: false,
            required: ["command", "description"],
            properties: {
              command: {
                type: "string",
              },
              description: {
                type: "string",
              },
            },
          },
        },
      ],
      toolChoice: { type: "required" },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolName).toBe("bash");
    expect(firstContentPart.input).toContain("Math.random");
    expect(result.finishReason.unified).toBe("tool-calls");
  });

  test("reuses session from conversationId header with single user turns", async () => {
    const conversationId = createUniqueCacheKey("conversation-header");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-header-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    const firstTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      headers: {
        "x-conversation-id": conversationId,
      },
    };

    await model.doGenerate(firstTurnOptions);

    const secondTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      headers: {
        "x-conversation-id": conversationId,
      },
    };

    await model.doGenerate(secondTurnOptions);

    expect(queryCalls.length).toBe(2);

    const resumedSessionId = readResumeFromQueryCall(queryCalls, 1);
    expect(resumedSessionId).toBe("session-header-1");

    const secondPrompt = readPromptFromQueryCall(queryCalls, 1);
    expect(secondPrompt).toBeDefined();

    if (secondPrompt === undefined) {
      return;
    }

    expect(secondPrompt.includes("첫 질문")).toBeFalse();
    expect(secondPrompt.includes("두 번째 질문")).toBeTrue();
  });

  test("reuses session from telemetry metadata conversationId", async () => {
    const conversationId = createUniqueCacheKey("conversation-telemetry");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-telemetry-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    const firstTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      experimental_telemetry: {
        metadata: {
          conversationId,
        },
      },
    };

    await model.doGenerate(firstTurnOptions);

    const secondTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      experimental_telemetry: {
        metadata: {
          conversationId,
        },
      },
    };

    await model.doGenerate(secondTurnOptions);

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("session-telemetry-1");
  });

  test("reuses session from providerOptions agentSdk conversationId", async () => {
    const conversationId = createUniqueCacheKey("conversation-provider-options");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-provider-options-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    const firstTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      providerOptions: {
        agentSdk: {
          conversationId,
        },
      },
    };

    await model.doGenerate(firstTurnOptions);

    const secondTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      providerOptions: {
        agentSdk: {
          conversationId,
        },
      },
    };

    await model.doGenerate(secondTurnOptions);

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("session-provider-options-1");
  });

  test("sends image attachment through SDKUserMessage prompt stream", async () => {
    const queryCalls: unknown[] = [];

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: "image-session-1",
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            { type: "text", text: "이 이미지를 설명해줘" },
            {
              type: "image",
              mediaType: "image/png",
              image: "data:image/png;base64,Zm9v",
            },
          ],
        },
      ],
    });

    expect(queryCalls.length).toBe(1);
    expect(readPromptFromQueryCall(queryCalls, 0)).toBeUndefined();

    const streamedPromptMessage = await readFirstPromptStreamMessageFromQueryCall(queryCalls, 0);
    expect(streamedPromptMessage).toBeDefined();

    if (streamedPromptMessage === undefined) {
      return;
    }

    const message = streamedPromptMessage.message;
    expect(isRecord(message)).toBeTrue();

    if (!isRecord(message)) {
      return;
    }

    const content = message.content;
    expect(Array.isArray(content)).toBeTrue();

    if (!Array.isArray(content)) {
      return;
    }

    const imageBlock = content.find((contentBlock) => {
      return isRecord(contentBlock) && contentBlock.type === "image";
    });

    expect(imageBlock).toBeDefined();

    if (!isRecord(imageBlock)) {
      return;
    }

    const source = imageBlock.source;
    expect(isRecord(source)).toBeTrue();

    if (!isRecord(source)) {
      return;
    }

    expect(source.type).toBe("base64");
    expect(source.media_type).toBe("image/png");
    expect(source.data).toBe("Zm9v");
  });

  test("legacy compatibility: reuses session from x-opencode-session header", async () => {
    const legacyConversationKey = createUniqueCacheKey("legacy-opencode-header");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `legacy-header-session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      headers: {
        "x-opencode-session": legacyConversationKey,
      },
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      headers: {
        "x-opencode-session": legacyConversationKey,
      },
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("legacy-header-session-1");
  });

  test("legacy compatibility: reuses session from telemetry metadata sessionId", async () => {
    const legacyTelemetrySessionKey = createUniqueCacheKey("legacy-telemetry-session-key");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `legacy-telemetry-session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      experimental_telemetry: {
        metadata: {
          sessionId: legacyTelemetrySessionKey,
        },
      },
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      experimental_telemetry: {
        metadata: {
          sessionId: legacyTelemetrySessionKey,
        },
      },
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("legacy-telemetry-session-1");
  });

  test("legacy compatibility: reuses session from providerOptions opencode promptCacheKey", async () => {
    const legacyPromptCacheKey = createUniqueCacheKey("legacy-provider-options-key");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `legacy-provider-options-session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      providerOptions: {
        opencode: {
          promptCacheKey: legacyPromptCacheKey,
        },
      },
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      providerOptions: {
        opencode: {
          promptCacheKey: legacyPromptCacheKey,
        },
      },
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("legacy-provider-options-session-1");
  });

  test("legacy compatibility: reuses session from providerOptions anthropic promptCacheKey", async () => {
    const legacyAnthropicPromptCacheKey = createUniqueCacheKey(
      "legacy-anthropic-provider-options-key",
    );
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `legacy-anthropic-provider-options-session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      providerOptions: {
        anthropic: {
          promptCacheKey: legacyAnthropicPromptCacheKey,
        },
      },
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      providerOptions: {
        anthropic: {
          promptCacheKey: legacyAnthropicPromptCacheKey,
        },
      },
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe(
      "legacy-anthropic-provider-options-session-1",
    );
  });

  test("discovers session from unknown providerOptions namespace via candidate keys", async () => {
    const unknownNamespaceKey = createUniqueCacheKey("unknown-namespace-key");
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `unknown-namespace-session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      providerOptions: {
        openai: {
          promptCacheKey: unknownNamespaceKey,
        },
      },
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      providerOptions: {
        openai: {
          promptCacheKey: unknownNamespaceKey,
        },
      },
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBe("unknown-namespace-session-1");
  });

  test("persists conversationId join across model instance reload", async () => {
    const previousXdgCacheHome = process.env.XDG_CACHE_HOME;
    const xdgCacheHome = join(tmpdir(), createUniqueCacheKey("agent-sdk-session-cache"));
    process.env.XDG_CACHE_HOME = xdgCacheHome;

    try {
      const conversationId = createUniqueCacheKey("conversation-persisted");
      const firstQueryCalls: unknown[] = [];

      const firstImport = await importIndexWithMockedQuery({
        queryCalls: firstQueryCalls,
        resultFactory: () => {
          return {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: "persisted-session-id",
          };
        },
      });

      const firstModel = firstImport.createAnthropic({})("claude-3-5-haiku-latest");
      await firstModel.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "첫 질문" }],
          },
        ],
        headers: {
          "x-conversation-id": conversationId,
        },
      });

      const secondQueryCalls: unknown[] = [];
      const secondImport = await importIndexWithMockedQuery({
        queryCalls: secondQueryCalls,
        resultFactory: () => {
          return {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: "fresh-session-id",
          };
        },
      });

      const secondModel = secondImport.createAnthropic({})("claude-3-5-haiku-latest");
      await secondModel.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "두 번째 질문" }],
          },
        ],
        headers: {
          "x-conversation-id": conversationId,
        },
      });

      expect(secondQueryCalls.length).toBe(1);
      expect(readResumeFromQueryCall(secondQueryCalls, 0)).toBe("persisted-session-id");
    } finally {
      if (previousXdgCacheHome === undefined) {
        delete process.env.XDG_CACHE_HOME;
      } else {
        process.env.XDG_CACHE_HOME = previousXdgCacheHome;
      }

      await rm(xdgCacheHome, { recursive: true, force: true }).catch(() => {
        return undefined;
      });
    }
  });

  test("does not reuse persisted session when model id differs", async () => {
    const previousXdgCacheHome = process.env.XDG_CACHE_HOME;
    const xdgCacheHome = join(tmpdir(), createUniqueCacheKey("agent-sdk-session-cache"));
    process.env.XDG_CACHE_HOME = xdgCacheHome;

    try {
      const conversationId = createUniqueCacheKey("conversation-model-scope");
      const firstQueryCalls: unknown[] = [];

      const firstImport = await importIndexWithMockedQuery({
        queryCalls: firstQueryCalls,
        resultFactory: () => {
          return {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: "model-a-session-id",
          };
        },
      });

      const firstModel = firstImport.createAnthropic({})("claude-3-5-haiku-latest");
      await firstModel.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "첫 질문" }],
          },
        ],
        headers: {
          "x-conversation-id": conversationId,
        },
      });

      const secondQueryCalls: unknown[] = [];
      const secondImport = await importIndexWithMockedQuery({
        queryCalls: secondQueryCalls,
        resultFactory: () => {
          return {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: "model-b-session-id",
          };
        },
      });

      const secondModel = secondImport.createAnthropic({})("claude-opus-4-1");
      await secondModel.doGenerate({
        prompt: [
          {
            role: "user",
            content: [{ type: "text", text: "두 번째 질문" }],
          },
        ],
        headers: {
          "x-conversation-id": conversationId,
        },
      });

      expect(secondQueryCalls.length).toBe(1);
      expect(readResumeFromQueryCall(secondQueryCalls, 0)).toBeUndefined();
    } finally {
      if (previousXdgCacheHome === undefined) {
        delete process.env.XDG_CACHE_HOME;
      } else {
        process.env.XDG_CACHE_HOME = previousXdgCacheHome;
      }

      await rm(xdgCacheHome, { recursive: true, force: true }).catch(() => {
        return undefined;
      });
    }
  });

  test("reuses claude session with appended prompt messages", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "첫 답변" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
    });

    expect(queryCalls.length).toBe(2);

    const secondPrompt = readPromptFromQueryCall(queryCalls, 1);
    expect(secondPrompt).toBeDefined();

    if (secondPrompt === undefined) {
      return;
    }

    expect(secondPrompt.includes("첫 질문")).toBeFalse();
    expect(secondPrompt.includes("첫 답변")).toBeFalse();
    expect(secondPrompt.includes("두 번째 질문")).toBeTrue();

    const resumedSessionId = readResumeFromQueryCall(queryCalls, 1);
    expect(resumedSessionId).toBe("session-1");
  });

  test("reuses session while preserving assistant tool-call context", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-tool-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "run bash" }],
        },
      ],
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "run bash" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_1",
              toolName: "bash",
              input: {
                command: 'bun -e "console.log(1)"',
              },
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_1",
              toolName: "bash",
              output: {
                type: "text",
                value: "1\n",
              },
            },
          ],
        },
        {
          role: "user",
          content: [{ type: "text", text: "summarize result" }],
        },
      ],
    });

    expect(queryCalls.length).toBe(2);

    const secondPrompt = readPromptFromQueryCall(queryCalls, 1);
    expect(secondPrompt).toBeDefined();

    if (secondPrompt === undefined) {
      return;
    }

    expect(secondPrompt.includes("run bash")).toBeFalse();
    expect(secondPrompt.includes("[tool-call:bash#call_1]")).toBeTrue();
    expect(secondPrompt.includes("[tool-result:bash#call_1]")).toBeTrue();
    expect(secondPrompt.includes("summarize result")).toBeTrue();

    const resumedSessionId = readResumeFromQueryCall(queryCalls, 1);
    expect(resumedSessionId).toBe("session-tool-1");
  });

  test("does not reuse claude session when prompt diverges", async () => {
    const queryCalls: unknown[] = [];

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: "session-diverge",
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "완전히 다른 질문" }],
        },
      ],
    });

    expect(queryCalls.length).toBe(2);
    expect(readResumeFromQueryCall(queryCalls, 1)).toBeUndefined();

    const secondPrompt = readPromptFromQueryCall(queryCalls, 1);
    expect(secondPrompt).toContain("완전히 다른 질문");
  });

  test("tracks multiple prompt histories and resumes matching branch", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        callCount += 1;

        return {
          type: "result",
          subtype: "success",
          stop_reason: "end_turn",
          result: "ok",
          usage: buildMockResultUsage(),
          session_id: `session-branch-${callCount}`,
        };
      },
    });

    const model = createAnthropic({})("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "A: 첫 질문" }],
        },
      ],
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "B: 첫 질문" }],
        },
      ],
    });

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "A: 첫 질문" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "A: 첫 답변" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "A: 두 번째 질문" }],
        },
      ],
    });

    expect(queryCalls.length).toBe(3);
    expect(readResumeFromQueryCall(queryCalls, 2)).toBe("session-branch-1");
  });

  test("runs claude-agent-sdk in isolated no-tool mode", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider("claude-3-5-haiku-latest");

    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Say hello." }],
        },
      ],
    });

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    expect(options.tools).toEqual([]);
    expect(options.allowedTools).toEqual([]);
    expect(options.settingSources).toEqual([]);
    expect(options.permissionMode).toBe("dontAsk");
    expect(options.maxTurns).toBe(1);
  });

  test("adds warnings for unsupported provider settings", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      headers: {
        "x-test-header": "enabled",
      },
      fetch: async (input: RequestInfo | URL, init?: RequestInit) => {
        return fetch(input, init);
      },
    });

    const model = provider("claude-3-5-haiku-latest");
    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Say hello." }],
        },
      ],
    });

    const features = result.warnings
      .filter((warning: unknown) => {
        return isRecord(warning) && "feature" in warning;
      })
      .map((warning: unknown) => {
        if (!isRecord(warning)) {
          return undefined;
        }

        const feature = warning.feature;
        return typeof feature === "string" ? feature : undefined;
      })
      .filter((feature): feature is string => {
        return typeof feature === "string";
      });

    expect(features.includes("providerSettings.headers")).toBeTrue();
    expect(features.includes("providerSettings.fetch")).toBeTrue();
  });

  test("forwards system role as query systemPrompt", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const model = createAnthropic({})("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "system",
          content: "Always answer in one short sentence.",
        },
        {
          role: "user",
          content: [{ type: "text", text: "Say hello." }],
        },
      ],
    });

    const systemPrompt = readSystemPromptFromFirstQueryCall(queryCalls);
    expect(systemPrompt).toBe("Always answer in one short sentence.");

    const prompt = readPromptFromFirstQueryCall(queryCalls);
    expect(prompt).toBeDefined();

    if (prompt === undefined) {
      return;
    }

    expect(prompt.includes("[system]")).toBeFalse();
    expect(prompt.includes("[user]")).toBeFalse();
    expect(prompt.includes("Say hello.")).toBeTrue();
  });

  test("joins multiple system messages into systemPrompt", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const model = createAnthropic({})("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "system",
          content: "Rule A",
        },
        {
          role: "system",
          content: "Rule B",
        },
        {
          role: "user",
          content: [{ type: "text", text: "Hello" }],
        },
      ],
    });

    const systemPrompt = readSystemPromptFromFirstQueryCall(queryCalls);
    expect(systemPrompt).toBe("Rule A\n\nRule B");
  });

  test("serializes tool-call and tool-result with toolCallId suffix", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const model = createAnthropic({})("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "Run bash" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool-call",
              toolCallId: "call_1",
              toolName: "bash",
              input: {
                command: "echo hi",
              },
            },
          ],
        },
        {
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: "call_1",
              toolName: "bash",
              output: {
                type: "text",
                value: "hi\n",
              },
            },
          ],
        },
        {
          role: "user",
          content: [{ type: "text", text: "Next" }],
        },
      ],
    });

    const prompt = readPromptFromFirstQueryCall(queryCalls);
    expect(prompt).toBeDefined();

    if (prompt === undefined) {
      return;
    }

    expect(prompt.includes("[tool-call:bash#call_1]")).toBeTrue();
    expect(prompt.includes("[tool-result:bash#call_1]")).toBeTrue();
  });

  test("does not serialize assistant reasoning content into prompt", async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const model = createAnthropic({})("claude-3-5-haiku-latest");
    await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "질문" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "reasoning",
              text: "secret reasoning block",
            },
            {
              type: "text",
              text: "visible assistant text",
            },
          ],
        },
        {
          role: "user",
          content: [{ type: "text", text: "다음 질문" }],
        },
      ],
    });

    const prompt = readPromptFromFirstQueryCall(queryCalls);
    expect(prompt).toBeDefined();

    if (prompt === undefined) {
      return;
    }

    expect(prompt.includes("secret reasoning block")).toBeFalse();
    expect(prompt.includes("[reasoning]")).toBeFalse();
    expect(prompt.includes("visible assistant text")).toBeTrue();
  });
});
