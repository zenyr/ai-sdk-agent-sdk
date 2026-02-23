import { afterEach, describe, expect, mock, test } from "bun:test";

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

describe("v2 entry compatibility contract", () => {
  test("doGenerate adds legacy finish/reason fields", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-v2-generate",
            session_id: "session-v2-generate",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-generate-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const model = anthropic("claude-3-5-haiku-latest");
    expect(model.specificationVersion).toBe("v2");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    expect(result.finishReason).toBe("stop");
    expect("rawFinishReason" in result).toBeTrue();

    expect("finish" in result).toBeTrue();
    expect("reason" in result).toBeTrue();

    if (!("finish" in result) || !("reason" in result)) {
      return;
    }

    expect(result.finish).toBe("stop");
    expect(result.reason).toBe("end_turn");
  });

  test("doStream finish part adds legacy finish/reason fields", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-v2",
                model: "mock-model",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "text",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_delta",
              index: 0,
              delta: {
                type: "text_delta",
                text: "hello",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_stop",
              index: 0,
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "message_delta",
              delta: {
                stop_reason: "end_turn",
              },
              usage: {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          };

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "done",
            usage: buildMockResultUsage(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-v2-stream",
            session_id: "session-v2-stream",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-stream-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const model = anthropic("claude-3-5-haiku-latest");
    expect(model.specificationVersion).toBe("v2");

    const streamResult = await model.doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (
      typeof finishPart !== "object" ||
      finishPart === null ||
      !("finishReason" in finishPart) ||
      !("finish" in finishPart) ||
      !("reason" in finishPart)
    ) {
      return;
    }

    expect(finishPart.finishReason).toBe("stop");
    expect(finishPart.finish).toBe("stop");
    expect(finishPart.reason).toBe("end_turn");
  });

  test("doGenerate maps empty tool routing output to legacy error fields", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "",
            usage: buildMockResultUsage(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-v2-empty-generate",
            session_id: "session-v2-empty-generate",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-empty-generate-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const result = await anthropic("claude-3-5-haiku-latest").doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "call tool" }],
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

    expect(result.finishReason).toBe("error");
    expect(result.finish).toBe("error");
    expect(result.reason).toBe("empty-tool-routing-output");
    expect(result.rawFinishReason).toBe("empty-tool-routing-output");
  });

  test("doGenerate maps legacy single tool-call object to tool-calls", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "",
            structured_output: {
              tool: "bash",
              parameters: {
                command: 'bun -e "console.log(Math.random())"',
                description: "Run Math.random once",
              },
            },
            usage: buildMockResultUsage(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-v2-legacy-tool-generate",
            session_id: "session-v2-legacy-tool-generate",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-legacy-tool-generate-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const result = await anthropic("claude-3-5-haiku-latest").doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "call bash" }],
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

    expect(result.finishReason).toBe("tool-calls");
    expect(result.finish).toBe("tool-calls");
    expect(result.reason).toBe("tool_use");

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolName).toBe("bash");
  });

  test("doGenerate maps native MCP tool-use error_max_turns to tool-calls", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "assistant",
            message: {
              content: [
                {
                  type: "tool_use",
                  id: "toolu_v2_native_1",
                  name: "mcp__ai_sdk_tool_bridge__bash",
                  input: {
                    command: 'bun -e "console.log(Math.random())"',
                    description: "Run Math.random once",
                  },
                },
              ],
            },
          };

          yield {
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
            uuid: "uuid-v2-native-tool-generate",
            session_id: "session-v2-native-tool-generate",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-native-tool-generate-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const result = await anthropic("claude-3-5-haiku-latest").doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "call bash" }],
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

    expect(result.finishReason).toBe("tool-calls");
    expect(result.finish).toBe("tool-calls");
    expect(result.reason).toBe("tool_use");

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolCallId).toBe("toolu_v2_native_1");
    expect(firstContentPart.toolName).toBe("bash");
  });

  test("doGenerate maps stream-only tool_use (no result message) to tool-calls", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-v2-no-result",
                model: "mock-model",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_start",
              index: 0,
              content_block: {
                type: "tool_use",
                id: "toolu_v2_no_result_1",
                name: "mcp__ai_sdk_tool_bridge__bash",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_delta",
              index: 0,
              delta: {
                type: "input_json_delta",
                partial_json: '{"command":"whoami","description":"현재 사용자 확인"}',
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "content_block_stop",
              index: 0,
            },
          };

          yield {
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
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-no-result-generate-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const result = await anthropic("claude-3-5-haiku-latest").doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "call bash" }],
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

    expect(result.finishReason).toBe("tool-calls");
    expect(result.finish).toBe("tool-calls");
    expect(result.reason).toBe("tool_use");

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe("tool-call");

    if (firstContentPart === undefined || firstContentPart.type !== "tool-call") {
      return;
    }

    expect(firstContentPart.toolCallId).toBe("toolu_v2_no_result_1");
    expect(firstContentPart.toolName).toBe("bash");
    expect(firstContentPart.input).toContain("whoami");
  });

  test("doStream maps empty tool routing output to legacy error fields", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-v2-empty",
                model: "mock-model",
              },
            },
          };

          yield {
            type: "stream_event",
            event: {
              type: "message_delta",
              delta: {
                stop_reason: "end_turn",
              },
              usage: {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          };

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "",
            usage: buildMockResultUsage(),
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-v2-empty-stream",
            session_id: "session-v2-empty-stream",
          };
        },
      };
    });

    const moduleId = `../v2.ts?v2-empty-stream-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "call tool" }],
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (
      typeof finishPart !== "object" ||
      finishPart === null ||
      !("finishReason" in finishPart) ||
      !("finish" in finishPart) ||
      !("reason" in finishPart)
    ) {
      return;
    }

    expect(finishPart.finishReason).toBe("error");
    expect(finishPart.finish).toBe("error");
    expect(finishPart.reason).toBe("empty-tool-routing-output");
  });
});
