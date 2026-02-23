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

describe("stream bridge contract", () => {
  test("doStream emits metadata from stream events", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-1",
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-${Date.now()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "say hello" }],
        },
      ],
    });

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const metadataPart = parts.find((part) => {
      return (
        typeof part === "object" &&
        part !== null &&
        "type" in part &&
        part.type === "response-metadata"
      );
    });

    expect(metadataPart).toBeDefined();

    if (
      typeof metadataPart !== "object" ||
      metadataPart === null ||
      !("id" in metadataPart) ||
      !("modelId" in metadataPart)
    ) {
      return;
    }

    expect(metadataPart.id).toBe("msg-1");
    expect(metadataPart.modelId).toBe("mock-model");

    const hasTextDelta = parts.some((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "text-delta"
      );
    });

    expect(hasTextDelta).toBeTrue();

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    if (
      typeof finishPart.finishReason !== "object" ||
      finishPart.finishReason === null ||
      !("unified" in finishPart.finishReason)
    ) {
      return;
    }

    expect(typeof finishPart.finishReason.unified).toBe("string");
  });

  test("doStream forwards system role as query systemPrompt", async () => {
    const queryCalls: unknown[] = [];

    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* (request: unknown) {
          queryCalls.push(request);

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-system-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
      prompt: [
        {
          role: "system",
          content: "Follow system rules.",
        },
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    for await (const _part of streamResult.stream) {
      // consume stream
    }

    const firstCall = queryCalls[0];
    expect(typeof firstCall).toBe("object");
    expect(firstCall).not.toBeNull();

    if (typeof firstCall !== "object" || firstCall === null) {
      return;
    }

    const options =
      typeof firstCall.options === "object" && firstCall.options !== null
        ? firstCall.options
        : undefined;
    const prompt = typeof firstCall.prompt === "string" ? firstCall.prompt : undefined;

    expect(options).toBeDefined();
    expect(prompt).toBeDefined();

    if (options === undefined || prompt === undefined) {
      return;
    }

    const systemPrompt =
      typeof options.systemPrompt === "string" ? options.systemPrompt : undefined;
    expect(systemPrompt).toBe("Follow system rules.");
    expect(prompt.includes("[system]")).toBeFalse();
    expect(prompt.includes("[user]")).toBeFalse();
    expect(prompt.includes("hello")).toBeTrue();
  });

  test("tool mode unwraps text envelope from buffered stream text", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-text",
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
                text: '{"type":"text","text":"안녕하세요! 무엇을 도와드릴까요?"}',
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-tool-text-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const textDeltas = parts.filter((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "text-delta"
      );
    });

    expect(textDeltas.length).toBe(1);

    const firstTextDelta = textDeltas[0];
    if (
      typeof firstTextDelta !== "object" ||
      firstTextDelta === null ||
      !("delta" in firstTextDelta)
    ) {
      return;
    }

    expect(firstTextDelta.delta).toBe("안녕하세요! 무엇을 도와드릴까요?");
  });

  test("doStream reuses claude session with appended prompt messages", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* (request: unknown) {
          queryCalls.push(request);
          callCount += 1;

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: `stream-session-${callCount}`,
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-resume-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);
    const model = anthropic("claude-3-5-haiku-latest");

    const firstStreamResult = await model.doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
    });

    for await (const _part of firstStreamResult.stream) {
      // consume to completion
    }

    const secondStreamResult = await model.doStream({
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

    for await (const _part of secondStreamResult.stream) {
      // consume to completion
    }

    expect(queryCalls.length).toBe(2);

    const secondCall = queryCalls[1];
    expect(typeof secondCall).toBe("object");
    expect(secondCall).not.toBeNull();

    if (typeof secondCall !== "object" || secondCall === null) {
      return;
    }

    const secondPrompt = typeof secondCall.prompt === "string" ? secondCall.prompt : undefined;
    expect(secondPrompt).toBeDefined();

    if (secondPrompt === undefined) {
      return;
    }

    expect(secondPrompt.includes("첫 질문")).toBeFalse();
    expect(secondPrompt.includes("두 번째 질문")).toBeTrue();

    const options =
      typeof secondCall.options === "object" && secondCall.options !== null
        ? secondCall.options
        : undefined;
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    const resume = typeof options.resume === "string" ? options.resume : undefined;
    expect(resume).toBe("stream-session-1");
  });

  test("doStream reuses session from conversationId header with single user turns", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* (request: unknown) {
          queryCalls.push(request);
          callCount += 1;

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: `stream-header-session-${callCount}`,
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-header-resume-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);
    const model = anthropic("claude-3-5-haiku-latest");

    const firstTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      headers: {
        "x-conversation-id": "conversation-stream-header-1",
      },
    };

    const firstStreamResult = await model.doStream(firstTurnOptions);
    for await (const _part of firstStreamResult.stream) {
      // consume to completion
    }

    const secondTurnOptions = {
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      headers: {
        "x-conversation-id": "conversation-stream-header-1",
      },
    };

    const secondStreamResult = await model.doStream(secondTurnOptions);
    for await (const _part of secondStreamResult.stream) {
      // consume to completion
    }

    expect(queryCalls.length).toBe(2);

    const secondCall = queryCalls[1];
    expect(typeof secondCall).toBe("object");
    expect(secondCall).not.toBeNull();

    if (typeof secondCall !== "object" || secondCall === null) {
      return;
    }

    const options =
      typeof secondCall.options === "object" && secondCall.options !== null
        ? secondCall.options
        : undefined;

    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    const resume = typeof options.resume === "string" ? options.resume : undefined;
    expect(resume).toBe("stream-header-session-1");

    const secondPrompt = typeof secondCall.prompt === "string" ? secondCall.prompt : undefined;
    expect(secondPrompt).toBeDefined();

    if (secondPrompt === undefined) {
      return;
    }

    expect(secondPrompt.includes("첫 질문")).toBeFalse();
    expect(secondPrompt.includes("두 번째 질문")).toBeTrue();
  });

  test("legacy compatibility: doStream reuses session from x-opencode-session header", async () => {
    const queryCalls: unknown[] = [];
    let callCount = 0;

    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* (request: unknown) {
          queryCalls.push(request);
          callCount += 1;

          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: buildMockResultUsage(),
            session_id: `stream-legacy-header-session-${callCount}`,
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-legacy-header-resume-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);
    const model = anthropic("claude-3-5-haiku-latest");

    const firstStreamResult = await model.doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "첫 질문" }],
        },
      ],
      headers: {
        "x-opencode-session": "legacy-stream-header-1",
      },
    });

    for await (const _part of firstStreamResult.stream) {
      // consume to completion
    }

    const secondStreamResult = await model.doStream({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "두 번째 질문" }],
        },
      ],
      headers: {
        "x-opencode-session": "legacy-stream-header-1",
      },
    });

    for await (const _part of secondStreamResult.stream) {
      // consume to completion
    }

    expect(queryCalls.length).toBe(2);

    const secondCall = queryCalls[1];
    expect(typeof secondCall).toBe("object");
    expect(secondCall).not.toBeNull();

    if (typeof secondCall !== "object" || secondCall === null) {
      return;
    }

    const options =
      typeof secondCall.options === "object" && secondCall.options !== null
        ? secondCall.options
        : undefined;

    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    const resume = typeof options.resume === "string" ? options.resume : undefined;
    expect(resume).toBe("stream-legacy-header-session-1");
  });

  test("tool mode emits tool-call from buffered stream text envelope", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-calls",
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
                text: '{"type":"tool-calls","calls":[{"toolName":"lookup_weather","input":{"city":"seoul"}}]}',
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-tool-calls-${Date.now()}-${Math.random()}`;
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

    const toolCallPart = parts.find((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "tool-call"
      );
    });

    expect(toolCallPart).toBeDefined();

    if (
      typeof toolCallPart !== "object" ||
      toolCallPart === null ||
      !("toolName" in toolCallPart) ||
      !("input" in toolCallPart)
    ) {
      return;
    }

    expect(toolCallPart.toolName).toBe("lookup_weather");
    expect(toolCallPart.input).toBe('{"city":"seoul"}');

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    const finishReason = finishPart.finishReason;
    if (typeof finishReason !== "object" || finishReason === null || !("unified" in finishReason)) {
      return;
    }

    expect(finishReason.unified).toBe("tool-calls");
  });

  test("tool mode emits tool-call from native MCP tool_use stream block", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-native-mcp",
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
                id: "toolu_stream_1",
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
                partial_json:
                  '{"command":"bun -e \\"console.log(Math.random())\\"","description":"Run Math.random once"}',
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
            uuid: "uuid-stream-native-tool",
            session_id: "session-stream-native-tool",
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-tool-native-mcp-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const toolCallPart = parts.find((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "tool-call"
      );
    });

    expect(toolCallPart).toBeDefined();

    if (
      typeof toolCallPart !== "object" ||
      toolCallPart === null ||
      !("toolCallId" in toolCallPart) ||
      !("toolName" in toolCallPart) ||
      !("input" in toolCallPart)
    ) {
      return;
    }

    expect(toolCallPart.toolCallId).toBe("toolu_stream_1");
    expect(toolCallPart.toolName).toBe("bash");
    expect(String(toolCallPart.input)).toContain("Math.random");

    const toolInputEndPart = parts.find((part) => {
      return (
        typeof part === "object" &&
        part !== null &&
        "type" in part &&
        part.type === "tool-input-end"
      );
    });

    expect(toolInputEndPart).toBeDefined();

    const errorPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "error";
    });

    expect(errorPart).toBeUndefined();

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    const finishReason = finishPart.finishReason;
    if (
      typeof finishReason !== "object" ||
      finishReason === null ||
      !("unified" in finishReason) ||
      !("raw" in finishReason)
    ) {
      return;
    }

    expect(finishReason.unified).toBe("tool-calls");
    expect(finishReason.raw).toBe("tool_use");
  });

  test("tool mode emits tool-call when stream ends at tool_use without result message", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-no-result",
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
                id: "toolu_no_result_stream_1",
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
                partial_json:
                  '{"command":"bun -e \\"console.log(Math.random())\\"","description":"Run Math.random once"}',
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

    const moduleId = `../index.ts?stream-contract-tool-no-result-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const toolCallPart = parts.find((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "tool-call"
      );
    });

    expect(toolCallPart).toBeDefined();

    if (
      typeof toolCallPart !== "object" ||
      toolCallPart === null ||
      !("toolCallId" in toolCallPart) ||
      !("toolName" in toolCallPart) ||
      !("input" in toolCallPart)
    ) {
      return;
    }

    expect(toolCallPart.toolCallId).toBe("toolu_no_result_stream_1");
    expect(toolCallPart.toolName).toBe("bash");
    expect(String(toolCallPart.input)).toContain("Math.random");

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    const finishReason = finishPart.finishReason;
    if (
      typeof finishReason !== "object" ||
      finishReason === null ||
      !("unified" in finishReason) ||
      !("raw" in finishReason)
    ) {
      return;
    }

    expect(finishReason.unified).toBe("tool-calls");
    expect(finishReason.raw).toBe("tool_use");
  });

  test("tool mode emits tool-call from legacy single-call JSON text", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-legacy-call",
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
                text: '{"tool":"bash","parameters":{"command":"bun -e \\"console.log(Math.random())\\"","description":"Run Math.random once"}}',
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-tool-legacy-call-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const toolCallPart = parts.find((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "tool-call"
      );
    });

    expect(toolCallPart).toBeDefined();

    if (
      typeof toolCallPart !== "object" ||
      toolCallPart === null ||
      !("toolName" in toolCallPart) ||
      !("input" in toolCallPart)
    ) {
      return;
    }

    expect(toolCallPart.toolName).toBe("bash");
    expect(String(toolCallPart.input)).toContain("Math.random");
  });

  test("tool mode emits explicit error when model returns empty successful output", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-tool-empty",
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-tool-empty-${Date.now()}-${Math.random()}`;
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

    const errorPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "error";
    });

    expect(errorPart).toBeDefined();

    if (typeof errorPart !== "object" || errorPart === null || !("error" in errorPart)) {
      return;
    }

    expect(String(errorPart.error)).toContain("Tool routing produced no tool call");

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    const finishReason = finishPart.finishReason;
    if (
      typeof finishReason !== "object" ||
      finishReason === null ||
      !("unified" in finishReason) ||
      !("raw" in finishReason)
    ) {
      return;
    }

    expect(finishReason.unified).toBe("error");
    expect(finishReason.raw).toBe("empty-tool-routing-output");
  });

  test("tool mode recovers from structured output retry exhaustion when text is recoverable", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "stream_event",
            event: {
              type: "message_start",
              message: {
                id: "msg-retry-recovered",
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
                text: '{"type":"text","text":"안녕하세요"}',
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
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-retry-recovered-${Date.now()}-${Math.random()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic("claude-3-5-haiku-latest").doStream({
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const errorPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "error";
    });

    expect(errorPart).toBeUndefined();

    const textDelta = parts.find((part) => {
      return (
        typeof part === "object" && part !== null && "type" in part && part.type === "text-delta"
      );
    });

    expect(textDelta).toBeDefined();

    if (typeof textDelta !== "object" || textDelta === null || !("delta" in textDelta)) {
      return;
    }

    expect(textDelta.delta).toBe("안녕하세요");

    const finishPart = parts.find((part) => {
      return typeof part === "object" && part !== null && "type" in part && part.type === "finish";
    });

    expect(finishPart).toBeDefined();

    if (typeof finishPart !== "object" || finishPart === null || !("finishReason" in finishPart)) {
      return;
    }

    const finishReason = finishPart.finishReason;
    if (typeof finishReason !== "object" || finishReason === null || !("unified" in finishReason)) {
      return;
    }

    expect(finishReason.unified).toBe("stop");
  });
});
