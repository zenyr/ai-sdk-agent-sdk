import { afterEach, describe, expect, mock, test } from "bun:test";

afterEach(() => {
  mock.restore();
});

const loadAdapterModule = async () => {
  const moduleId = `../index.ts?adapter-v3-contract-${Date.now()}-${Math.random()}`;
  return await import(moduleId);
};

const buildMockResultUsage = () => {
  return {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
};

describe("adapter-v3 exports contract", () => {
  test("exports provider factory and helper", async () => {
    const { anthropic, createAnthropic, forwardAnthropicContainerIdFromLastStep, VERSION } = await loadAdapterModule();

    expect(typeof VERSION).toBe("string");
    expect(typeof anthropic).toBe("function");
    expect(typeof createAnthropic).toBe("function");
    expect(typeof forwardAnthropicContainerIdFromLastStep).toBe("function");
  });

  test("createAnthropic returns v3 provider surface at runtime", async () => {
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
            uuid: "uuid-adapter-v3-generate",
            session_id: "session-adapter-v3-generate",
          };
        },
      };
    });

    const { createAnthropic } = await loadAdapterModule();
    const provider = createAnthropic({});

    expect(typeof provider).toBe("function");

    const model = provider("claude-3-5-haiku-latest");
    expect(model.specificationVersion).toBe("v3");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    expect(typeof result.finishReason.unified).toBe("string");
    expect(typeof result.finishReason.raw).toBe("string");
    expect("finish" in result).toBeFalse();
    expect("reason" in result).toBeFalse();
    expect("rawFinishReason" in result).toBeFalse();
  });
});
