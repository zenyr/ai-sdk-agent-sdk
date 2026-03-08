import { afterEach, describe, expect, mock, test } from "bun:test";

afterEach(() => {
  mock.restore();
});

const loadCoreModule = async () => {
  const moduleId = `../core.ts?core-contract-${Date.now()}-${Math.random()}`;
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

describe("core entry contract", () => {
  test("core exports pure provider entrypoints", async () => {
    const { VERSION, anthropic, createAnthropic, forwardAnthropicContainerIdFromLastStep } =
      await loadCoreModule();

    expect(typeof VERSION).toBe("string");
    expect(typeof anthropic).toBe("function");
    expect(typeof createAnthropic).toBe("function");
    expect(typeof forwardAnthropicContainerIdFromLastStep).toBe("function");
  });

  test("core doGenerate keeps canonical v3 result shape", async () => {
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
            uuid: "uuid-core-generate",
            session_id: "session-core-generate",
          };
        },
      };
    });

    const { anthropic } = await loadCoreModule();
    const model = anthropic("claude-3-5-haiku-latest");

    expect(model.specificationVersion).toBe("v3");

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    expect(result.finishReason.unified).toBe("stop");
    expect("finish" in result).toBeFalse();
    expect("reason" in result).toBeFalse();
    expect("rawFinishReason" in result).toBeFalse();
  });
});
