import { afterEach, describe, expect, mock, test } from "bun:test";
import { anthropic as upstreamAnthropic } from "@ai-sdk/anthropic";

afterEach(() => {
  mock.restore();
});

const loadMainModule = async () => {
  const moduleId = `../index.ts?provider-contract-${Date.now()}-${Math.random()}`;
  return await import(moduleId);
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

describe("runtime surface contract", () => {
  test("root exports are available", async () => {
    const { VERSION, anthropic, createAnthropic, forwardAnthropicContainerIdFromLastStep } =
      await loadMainModule();

    expect(typeof VERSION).toBe("string");
    expect(typeof anthropic).toBe("function");
    expect(typeof createAnthropic).toBe("function");
    expect(typeof forwardAnthropicContainerIdFromLastStep).toBe("function");
  });

  test("VERSION matches package version", async () => {
    const { VERSION } = await loadMainModule();
    const packageJson = JSON.parse(
      await Bun.file(new URL("../package.json", import.meta.url)).text(),
    );

    expect(VERSION).toBe(packageJson.version);
  });

  test("provider exposes same tool keys as upstream anthropic", async () => {
    const { anthropic } = await loadMainModule();

    const localToolKeys = Object.keys(anthropic.tools).sort();
    const upstreamToolKeys = Object.keys(upstreamAnthropic.tools).sort();

    expect(localToolKeys).toEqual(upstreamToolKeys);
  });

  test("createAnthropic rejects apiKey and authToken together", async () => {
    const { createAnthropic } = await loadMainModule();

    expect(() => {
      createAnthropic({ apiKey: "api-key", authToken: "auth-token" });
    }).toThrow();
  });

  test("provider specification version is v3", async () => {
    const { anthropic } = await loadMainModule();

    expect(anthropic.specificationVersion).toBe("v3");

    const model = anthropic("claude-3-5-haiku-latest");

    expect(model.specificationVersion).toBe("v3");
    expect(model.provider).toBe("anthropic.messages");
    expect(model.modelId).toBe("claude-3-5-haiku-latest");
  });

  test("root entry exposes explicit compat helper only", async () => {
    const mainModule = await loadMainModule();

    expect("withOpenCodeCompatibility" in mainModule).toBeTrue();
    expect("isOpenCode" in mainModule).toBeFalse();
  });

  test("root entry adds legacy finish overlay for compat consumers", async () => {
    mock.module("@anthropic-ai/claude-agent-sdk", () => {
      return {
        query: async function* () {
          yield {
            type: "result",
            subtype: "success",
            stop_reason: "end_turn",
            result: "ok",
            usage: {
              input_tokens: 10,
              output_tokens: 5,
              cache_read_input_tokens: 0,
              cache_creation_input_tokens: 0,
            },
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: false,
            num_turns: 1,
            total_cost_usd: 0,
            modelUsage: {},
            permission_denials: [],
            uuid: "uuid-root-compat",
            session_id: "session-root-compat",
          };
        },
      };
    });

    const { anthropic } = await loadMainModule();
    const result = await anthropic("claude-3-5-haiku-latest").doGenerate({
      prompt: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
    });

    expect(result.finishReason.unified).toBe("stop");
    expect("finish" in result).toBeTrue();

    if (!("finish" in result) || !("reason" in result)) {
      return;
    }

    expect(result.finish).toBe("stop");
    expect(result.reason).toBe("end_turn");
  });

  test("forward helper returns undefined with no container metadata", async () => {
    const { forwardAnthropicContainerIdFromLastStep } = await loadMainModule();

    const output = forwardAnthropicContainerIdFromLastStep({
      steps: [{}, { providerMetadata: {} }],
    });

    expect(output).toBeUndefined();
  });

  test("forward helper picks latest container id", async () => {
    const { forwardAnthropicContainerIdFromLastStep } = await loadMainModule();

    const output = forwardAnthropicContainerIdFromLastStep({
      steps: [
        {
          providerMetadata: {
            anthropic: {
              container: { id: "container-old" },
            },
          },
        },
        {
          providerMetadata: {
            anthropic: {
              container: { id: "container-new" },
            },
          },
        },
      ],
    });

    expect(output).toBeDefined();
    expect(isRecord(output)).toBeTrue();

    if (!isRecord(output)) {
      return;
    }

    const providerOptions = output.providerOptions;
    expect(isRecord(providerOptions)).toBeTrue();

    if (!isRecord(providerOptions)) {
      return;
    }

    const anthropicOptions = providerOptions.anthropic;
    expect(isRecord(anthropicOptions)).toBeTrue();

    if (!isRecord(anthropicOptions)) {
      return;
    }

    const container = anthropicOptions.container;
    expect(isRecord(container)).toBeTrue();

    if (!isRecord(container)) {
      return;
    }

    expect(container.id).toBe("container-new");
  });
});
