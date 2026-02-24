import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { anthropic as upstreamAnthropic } from "@ai-sdk/anthropic";

const originalOpenCode = process.env.OPENCODE;

beforeEach(() => {
  delete process.env.OPENCODE;
});

afterEach(() => {
  if (originalOpenCode === undefined) {
    delete process.env.OPENCODE;
    return;
  }

  process.env.OPENCODE = originalOpenCode;
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
