import { describe, expect, test } from "bun:test";
import type { LanguageModelV3Content, LanguageModelV3StreamPart } from "@ai-sdk/provider";

import { createAnthropic } from "../../index";

const E2E_FLAG = "AI_SDK_AGENT_E2E";
const DEFAULT_MODEL_ID = "claude-3-5-haiku-latest";

const readNonEmptyEnv = (name: string): string | undefined => {
  const value = Bun.env[name];
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const isTextContent = (
  part: LanguageModelV3Content,
): part is Extract<LanguageModelV3Content, { type: "text" }> => {
  return part.type === "text";
};

const isFinishPart = (
  part: LanguageModelV3StreamPart,
): part is Extract<LanguageModelV3StreamPart, { type: "finish" }> => {
  return part.type === "finish";
};

const isErrorPart = (
  part: LanguageModelV3StreamPart,
): part is Extract<LanguageModelV3StreamPart, { type: "error" }> => {
  return part.type === "error";
};

const resolveModelId = (): string => {
  return readNonEmptyEnv("AI_SDK_AGENT_E2E_MODEL") ?? DEFAULT_MODEL_ID;
};

const resolveProvider = () => {
  const apiKey = readNonEmptyEnv("ANTHROPIC_API_KEY");
  if (apiKey !== undefined) {
    return createAnthropic({ apiKey });
  }

  const authToken = readNonEmptyEnv("ANTHROPIC_AUTH_TOKEN");
  if (authToken !== undefined) {
    return createAnthropic({ authToken });
  }

  throw new Error(
    "Missing ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN. Set one before running bun run test:e2e.",
  );
};

const isE2eEnabled = readNonEmptyEnv(E2E_FLAG) === "1";
const e2eTest = isE2eEnabled ? test : test.skip;

describe("real api e2e", () => {
  e2eTest("doGenerate completes against Anthropic API", async () => {
    const provider = resolveProvider();
    const model = provider(resolveModelId());

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: 'Reply with exactly "OK".',
            },
          ],
        },
      ],
    });

    expect(result.finishReason.unified).not.toBe("error");

    const text = result.content
      .filter(isTextContent)
      .map((contentPart) => contentPart.text)
      .join(" ")
      .trim();

    expect(text.length).toBeGreaterThan(0);

    const outputTokens = result.usage.outputTokens.total;
    expect(typeof outputTokens).toBe("number");

    if (typeof outputTokens !== "number") {
      return;
    }

    expect(outputTokens).toBeGreaterThan(0);
  });

  e2eTest("doStream completes against Anthropic API", async () => {
    const provider = resolveProvider();
    const model = provider(resolveModelId());

    const streamResult = await model.doStream({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Say hello in one short sentence.",
            },
          ],
        },
      ],
    });

    const parts: LanguageModelV3StreamPart[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    expect(parts.length).toBeGreaterThan(0);

    const errorParts = parts.filter(isErrorPart);
    expect(errorParts.length).toBe(0);

    const finishPart = parts.find(isFinishPart);
    expect(finishPart).toBeDefined();

    if (finishPart === undefined) {
      return;
    }

    expect(finishPart.finishReason.unified).not.toBe("error");

    const outputTokens = finishPart.usage.outputTokens.total;
    expect(typeof outputTokens).toBe("number");

    if (typeof outputTokens !== "number") {
      return;
    }

    expect(outputTokens).toBeGreaterThan(0);
  });
});
