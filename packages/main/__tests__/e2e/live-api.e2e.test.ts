import { describe, expect, test } from "bun:test";
import type { LanguageModelV3Content } from "@ai-sdk/provider";

import { createAnthropic } from "../../index";

const E2E_FLAG = "AI_SDK_AGENT_E2E";
const API_KEY_FLAG = "ANTHROPIC_API_KEY";
const DEFAULT_MODEL_ID = "claude-haiku-4-5";

const readNonEmptyEnv = (name: string): string | undefined => {
  const value = Bun.env[name];
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
};

const isTextContent = (part: LanguageModelV3Content): part is Extract<LanguageModelV3Content, { type: "text" }> => {
  return part.type === "text";
};

const isToolCallContent = (
  part: LanguageModelV3Content
): part is Extract<LanguageModelV3Content, { type: "tool-call" }> => {
  return part.type === "tool-call";
};

const resolveModelId = (): string => {
  return readNonEmptyEnv("AI_SDK_AGENT_E2E_MODEL") ?? DEFAULT_MODEL_ID;
};

const createConversationToken = (): string => {
  return `token-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const resolveProvider = () => {
  return createAnthropic({});
};

const resolveApiKeyProvider = () => {
  const apiKey = readNonEmptyEnv(API_KEY_FLAG);
  if (apiKey === undefined) {
    throw new Error(`Missing ${API_KEY_FLAG}. Set it before running the apiKey override e2e.`);
  }

  return createAnthropic({ apiKey });
};

const isE2eEnabled = readNonEmptyEnv(E2E_FLAG) === "1";
const e2eTest = isE2eEnabled ? test : test.skip;
const apiKeyE2eTest = isE2eEnabled && readNonEmptyEnv(API_KEY_FLAG) !== undefined ? test : test.skip;

describe("real api e2e", () => {
  e2eTest("ambient auth remains the default path", async () => {
    const provider = resolveProvider();
    const model = provider(resolveModelId());

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: 'Reply with exactly "ambient".',
            },
          ],
        },
      ],
    });

    expect(result.finishReason).toBe("stop");

    const text = result.content
      .filter(isTextContent)
      .map(contentPart => contentPart.text)
      .join(" ")
      .trim()
      .toLowerCase();

    expect(text).toContain("ambient");
  });

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

    expect(result.finishReason).not.toBe("error");

    const text = result.content
      .filter(isTextContent)
      .map(contentPart => contentPart.text)
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

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    expect(parts.length).toBeGreaterThan(0);

    const errorParts = parts.filter(part => {
      return isRecord(part) && part.type === "error";
    });
    expect(errorParts.length).toBe(0);

    const finishPart = parts.find(part => {
      return isRecord(part) && part.type === "finish";
    });
    expect(finishPart).toBeDefined();

    if (!isRecord(finishPart)) {
      return;
    }

    expect(finishPart.finishReason).not.toBe("error");

    if (!isRecord(finishPart.usage) || !isRecord(finishPart.usage.outputTokens)) {
      return;
    }

    const outputTokens = finishPart.usage.outputTokens.total;
    expect(typeof outputTokens).toBe("number");

    if (typeof outputTokens !== "number") {
      return;
    }

    expect(outputTokens).toBeGreaterThan(0);
  });

  e2eTest("doGenerate returns tool-call content with required tool choice", async () => {
    const provider = resolveProvider();
    const model = provider(resolveModelId());
    const expectedCity = "Seoul";

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Call the tool exactly once with city set to Seoul. Do not output plain text.",
            },
          ],
        },
      ],
      tools: [
        {
          type: "function",
          name: "lookup_weather",
          description: "Lookup weather by city",
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

    expect(result.finishReason).toBe("tool-calls");

    const firstToolCall = result.content.find(isToolCallContent);
    expect(firstToolCall).toBeDefined();

    if (firstToolCall === undefined) {
      return;
    }

    expect(firstToolCall.toolName).toBe("lookup_weather");
    expect(firstToolCall.input).toContain(expectedCity);
  });

  e2eTest("doGenerate resumes conversation using x-conversation-id", async () => {
    const provider = resolveProvider();
    const model = provider(resolveModelId());
    const rememberedToken = createConversationToken();
    const conversationId = `conv-${createConversationToken()}`;

    const firstTurn = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: `Remember this token exactly for this conversation: ${rememberedToken}. Reply with only OK.`,
            },
          ],
        },
      ],
      headers: {
        "x-conversation-id": conversationId,
      },
    });

    expect(firstTurn.finishReason).not.toBe("error");

    const secondTurn = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Reply with only the remembered token from earlier in this same conversation.",
            },
          ],
        },
      ],
      headers: {
        "x-conversation-id": conversationId,
      },
    });

    expect(secondTurn.finishReason).not.toBe("error");

    const responseText = secondTurn.content
      .filter(isTextContent)
      .map(contentPart => contentPart.text)
      .join(" ")
      .trim();

    expect(responseText).toContain(rememberedToken);
  });

  apiKeyE2eTest("explicit apiKey overrides ambient auth when provided", async () => {
    const provider = resolveApiKeyProvider();
    const model = provider(resolveModelId());

    const result = await model.doGenerate({
      prompt: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: 'Reply with exactly "api-key".',
            },
          ],
        },
      ],
    });

    expect(result.finishReason).toBe("stop");

    const text = result.content
      .filter(isTextContent)
      .map(contentPart => contentPart.text)
      .join(" ")
      .trim()
      .toLowerCase();

    expect(text).toContain("api-key");
  });
});
