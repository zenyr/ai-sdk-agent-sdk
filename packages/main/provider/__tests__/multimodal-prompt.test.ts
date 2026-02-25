import { describe, expect, test } from "bun:test";
import type { SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";

import { buildMultimodalQueryPrompt } from "../domain/multimodal-prompt";

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const readFirstPromptMessage = async (
  prompt: AsyncIterable<SDKUserMessage> | undefined,
): Promise<SDKUserMessage | undefined> => {
  if (prompt === undefined) {
    return undefined;
  }

  for await (const message of prompt) {
    return message;
  }

  return undefined;
};

const readTextBlocks = (contentBlocks: unknown[]): string[] => {
  return contentBlocks
    .filter((contentBlock) => {
      return isRecord(contentBlock) && contentBlock.type === "text";
    })
    .map((contentBlock) => {
      if (!isRecord(contentBlock) || typeof contentBlock.text !== "string") {
        return "";
      }

      return contentBlock.text;
    })
    .filter((text) => {
      return text.length > 0;
    });
};

describe("multimodal-prompt", () => {
  test("returns undefined when prompt has no image content", async () => {
    const queryPrompt = await buildMultimodalQueryPrompt({
      promptMessages: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
      ],
      resumeSessionId: undefined,
      preambleText: undefined,
    });

    expect(queryPrompt).toBeUndefined();
  });

  test("builds multimodal prompt with preamble and previous context", async () => {
    const queryPrompt = await buildMultimodalQueryPrompt({
      promptMessages: [
        {
          role: "user",
          content: [{ type: "text", text: "first question" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "first answer" }],
        },
        {
          role: "user",
          content: [
            { type: "text", text: "describe this image" },
            {
              type: "file",
              mediaType: "image/png",
              data: "data:image/png;base64,Zm9v",
            },
          ],
        },
      ],
      resumeSessionId: undefined,
      preambleText: "Return JSON only",
    });

    const firstMessage = await readFirstPromptMessage(queryPrompt);
    expect(firstMessage).toBeDefined();

    if (firstMessage === undefined) {
      return;
    }

    expect(firstMessage.session_id).toBe("default");

    const contentBlocks = firstMessage.message.content;
    expect(Array.isArray(contentBlocks)).toBeTrue();

    if (!Array.isArray(contentBlocks)) {
      return;
    }

    const textBlocks = readTextBlocks(contentBlocks);
    expect(textBlocks.length).toBeGreaterThan(1);
    expect(textBlocks[0]).toBe("Return JSON only");
    expect(
      textBlocks.some((textBlock) => {
        return (
          textBlock.includes("Previous conversation context") &&
          textBlock.includes("first question") &&
          textBlock.includes("first answer")
        );
      }),
    ).toBeTrue();

    const imageBlock = contentBlocks.find((contentBlock) => {
      return isRecord(contentBlock) && contentBlock.type === "image";
    });

    expect(imageBlock).toBeDefined();

    if (!isRecord(imageBlock) || !isRecord(imageBlock.source)) {
      return;
    }

    expect(imageBlock.source.type).toBe("base64");
    expect(imageBlock.source.media_type).toBe("image/png");
    expect(imageBlock.source.data).toBe("Zm9v");
  });

  test("omits previous context text when resuming existing session", async () => {
    const queryPrompt = await buildMultimodalQueryPrompt({
      promptMessages: [
        {
          role: "user",
          content: [{ type: "text", text: "first question" }],
        },
        {
          role: "assistant",
          content: [{ type: "text", text: "first answer" }],
        },
        {
          role: "user",
          content: [
            { type: "text", text: "describe this image" },
            {
              type: "file",
              mediaType: "image/png",
              data: "data:image/png;base64,Zm9v",
            },
          ],
        },
      ],
      resumeSessionId: "resume-session-1",
      preambleText: undefined,
    });

    const firstMessage = await readFirstPromptMessage(queryPrompt);
    expect(firstMessage).toBeDefined();

    if (firstMessage === undefined) {
      return;
    }

    expect(firstMessage.session_id).toBe("resume-session-1");

    const textBlocks = readTextBlocks(firstMessage.message.content);
    expect(
      textBlocks.some((textBlock) => {
        return textBlock.includes("Previous conversation context");
      }),
    ).toBeFalse();
  });

  test("ignores non-image file attachments", async () => {
    const queryPrompt = await buildMultimodalQueryPrompt({
      promptMessages: [
        {
          role: "user",
          content: [
            {
              type: "file",
              mediaType: "text/plain",
              data: "Zm9v",
            },
          ],
        },
      ],
      resumeSessionId: undefined,
      preambleText: undefined,
    });

    expect(queryPrompt).toBeUndefined();
  });
});
