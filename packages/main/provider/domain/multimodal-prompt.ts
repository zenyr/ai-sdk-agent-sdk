import type { LanguageModelV3Message } from "@ai-sdk/provider";
import type { SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";

import {
  joinSerializedPromptMessages,
  serializePromptMessagesWithoutSystem,
} from "../../bridge/prompt-serializer";
import { isRecord, readString } from "../../shared/type-readers";

type AgentUserTextContentBlock = {
  type: "text";
  text: string;
};

type AgentUserImageContentBlock = {
  type: "image";
  source: {
    type: "base64";
    media_type: string;
    data: string;
  };
};

type AgentUserContentBlock = AgentUserTextContentBlock | AgentUserImageContentBlock;

const normalizeMediaType = (mediaType: string | undefined): string | undefined => {
  if (mediaType === undefined) {
    return undefined;
  }

  const normalizedMediaType = mediaType.split(";")[0]?.trim().toLowerCase();
  if (normalizedMediaType === undefined || normalizedMediaType.length === 0) {
    return undefined;
  }

  return normalizedMediaType;
};

const isImageMediaType = (mediaType: string | undefined): boolean => {
  return typeof mediaType === "string" && mediaType.startsWith("image/");
};

const readMessageContentParts = (message: LanguageModelV3Message): unknown[] => {
  if (message.role === "system") {
    return [];
  }

  const messageContent: unknown = message.content;

  if (typeof messageContent === "string") {
    if (messageContent.length === 0) {
      return [];
    }

    return [
      {
        type: "text",
        text: messageContent,
      },
    ];
  }

  if (!Array.isArray(messageContent)) {
    return [];
  }

  return messageContent;
};

const findLastUserMessageIndex = (promptMessages: LanguageModelV3Message[]): number | undefined => {
  for (let index = promptMessages.length - 1; index >= 0; index -= 1) {
    const promptMessage = promptMessages[index];
    if (promptMessage?.role === "user") {
      return index;
    }
  }

  return undefined;
};

const hasImagePart = (contentParts: unknown[]): boolean => {
  for (const contentPart of contentParts) {
    if (!isRecord(contentPart)) {
      continue;
    }

    const contentPartType = readString(contentPart, "type");
    if (contentPartType === "image") {
      return true;
    }

    if (contentPartType !== "file") {
      continue;
    }

    const mediaType = normalizeMediaType(
      readString(contentPart, "mediaType") ?? readString(contentPart, "mimeType"),
    );
    if (isImageMediaType(mediaType)) {
      return true;
    }
  }

  return false;
};

const readImageDataUrl = (value: string): { mediaType: string; data: string } | undefined => {
  if (!value.startsWith("data:")) {
    return undefined;
  }

  const metadataEndIndex = value.indexOf(",");
  if (metadataEndIndex <= 5) {
    return undefined;
  }

  const metadata = value.slice(5, metadataEndIndex);
  const data = value.slice(metadataEndIndex + 1);
  if (!metadata.toLowerCase().includes(";base64") || data.length === 0) {
    return undefined;
  }

  const mediaType = normalizeMediaType(metadata.split(";")[0]);
  if (mediaType === undefined || !isImageMediaType(mediaType)) {
    return undefined;
  }

  return {
    mediaType,
    data,
  };
};

const readHttpUrl = (value: string): URL | undefined => {
  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return undefined;
    }

    return url;
  } catch {
    return undefined;
  }
};

const readBinaryDataAsUint8Array = (value: unknown): Uint8Array | undefined => {
  if (value instanceof Uint8Array) {
    return value;
  }

  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value);
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }

  return undefined;
};

const readImageDataFromUrl = async (
  url: URL,
): Promise<{ mediaType: string | undefined; data: string }> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to download image attachment: ${url.toString()}`);
  }

  const responseMediaType = normalizeMediaType(response.headers.get("content-type") ?? undefined);
  if (responseMediaType !== undefined && !isImageMediaType(responseMediaType)) {
    throw new Error(`unsupported image media type from URL: ${responseMediaType}`);
  }

  const responseBytes = new Uint8Array(await response.arrayBuffer());

  return {
    mediaType: responseMediaType,
    data: Buffer.from(responseBytes).toString("base64"),
  };
};

const normalizeImageAttachment = async (args: {
  data: unknown;
  mediaTypeHint: string | undefined;
}): Promise<{ mediaType: string; data: string }> => {
  const normalizedMediaTypeHint = normalizeMediaType(args.mediaTypeHint);
  if (normalizedMediaTypeHint !== undefined && !isImageMediaType(normalizedMediaTypeHint)) {
    throw new Error(`unsupported image media type: ${normalizedMediaTypeHint}`);
  }

  if (args.data instanceof URL) {
    const downloadedImage = await readImageDataFromUrl(args.data);
    const mediaType = downloadedImage.mediaType ?? normalizedMediaTypeHint;
    if (mediaType === undefined || !isImageMediaType(mediaType)) {
      throw new Error("missing media type for image URL attachment");
    }

    return {
      mediaType,
      data: downloadedImage.data,
    };
  }

  if (typeof args.data === "string") {
    const parsedDataUrl = readImageDataUrl(args.data);
    if (parsedDataUrl !== undefined) {
      return parsedDataUrl;
    }

    const imageUrl = readHttpUrl(args.data);
    if (imageUrl !== undefined) {
      const downloadedImage = await readImageDataFromUrl(imageUrl);
      const mediaType = downloadedImage.mediaType ?? normalizedMediaTypeHint;
      if (mediaType === undefined || !isImageMediaType(mediaType)) {
        throw new Error("missing media type for image URL attachment");
      }

      return {
        mediaType,
        data: downloadedImage.data,
      };
    }

    if (normalizedMediaTypeHint === undefined || !isImageMediaType(normalizedMediaTypeHint)) {
      throw new Error("missing media type for image base64 attachment");
    }

    return {
      mediaType: normalizedMediaTypeHint,
      data: args.data,
    };
  }

  const binaryData = readBinaryDataAsUint8Array(args.data);
  if (binaryData === undefined) {
    throw new Error("unsupported image attachment data type");
  }

  if (normalizedMediaTypeHint === undefined || !isImageMediaType(normalizedMediaTypeHint)) {
    throw new Error("missing media type for image binary attachment");
  }

  return {
    mediaType: normalizedMediaTypeHint,
    data: Buffer.from(binaryData).toString("base64"),
  };
};

const mapContentPartToMultimodalBlocks = async (
  contentPart: unknown,
): Promise<AgentUserContentBlock[]> => {
  if (!isRecord(contentPart)) {
    return [];
  }

  const contentPartType = readString(contentPart, "type");
  if (contentPartType === "text") {
    const text = readString(contentPart, "text") ?? "";
    if (text.length === 0) {
      return [];
    }

    return [{ type: "text", text }];
  }

  if (contentPartType !== "image" && contentPartType !== "file") {
    return [];
  }

  const mediaTypeHint = readString(contentPart, "mediaType") ?? readString(contentPart, "mimeType");
  if (contentPartType === "file" && !isImageMediaType(normalizeMediaType(mediaTypeHint))) {
    return [];
  }

  const imageData = contentPartType === "image" ? contentPart.image : contentPart.data;
  const normalizedImageAttachment = await normalizeImageAttachment({
    data: imageData,
    mediaTypeHint,
  });

  return [
    {
      type: "image",
      source: {
        type: "base64",
        media_type: normalizedImageAttachment.mediaType,
        data: normalizedImageAttachment.data,
      },
    },
  ];
};

const createSingleUserMessagePrompt = (
  userMessage: SDKUserMessage,
): AsyncIterable<SDKUserMessage> => {
  const promptStream = {
    async *[Symbol.asyncIterator]() {
      yield userMessage;
    },
  };

  return promptStream;
};

export const buildMultimodalQueryPrompt = async (args: {
  promptMessages: LanguageModelV3Message[];
  resumeSessionId: string | undefined;
  preambleText: string | undefined;
}): Promise<AsyncIterable<SDKUserMessage> | undefined> => {
  const lastUserMessageIndex = findLastUserMessageIndex(args.promptMessages);
  if (lastUserMessageIndex === undefined) {
    return undefined;
  }

  const lastUserMessage = args.promptMessages[lastUserMessageIndex];
  if (lastUserMessage === undefined) {
    return undefined;
  }

  const lastUserMessageContentParts = readMessageContentParts(lastUserMessage);
  if (!hasImagePart(lastUserMessageContentParts)) {
    return undefined;
  }

  const contentBlocks: AgentUserContentBlock[] = [];

  if (args.preambleText !== undefined && args.preambleText.length > 0) {
    contentBlocks.push({
      type: "text",
      text: args.preambleText,
    });
  }

  if (args.resumeSessionId === undefined) {
    const previousPromptMessages = args.promptMessages.slice(0, lastUserMessageIndex);
    const serializedPreviousPromptMessages =
      serializePromptMessagesWithoutSystem(previousPromptMessages);
    const previousPromptText = joinSerializedPromptMessages(serializedPreviousPromptMessages);

    if (previousPromptText.length > 0) {
      contentBlocks.push({
        type: "text",
        text: `Previous conversation context:\n\n${previousPromptText}`,
      });
    }
  }

  for (const contentPart of lastUserMessageContentParts) {
    const mappedContentBlocks = await mapContentPartToMultimodalBlocks(contentPart);
    for (const mappedContentBlock of mappedContentBlocks) {
      contentBlocks.push(mappedContentBlock);
    }
  }

  if (contentBlocks.length === 0) {
    return undefined;
  }

  const userMessage: SDKUserMessage = {
    type: "user",
    session_id: args.resumeSessionId ?? "default",
    parent_tool_use_id: null,
    message: {
      role: "user",
      content: contentBlocks,
    },
  };

  return createSingleUserMessagePrompt(userMessage);
};
