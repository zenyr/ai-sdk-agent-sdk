import type {
  LanguageModelV3Content,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider";
import { generateId } from "@ai-sdk/provider-utils";
import type { StreamEventState } from "../shared/stream-types";
import { isRecord, readNumber, readRecord, readString } from "../shared/type-readers";
import { mapUsageFromMessageDelta } from "./result-mapping";

type StreamPartEnqueuer = {
  enqueue: (part: LanguageModelV3StreamPart) => void;
};

export const appendStreamPartsFromRawEvent = (
  rawEvent: unknown,
  streamState: StreamEventState,
): LanguageModelV3StreamPart[] => {
  if (!isRecord(rawEvent)) {
    return [];
  }

  const eventType = readString(rawEvent, "type");
  if (eventType === undefined) {
    return [];
  }

  if (eventType === "message_start") {
    const message = readRecord(rawEvent, "message");
    if (!isRecord(message)) {
      return [];
    }

    streamState.emittedResponseMetadata = true;

    return [
      {
        type: "response-metadata",
        id: readString(message, "id"),
        modelId: readString(message, "model"),
      },
    ];
  }

  if (eventType === "content_block_start") {
    const index = readNumber(rawEvent, "index");
    const contentBlock = readRecord(rawEvent, "content_block");

    if (index === undefined || contentBlock === undefined) {
      return [];
    }

    const contentBlockType = readString(contentBlock, "type");
    if (contentBlockType === "text") {
      const blockId = generateId();
      streamState.blockStates.set(index, { kind: "text", id: blockId });
      return [{ type: "text-start", id: blockId }];
    }

    if (contentBlockType === "thinking") {
      const blockId = generateId();
      streamState.blockStates.set(index, { kind: "reasoning", id: blockId });
      return [{ type: "reasoning-start", id: blockId }];
    }

    if (
      contentBlockType === "tool_use" ||
      contentBlockType === "server_tool_use" ||
      contentBlockType === "mcp_tool_use"
    ) {
      const blockId = readString(contentBlock, "id") ?? generateId();
      const toolName = readString(contentBlock, "name") ?? "unknown_tool";
      const providerExecuted =
        contentBlockType === "server_tool_use" || contentBlockType === "mcp_tool_use";
      const dynamic = contentBlockType === "mcp_tool_use";

      streamState.blockStates.set(index, { kind: "tool-input", id: blockId });

      return [
        {
          type: "tool-input-start",
          id: blockId,
          toolName,
          providerExecuted,
          dynamic,
        },
      ];
    }

    return [];
  }

  if (eventType === "content_block_delta") {
    const index = readNumber(rawEvent, "index");
    const delta = readRecord(rawEvent, "delta");

    if (index === undefined || delta === undefined) {
      return [];
    }

    const blockState = streamState.blockStates.get(index);
    if (blockState === undefined) {
      return [];
    }

    const deltaType = readString(delta, "type");

    if (blockState.kind === "text" && deltaType === "text_delta") {
      return [
        {
          type: "text-delta",
          id: blockState.id,
          delta: readString(delta, "text") ?? "",
        },
      ];
    }

    if (blockState.kind === "reasoning" && deltaType === "thinking_delta") {
      return [
        {
          type: "reasoning-delta",
          id: blockState.id,
          delta: readString(delta, "thinking") ?? "",
        },
      ];
    }

    if (blockState.kind === "tool-input" && deltaType === "input_json_delta") {
      return [
        {
          type: "tool-input-delta",
          id: blockState.id,
          delta: readString(delta, "partial_json") ?? "",
        },
      ];
    }

    return [];
  }

  if (eventType === "content_block_stop") {
    const index = readNumber(rawEvent, "index");
    if (index === undefined) {
      return [];
    }

    const blockState = streamState.blockStates.get(index);
    if (blockState === undefined) {
      return [];
    }

    streamState.blockStates.delete(index);

    if (blockState.kind === "text") {
      return [{ type: "text-end", id: blockState.id }];
    }

    if (blockState.kind === "reasoning") {
      return [{ type: "reasoning-end", id: blockState.id }];
    }

    return [{ type: "tool-input-end", id: blockState.id }];
  }

  if (eventType === "message_delta") {
    const delta = readRecord(rawEvent, "delta");
    if (delta !== undefined) {
      const stopReason = delta.stop_reason;
      if (stopReason === null || typeof stopReason === "string") {
        streamState.latestStopReason = stopReason;
      }
    }

    const usage = mapUsageFromMessageDelta(readRecord(rawEvent, "usage"));
    if (usage !== undefined) {
      streamState.latestUsage = usage;
    }

    return [];
  }

  return [];
};

export const closePendingStreamBlocks = (
  streamState: StreamEventState,
): LanguageModelV3StreamPart[] => {
  const parts: LanguageModelV3StreamPart[] = [];

  for (const blockState of streamState.blockStates.values()) {
    if (blockState.kind === "text") {
      parts.push({ type: "text-end", id: blockState.id });
      continue;
    }

    if (blockState.kind === "reasoning") {
      parts.push({ type: "reasoning-end", id: blockState.id });
      continue;
    }

    parts.push({ type: "tool-input-end", id: blockState.id });
  }

  streamState.blockStates.clear();

  return parts;
};

export const enqueueSingleTextBlock = (
  enqueuer: StreamPartEnqueuer,
  idGenerator: () => string,
  text: string,
): void => {
  const blockId = idGenerator();
  enqueuer.enqueue({ type: "text-start", id: blockId });
  enqueuer.enqueue({ type: "text-delta", id: blockId, delta: text });
  enqueuer.enqueue({ type: "text-end", id: blockId });
};

export const contentToStreamParts = (
  content: LanguageModelV3Content[],
): LanguageModelV3StreamPart[] => {
  const streamParts: LanguageModelV3StreamPart[] = [];

  for (const contentPart of content) {
    if (contentPart.type === "text") {
      const textId = generateId();
      streamParts.push({ type: "text-start", id: textId });
      streamParts.push({
        type: "text-delta",
        id: textId,
        delta: contentPart.text,
      });
      streamParts.push({ type: "text-end", id: textId });
      continue;
    }

    if (contentPart.type === "reasoning") {
      const reasoningId = generateId();
      streamParts.push({ type: "reasoning-start", id: reasoningId });
      streamParts.push({
        type: "reasoning-delta",
        id: reasoningId,
        delta: contentPart.text,
      });
      streamParts.push({ type: "reasoning-end", id: reasoningId });
      continue;
    }

    streamParts.push(contentPart);
  }

  return streamParts;
};

export const appendStreamBlocks = (
  blocks: Array<{ kind: "text" | "reasoning"; id: string }>,
): LanguageModelV3StreamPart[] => {
  const streamParts: LanguageModelV3StreamPart[] = [];

  for (const block of blocks) {
    streamParts.push({
      type: block.kind === "text" ? "text-start" : "reasoning-start",
      id: block.id,
    });
    streamParts.push({
      type: block.kind === "text" ? "text-end" : "reasoning-end",
      id: block.id,
    });
  }

  return streamParts;
};

export const collectUsage = (
  deltas: Array<LanguageModelV3Usage | undefined>,
): LanguageModelV3Usage => {
  return deltas.reduce<LanguageModelV3Usage>(
    (acc, item) => {
      if (item === undefined) {
        return acc;
      }

      return {
        inputTokens: {
          total: item.inputTokens.total ?? acc.inputTokens.total,
          noCache: item.inputTokens.noCache ?? acc.inputTokens.noCache,
          cacheRead: item.inputTokens.cacheRead ?? acc.inputTokens.cacheRead,
          cacheWrite: item.inputTokens.cacheWrite ?? acc.inputTokens.cacheWrite,
        },
        outputTokens: {
          total: item.outputTokens.total ?? acc.outputTokens.total,
          text: item.outputTokens.text ?? acc.outputTokens.text,
          reasoning: item.outputTokens.reasoning ?? acc.outputTokens.reasoning,
        },
      };
    },
    {
      inputTokens: {
        total: undefined,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: undefined,
        text: undefined,
        reasoning: undefined,
      },
    },
  );
};
