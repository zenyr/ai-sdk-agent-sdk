import type { AnthropicMessageMetadata, AnthropicUsageIteration } from "@ai-sdk/anthropic";
import type {
  JSONObject,
  LanguageModelV3FinishReason,
  LanguageModelV3Usage,
} from "@ai-sdk/provider";
import type { SDKResultMessage } from "@anthropic-ai/claude-agent-sdk";

import { isRecord, readNumber, readString } from "../shared/type-readers";

export const mapFinishReason = (rawFinishReason: string | null): LanguageModelV3FinishReason => {
  if (rawFinishReason === "tool_use") {
    return { unified: "tool-calls", raw: rawFinishReason };
  }

  if (rawFinishReason === "max_tokens" || rawFinishReason === "model_context_window_exceeded") {
    return { unified: "length", raw: rawFinishReason };
  }

  if (rawFinishReason === "refusal") {
    return { unified: "content-filter", raw: rawFinishReason };
  }

  if (rawFinishReason === "pause_turn" || rawFinishReason === "compaction") {
    return { unified: "other", raw: rawFinishReason };
  }

  if (rawFinishReason === null) {
    return { unified: "stop", raw: undefined };
  }

  return { unified: "stop", raw: rawFinishReason };
};

export const mapUsage = (resultMessage: SDKResultMessage): LanguageModelV3Usage => {
  const totalInput = resultMessage.usage.input_tokens;
  const cacheRead = resultMessage.usage.cache_read_input_tokens;
  const cacheWrite = resultMessage.usage.cache_creation_input_tokens;
  const noCache = totalInput - cacheRead - cacheWrite;

  return {
    inputTokens: {
      total: totalInput,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: resultMessage.usage.output_tokens,
      text: resultMessage.usage.output_tokens,
      reasoning: undefined,
    },
  };
};

export const mapIterations = (
  resultMessage: SDKResultMessage,
): AnthropicUsageIteration[] | null => {
  const iterations = resultMessage.usage.iterations;
  if (!Array.isArray(iterations)) {
    return null;
  }

  const mapped = iterations
    .map((iteration) => {
      if (!isRecord(iteration)) {
        return undefined;
      }

      const type = readString(iteration, "type");
      const inputTokens = readNumber(iteration, "input_tokens");
      const outputTokens = readNumber(iteration, "output_tokens");

      if (
        (type !== "compaction" && type !== "message") ||
        typeof inputTokens !== "number" ||
        typeof outputTokens !== "number"
      ) {
        return undefined;
      }

      return {
        type,
        inputTokens,
        outputTokens,
      };
    })
    .filter((value): value is AnthropicUsageIteration => value !== undefined);

  return mapped.length > 0 ? mapped : null;
};

export const mapMetadata = (resultMessage: SDKResultMessage): AnthropicMessageMetadata => {
  return {
    usage: {},
    cacheCreationInputTokens: resultMessage.usage.cache_creation_input_tokens,
    stopSequence: null,
    iterations: mapIterations(resultMessage),
    container: null,
    contextManagement: null,
  };
};

export const buildProviderMetadata = (
  resultMessage: SDKResultMessage,
): Record<string, JSONObject> => {
  const metadata = mapMetadata(resultMessage);

  return {
    anthropic: {
      usage: metadata.usage,
      cacheCreationInputTokens: metadata.cacheCreationInputTokens,
      stopSequence: metadata.stopSequence,
      iterations:
        metadata.iterations?.map((iteration) => {
          return {
            type: iteration.type,
            inputTokens: iteration.inputTokens,
            outputTokens: iteration.outputTokens,
          };
        }) ?? null,
      container: null,
      contextManagement: null,
    },
  };
};

export const mapUsageFromMessageDelta = (usageValue: unknown): LanguageModelV3Usage | undefined => {
  if (!isRecord(usageValue)) {
    return undefined;
  }

  const totalInput = readNumber(usageValue, "input_tokens");
  const cacheRead = readNumber(usageValue, "cache_read_input_tokens");
  const cacheWrite = readNumber(usageValue, "cache_creation_input_tokens");
  const outputTokens = readNumber(usageValue, "output_tokens");

  if (
    totalInput === undefined &&
    cacheRead === undefined &&
    cacheWrite === undefined &&
    outputTokens === undefined
  ) {
    return undefined;
  }

  let noCache: number | undefined;
  if (typeof totalInput === "number") {
    noCache = totalInput - (cacheRead ?? 0) - (cacheWrite ?? 0);
  }

  return {
    inputTokens: {
      total: totalInput,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: outputTokens,
      text: outputTokens,
      reasoning: undefined,
    },
  };
};
