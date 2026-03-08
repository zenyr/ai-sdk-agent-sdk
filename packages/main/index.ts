import type { AnthropicLanguageModelOptions, AnthropicProvider } from "@ai-sdk/anthropic";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";
import {
  anthropic as anthropicV3,
  createAnthropic as createAnthropicV3,
} from "./provider/create-anthropic";
import { anthropic as anthropicV2, createAnthropic as createAnthropicV2 } from "./v2";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export { forwardAnthropicContainerIdFromLastStep } from "./provider/forward-container";
export { VERSION } from "./shared/constants";
export type {
  AgentSdkProviderSettings,
  ToolExecutor,
  ToolExecutorMap,
} from "./shared/tool-executor";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export const isOpenCode = (): boolean => {
  const value = process.env.OPENCODE;
  return typeof value === "string" && value.length > 0 && value !== "0" && value !== "false";
};

type V2FinishReason =
  | "stop"
  | "length"
  | "content-filter"
  | "tool-calls"
  | "error"
  | "other"
  | "unknown";

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const isV2FinishReason = (value: unknown): value is V2FinishReason => {
  return (
    value === "stop" ||
    value === "length" ||
    value === "content-filter" ||
    value === "tool-calls" ||
    value === "error" ||
    value === "other" ||
    value === "unknown"
  );
};

const mapLegacyFinish = (finishReason: unknown) => {
  if (!isRecord(finishReason)) {
    return {
      finish: "other",
      reason: "other",
      rawFinishReason: undefined,
    };
  }

  const unified = finishReason.unified;
  const finish = isV2FinishReason(unified) ? unified : "other";
  const raw = typeof finishReason.raw === "string" ? finishReason.raw : undefined;

  return {
    finish,
    reason: raw ?? finish,
    rawFinishReason: raw,
  };
};

class OpenCodeCompatLanguageModel {
  readonly specificationVersion: "v3" = "v3";
  readonly provider: string;
  readonly modelId: string;
  readonly supportedUrls: LanguageModelV3["supportedUrls"];

  private readonly baseModel: LanguageModelV3;

  constructor(baseModel: LanguageModelV3) {
    this.baseModel = baseModel;
    this.provider = baseModel.provider;
    this.modelId = baseModel.modelId;
    this.supportedUrls = baseModel.supportedUrls;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const result = await this.baseModel.doGenerate(options);
    const legacyFinish = mapLegacyFinish(result.finishReason);
    const legacyResult = {
      ...result,
      finish: legacyFinish.finish,
      reason: legacyFinish.reason,
      rawFinishReason: legacyFinish.rawFinishReason,
    };

    return legacyResult;
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const streamResult = await this.baseModel.doStream(options);

    const stream = streamResult.stream.pipeThrough(
      new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
        transform: (part, controller) => {
          if (part.type !== "finish") {
            controller.enqueue(part);
            return;
          }

          const legacyFinish = mapLegacyFinish(part.finishReason);
          const legacyFinishPart = {
            ...part,
            finish: legacyFinish.finish,
            reason: legacyFinish.reason,
            rawFinishReason: legacyFinish.rawFinishReason,
          };

          controller.enqueue(legacyFinishPart);
        },
      }),
    );

    return {
      ...streamResult,
      stream,
    };
  }
}

const withOpenCodeCompatibility = (provider: AnthropicProvider): AnthropicProvider => {
  const createLanguageModel = (modelId: string): LanguageModelV3 => {
    return new OpenCodeCompatLanguageModel(provider(modelId));
  };

  const specificationVersion: "v3" = "v3";

  const wrappedProvider: AnthropicProvider = Object.assign(createLanguageModel, {
    specificationVersion,
    languageModel: createLanguageModel,
    chat: createLanguageModel,
    messages: createLanguageModel,
    embeddingModel: provider.embeddingModel,
    textEmbeddingModel: provider.textEmbeddingModel,
    imageModel: provider.imageModel,
    tools: provider.tools,
  });

  return wrappedProvider;
};

// @ts-expect-error Runtime OPENCODE path requires legacy v2 finish reason shape.
export const createAnthropic: typeof createAnthropicV3 = (
  ...args: Parameters<typeof createAnthropicV3>
) => {
  if (isOpenCode()) {
    return createAnthropicV2(...args);
  }

  const provider = createAnthropicV3(...args);
  return withOpenCodeCompatibility(provider);
};

const resolvedAnthropicProvider = isOpenCode()
  ? anthropicV2
  : withOpenCodeCompatibility(anthropicV3);

// @ts-expect-error Runtime OPENCODE path requires legacy v2 finish reason shape.
export const anthropic: AnthropicProvider = resolvedAnthropicProvider;
