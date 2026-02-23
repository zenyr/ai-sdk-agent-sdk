import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";
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
  forwardAnthropicContainerIdFromLastStep,
  VERSION,
} from "./index";

import type { AgentSdkProviderSettings } from "./shared/tool-executor";
import { isRecord } from "./shared/type-readers";

type V2FinishReason =
  | "stop"
  | "length"
  | "content-filter"
  | "tool-calls"
  | "error"
  | "other"
  | "unknown";

type LegacyFinishPart = Omit<
  Extract<LanguageModelV3StreamPart, { type: "finish" }>,
  "finishReason"
> & {
  finishReason: V2FinishReason;
  rawFinishReason?: string;
  finish: V2FinishReason;
  reason: string;
};

type LegacyGenerateResult = Omit<LanguageModelV3GenerateResult, "finishReason"> & {
  finishReason: V2FinishReason;
  rawFinishReason?: string;
  finish: V2FinishReason;
  reason: string;
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

const mapFinishReasonToV2 = (value: unknown): V2FinishReason => {
  if (!isRecord(value)) {
    return "other";
  }

  const unified = value.unified;
  if (isV2FinishReason(unified)) {
    return unified;
  }

  return "other";
};

const readRawFinishReason = (value: unknown): string | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const raw = value.raw;
  return typeof raw === "string" ? raw : undefined;
};

const withLegacyFinish = (
  part: Extract<LanguageModelV3StreamPart, { type: "finish" }>,
): LegacyFinishPart => {
  const finish = mapFinishReasonToV2(part.finishReason);
  const rawFinishReason = readRawFinishReason(part.finishReason);
  const reason = rawFinishReason ?? finish;

  return {
    ...part,
    finishReason: finish,
    rawFinishReason,
    finish,
    reason,
  };
};

class AgentSdkAnthropicLanguageModelV2Adapter {
  readonly specificationVersion: "v2" = "v2";
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

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LegacyGenerateResult> {
    const result = await this.baseModel.doGenerate(options);
    const finish = mapFinishReasonToV2(result.finishReason);
    const rawFinishReason = readRawFinishReason(result.finishReason);
    const reason = rawFinishReason ?? finish;

    const legacyResult: LegacyGenerateResult = {
      ...result,
      finishReason: finish,
      rawFinishReason,
      finish,
      reason,
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

          controller.enqueue(withLegacyFinish(part));
        },
      }),
    );

    return {
      ...streamResult,
      stream,
    };
  }
}

const wrapProviderWithLegacyFinish = (provider: ReturnType<typeof createAnthropicV3>) => {
  const createLanguageModel = (modelId: string) => {
    return new AgentSdkAnthropicLanguageModelV2Adapter(provider(modelId));
  };

  const legacyProvider = Object.assign(createLanguageModel, {
    specificationVersion: "v2",
    languageModel: createLanguageModel,
    chat: createLanguageModel,
    messages: createLanguageModel,
    embeddingModel: provider.embeddingModel,
    textEmbeddingModel: provider.textEmbeddingModel,
    imageModel: provider.imageModel,
    tools: provider.tools,
  });

  return legacyProvider;
};

export const createAnthropic = (
  options: AgentSdkProviderSettings = {},
): ReturnType<typeof createAnthropicV3> => {
  const provider = createAnthropicV3(options);
  return wrapProviderWithLegacyFinish(provider);
};

export const anthropic = wrapProviderWithLegacyFinish(anthropicV3);

export { VERSION, forwardAnthropicContainerIdFromLastStep };

export type {
  AgentSdkProviderSettings,
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;
