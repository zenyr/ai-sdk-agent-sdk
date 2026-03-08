import type { AnthropicProvider } from "@ai-sdk/anthropic";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
} from "@ai-sdk/provider";

type V2FinishReason =
  | "stop"
  | "length"
  | "content-filter"
  | "tool-calls"
  | "error"
  | "other"
  | "unknown";

type LegacyFinishOverlay = {
  finish: V2FinishReason;
  reason: string;
  rawFinishReason?: string;
};

type OpenCodeCompatGenerateResult = LanguageModelV3GenerateResult & LegacyFinishOverlay;

type OpenCodeCompatFinishPart = Extract<LanguageModelV3StreamPart, { type: "finish" }> &
  LegacyFinishOverlay;

type OpenCodeCompatStreamPart =
  | Exclude<LanguageModelV3StreamPart, { type: "finish" }>
  | OpenCodeCompatFinishPart;

type OpenCodeCompatStreamResult = Omit<LanguageModelV3StreamResult, "stream"> & {
  stream: ReadableStream<OpenCodeCompatStreamPart>;
};

type OpenCodeCompatLanguageModelType = Omit<LanguageModelV3, "doGenerate" | "doStream"> & {
  doGenerate: (options: LanguageModelV3CallOptions) => Promise<OpenCodeCompatGenerateResult>;
  doStream: (options: LanguageModelV3CallOptions) => Promise<OpenCodeCompatStreamResult>;
};

export type OpenCodeCompatAnthropicProvider = Omit<
  AnthropicProvider,
  "languageModel" | "chat" | "messages"
> & {
  (modelId: string): OpenCodeCompatLanguageModelType;
  languageModel: (modelId: string) => OpenCodeCompatLanguageModelType;
  chat: (modelId: string) => OpenCodeCompatLanguageModelType;
  messages: (modelId: string) => OpenCodeCompatLanguageModelType;
};

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

const mapLegacyFinish = (finishReason: unknown): LegacyFinishOverlay => {
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

  async doGenerate(options: LanguageModelV3CallOptions): Promise<OpenCodeCompatGenerateResult> {
    const result = await this.baseModel.doGenerate(options);
    const legacyFinish = mapLegacyFinish(result.finishReason);

    return {
      ...result,
      finish: legacyFinish.finish,
      reason: legacyFinish.reason,
      rawFinishReason: legacyFinish.rawFinishReason,
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<OpenCodeCompatStreamResult> {
    const streamResult = await this.baseModel.doStream(options);

    const stream = streamResult.stream.pipeThrough(
      new TransformStream<LanguageModelV3StreamPart, OpenCodeCompatStreamPart>({
        transform: (part, controller) => {
          if (part.type !== "finish") {
            controller.enqueue(part);
            return;
          }

          const legacyFinish = mapLegacyFinish(part.finishReason);
          controller.enqueue({
            ...part,
            finish: legacyFinish.finish,
            reason: legacyFinish.reason,
            rawFinishReason: legacyFinish.rawFinishReason,
          });
        },
      }),
    );

    return {
      ...streamResult,
      stream,
    };
  }
}

export const withOpenCodeCompatibility = (
  provider: AnthropicProvider,
): OpenCodeCompatAnthropicProvider => {
  const createLanguageModel = (modelId: string): OpenCodeCompatLanguageModelType => {
    return new OpenCodeCompatLanguageModel(provider(modelId));
  };

  return Object.assign(createLanguageModel, {
    specificationVersion: "v3" as const,
    languageModel: createLanguageModel,
    chat: createLanguageModel,
    messages: createLanguageModel,
    embeddingModel: provider.embeddingModel,
    textEmbeddingModel: provider.textEmbeddingModel,
    imageModel: provider.imageModel,
    tools: provider.tools,
  });
};
