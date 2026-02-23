import {
  InvalidArgumentError,
  type LanguageModelV3,
  NoSuchModelError,
} from "@ai-sdk/provider";
import { anthropic as upstreamAnthropic } from "@ai-sdk/anthropic";
import { generateId } from "@ai-sdk/provider-utils";

import {
  type AnthropicProvider,
  type AnthropicProviderSettings,
} from "@ai-sdk/anthropic";

import { AgentSdkAnthropicLanguageModel } from "./agent-sdk-language-model";

const anthropicTools = upstreamAnthropic.tools;

const createLanguageModelFactory = (args: {
  providerName: string;
  settings: AnthropicProviderSettings;
  idGenerator: () => string;
}) => {
  return (modelId: string): LanguageModelV3 => {
    return new AgentSdkAnthropicLanguageModel({
      modelId,
      provider: args.providerName,
      settings: args.settings,
      idGenerator: args.idGenerator,
    });
  };
};

export const createAnthropic = (
  options: AnthropicProviderSettings = {},
): AnthropicProvider => {
  if (
    typeof options.apiKey === "string" &&
    options.apiKey.length > 0 &&
    typeof options.authToken === "string" &&
    options.authToken.length > 0
  ) {
    throw new InvalidArgumentError({
      argument: "apiKey/authToken",
      message:
        "Both apiKey and authToken were provided. Please use only one authentication method.",
    });
  }

  const providerName =
    typeof options.name === "string" && options.name.length > 0
      ? options.name
      : "anthropic.messages";

  const idGenerator =
    typeof options.generateId === "function" ? options.generateId : generateId;

  const createLanguageModel = createLanguageModelFactory({
    providerName,
    settings: options,
    idGenerator,
  });

  const specificationVersion: "v3" = "v3";

  const provider: AnthropicProvider = Object.assign(
    (modelId: string) => createLanguageModel(modelId),
    {
      specificationVersion,
      languageModel: createLanguageModel,
      chat: createLanguageModel,
      messages: createLanguageModel,
      embeddingModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: "embeddingModel",
        });
      },
      textEmbeddingModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: "embeddingModel",
        });
      },
      imageModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: "imageModel",
        });
      },
      tools: anthropicTools,
    },
  );

  return provider;
};

export const anthropic = createAnthropic();
