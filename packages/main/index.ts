import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";

export { VERSION } from "./shared/constants";
export { anthropic, createAnthropic } from "./provider/create-anthropic";
export { forwardAnthropicContainerIdFromLastStep } from "./provider/forward-container";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;
