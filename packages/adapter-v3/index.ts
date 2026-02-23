import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";

export {
  VERSION,
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
} from "ai-sdk-agent-sdk";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;
