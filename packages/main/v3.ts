import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export { anthropic, createAnthropic } from "./provider/create-anthropic";
export { forwardAnthropicContainerIdFromLastStep } from "./provider/forward-container";
export { VERSION } from "./shared/constants";
export type {
  AgentSdkProviderSettings,
  ToolExecutor,
  ToolExecutorMap,
} from "./shared/tool-executor";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;
