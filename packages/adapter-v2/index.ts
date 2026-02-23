import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export type {
  AgentSdkProviderSettings,
  ToolExecutor,
  ToolExecutorMap,
} from "ai-sdk-agent-sdk/v2";
export {
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
  VERSION,
} from "ai-sdk-agent-sdk/v2";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;
