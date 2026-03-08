import type { AnthropicLanguageModelOptions, AnthropicProviderSettings } from "@ai-sdk/anthropic";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export {
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
  VERSION,
} from "ai-sdk-agent-sdk/v3";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export type ToolExecutor = (input: Record<string, unknown>) => unknown | Promise<unknown>;

export type ToolExecutorMap = Record<string, ToolExecutor>;

export type AgentSdkProviderSettings = AnthropicProviderSettings & {
  toolExecutors?: ToolExecutorMap;
  maxTurns?: number;
};
