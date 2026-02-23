import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";

export type ToolExecutor = (input: Record<string, unknown>) => unknown | Promise<unknown>;

export type ToolExecutorMap = Record<string, ToolExecutor>;

export type AgentSdkProviderSettings = AnthropicProviderSettings & {
  toolExecutors?: ToolExecutorMap;
  maxTurns?: number;
};
