import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type { Options as AgentQueryOptions } from "@anthropic-ai/claude-agent-sdk";

export type ToolExecutor = (input: Record<string, unknown>) => unknown | Promise<unknown>;

export type ToolExecutorMap = Record<string, ToolExecutor>;

type ReservedAgentQueryOptionKeys =
  | "model"
  | "tools"
  | "allowedTools"
  | "resume"
  | "systemPrompt"
  | "maxTurns"
  | "abortController"
  | "env"
  | "mcpServers"
  | "outputFormat"
  | "effort"
  | "thinking"
  | "includePartialMessages";

export type AgentSdkQueryOptions = Omit<AgentQueryOptions, ReservedAgentQueryOptionKeys>;

export type AgentSdkProviderSettings = AnthropicProviderSettings & {
  toolExecutors?: ToolExecutorMap;
  maxTurns?: number;
  experimental_agentSdk?: AgentSdkQueryOptions;
};
