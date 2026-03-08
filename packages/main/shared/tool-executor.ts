import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type { Options as AgentQueryOptions } from "@anthropic-ai/claude-agent-sdk";

export type ToolExecutor = (input: Record<string, unknown>) => unknown | Promise<unknown>;

export type ToolExecutorMap = Record<string, ToolExecutor>;

export type ToolExecutionRequest = {
  toolName: string;
  input: Record<string, unknown>;
};

export type ToolCallDelegate = (request: ToolExecutionRequest) => unknown | Promise<unknown>;

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
  toolCallDelegate?: ToolCallDelegate;
  maxTurns?: number;
  experimental_agentSdk?: AgentSdkQueryOptions;
};
