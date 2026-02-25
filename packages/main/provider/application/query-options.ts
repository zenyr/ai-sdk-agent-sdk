import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type { Options as AgentQueryOptions } from "@anthropic-ai/claude-agent-sdk";

import { buildQueryEnv } from "../domain/query-env";

type BuildAgentQueryOptionsArgs = {
  modelId: string;
  settings: AnthropicProviderSettings;
  allowedTools: AgentQueryOptions["allowedTools"];
  mcpServers: AgentQueryOptions["mcpServers"];
  resumeSessionId: AgentQueryOptions["resume"];
  systemPrompt: AgentQueryOptions["systemPrompt"];
  maxTurns: number | undefined;
  useNativeToolExecution: boolean;
  abortController: AbortController;
  outputFormat: AgentQueryOptions["outputFormat"];
  effort: AgentQueryOptions["effort"];
  thinking: AgentQueryOptions["thinking"];
  includePartialMessages: boolean;
};

export const buildAgentQueryOptions = (args: BuildAgentQueryOptionsArgs): AgentQueryOptions => {
  return {
    model: args.modelId,
    tools: [],
    allowedTools: args.allowedTools,
    resume: args.resumeSessionId,
    systemPrompt: args.systemPrompt,
    permissionMode: "dontAsk",
    settingSources: [],
    maxTurns: args.useNativeToolExecution ? args.maxTurns : 1,
    abortController: args.abortController,
    env: buildQueryEnv(args.settings),
    hooks: {},
    plugins: [],
    mcpServers: args.mcpServers,
    outputFormat: args.outputFormat,
    effort: args.effort,
    thinking: args.thinking,
    cwd: process.cwd(),
    includePartialMessages: args.includePartialMessages,
  };
};
