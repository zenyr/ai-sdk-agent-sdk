import type { Options as AgentQueryOptions } from "@anthropic-ai/claude-agent-sdk";

import type { AgentSdkProviderSettings, AgentSdkQueryOptions } from "../../shared/tool-executor";
import { buildQueryEnv } from "../domain/query-env";

type BuildAgentQueryOptionsArgs = {
  modelId: string;
  settings: AgentSdkProviderSettings;
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
  onStderr: (data: string) => void;
};

const resolveAgentSdkQueryOptions = (settings: AgentSdkProviderSettings): AgentSdkQueryOptions | undefined => {
  return settings.experimental_agentSdk;
};

export const buildAgentQueryOptions = (args: BuildAgentQueryOptionsArgs): AgentQueryOptions => {
  const passthroughOptions = resolveAgentSdkQueryOptions(args.settings);
  const cwd = passthroughOptions?.cwd ?? process.cwd();

  return {
    ...passthroughOptions,
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
    cwd,
    includePartialMessages: args.includePartialMessages,
    stderr: data => {
      passthroughOptions?.stderr?.(data);
      args.onStderr(data);
    },
  };
};
