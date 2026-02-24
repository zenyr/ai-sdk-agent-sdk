import type {
  Options as AgentQueryOptions,
  SDKMessage,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";

export type AgentRuntimePort = {
  query(args: {
    prompt: string | AsyncIterable<SDKUserMessage>;
    options: AgentQueryOptions;
  }): AsyncIterable<SDKMessage>;
};
