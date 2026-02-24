import * as agentSdk from "@anthropic-ai/claude-agent-sdk";

import type { AgentRuntimePort } from "../ports/agent-runtime-port";

export const claudeAgentRuntime: AgentRuntimePort = {
  query: (args) => {
    return agentSdk.query(args);
  },
};
