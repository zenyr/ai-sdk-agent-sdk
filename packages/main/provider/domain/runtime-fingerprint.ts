import { createHash } from "node:crypto";
import { withoutTrailingSlash } from "@ai-sdk/provider-utils";
import type { AgentSdkProviderSettings } from "../../shared/tool-executor";

const readNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  if (value.length === 0) {
    return undefined;
  }

  return value;
};

export const buildRuntimeFingerprint = (args: { provider: string; settings: AgentSdkProviderSettings }): string => {
  const baseURL = readNonEmptyString(args.settings.baseURL);
  const normalizedBaseURL = baseURL === undefined ? undefined : withoutTrailingSlash(baseURL);
  const cwd = readNonEmptyString(args.settings.experimental_agentSdk?.cwd) ?? process.cwd();
  const pathToClaudeCodeExecutable = readNonEmptyString(
    args.settings.experimental_agentSdk?.pathToClaudeCodeExecutable
  );

  const fingerprintPayload = JSON.stringify({
    provider: args.provider,
    baseURL: normalizedBaseURL,
    cwd,
    pathToClaudeCodeExecutable,
  });

  return createHash("sha256").update(fingerprintPayload).digest("hex").slice(0, 16);
};
