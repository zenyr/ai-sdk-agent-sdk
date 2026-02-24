import type { SharedV3ProviderOptions, SharedV3Warning } from "@ai-sdk/provider";

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const DEGRADE_ONLY_OPTIONS = new Set([
  "sendReasoning",
  "structuredOutputMode",
  "disableParallelToolUse",
  "toolStreaming",
]);

const UNSUPPORTED_OPTION_DETAILS: Record<string, string> = {
  cacheControl:
    "This option is not supported on the Agent SDK backend. Prompt cache TTL (including 1h cache) cannot be configured via claude-agent-sdk options.",
  mcpServers: "This option is not supported on the Agent SDK backend.",
  container: "This option is not supported on the Agent SDK backend.",
  speed: "This option is not supported on the Agent SDK backend.",
  contextManagement: "This option is not supported on the Agent SDK backend.",
};

const MAPPED_OPTIONS = new Set(["effort", "thinking"]);

export const collectAnthropicProviderOptionWarnings = (
  providerOptions: SharedV3ProviderOptions | undefined,
): SharedV3Warning[] => {
  if (!isRecord(providerOptions)) {
    return [];
  }

  const anthropicOptions = providerOptions.anthropic;
  if (!isRecord(anthropicOptions)) {
    return [];
  }

  const warnings: SharedV3Warning[] = [];

  for (const optionName of Object.keys(anthropicOptions)) {
    if (MAPPED_OPTIONS.has(optionName)) {
      continue;
    }

    if (DEGRADE_ONLY_OPTIONS.has(optionName)) {
      warnings.push({
        type: "compatibility",
        feature: `providerOptions.anthropic.${optionName}`,
        details: "This option is accepted but behavior may differ on the Agent SDK backend.",
      });
      continue;
    }

    if (optionName in UNSUPPORTED_OPTION_DETAILS) {
      warnings.push({
        type: "unsupported",
        feature: `providerOptions.anthropic.${optionName}`,
        details: UNSUPPORTED_OPTION_DETAILS[optionName],
      });
      continue;
    }

    warnings.push({
      type: "other",
      message: `Unknown anthropic provider option '${optionName}' is ignored by this backend.`,
    });
  }

  return warnings;
};
