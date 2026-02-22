import type { SharedV3ProviderOptions, SharedV3Warning } from '@ai-sdk/provider';

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

const DEGRADE_ONLY_OPTIONS = new Set([
  'sendReasoning',
  'structuredOutputMode',
  'disableParallelToolUse',
  'toolStreaming',
]);

const UNSUPPORTED_OPTIONS = new Set([
  'cacheControl',
  'mcpServers',
  'container',
  'speed',
  'contextManagement',
]);

const MAPPED_OPTIONS = new Set(['effort', 'thinking']);

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
        type: 'compatibility',
        feature: `providerOptions.anthropic.${optionName}`,
        details:
          'This option is accepted but behavior may differ on the Agent SDK backend.',
      });
      continue;
    }

    if (UNSUPPORTED_OPTIONS.has(optionName)) {
      warnings.push({
        type: 'unsupported',
        feature: `providerOptions.anthropic.${optionName}`,
        details: 'This option is not supported on the Agent SDK backend.',
      });
      continue;
    }

    warnings.push({
      type: 'other',
      message: `Unknown anthropic provider option '${optionName}' is ignored by this backend.`,
    });
  }

  return warnings;
};
