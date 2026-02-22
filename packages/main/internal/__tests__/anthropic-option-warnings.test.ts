import { describe, expect, test } from 'bun:test';

import { collectAnthropicProviderOptionWarnings } from '../anthropic-option-warnings';

describe('collectAnthropicProviderOptionWarnings', () => {
  test('returns empty when provider options are missing', () => {
    const warnings = collectAnthropicProviderOptionWarnings(undefined);
    expect(warnings).toEqual([]);
  });

  test('does not warn for mapped options', () => {
    const warnings = collectAnthropicProviderOptionWarnings({
      anthropic: {
        effort: 'low',
        thinking: { type: 'enabled', budgetTokens: 1024 },
      },
    });

    expect(warnings).toEqual([]);
  });

  test('warns for degraded, unsupported, and unknown options', () => {
    const warnings = collectAnthropicProviderOptionWarnings({
      anthropic: {
        sendReasoning: true,
        cacheControl: { type: 'ephemeral' },
        unknownOption: true,
      },
    });

    expect(warnings.length).toBe(3);

    const features = warnings
      .filter(warning => 'feature' in warning)
      .map(warning => warning.feature)
      .sort();

    expect(features).toEqual([
      'providerOptions.anthropic.cacheControl',
      'providerOptions.anthropic.sendReasoning',
    ]);

    const otherMessages = warnings
      .filter(warning => warning.type === 'other')
      .map(warning => warning.message);

    expect(otherMessages.length).toBe(1);
    expect(otherMessages[0]?.includes('unknownOption')).toBeTrue();
  });
});
