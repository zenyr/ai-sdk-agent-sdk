import { describe, expect, test } from 'bun:test';
import { anthropic as upstreamAnthropic } from '@ai-sdk/anthropic';

import {
  VERSION,
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
} from '../index';

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

describe('runtime surface contract', () => {
  test('root exports are available', () => {
    expect(typeof VERSION).toBe('string');
    expect(typeof anthropic).toBe('function');
    expect(typeof createAnthropic).toBe('function');
    expect(typeof forwardAnthropicContainerIdFromLastStep).toBe('function');
  });

  test('provider exposes same tool keys as upstream anthropic', () => {
    const localToolKeys = Object.keys(anthropic.tools).sort();
    const upstreamToolKeys = Object.keys(upstreamAnthropic.tools).sort();

    expect(localToolKeys).toEqual(upstreamToolKeys);
  });

  test('createAnthropic rejects apiKey and authToken together', () => {
    expect(() => {
      createAnthropic({ apiKey: 'api-key', authToken: 'auth-token' });
    }).toThrow();
  });

  test('provider specification version is v3', () => {
    expect(anthropic.specificationVersion).toBe('v3');

    const model = anthropic('claude-3-5-haiku-latest');

    expect(model.specificationVersion).toBe('v3');
    expect(model.provider).toBe('anthropic.messages');
    expect(model.modelId).toBe('claude-3-5-haiku-latest');
  });

  test('forward helper returns undefined with no container metadata', () => {
    const output = forwardAnthropicContainerIdFromLastStep({
      steps: [{}, { providerMetadata: {} }],
    });

    expect(output).toBeUndefined();
  });

  test('forward helper picks latest container id', () => {
    const output = forwardAnthropicContainerIdFromLastStep({
      steps: [
        {
          providerMetadata: {
            anthropic: {
              container: { id: 'container-old' },
            },
          },
        },
        {
          providerMetadata: {
            anthropic: {
              container: { id: 'container-new' },
            },
          },
        },
      ],
    });

    expect(output).toBeDefined();
    expect(isRecord(output)).toBeTrue();

    if (!isRecord(output)) {
      return;
    }

    const providerOptions = output.providerOptions;
    expect(isRecord(providerOptions)).toBeTrue();

    if (!isRecord(providerOptions)) {
      return;
    }

    const anthropicOptions = providerOptions.anthropic;
    expect(isRecord(anthropicOptions)).toBeTrue();

    if (!isRecord(anthropicOptions)) {
      return;
    }

    const container = anthropicOptions.container;
    expect(isRecord(container)).toBeTrue();

    if (!isRecord(container)) {
      return;
    }

    expect(container.id).toBe('container-new');
  });
});
