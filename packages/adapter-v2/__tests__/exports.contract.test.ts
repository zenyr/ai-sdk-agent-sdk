import { describe, expect, test } from 'bun:test';

import {
  VERSION,
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
} from '../index';

describe('adapter-v2 exports contract', () => {
  test('exports provider factory and helper', () => {
    expect(typeof VERSION).toBe('string');
    expect(typeof anthropic).toBe('function');
    expect(typeof createAnthropic).toBe('function');
    expect(typeof forwardAnthropicContainerIdFromLastStep).toBe('function');
  });

  test('createAnthropic returns callable provider', () => {
    const provider = createAnthropic({});
    expect(typeof provider).toBe('function');
  });
});
