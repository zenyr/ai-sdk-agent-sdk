import { describe, expect, mock, test } from 'bun:test';

const buildMockResultUsage = () => {
  return {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
};

const buildSuccessfulResult = () => {
  return {
    type: 'result',
    subtype: 'success',
    stop_reason: 'end_turn',
    result: 'ok',
    usage: buildMockResultUsage(),
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

const readEnvFromFirstQueryCall = (
  queryCalls: unknown[],
): Record<string, unknown> | undefined => {
  const firstCall = queryCalls[0];
  if (!isRecord(firstCall)) {
    return undefined;
  }

  const options = firstCall.options;
  if (!isRecord(options)) {
    return undefined;
  }

  const env = options.env;
  if (!isRecord(env)) {
    return undefined;
  }

  return env;
};

const readPromptFromFirstQueryCall = (queryCalls: unknown[]): string | undefined => {
  const firstCall = queryCalls[0];
  if (!isRecord(firstCall)) {
    return undefined;
  }

  const prompt = firstCall.prompt;
  if (typeof prompt !== 'string') {
    return undefined;
  }

  return prompt;
};

const readOutputFormatFromFirstQueryCall = (
  queryCalls: unknown[],
): Record<string, unknown> | undefined => {
  const firstCall = queryCalls[0];
  if (!isRecord(firstCall)) {
    return undefined;
  }

  const options = firstCall.options;
  if (!isRecord(options)) {
    return undefined;
  }

  const outputFormat = options.outputFormat;
  if (!isRecord(outputFormat)) {
    return undefined;
  }

  return outputFormat;
};

const readOptionsFromFirstQueryCall = (
  queryCalls: unknown[],
): Record<string, unknown> | undefined => {
  const firstCall = queryCalls[0];
  if (!isRecord(firstCall)) {
    return undefined;
  }

  const options = firstCall.options;
  if (!isRecord(options)) {
    return undefined;
  }

  return options;
};

const importIndexWithMockedQuery = async (args: {
  queryCalls: unknown[];
  resultFactory?: () => unknown;
  messagesFactory?: () => unknown[];
}) => {
  mock.module('@anthropic-ai/claude-agent-sdk', () => {
    return {
      createSdkMcpServer: (options: { name: string }) => {
        return {
          type: 'sdk',
          name: options.name,
          instance: {},
        };
      },
      query: async function* (request: unknown) {
        args.queryCalls.push(request);

        const messages = args.messagesFactory?.();
        if (Array.isArray(messages)) {
          for (const message of messages) {
            yield message;
          }

          return;
        }

        yield args.resultFactory?.() ?? buildSuccessfulResult();
      },
    };
  });

  const moduleId = `../index.ts?provider-settings-${Date.now()}-${Math.random()}`;
  return import(moduleId);
};

describe('provider settings contract', () => {
  test('forwards baseURL and apiKey into claude-agent-sdk env', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      apiKey: 'api-key-test',
      baseURL: 'https://proxy.example/v1/',
    });

    const model = provider('claude-3-5-haiku-latest');
    await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say hello.' }],
        },
      ],
    });

    const env = readEnvFromFirstQueryCall(queryCalls);
    expect(env).toBeDefined();

    if (env === undefined) {
      return;
    }

    expect(env.ANTHROPIC_API_KEY).toBe('api-key-test');
    expect(env.ANTHROPIC_BASE_URL).toBe('https://proxy.example/v1');
  });

  test('forwards authToken into claude-agent-sdk env', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      authToken: 'auth-token-test',
      baseURL: 'https://auth-proxy.example/v1/',
    });

    const model = provider('claude-3-5-haiku-latest');
    await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say hello.' }],
        },
      ],
    });

    const env = readEnvFromFirstQueryCall(queryCalls);
    expect(env).toBeDefined();

    if (env === undefined) {
      return;
    }

    expect(env.ANTHROPIC_AUTH_TOKEN).toBe('auth-token-test');
    expect(env.ANTHROPIC_BASE_URL).toBe('https://auth-proxy.example/v1');
  });

  test('preserves custom provider name', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      name: 'anthropic.proxy',
    });

    const model = provider('claude-3-5-haiku-latest');
    expect(model.provider).toBe('anthropic.proxy');
  });

  test('uses custom generateId for tool-call id generation', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: 'result',
          subtype: 'success',
          stop_reason: 'tool_use',
          result: '',
          structured_output: {
            type: 'tool-calls',
            calls: [
              {
                toolName: 'lookup_weather',
                input: {
                  city: 'seoul',
                },
              },
            ],
          },
          usage: buildMockResultUsage(),
        };
      },
    });

    const provider = createAnthropic({
      generateId: () => 'fixed-tool-call-id',
    });

    const model = provider('claude-3-5-haiku-latest');
    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Call tool.' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'lookup_weather',
          description: 'Lookup weather',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['city'],
            properties: {
              city: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('tool-call');

    if (firstContentPart === undefined || firstContentPart.type !== 'tool-call') {
      return;
    }

    expect(firstContentPart.toolCallId).toBe('fixed-tool-call-id');
  });

  test('tool mode uses in-process MCP bridge instead of output schema prompting', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Call tool when needed.' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'lookup_weather',
          description: 'Lookup weather',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['city'],
            properties: {
              city: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const prompt = readPromptFromFirstQueryCall(queryCalls);
    expect(prompt).toBeDefined();

    if (prompt === undefined) {
      return;
    }

    expect(prompt.includes('Call tool when needed.')).toBeTrue();
    expect(prompt.includes('You are in tool routing mode.')).toBeFalse();

    const outputFormat = readOutputFormatFromFirstQueryCall(queryCalls);
    expect(outputFormat).toBeUndefined();

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    if (Array.isArray(options.allowedTools) && options.allowedTools.length > 0) {
      expect(options.allowedTools).toContain(
        'mcp__ai_sdk_tool_bridge__lookup_weather',
      );

      const mcpServers = options.mcpServers;
      expect(isRecord(mcpServers)).toBeTrue();

      if (!isRecord(mcpServers)) {
        return;
      }

      const bridgeServer = mcpServers.ai_sdk_tool_bridge;
      expect(isRecord(bridgeServer)).toBeTrue();

      if (!isRecord(bridgeServer)) {
        return;
      }

      expect(bridgeServer.type).toBe('sdk');
      expect(bridgeServer.name).toBe('ai_sdk_tool_bridge');
    }
  });

  test('tool mode preserves configured thinking', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Call tool when needed.' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'lookup_weather',
          description: 'Lookup weather',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['city'],
            properties: {
              city: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
      providerOptions: {
        anthropic: {
          thinking: {
            type: 'adaptive',
          },
        },
      },
    });

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    expect(isRecord(options.thinking)).toBeTrue();

    if (!isRecord(options.thinking)) {
      return;
    }

    expect(options.thinking.type).toBe('adaptive');
  });

  test('tool mode returns explicit error for empty successful output', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      resultFactory: () => {
        return {
          type: 'result',
          subtype: 'success',
          stop_reason: 'end_turn',
          result: '',
          usage: buildMockResultUsage(),
        };
      },
    });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Call tool when needed.' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'lookup_weather',
          description: 'Lookup weather',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['city'],
            properties: {
              city: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    expect(result.finishReason.unified).toBe('error');
    expect(result.finishReason.raw).toBe('empty-tool-routing-output');

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('text');

    if (firstContentPart === undefined || firstContentPart.type !== 'text') {
      return;
    }

    expect(firstContentPart.text).toContain('Tool routing produced no tool call');
  });

  test('tool mode recovers native MCP tool-use from error_max_turns', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: 'assistant',
            message: {
              content: [
                {
                  type: 'tool_use',
                  id: 'toolu_native_1',
                  name: 'mcp__ai_sdk_tool_bridge__bash',
                  input: {
                    command: 'bun -e "console.log(Math.random())"',
                    description: 'Run Math.random once',
                  },
                },
              ],
            },
          },
          {
            type: 'result',
            subtype: 'error_max_turns',
            stop_reason: null,
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: [],
            uuid: 'uuid-native-tool-1',
            session_id: 'session-native-tool-1',
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'run bash' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'bash',
          description: 'Run shell command',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['command', 'description'],
            properties: {
              command: {
                type: 'string',
              },
              description: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('tool-call');

    if (firstContentPart === undefined || firstContentPart.type !== 'tool-call') {
      return;
    }

    expect(firstContentPart.toolCallId).toBe('toolu_native_1');
    expect(firstContentPart.toolName).toBe('bash');
    expect(firstContentPart.input).toContain('Math.random');
    expect(result.finishReason.unified).toBe('tool-calls');
    expect(result.finishReason.raw).toBe('tool_use');
  });

  test('tool mode recovers native MCP tool-use when query returns no result message', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: 'stream_event',
            event: {
              type: 'message_start',
              message: {
                id: 'msg-no-result-tool',
                model: 'mock-model',
              },
            },
          },
          {
            type: 'stream_event',
            event: {
              type: 'content_block_start',
              index: 0,
              content_block: {
                type: 'tool_use',
                id: 'toolu_no_result_1',
                name: 'mcp__ai_sdk_tool_bridge__bash',
              },
            },
          },
          {
            type: 'stream_event',
            event: {
              type: 'content_block_delta',
              index: 0,
              delta: {
                type: 'input_json_delta',
                partial_json:
                  '{"command":"bun -e \\\"console.log(Math.random())\\\"","description":"Run Math.random once"}',
              },
            },
          },
          {
            type: 'stream_event',
            event: {
              type: 'content_block_stop',
              index: 0,
            },
          },
          {
            type: 'stream_event',
            event: {
              type: 'message_delta',
              delta: {
                stop_reason: 'tool_use',
              },
              usage: {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'run bash' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'bash',
          description: 'Run shell command',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['command', 'description'],
            properties: {
              command: {
                type: 'string',
              },
              description: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('tool-call');

    if (firstContentPart === undefined || firstContentPart.type !== 'tool-call') {
      return;
    }

    expect(firstContentPart.toolCallId).toBe('toolu_no_result_1');
    expect(firstContentPart.toolName).toBe('bash');
    expect(firstContentPart.input).toContain('Math.random');
    expect(result.finishReason.unified).toBe('tool-calls');
    expect(result.finishReason.raw).toBe('tool_use');
  });

  test('tool mode recovers from structured output retry exhaustion in doGenerate', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: 'assistant',
            message: {
              content: [{ type: 'text', text: '{"type":"text","text":"안녕하세요"}' }],
            },
          },
          {
            type: 'result',
            subtype: 'error_max_structured_output_retries',
            stop_reason: 'end_turn',
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: [
              '[{"expected":"string","code":"invalid_type","path":["reason"]}]',
            ],
            uuid: 'uuid-1',
            session_id: 'session-1',
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'hello' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'lookup_weather',
          description: 'Lookup weather',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['city'],
            properties: {
              city: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('text');

    if (firstContentPart === undefined || firstContentPart.type !== 'text') {
      return;
    }

    expect(firstContentPart.text).toBe('안녕하세요');
    expect(result.finishReason.unified).toBe('stop');
  });

  test('tool mode recovers legacy single tool-call object from assistant text', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({
      queryCalls,
      messagesFactory: () => {
        return [
          {
            type: 'assistant',
            message: {
              content: [
                {
                  type: 'text',
                  text: '{"tool":"bash","parameters":{"command":"bun -e \\\"console.log(Math.random())\\\"","description":"Run Math.random once"}}',
                },
              ],
            },
          },
          {
            type: 'result',
            subtype: 'error_max_structured_output_retries',
            stop_reason: 'end_turn',
            duration_ms: 1,
            duration_api_ms: 1,
            is_error: true,
            num_turns: 1,
            total_cost_usd: 0,
            usage: buildMockResultUsage(),
            modelUsage: {},
            permission_denials: [],
            errors: [
              '[{"expected":"string","code":"invalid_type","path":["reason"]}]',
            ],
            uuid: 'uuid-legacy-tool-1',
            session_id: 'session-legacy-tool-1',
          },
        ];
      },
    });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'run bash' }],
        },
      ],
      tools: [
        {
          type: 'function',
          name: 'bash',
          description: 'Run shell command',
          inputSchema: {
            type: 'object',
            additionalProperties: false,
            required: ['command', 'description'],
            properties: {
              command: {
                type: 'string',
              },
              description: {
                type: 'string',
              },
            },
          },
        },
      ],
      toolChoice: { type: 'required' },
    });

    const firstContentPart = result.content[0];
    expect(firstContentPart?.type).toBe('tool-call');

    if (firstContentPart === undefined || firstContentPart.type !== 'tool-call') {
      return;
    }

    expect(firstContentPart.toolName).toBe('bash');
    expect(firstContentPart.input).toContain('Math.random');
    expect(result.finishReason.unified).toBe('tool-calls');
  });

  test('runs claude-agent-sdk in isolated no-tool mode', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({});
    const model = provider('claude-3-5-haiku-latest');

    await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say hello.' }],
        },
      ],
    });

    const options = readOptionsFromFirstQueryCall(queryCalls);
    expect(options).toBeDefined();

    if (options === undefined) {
      return;
    }

    expect(options.tools).toEqual([]);
    expect(options.allowedTools).toEqual([]);
    expect(options.settingSources).toEqual([]);
    expect(options.permissionMode).toBe('dontAsk');
    expect(options.maxTurns).toBe(1);
  });

  test('adds warnings for unsupported provider settings', async () => {
    const queryCalls: unknown[] = [];
    const { createAnthropic } = await importIndexWithMockedQuery({ queryCalls });

    const provider = createAnthropic({
      headers: {
        'x-test-header': 'enabled',
      },
      fetch: async (input, init) => {
        return fetch(input, init);
      },
    });

    const model = provider('claude-3-5-haiku-latest');
    const result = await model.doGenerate({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say hello.' }],
        },
      ],
    });

    const features = result.warnings
      .filter((warning) => 'feature' in warning)
      .map((warning) => warning.feature);

    expect(features.includes('providerSettings.headers')).toBeTrue();
    expect(features.includes('providerSettings.fetch')).toBeTrue();
  });
});
