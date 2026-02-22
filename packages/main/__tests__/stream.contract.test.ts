import { describe, expect, mock, test } from 'bun:test';

const buildMockResultUsage = () => {
  return {
    input_tokens: 10,
    output_tokens: 5,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  };
};

describe('stream bridge contract', () => {
  test('doStream emits metadata from stream events', async () => {
    mock.module('@anthropic-ai/claude-agent-sdk', () => {
      return {
        query: async function* () {
          yield {
            type: 'stream_event',
            event: {
              type: 'message_start',
              message: {
                id: 'msg-1',
                model: 'mock-model',
              },
            },
          };

          yield {
            type: 'stream_event',
            event: {
              type: 'content_block_start',
              index: 0,
              content_block: {
                type: 'text',
              },
            },
          };

          yield {
            type: 'stream_event',
            event: {
              type: 'content_block_delta',
              index: 0,
              delta: {
                type: 'text_delta',
                text: 'hello',
              },
            },
          };

          yield {
            type: 'stream_event',
            event: {
              type: 'content_block_stop',
              index: 0,
            },
          };

          yield {
            type: 'stream_event',
            event: {
              type: 'message_delta',
              delta: {
                stop_reason: 'end_turn',
              },
              usage: {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
              },
            },
          };

          yield {
            type: 'result',
            subtype: 'success',
            stop_reason: 'end_turn',
            result: 'done',
            usage: buildMockResultUsage(),
          };
        },
      };
    });

    const moduleId = `../index.ts?stream-contract-${Date.now()}`;
    const { anthropic } = await import(moduleId);

    const streamResult = await anthropic('claude-3-5-haiku-latest').doStream({
      prompt: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'say hello' }],
        },
      ],
    });

    const parts: unknown[] = [];
    for await (const part of streamResult.stream) {
      parts.push(part);
    }

    const metadataPart = parts.find(part => {
      return (
        typeof part === 'object' &&
        part !== null &&
        'type' in part &&
        part.type === 'response-metadata'
      );
    });

    expect(metadataPart).toBeDefined();

    if (
      typeof metadataPart !== 'object' ||
      metadataPart === null ||
      !('id' in metadataPart) ||
      !('modelId' in metadataPart)
    ) {
      return;
    }

    expect(metadataPart.id).toBe('msg-1');
    expect(metadataPart.modelId).toBe('mock-model');

    const hasTextDelta = parts.some(part => {
      return (
        typeof part === 'object' &&
        part !== null &&
        'type' in part &&
        part.type === 'text-delta'
      );
    });

    expect(hasTextDelta).toBeTrue();

    const finishPart = parts.find(part => {
      return (
        typeof part === 'object' &&
        part !== null &&
        'type' in part &&
        part.type === 'finish'
      );
    });

    expect(finishPart).toBeDefined();
  });
});
