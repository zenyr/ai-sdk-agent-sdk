import {
  InvalidArgumentError,
  type JSONObject,
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
  type LanguageModelV3Content,
  type LanguageModelV3FinishReason,
  type LanguageModelV3FunctionTool,
  type LanguageModelV3GenerateResult,
  type LanguageModelV3Message,
  type LanguageModelV3Prompt,
  type LanguageModelV3ProviderTool,
  type LanguageModelV3StreamPart,
  type LanguageModelV3StreamResult,
  type LanguageModelV3Usage,
  NoSuchModelError,
  type SharedV3Warning,
} from '@ai-sdk/provider';
import { anthropic as upstreamAnthropic } from '@ai-sdk/anthropic';
import { generateId } from '@ai-sdk/provider-utils';
import {
  query,
  type Options as AgentQueryOptions,
  type SDKAssistantMessage,
  type SDKMessage,
  type SDKResultMessage,
  type ThinkingConfig,
} from '@anthropic-ai/claude-agent-sdk';

import { collectAnthropicProviderOptionWarnings } from './internal/anthropic-option-warnings';

import type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from '@ai-sdk/anthropic';

type ParsedAnthropicProviderOptions = {
  effort?: 'low' | 'medium' | 'high' | 'max';
  thinking?: ThinkingConfig;
};

type CompletionMode =
  | {
      type: 'plain-text';
    }
  | {
      type: 'json';
      schema: Record<string, unknown>;
    }
  | {
      type: 'tools';
      schema: Record<string, unknown>;
      tools: LanguageModelV3FunctionTool[];
    };

type StructuredToolCall = {
  toolName: string;
  input: unknown;
};

type StructuredToolEnvelope = {
  type: 'tool-calls';
  calls: StructuredToolCall[];
};

type StructuredTextEnvelope = {
  type: 'text';
  text: string;
};

const DEFAULT_SUPPORTED_URLS: Record<string, RegExp[]> = {
  'image/*': [/^https?:\/\/.*$/],
  'application/pdf': [/^https?:\/\/.*$/],
};

const VERSION = '0.0.1';

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

const readRecord = (
  record: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined => {
  const value = record[key];
  if (!isRecord(value)) {
    return undefined;
  }

  return value;
};

const readString = (
  record: Record<string, unknown>,
  key: string,
): string | undefined => {
  const value = record[key];
  if (typeof value !== 'string') {
    return undefined;
  }

  return value;
};

const readNumber = (
  record: Record<string, unknown>,
  key: string,
): number | undefined => {
  const value = record[key];
  if (typeof value !== 'number') {
    return undefined;
  }

  return value;
};

const readArray = (
  record: Record<string, unknown>,
  key: string,
): unknown[] | undefined => {
  const value = record[key];
  if (!Array.isArray(value)) {
    return undefined;
  }

  return value;
};

const safeJsonStringify = (value: unknown): string => {
  try {
    return JSON.stringify(value);
  } catch {
    return 'null';
  }
};

const isFunctionTool = (
  tool: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool,
): tool is LanguageModelV3FunctionTool => {
  return tool.type === 'function';
};

const isAssistantMessage = (message: SDKMessage): message is SDKAssistantMessage => {
  return message.type === 'assistant';
};

const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === 'result';
};

const isStructuredTextEnvelope = (value: unknown): value is StructuredTextEnvelope => {
  if (!isRecord(value)) {
    return false;
  }

  return value.type === 'text' && typeof value.text === 'string';
};

const isStructuredToolEnvelope = (value: unknown): value is StructuredToolEnvelope => {
  if (!isRecord(value)) {
    return false;
  }

  if (value.type !== 'tool-calls' || !Array.isArray(value.calls)) {
    return false;
  }

  return value.calls.every(call => {
    if (!isRecord(call)) {
      return false;
    }

    return typeof call.toolName === 'string' && 'input' in call;
  });
};

const parseAnthropicProviderOptions = (
  options: LanguageModelV3CallOptions,
): ParsedAnthropicProviderOptions => {
  const providerOptions = options.providerOptions;
  if (!isRecord(providerOptions)) {
    return {};
  }

  const anthropicOptions = readRecord(providerOptions, 'anthropic');
  if (anthropicOptions === undefined) {
    return {};
  }

  const parsed: ParsedAnthropicProviderOptions = {};

  const effort = readString(anthropicOptions, 'effort');
  if (
    effort === 'low' ||
    effort === 'medium' ||
    effort === 'high' ||
    effort === 'max'
  ) {
    parsed.effort = effort;
  }

  const thinkingRecord = readRecord(anthropicOptions, 'thinking');
  if (thinkingRecord !== undefined) {
    const thinkingType = readString(thinkingRecord, 'type');

    if (thinkingType === 'adaptive') {
      parsed.thinking = { type: 'adaptive' };
    }

    if (thinkingType === 'disabled') {
      parsed.thinking = { type: 'disabled' };
    }

    if (thinkingType === 'enabled') {
      const budgetTokens = readNumber(thinkingRecord, 'budgetTokens');

      if (typeof budgetTokens === 'number') {
        parsed.thinking = {
          type: 'enabled',
          budgetTokens,
        };
      }

      if (typeof budgetTokens !== 'number') {
        parsed.thinking = {
          type: 'enabled',
        };
      }
    }
  }

  return parsed;
};

const mapFinishReason = (rawFinishReason: string | null): LanguageModelV3FinishReason => {
  if (rawFinishReason === 'tool_use') {
    return { unified: 'tool-calls', raw: rawFinishReason };
  }

  if (
    rawFinishReason === 'max_tokens' ||
    rawFinishReason === 'model_context_window_exceeded'
  ) {
    return { unified: 'length', raw: rawFinishReason };
  }

  if (rawFinishReason === 'refusal') {
    return { unified: 'content-filter', raw: rawFinishReason };
  }

  if (rawFinishReason === 'pause_turn' || rawFinishReason === 'compaction') {
    return { unified: 'other', raw: rawFinishReason };
  }

  if (rawFinishReason === null) {
    return { unified: 'stop', raw: undefined };
  }

  return { unified: 'stop', raw: rawFinishReason };
};

const mapUsage = (resultMessage: SDKResultMessage): LanguageModelV3Usage => {
  const totalInput = resultMessage.usage.input_tokens;
  const cacheRead = resultMessage.usage.cache_read_input_tokens;
  const cacheWrite = resultMessage.usage.cache_creation_input_tokens;
  const noCache = totalInput - cacheRead - cacheWrite;

  return {
    inputTokens: {
      total: totalInput,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: resultMessage.usage.output_tokens,
      text: resultMessage.usage.output_tokens,
      reasoning: undefined,
    },
  };
};

const mapIterations = (resultMessage: SDKResultMessage): AnthropicUsageIteration[] | null => {
  const iterations = resultMessage.usage.iterations;
  if (!Array.isArray(iterations)) {
    return null;
  }

  const mapped = iterations
    .map(iteration => {
      if (!isRecord(iteration)) {
        return undefined;
      }

      const type = readString(iteration, 'type');
      const inputTokens = readNumber(iteration, 'input_tokens');
      const outputTokens = readNumber(iteration, 'output_tokens');

      if (
        (type !== 'compaction' && type !== 'message') ||
        typeof inputTokens !== 'number' ||
        typeof outputTokens !== 'number'
      ) {
        return undefined;
      }

      return {
        type,
        inputTokens,
        outputTokens,
      };
    })
    .filter((value): value is AnthropicUsageIteration => value !== undefined);

  return mapped.length > 0 ? mapped : null;
};

const mapMetadata = (resultMessage: SDKResultMessage): AnthropicMessageMetadata => {
  return {
    usage: {},
    cacheCreationInputTokens: resultMessage.usage.cache_creation_input_tokens,
    stopSequence: null,
    iterations: mapIterations(resultMessage),
    container: null,
    contextManagement: null,
  };
};

const contentPartToText = (part: unknown): string => {
  if (!isRecord(part)) {
    return safeJsonStringify(part);
  }

  const type = readString(part, 'type');
  if (type === 'text') {
    const text = readString(part, 'text');
    return typeof text === 'string' ? text : '';
  }

  if (type === 'file') {
    const mediaType = readString(part, 'mediaType') ?? 'application/octet-stream';
    return `[file:${mediaType}]`;
  }

  if (type === 'reasoning') {
    const text = readString(part, 'text');
    return typeof text === 'string' ? `[reasoning]\n${text}` : '[reasoning]';
  }

  if (type === 'tool-call') {
    const toolName = readString(part, 'toolName') ?? 'unknown_tool';
    const input = part.input;
    return `[tool-call:${toolName}] ${safeJsonStringify(input)}`;
  }

  if (type === 'tool-result') {
    const toolName = readString(part, 'toolName') ?? 'unknown_tool';
    const output = part.output;
    return `[tool-result:${toolName}] ${safeJsonStringify(output)}`;
  }

  return safeJsonStringify(part);
};

const serializeMessage = (message: LanguageModelV3Message): string => {
  if (message.role === 'system') {
    return `[system]\n${message.content}`;
  }

  const serializedContent = message.content.map(contentPartToText).join('\n');
  return `[${message.role}]\n${serializedContent}`;
};

const serializePrompt = (prompt: LanguageModelV3Prompt): string => {
  return prompt.map(serializeMessage).join('\n\n');
};

const buildToolInstruction = (
  tools: LanguageModelV3FunctionTool[],
  toolChoice: LanguageModelV3CallOptions['toolChoice'],
): string => {
  const toolLines = tools
    .map(toolDefinition => {
      const description = toolDefinition.description ?? 'No description';
      const schema = safeJsonStringify(toolDefinition.inputSchema);

      return `- ${toolDefinition.name}: ${description}\n  schema: ${schema}`;
    })
    .join('\n');

  let toolChoiceInstruction = 'Choose tools automatically when necessary.';
  if (toolChoice?.type === 'required') {
    toolChoiceInstruction = 'You must return at least one tool call.';
  }

  if (toolChoice?.type === 'tool') {
    toolChoiceInstruction = `You must call exactly this tool: ${toolChoice.toolName}.`;
  }

  return [
    'You are in tool routing mode.',
    toolChoiceInstruction,
    'Return strictly valid JSON and no markdown.',
    'If a tool is needed, return {"type":"tool-calls","calls":[{"toolName":"...","input":{...}}]}.',
    'If no tool is needed, return {"type":"text","text":"..."}.',
    'Available tools:',
    toolLines,
  ].join('\n');
};

const buildToolSchema = (
  tools: LanguageModelV3FunctionTool[],
  toolChoice: LanguageModelV3CallOptions['toolChoice'],
): Record<string, unknown> => {
  const filteredTools =
    toolChoice?.type === 'tool'
      ? tools.filter(toolDefinition => toolDefinition.name === toolChoice.toolName)
      : tools;

  const requiresAtLeastOneCall =
    toolChoice?.type === 'required' || toolChoice?.type === 'tool';

  const callVariants = filteredTools.map(toolDefinition => {
    return {
      type: 'object',
      additionalProperties: false,
      required: ['toolName', 'input'],
      properties: {
        toolName: { const: toolDefinition.name },
        input: toolDefinition.inputSchema,
      },
    };
  });

  return {
    type: 'object',
    oneOf: [
      {
        type: 'object',
        additionalProperties: false,
        required: ['type', 'text'],
        properties: {
          type: { const: 'text' },
          text: { type: 'string' },
        },
      },
      {
        type: 'object',
        additionalProperties: false,
        required: ['type', 'calls'],
        properties: {
          type: { const: 'tool-calls' },
          calls: {
            type: 'array',
            minItems: requiresAtLeastOneCall ? 1 : undefined,
            items:
              callVariants.length > 0
                ? { oneOf: callVariants }
                : { type: 'object', additionalProperties: true },
          },
        },
      },
    ],
  };
};

const buildCompletionMode = (options: LanguageModelV3CallOptions): CompletionMode => {
  const tools = options.tools?.filter(isFunctionTool) ?? [];
  const hasToolMode = tools.length > 0 && options.toolChoice?.type !== 'none';

  if (hasToolMode) {
    return {
      type: 'tools',
      schema: buildToolSchema(tools, options.toolChoice),
      tools,
    };
  }

  if (options.responseFormat?.type === 'json') {
    const schema =
      isRecord(options.responseFormat.schema) ? options.responseFormat.schema : {};

    return {
      type: 'json',
      schema,
    };
  }

  return {
    type: 'plain-text',
  };
};

const extractAssistantText = (assistantMessage: SDKAssistantMessage | undefined): string => {
  if (assistantMessage === undefined) {
    return '';
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return '';
  }

  const text = contentBlocks
    .map(block => {
      if (!isRecord(block)) {
        return '';
      }

      if (block.type !== 'text') {
        return '';
      }

      const textPart = readString(block, 'text');
      return typeof textPart === 'string' ? textPart : '';
    })
    .join('');

  return text;
};

const collectWarnings = (
  options: LanguageModelV3CallOptions,
  mode: CompletionMode,
): SharedV3Warning[] => {
  const warnings: SharedV3Warning[] = collectAnthropicProviderOptionWarnings(
    options.providerOptions,
  );

  if (typeof options.temperature === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'temperature',
      details: 'claude-agent-sdk backend does not expose direct temperature control.',
    });
  }

  if (typeof options.topP === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'topP',
      details: 'claude-agent-sdk backend does not expose direct topP control.',
    });
  }

  if (typeof options.topK === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'topK',
      details: 'claude-agent-sdk backend does not expose direct topK control.',
    });
  }

  if (typeof options.presencePenalty === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'presencePenalty',
      details:
        'claude-agent-sdk backend does not expose direct presence penalty control.',
    });
  }

  if (typeof options.frequencyPenalty === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'frequencyPenalty',
      details:
        'claude-agent-sdk backend does not expose direct frequency penalty control.',
    });
  }

  if (typeof options.seed === 'number') {
    warnings.push({
      type: 'unsupported',
      feature: 'seed',
      details: 'claude-agent-sdk backend does not expose deterministic seed control.',
    });
  }

  if (typeof options.maxOutputTokens === 'number') {
    warnings.push({
      type: 'compatibility',
      feature: 'maxOutputTokens',
      details:
        'maxOutputTokens is best-effort only because claude-agent-sdk controls decoding internally.',
    });
  }

  if (
    mode.type === 'tools' &&
    options.tools !== undefined &&
    options.tools.some(
      (toolDefinition: LanguageModelV3FunctionTool | LanguageModelV3ProviderTool) => {
        return toolDefinition.type === 'provider';
      },
    )
  ) {
    warnings.push({
      type: 'unsupported',
      feature: 'provider-defined tools',
      details:
        'provider-defined tools are ignored when using claude-agent-sdk compatibility backend.',
    });
  }

  return warnings;
};

const contentToStreamParts = (
  content: LanguageModelV3Content[],
): LanguageModelV3StreamPart[] => {
  const streamParts: LanguageModelV3StreamPart[] = [];

  for (const contentPart of content) {
    if (contentPart.type === 'text') {
      const textId = generateId();
      streamParts.push({ type: 'text-start', id: textId });
      streamParts.push({ type: 'text-delta', id: textId, delta: contentPart.text });
      streamParts.push({ type: 'text-end', id: textId });
      continue;
    }

    if (contentPart.type === 'reasoning') {
      const reasoningId = generateId();
      streamParts.push({ type: 'reasoning-start', id: reasoningId });
      streamParts.push({
        type: 'reasoning-delta',
        id: reasoningId,
        delta: contentPart.text,
      });
      streamParts.push({ type: 'reasoning-end', id: reasoningId });
      continue;
    }

    streamParts.push(contentPart);
  }

  return streamParts;
};

const buildProviderMetadata = (
  resultMessage: SDKResultMessage,
): Record<string, JSONObject> => {
  const metadata = mapMetadata(resultMessage);

  return {
    anthropic: {
      usage: metadata.usage,
      cacheCreationInputTokens: metadata.cacheCreationInputTokens,
      stopSequence: metadata.stopSequence,
      iterations:
        metadata.iterations?.map(iteration => {
          return {
            type: iteration.type,
            inputTokens: iteration.inputTokens,
            outputTokens: iteration.outputTokens,
          };
        }) ?? null,
      container: null,
      contextManagement: null,
    },
  };
};

class AgentSdkAnthropicLanguageModel implements LanguageModelV3 {
  readonly specificationVersion: 'v3' = 'v3';
  readonly provider: string;
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]>;

  private readonly settings: AnthropicProviderSettings;
  private readonly idGenerator: () => string;

  constructor(args: {
    modelId: string;
    provider: string;
    settings: AnthropicProviderSettings;
    idGenerator: () => string;
  }) {
    this.modelId = args.modelId;
    this.provider = args.provider;
    this.settings = args.settings;
    this.idGenerator = args.idGenerator;
    this.supportedUrls = DEFAULT_SUPPORTED_URLS;
  }

  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    const completionMode = buildCompletionMode(options);
    const anthropicOptions = parseAnthropicProviderOptions(options);

    const warnings = collectWarnings(options, completionMode);

    const basePrompt = serializePrompt(options.prompt);
    let prompt = basePrompt;
    let outputFormat: AgentQueryOptions['outputFormat'];

    if (completionMode.type === 'tools') {
      prompt = `${buildToolInstruction(completionMode.tools, options.toolChoice)}\n\n${basePrompt}`;
      outputFormat = {
        type: 'json_schema',
        schema: completionMode.schema,
      };
    }

    if (completionMode.type === 'json') {
      prompt = `Return only JSON that matches the required schema.\n\n${basePrompt}`;
      outputFormat = {
        type: 'json_schema',
        schema: completionMode.schema,
      };
    }

    const abortController = new AbortController();
    const externalAbortSignal = options.abortSignal;
    const abortFromExternalSignal = () => {
      abortController.abort();
    };

    if (externalAbortSignal !== undefined) {
      if (externalAbortSignal.aborted) {
        abortController.abort();
      }

      externalAbortSignal.addEventListener('abort', abortFromExternalSignal, {
        once: true,
      });
    }

    const env: Record<string, string | undefined> = {
      ...process.env,
    };

    if (typeof this.settings.apiKey === 'string' && this.settings.apiKey.length > 0) {
      env.ANTHROPIC_API_KEY = this.settings.apiKey;
    }

    if (
      typeof this.settings.authToken === 'string' &&
      this.settings.authToken.length > 0
    ) {
      env.ANTHROPIC_AUTH_TOKEN = this.settings.authToken;
    }

    const queryOptions: AgentQueryOptions = {
      model: this.modelId,
      tools: [],
      permissionMode: 'dontAsk',
      maxTurns: 1,
      abortController,
      env,
      outputFormat,
      effort: anthropicOptions.effort,
      thinking: anthropicOptions.thinking,
      cwd: process.cwd(),
    };

    let lastAssistantMessage: SDKAssistantMessage | undefined;
    let finalResultMessage: SDKResultMessage | undefined;

    try {
      for await (const message of query({ prompt, options: queryOptions })) {
        if (isAssistantMessage(message)) {
          lastAssistantMessage = message;
        }

        if (isResultMessage(message)) {
          finalResultMessage = message;
        }
      }
    } finally {
      if (externalAbortSignal !== undefined) {
        externalAbortSignal.removeEventListener('abort', abortFromExternalSignal);
      }
    }

    if (finalResultMessage === undefined) {
      const emptyUsage: LanguageModelV3Usage = {
        inputTokens: {
          total: undefined,
          noCache: undefined,
          cacheRead: undefined,
          cacheWrite: undefined,
        },
        outputTokens: {
          total: undefined,
          text: undefined,
          reasoning: undefined,
        },
      };

      return {
        content: [{ type: 'text', text: '' }],
        finishReason: {
          unified: 'error',
          raw: 'agent-sdk-no-result',
        },
        usage: emptyUsage,
        warnings,
      };
    }

    const usage = mapUsage(finalResultMessage);
    const providerMetadata = buildProviderMetadata(finalResultMessage);

    let content: LanguageModelV3Content[] = [];
    let finishReason = mapFinishReason(finalResultMessage.stop_reason);

    if (finalResultMessage.subtype === 'success') {
      const structuredOutput = finalResultMessage.structured_output;

      if (
        completionMode.type === 'tools' &&
        isStructuredToolEnvelope(structuredOutput)
      ) {
        const toolCalls = structuredOutput.calls.map(call => {
          const toolCall: LanguageModelV3Content = {
            type: 'tool-call',
            toolCallId: this.idGenerator(),
            toolName: call.toolName,
            input: safeJsonStringify(call.input),
            providerExecuted: false,
          };

          return toolCall;
        });

        if (toolCalls.length > 0) {
          content = toolCalls;
          finishReason = {
            unified: 'tool-calls',
            raw: 'tool_use',
          };
        }
      }

      if (
        content.length === 0 &&
        completionMode.type === 'tools' &&
        isStructuredTextEnvelope(structuredOutput)
      ) {
        content = [{ type: 'text', text: structuredOutput.text }];
      }

      if (content.length === 0 && completionMode.type === 'json') {
        if (structuredOutput !== undefined) {
          content = [{ type: 'text', text: safeJsonStringify(structuredOutput) }];
        }
      }

      if (content.length === 0) {
        const assistantText = extractAssistantText(lastAssistantMessage);
        if (assistantText.length > 0) {
          content = [{ type: 'text', text: assistantText }];
        }
      }

      if (content.length === 0) {
        content = [{ type: 'text', text: finalResultMessage.result }];
      }
    }

    if (finalResultMessage.subtype !== 'success') {
      const errorText = finalResultMessage.errors.join('\n');
      content = [{ type: 'text', text: errorText }];
      finishReason = {
        unified: 'error',
        raw: finalResultMessage.subtype,
      };
    }

    return {
      content,
      finishReason,
      usage,
      warnings,
      providerMetadata,
      request: {
        body: {
          prompt,
          completionMode: completionMode.type,
        },
      },
      response: {
        modelId: this.modelId,
        timestamp: new Date(),
      },
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const generated = await this.doGenerate(options);
    const streamParts: LanguageModelV3StreamPart[] = [
      {
        type: 'stream-start',
        warnings: generated.warnings,
      },
      {
        type: 'response-metadata',
        modelId: generated.response?.modelId,
        timestamp: generated.response?.timestamp,
        id: generated.response?.id,
      },
      ...contentToStreamParts(generated.content),
      {
        type: 'finish',
        usage: generated.usage,
        finishReason: generated.finishReason,
        providerMetadata: generated.providerMetadata,
      },
    ];

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: controller => {
        for (const streamPart of streamParts) {
          controller.enqueue(streamPart);
        }

        controller.close();
      },
    });

    return {
      stream,
      request: generated.request,
      response: {
        headers: undefined,
      },
    };
  }
}

const createLanguageModelFactory = (args: {
  providerName: string;
  settings: AnthropicProviderSettings;
  idGenerator: () => string;
}) => {
  return (modelId: string): LanguageModelV3 => {
    return new AgentSdkAnthropicLanguageModel({
      modelId,
      provider: args.providerName,
      settings: args.settings,
      idGenerator: args.idGenerator,
    });
  };
};

const anthropicTools = upstreamAnthropic.tools;

const createAnthropic = (options: AnthropicProviderSettings = {}): AnthropicProvider => {
  if (
    typeof options.apiKey === 'string' &&
    options.apiKey.length > 0 &&
    typeof options.authToken === 'string' &&
    options.authToken.length > 0
  ) {
    throw new InvalidArgumentError({
      argument: 'apiKey/authToken',
      message:
        'Both apiKey and authToken were provided. Please use only one authentication method.',
    });
  }

  const providerName =
    typeof options.name === 'string' && options.name.length > 0
      ? options.name
      : 'anthropic.messages';

  const idGenerator =
    typeof options.generateId === 'function' ? options.generateId : generateId;

  const createLanguageModel = createLanguageModelFactory({
    providerName,
    settings: options,
    idGenerator,
  });

  const specificationVersion: 'v3' = 'v3';

  const provider: AnthropicProvider = Object.assign(
    (modelId: string) => createLanguageModel(modelId),
    {
      specificationVersion,
      languageModel: createLanguageModel,
      chat: createLanguageModel,
      messages: createLanguageModel,
      embeddingModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: 'embeddingModel',
        });
      },
      textEmbeddingModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: 'embeddingModel',
        });
      },
      imageModel: (modelId: string) => {
        throw new NoSuchModelError({
          modelId,
          modelType: 'imageModel',
        });
      },
      tools: anthropicTools,
    },
  );

  return provider;
};

const anthropic = createAnthropic();

const forwardAnthropicContainerIdFromLastStep = ({
  steps,
}: {
  steps: Array<{
    providerMetadata?: Record<string, JSONObject>;
  }>;
}): undefined | { providerOptions?: Record<string, JSONObject> } => {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step === undefined) {
      continue;
    }

    const metadata = step.providerMetadata;
    if (metadata === undefined) {
      continue;
    }

    const anthropicMetadata = metadata.anthropic;
    if (!isRecord(anthropicMetadata)) {
      continue;
    }

    const container = readRecord(anthropicMetadata, 'container');
    if (container === undefined) {
      continue;
    }

    const containerId = readString(container, 'id');
    if (typeof containerId !== 'string') {
      continue;
    }

    return {
      providerOptions: {
        anthropic: {
          container: {
            id: containerId,
          },
        },
      },
    };
  }

  return undefined;
};

type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export {
  VERSION,
  anthropic,
  createAnthropic,
  forwardAnthropicContainerIdFromLastStep,
};

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderOptions,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
};
