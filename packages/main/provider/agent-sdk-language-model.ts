import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type {
  JSONObject,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3GenerateResult,
  LanguageModelV3Message,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import { withoutTrailingSlash } from "@ai-sdk/provider-utils";
import type {
  Options as AgentQueryOptions,
  SDKAssistantMessage,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";
import * as agentSdk from "@anthropic-ai/claude-agent-sdk";

import { buildCompletionMode } from "../bridge/completion-mode";
import {
  collectWarnings,
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseAnthropicProviderOptions,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../bridge/parse-utils";
import {
  extractSystemPromptFromMessages,
  joinSerializedPromptMessages,
  serializePromptMessages,
  serializePromptMessagesForResumeQuery,
  serializePromptMessagesWithoutSystem,
} from "../bridge/prompt-serializer";
import { buildProviderMetadata, mapFinishReason, mapUsage } from "../bridge/result-mapping";
import {
  appendStreamPartsFromRawEvent,
  closePendingStreamBlocks,
  enqueueSingleTextBlock,
} from "../bridge/stream-event-mapper";
import { buildZodRawShapeFromToolInputSchema } from "../bridge/tool-schema-to-zod-shape";
import { DEFAULT_SUPPORTED_URLS } from "../shared/constants";
import {
  createEmptyUsage,
  type StreamBlockState,
  type StreamEventState,
} from "../shared/stream-types";
import type { ToolExecutorMap } from "../shared/tool-executor";
import { isRecord, readString, safeJsonStringify } from "../shared/type-readers";

const isAssistantMessage = (message: SDKMessage): message is SDKAssistantMessage => {
  return message.type === "assistant";
};

const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === "result";
};

const isPartialAssistantMessage = (message: SDKMessage): message is SDKPartialAssistantMessage => {
  return message.type === "stream_event";
};

const extractAssistantText = (assistantMessage: SDKAssistantMessage | undefined): string => {
  if (assistantMessage === undefined) {
    return "";
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return "";
  }

  const text = contentBlocks
    .map((block) => {
      if (!isRecord(block)) {
        return "";
      }

      if (block.type !== "text") {
        return "";
      }

      const textPart = readString(block, "text");
      return typeof textPart === "string" ? textPart : "";
    })
    .join("");

  return text;
};

const readNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  if (value.length === 0) {
    return undefined;
  }

  return value;
};

const isStructuredOutputRetryExhausted = (resultMessage: SDKResultMessage): boolean => {
  return resultMessage.subtype === "error_max_structured_output_retries";
};

const EMPTY_TOOL_ROUTING_OUTPUT_ERROR = "empty-tool-routing-output";
const EMPTY_TOOL_ROUTING_OUTPUT_TEXT = "Tool routing produced no tool call or text response.";
const TOOL_BRIDGE_SERVER_NAME = "ai_sdk_tool_bridge";
const TOOL_BRIDGE_NAME_PREFIX = `mcp__${TOOL_BRIDGE_SERVER_NAME}__`;

type ToolBridgeConfig = {
  allowedTools: string[];
  mcpServers: NonNullable<AgentQueryOptions["mcpServers"]>;
  hasAnyExecutor: boolean;
  allToolsHaveExecutors: boolean;
  missingExecutorToolNames: string[];
};

type PromptSessionState = {
  sessionId: string;
  serializedPromptMessages: string[];
};

type PromptQueryInput = {
  prompt: string;
  serializedPromptMessages: string[];
  resumeSessionId?: string;
};

const MAX_PROMPT_SESSION_STATES = 20;

const toBridgeToolName = (toolName: string): string => {
  return `${TOOL_BRIDGE_NAME_PREFIX}${toolName}`;
};

const fromBridgeToolName = (toolName: string): string => {
  if (!toolName.startsWith(TOOL_BRIDGE_NAME_PREFIX)) {
    return toolName;
  }

  const mappedToolName = toolName.slice(TOOL_BRIDGE_NAME_PREFIX.length);
  return mappedToolName.length > 0 ? mappedToolName : toolName;
};

const isBridgeToolName = (toolName: string): boolean => {
  return toolName.startsWith(TOOL_BRIDGE_NAME_PREFIX);
};

const normalizeToolInputJson = (value: string): string => {
  const trimmedValue = value.trim();
  if (trimmedValue.length === 0) {
    return "{}";
  }

  try {
    const parsedValue: unknown = JSON.parse(trimmedValue);
    return safeJsonStringify(parsedValue);
  } catch {
    return trimmedValue;
  }
};

const hasSerializedPromptPrefix = (prefix: string[], target: string[]): boolean => {
  if (prefix.length > target.length) {
    return false;
  }

  for (let index = 0; index < prefix.length; index += 1) {
    if (prefix[index] !== target[index]) {
      return false;
    }
  }

  return true;
};

const hasIdenticalSerializedPrompt = (source: string[], target: string[]): boolean => {
  if (source.length !== target.length) {
    return false;
  }

  for (let index = 0; index < source.length; index += 1) {
    if (source[index] !== target[index]) {
      return false;
    }
  }

  return true;
};

const findBestPromptSessionState = (args: {
  serializedPromptMessages: string[];
  previousSessionStates: PromptSessionState[];
}): PromptSessionState | undefined => {
  let bestState: PromptSessionState | undefined;

  for (const sessionState of args.previousSessionStates) {
    if (
      !hasSerializedPromptPrefix(
        sessionState.serializedPromptMessages,
        args.serializedPromptMessages,
      )
    ) {
      continue;
    }

    if (
      bestState === undefined ||
      sessionState.serializedPromptMessages.length > bestState.serializedPromptMessages.length
    ) {
      bestState = sessionState;
    }
  }

  return bestState;
};

const mergePromptSessionState = (args: {
  previousSessionStates: PromptSessionState[];
  nextSessionState: PromptSessionState;
}): PromptSessionState[] => {
  const dedupedStates = args.previousSessionStates.filter((sessionState) => {
    if (sessionState.sessionId === args.nextSessionState.sessionId) {
      return false;
    }

    return !hasIdenticalSerializedPrompt(
      sessionState.serializedPromptMessages,
      args.nextSessionState.serializedPromptMessages,
    );
  });

  return [args.nextSessionState, ...dedupedStates].slice(0, MAX_PROMPT_SESSION_STATES);
};

const buildPromptQueryInput = (args: {
  promptMessages: LanguageModelV3Message[];
  previousSessionStates: PromptSessionState[];
}): PromptQueryInput => {
  const serializedPromptMessages = serializePromptMessages(args.promptMessages);
  const serializedPromptMessagesForQuery = serializePromptMessagesWithoutSystem(
    args.promptMessages,
  );
  const fullPrompt = joinSerializedPromptMessages(serializedPromptMessagesForQuery);
  const previousSessionState = findBestPromptSessionState({
    serializedPromptMessages,
    previousSessionStates: args.previousSessionStates,
  });

  if (previousSessionState === undefined) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const previousPromptMessages = previousSessionState.serializedPromptMessages;
  if (!hasSerializedPromptPrefix(previousPromptMessages, serializedPromptMessages)) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const appendedPromptMessages = serializedPromptMessages.slice(previousPromptMessages.length);
  if (appendedPromptMessages.length === 0) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const appendedSourceMessages = args.promptMessages.slice(previousPromptMessages.length);
  const appendedPromptMessagesForQuery =
    serializePromptMessagesForResumeQuery(appendedSourceMessages);

  if (appendedPromptMessagesForQuery.length === 0) {
    const fallbackAppendedPromptMessages =
      serializePromptMessagesWithoutSystem(appendedSourceMessages);

    if (fallbackAppendedPromptMessages.length === 0) {
      return {
        prompt: fullPrompt,
        serializedPromptMessages,
      };
    }

    return {
      prompt: joinSerializedPromptMessages(fallbackAppendedPromptMessages),
      serializedPromptMessages,
      resumeSessionId: previousSessionState.sessionId,
    };
  }

  return {
    prompt: joinSerializedPromptMessages(appendedPromptMessagesForQuery),
    serializedPromptMessages,
    resumeSessionId: previousSessionState.sessionId,
  };
};

const readSessionIdFromQueryMessages = (args: {
  resultMessage: SDKResultMessage | undefined;
  assistantMessage: SDKAssistantMessage | undefined;
}): string | undefined => {
  if (isRecord(args.resultMessage)) {
    const sessionIdFromResult = readString(args.resultMessage, "session_id");
    if (sessionIdFromResult !== undefined) {
      return sessionIdFromResult;
    }
  }

  if (!isRecord(args.assistantMessage)) {
    return undefined;
  }

  return readString(args.assistantMessage, "session_id");
};

const buildToolBridgeConfig = (
  tools: Array<{ name: string; description?: string; inputSchema: unknown }>,
  toolExecutors: ToolExecutorMap | undefined,
): ToolBridgeConfig | undefined => {
  if (tools.length === 0) {
    return undefined;
  }

  const createBridgeServer = agentSdk.createSdkMcpServer;
  if (typeof createBridgeServer !== "function") {
    return undefined;
  }

  const buildMcpTool = agentSdk.tool;

  const stringifyToolExecutorOutput = (value: unknown): string => {
    if (typeof value === "string") {
      return value;
    }

    return safeJsonStringify(value);
  };

  const stringifyToolExecutorError = (error: unknown): string => {
    if (error instanceof Error && error.message.length > 0) {
      return error.message;
    }

    if (typeof error === "string" && error.length > 0) {
      return error;
    }

    return safeJsonStringify(error);
  };

  const buildDisabledHandler = async () => {
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: "Provider-side execution is disabled for AI SDK bridge tools.",
        },
      ],
    };
  };

  const missingExecutorToolNames: string[] = [];
  let hasAnyExecutor = false;

  const mcpTools = tools.map((toolDefinition) => {
    const zodRawShape = buildZodRawShapeFromToolInputSchema(toolDefinition.inputSchema);
    const toolExecutor = toolExecutors?.[toolDefinition.name];

    if (toolExecutor !== undefined) {
      hasAnyExecutor = true;
    } else {
      missingExecutorToolNames.push(toolDefinition.name);
    }

    const toolHandler =
      toolExecutor === undefined
        ? buildDisabledHandler
        : async (args: unknown) => {
            const input = isRecord(args) ? args : {};

            try {
              const output = await toolExecutor(input);
              return {
                content: [
                  {
                    type: "text",
                    text: stringifyToolExecutorOutput(output),
                  },
                ],
              };
            } catch (error) {
              return {
                isError: true,
                content: [
                  {
                    type: "text",
                    text: stringifyToolExecutorError(error),
                  },
                ],
              };
            }
          };

    if (typeof buildMcpTool === "function") {
      return buildMcpTool(
        toolDefinition.name,
        toolDefinition.description ?? "No description",
        zodRawShape,
        toolHandler,
      );
    }

    return {
      name: toolDefinition.name,
      description: toolDefinition.description ?? "No description",
      inputSchema: zodRawShape,
      handler: toolHandler,
    };
  });

  const mcpServer = createBridgeServer({
    name: TOOL_BRIDGE_SERVER_NAME,
    tools: mcpTools,
  });

  return {
    allowedTools: tools.map((toolDefinition) => {
      return toBridgeToolName(toolDefinition.name);
    }),
    mcpServers: {
      [TOOL_BRIDGE_SERVER_NAME]: mcpServer,
    },
    hasAnyExecutor,
    allToolsHaveExecutors: hasAnyExecutor && missingExecutorToolNames.length === 0,
    missingExecutorToolNames,
  };
};

const hasToolModePrimaryContent = (content: LanguageModelV3Content[]): boolean => {
  return content.some((part) => {
    if (part.type === "tool-call") {
      return true;
    }

    if (part.type === "text") {
      return part.text.trim().length > 0;
    }

    return false;
  });
};

const recoverToolModeToolCallsFromAssistant = (
  assistantMessage: SDKAssistantMessage | undefined,
  idGenerator: () => string,
): LanguageModelV3Content[] => {
  if (assistantMessage === undefined) {
    return [];
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return [];
  }

  const toolCalls: LanguageModelV3Content[] = [];

  for (const block of contentBlocks) {
    if (!isRecord(block)) {
      continue;
    }

    const blockType = readString(block, "type");
    if (
      blockType !== "tool_use" &&
      blockType !== "mcp_tool_use" &&
      blockType !== "server_tool_use"
    ) {
      continue;
    }

    const rawToolName = readString(block, "name");
    if (rawToolName === undefined) {
      continue;
    }

    const toolCallId = readString(block, "id") ?? idGenerator();
    const inputValue = "input" in block ? block.input : {};

    toolCalls.push({
      type: "tool-call",
      toolCallId,
      toolName: fromBridgeToolName(rawToolName),
      input: safeJsonStringify(inputValue),
      providerExecuted: false,
    });
  }

  return toolCalls;
};

const recoverToolModeContentFromAssistantText = (
  assistantMessage: SDKAssistantMessage | undefined,
  idGenerator: () => string,
): LanguageModelV3Content[] => {
  const assistantText = extractAssistantText(assistantMessage);
  if (assistantText.length === 0) {
    return [];
  }

  const parsedEnvelope = parseStructuredEnvelopeFromText(assistantText);

  if (isStructuredToolEnvelope(parsedEnvelope)) {
    const toolCalls = mapStructuredToolCallsToContent(parsedEnvelope.calls, idGenerator);

    if (toolCalls.length > 0) {
      return toolCalls;
    }
  }

  if (isStructuredTextEnvelope(parsedEnvelope)) {
    return [{ type: "text", text: parsedEnvelope.text }];
  }

  return [{ type: "text", text: assistantText }];
};

const buildQueryEnv = (settings: AnthropicProviderSettings): Record<string, string | undefined> => {
  const env: Record<string, string | undefined> = {
    ...process.env,
  };

  const apiKey = readNonEmptyString(settings.apiKey);
  if (apiKey !== undefined) {
    env.ANTHROPIC_API_KEY = apiKey;
  }

  const authToken = readNonEmptyString(settings.authToken);
  if (authToken !== undefined) {
    env.ANTHROPIC_AUTH_TOKEN = authToken;
  }

  const baseURL = readNonEmptyString(settings.baseURL);
  if (baseURL !== undefined) {
    const normalizedBaseURL = withoutTrailingSlash(baseURL);
    if (normalizedBaseURL !== undefined) {
      env.ANTHROPIC_BASE_URL = normalizedBaseURL;
    }
  }

  return env;
};

const collectProviderSettingWarnings = (settings: AnthropicProviderSettings): SharedV3Warning[] => {
  const warnings: SharedV3Warning[] = [];

  const headers = settings.headers;
  if (isRecord(headers) && Object.keys(headers).length > 0) {
    warnings.push({
      type: "unsupported",
      feature: "providerSettings.headers",
      details: "createAnthropic({ headers }) is not forwarded on claude-agent-sdk backend.",
    });
  }

  if (typeof settings.fetch === "function") {
    warnings.push({
      type: "unsupported",
      feature: "providerSettings.fetch",
      details: "createAnthropic({ fetch }) is not forwarded on claude-agent-sdk backend.",
    });
  }

  return warnings;
};

const buildPartialToolExecutorWarning = (missingExecutorToolNames: string[]): SharedV3Warning => {
  return {
    type: "compatibility",
    feature: "toolExecutors.partial",
    details:
      `toolExecutors is missing handlers for: ${missingExecutorToolNames.join(", ")}. ` +
      "Falling back to AI SDK tool loop with maxTurns=1.",
  };
};

export class AgentSdkAnthropicLanguageModel implements LanguageModelV3 {
  readonly specificationVersion: "v3" = "v3";
  readonly provider: string;
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]>;

  private readonly settings: AnthropicProviderSettings;
  private readonly idGenerator: () => string;
  private readonly toolExecutors: ToolExecutorMap | undefined;
  private readonly maxTurns: number | undefined;
  private readonly providerSettingWarnings: SharedV3Warning[];
  private promptSessionStates: PromptSessionState[] = [];

  constructor(args: {
    modelId: string;
    provider: string;
    settings: AnthropicProviderSettings;
    idGenerator: () => string;
    toolExecutors?: ToolExecutorMap;
    maxTurns?: number;
  }) {
    this.modelId = args.modelId;
    this.provider = args.provider;
    this.settings = args.settings;
    this.idGenerator = args.idGenerator;
    this.toolExecutors = args.toolExecutors;
    this.maxTurns = args.maxTurns;
    this.providerSettingWarnings = collectProviderSettingWarnings(this.settings);
    this.supportedUrls = DEFAULT_SUPPORTED_URLS;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const completionMode = buildCompletionMode(options);
    const anthropicOptions = parseAnthropicProviderOptions(options);
    const warnings = [...collectWarnings(options, completionMode), ...this.providerSettingWarnings];

    const promptQueryInput = buildPromptQueryInput({
      promptMessages: options.prompt,
      previousSessionStates: this.promptSessionStates,
    });
    const basePrompt = promptQueryInput.prompt;
    const systemPrompt = extractSystemPromptFromMessages(options.prompt);
    let prompt = basePrompt;
    let outputFormat: AgentQueryOptions["outputFormat"];
    const toolBridgeConfig =
      completionMode.type === "tools"
        ? buildToolBridgeConfig(completionMode.tools, this.toolExecutors)
        : undefined;
    const useNativeToolExecution =
      completionMode.type === "tools" && toolBridgeConfig?.allToolsHaveExecutors === true;

    if (
      completionMode.type === "tools" &&
      toolBridgeConfig?.hasAnyExecutor === true &&
      !useNativeToolExecution
    ) {
      warnings.push(buildPartialToolExecutorWarning(toolBridgeConfig.missingExecutorToolNames));
    }

    if (completionMode.type === "json") {
      prompt = `Return only JSON that matches the required schema.\n\n${basePrompt}`;
      outputFormat = {
        type: "json_schema",
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

      externalAbortSignal.addEventListener("abort", abortFromExternalSignal, {
        once: true,
      });
    }

    const queryOptions: AgentQueryOptions = {
      model: this.modelId,
      tools: [],
      allowedTools: toolBridgeConfig?.allowedTools ?? [],
      resume: promptQueryInput.resumeSessionId,
      systemPrompt,
      permissionMode: "dontAsk",
      settingSources: [],
      maxTurns: useNativeToolExecution ? this.maxTurns : 1,
      abortController,
      env: buildQueryEnv(this.settings),
      hooks: {},
      plugins: [],
      mcpServers: toolBridgeConfig?.mcpServers,
      outputFormat,
      effort: anthropicOptions.effort,
      thinking: anthropicOptions.thinking,
      cwd: process.cwd(),
      includePartialMessages: completionMode.type === "tools",
    };

    let lastAssistantMessage: SDKAssistantMessage | undefined;
    let finalResultMessage: SDKResultMessage | undefined;
    const partialStreamState: StreamEventState = {
      blockStates: new Map<number, StreamBlockState>(),
      emittedResponseMetadata: false,
      latestStopReason: null,
      latestUsage: undefined,
    };
    const pendingBridgeToolInputs = new Map<string, { toolName: string; deltas: string[] }>();
    const recoveredToolCallsFromStream: LanguageModelV3Content[] = [];

    const appendRecoveredToolCall = (args: {
      toolCallId: string;
      toolName: string;
      rawInput: string;
    }) => {
      recoveredToolCallsFromStream.push({
        type: "tool-call",
        toolCallId: args.toolCallId,
        toolName: fromBridgeToolName(args.toolName),
        input: normalizeToolInputJson(args.rawInput),
        providerExecuted: useNativeToolExecution,
      });
    };

    try {
      for await (const message of agentSdk.query({
        prompt,
        options: queryOptions,
      })) {
        if (isPartialAssistantMessage(message)) {
          const mappedParts = appendStreamPartsFromRawEvent(message.event, partialStreamState);

          for (const mappedPart of mappedParts) {
            if (
              completionMode.type === "tools" &&
              !useNativeToolExecution &&
              mappedPart.type === "tool-input-start" &&
              isBridgeToolName(mappedPart.toolName)
            ) {
              pendingBridgeToolInputs.set(mappedPart.id, {
                toolName: mappedPart.toolName,
                deltas: [],
              });
              continue;
            }

            if (
              completionMode.type === "tools" &&
              !useNativeToolExecution &&
              mappedPart.type === "tool-input-delta"
            ) {
              const pendingBridgeInput = pendingBridgeToolInputs.get(mappedPart.id);

              if (pendingBridgeInput !== undefined) {
                pendingBridgeInput.deltas.push(mappedPart.delta);
              }

              continue;
            }

            if (
              completionMode.type === "tools" &&
              !useNativeToolExecution &&
              mappedPart.type === "tool-input-end"
            ) {
              const pendingBridgeInput = pendingBridgeToolInputs.get(mappedPart.id);

              if (pendingBridgeInput !== undefined) {
                appendRecoveredToolCall({
                  toolCallId: mappedPart.id,
                  toolName: pendingBridgeInput.toolName,
                  rawInput: pendingBridgeInput.deltas.join(""),
                });

                pendingBridgeToolInputs.delete(mappedPart.id);
              }
            }
          }

          continue;
        }

        if (isAssistantMessage(message)) {
          lastAssistantMessage = message;
        }

        if (isResultMessage(message)) {
          finalResultMessage = message;
        }
      }

      const remainingParts = closePendingStreamBlocks(partialStreamState);
      for (const remainingPart of remainingParts) {
        if (
          completionMode.type === "tools" &&
          !useNativeToolExecution &&
          remainingPart.type === "tool-input-end"
        ) {
          const pendingBridgeInput = pendingBridgeToolInputs.get(remainingPart.id);

          if (pendingBridgeInput !== undefined) {
            appendRecoveredToolCall({
              toolCallId: remainingPart.id,
              toolName: pendingBridgeInput.toolName,
              rawInput: pendingBridgeInput.deltas.join(""),
            });

            pendingBridgeToolInputs.delete(remainingPart.id);
          }
        }
      }
    } finally {
      if (externalAbortSignal !== undefined) {
        externalAbortSignal.removeEventListener("abort", abortFromExternalSignal);
      }
    }

    const sessionId = readSessionIdFromQueryMessages({
      resultMessage: finalResultMessage,
      assistantMessage: lastAssistantMessage,
    });

    if (sessionId !== undefined) {
      this.promptSessionStates = mergePromptSessionState({
        previousSessionStates: this.promptSessionStates,
        nextSessionState: {
          sessionId,
          serializedPromptMessages: promptQueryInput.serializedPromptMessages,
        },
      });
    }

    if (finalResultMessage === undefined) {
      if (completionMode.type === "tools" && !useNativeToolExecution) {
        if (recoveredToolCallsFromStream.length > 0) {
          return {
            content: recoveredToolCallsFromStream,
            finishReason: {
              unified: "tool-calls",
              raw: "tool_use",
            },
            usage: partialStreamState.latestUsage ?? createEmptyUsage(),
            warnings,
            request: {
              body: {
                prompt,
                systemPrompt,
                completionMode: completionMode.type,
              },
            },
            response: {
              modelId: this.modelId,
              timestamp: new Date(),
            },
          };
        }

        const recoveredToolCalls = recoverToolModeToolCallsFromAssistant(
          lastAssistantMessage,
          this.idGenerator,
        );

        if (recoveredToolCalls.length > 0) {
          return {
            content: recoveredToolCalls,
            finishReason: {
              unified: "tool-calls",
              raw: "tool_use",
            },
            usage: partialStreamState.latestUsage ?? createEmptyUsage(),
            warnings,
            request: {
              body: {
                prompt,
                systemPrompt,
                completionMode: completionMode.type,
              },
            },
            response: {
              modelId: this.modelId,
              timestamp: new Date(),
            },
          };
        }
      }

      return {
        content: [{ type: "text", text: extractAssistantText(lastAssistantMessage) }],
        finishReason: {
          unified: "error",
          raw: "agent-sdk-no-result",
        },
        usage: partialStreamState.latestUsage ?? createEmptyUsage(),
        warnings,
      };
    }

    const usage = mapUsage(finalResultMessage);
    const providerMetadata = buildProviderMetadata(finalResultMessage);

    let content: LanguageModelV3Content[] = [];
    let finishReason = mapFinishReason(finalResultMessage.stop_reason);

    if (completionMode.type === "tools" && useNativeToolExecution) {
      if (finalResultMessage.subtype === "success") {
        const assistantText = extractAssistantText(lastAssistantMessage);
        const text = assistantText.length > 0 ? assistantText : finalResultMessage.result;
        content = [{ type: "text", text }];
      } else {
        const errorText = finalResultMessage.errors.join("\n");
        content = [
          {
            type: "text",
            text: errorText.length > 0 ? errorText : finalResultMessage.result,
          },
        ];
        finishReason = {
          unified: "error",
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
            systemPrompt,
            completionMode: completionMode.type,
          },
        },
        response: {
          modelId: this.modelId,
          timestamp: new Date(),
        },
      };
    }

    if (finalResultMessage.subtype === "success") {
      const structuredOutput = finalResultMessage.structured_output;
      const parsedStructuredOutput =
        completionMode.type === "tools"
          ? parseStructuredEnvelopeFromUnknown(structuredOutput)
          : undefined;

      if (completionMode.type === "tools" && isStructuredToolEnvelope(parsedStructuredOutput)) {
        const toolCalls = mapStructuredToolCallsToContent(
          parsedStructuredOutput.calls,
          this.idGenerator,
        );

        if (toolCalls.length > 0) {
          content = toolCalls;
          finishReason = {
            unified: "tool-calls",
            raw: "tool_use",
          };
        }
      }

      if (
        content.length === 0 &&
        completionMode.type === "tools" &&
        recoveredToolCallsFromStream.length > 0
      ) {
        content = recoveredToolCallsFromStream;
        finishReason = {
          unified: "tool-calls",
          raw: "tool_use",
        };
      }

      if (content.length === 0 && completionMode.type === "tools") {
        const nativeToolCalls = recoverToolModeToolCallsFromAssistant(
          lastAssistantMessage,
          this.idGenerator,
        );

        if (nativeToolCalls.length > 0) {
          content = nativeToolCalls;
          finishReason = {
            unified: "tool-calls",
            raw: "tool_use",
          };
        }
      }

      if (
        content.length === 0 &&
        completionMode.type === "tools" &&
        isStructuredTextEnvelope(parsedStructuredOutput)
      ) {
        content = [{ type: "text", text: parsedStructuredOutput.text }];
      }

      if (content.length === 0 && completionMode.type === "json") {
        if (structuredOutput !== undefined) {
          content = [{ type: "text", text: safeJsonStringify(structuredOutput) }];
        }
      }

      if (content.length === 0) {
        const assistantText = extractAssistantText(lastAssistantMessage);
        if (assistantText.length > 0) {
          if (completionMode.type === "tools") {
            const parsedEnvelope = parseStructuredEnvelopeFromText(assistantText);

            if (isStructuredToolEnvelope(parsedEnvelope)) {
              const toolCalls = mapStructuredToolCallsToContent(
                parsedEnvelope.calls,
                this.idGenerator,
              );

              if (toolCalls.length > 0) {
                content = toolCalls;
                finishReason = {
                  unified: "tool-calls",
                  raw: "tool_use",
                };
              }
            }

            if (content.length === 0 && isStructuredTextEnvelope(parsedEnvelope)) {
              content = [{ type: "text", text: parsedEnvelope.text }];
            }
          }

          if (content.length === 0) {
            content = [{ type: "text", text: assistantText }];
          }
        }
      }

      if (content.length === 0) {
        content = [{ type: "text", text: finalResultMessage.result }];
      }
    }

    if (finalResultMessage.subtype !== "success") {
      if (completionMode.type === "tools" && recoveredToolCallsFromStream.length > 0) {
        content = recoveredToolCallsFromStream;
        finishReason = {
          unified: "tool-calls",
          raw: "tool_use",
        };
      }

      if (completionMode.type === "tools") {
        const recoveredToolCalls = recoverToolModeToolCallsFromAssistant(
          lastAssistantMessage,
          this.idGenerator,
        );

        if (recoveredToolCalls.length > 0) {
          content = recoveredToolCalls;
          finishReason = {
            unified: "tool-calls",
            raw: "tool_use",
          };
        }
      }

      const canRecoverFromStructuredOutputRetry =
        completionMode.type === "tools" &&
        content.length === 0 &&
        isStructuredOutputRetryExhausted(finalResultMessage);

      if (canRecoverFromStructuredOutputRetry) {
        const recoveredContent = recoverToolModeContentFromAssistantText(
          lastAssistantMessage,
          this.idGenerator,
        );

        if (recoveredContent.length > 0) {
          content = recoveredContent;

          const hasRecoveredToolCalls = recoveredContent.some((part) => {
            return part.type === "tool-call";
          });

          finishReason = {
            unified: hasRecoveredToolCalls ? "tool-calls" : "stop",
            raw: "error_max_structured_output_retries_recovered",
          };
        }
      }

      if (content.length === 0) {
        const errorText = finalResultMessage.errors.join("\n");
        content = [{ type: "text", text: errorText }];
        finishReason = {
          unified: "error",
          raw: finalResultMessage.subtype,
        };
      }
    }

    if (
      completionMode.type === "tools" &&
      !hasToolModePrimaryContent(content) &&
      finishReason.unified !== "error"
    ) {
      content = [
        {
          type: "text",
          text: EMPTY_TOOL_ROUTING_OUTPUT_TEXT,
        },
      ];
      finishReason = {
        unified: "error",
        raw: EMPTY_TOOL_ROUTING_OUTPUT_ERROR,
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
          systemPrompt,
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
    const completionMode = buildCompletionMode(options);
    const anthropicOptions = parseAnthropicProviderOptions(options);
    const warnings = [...collectWarnings(options, completionMode), ...this.providerSettingWarnings];

    const promptQueryInput = buildPromptQueryInput({
      promptMessages: options.prompt,
      previousSessionStates: this.promptSessionStates,
    });
    const basePrompt = promptQueryInput.prompt;
    const systemPrompt = extractSystemPromptFromMessages(options.prompt);
    let prompt = basePrompt;
    let outputFormat: AgentQueryOptions["outputFormat"];
    const toolBridgeConfig =
      completionMode.type === "tools"
        ? buildToolBridgeConfig(completionMode.tools, this.toolExecutors)
        : undefined;
    const useNativeToolExecution =
      completionMode.type === "tools" && toolBridgeConfig?.allToolsHaveExecutors === true;

    if (
      completionMode.type === "tools" &&
      toolBridgeConfig?.hasAnyExecutor === true &&
      !useNativeToolExecution
    ) {
      warnings.push(buildPartialToolExecutorWarning(toolBridgeConfig.missingExecutorToolNames));
    }

    if (completionMode.type === "json") {
      prompt = `Return only JSON that matches the required schema.\n\n${basePrompt}`;
      outputFormat = {
        type: "json_schema",
        schema: completionMode.schema,
      };
    }

    const abortController = new AbortController();
    const externalAbortSignal = options.abortSignal;

    const cleanupAbortListener = () => {
      if (externalAbortSignal !== undefined) {
        externalAbortSignal.removeEventListener("abort", abortFromExternalSignal);
      }
    };

    const abortFromExternalSignal = () => {
      abortController.abort();
    };

    if (externalAbortSignal !== undefined) {
      if (externalAbortSignal.aborted) {
        abortController.abort();
      }

      externalAbortSignal.addEventListener("abort", abortFromExternalSignal, {
        once: true,
      });
    }

    const queryOptions: AgentQueryOptions = {
      model: this.modelId,
      tools: [],
      allowedTools: toolBridgeConfig?.allowedTools ?? [],
      resume: promptQueryInput.resumeSessionId,
      systemPrompt,
      permissionMode: "dontAsk",
      settingSources: [],
      maxTurns: useNativeToolExecution ? this.maxTurns : 1,
      abortController,
      env: buildQueryEnv(this.settings),
      hooks: {},
      plugins: [],
      mcpServers: toolBridgeConfig?.mcpServers,
      outputFormat,
      effort: anthropicOptions.effort,
      thinking: anthropicOptions.thinking,
      cwd: process.cwd(),
      includePartialMessages: true,
    };

    const streamState: StreamEventState = {
      blockStates: new Map<number, StreamBlockState>(),
      emittedResponseMetadata: false,
      latestStopReason: null,
      latestUsage: undefined,
    };
    const shouldBufferToolModeText = completionMode.type === "tools" && !useNativeToolExecution;

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: async (controller) => {
        let lastAssistantMessage: SDKAssistantMessage | undefined;
        let finalResultMessage: SDKResultMessage | undefined;
        let emittedToolModeToolCalls = false;
        let emittedToolModeText = false;
        const bufferedToolModeText: string[] = [];
        const pendingBridgeToolInputs = new Map<string, { toolName: string; deltas: string[] }>();

        controller.enqueue({
          type: "stream-start",
          warnings,
        });

        try {
          for await (const message of agentSdk.query({
            prompt,
            options: queryOptions,
          })) {
            if (isPartialAssistantMessage(message)) {
              const mappedParts = appendStreamPartsFromRawEvent(message.event, streamState);

              for (const mappedPart of mappedParts) {
                if (
                  completionMode.type === "tools" &&
                  mappedPart.type === "tool-input-start" &&
                  isBridgeToolName(mappedPart.toolName)
                ) {
                  controller.enqueue({
                    type: "tool-input-start",
                    id: mappedPart.id,
                    toolName: fromBridgeToolName(mappedPart.toolName),
                    providerMetadata: mappedPart.providerMetadata,
                    providerExecuted: useNativeToolExecution,
                    dynamic: mappedPart.dynamic,
                  });

                  pendingBridgeToolInputs.set(mappedPart.id, {
                    toolName: mappedPart.toolName,
                    deltas: [],
                  });
                  continue;
                }

                if (completionMode.type === "tools" && mappedPart.type === "tool-input-delta") {
                  const pendingBridgeInput = pendingBridgeToolInputs.get(mappedPart.id);

                  if (pendingBridgeInput !== undefined) {
                    pendingBridgeInput.deltas.push(mappedPart.delta);
                    controller.enqueue(mappedPart);
                    continue;
                  }
                }

                if (completionMode.type === "tools" && mappedPart.type === "tool-input-end") {
                  const pendingBridgeInput = pendingBridgeToolInputs.get(mappedPart.id);

                  if (pendingBridgeInput !== undefined) {
                    controller.enqueue(mappedPart);

                    const input = normalizeToolInputJson(pendingBridgeInput.deltas.join(""));

                    controller.enqueue({
                      type: "tool-call",
                      toolCallId: mappedPart.id,
                      toolName: fromBridgeToolName(pendingBridgeInput.toolName),
                      input,
                      providerExecuted: useNativeToolExecution,
                    });

                    emittedToolModeToolCalls = true;
                    pendingBridgeToolInputs.delete(mappedPart.id);
                    continue;
                  }
                }

                if (
                  shouldBufferToolModeText &&
                  (mappedPart.type === "text-start" ||
                    mappedPart.type === "text-delta" ||
                    mappedPart.type === "text-end")
                ) {
                  if (mappedPart.type === "text-delta") {
                    bufferedToolModeText.push(mappedPart.delta);
                  }

                  continue;
                }

                controller.enqueue(mappedPart);
              }

              continue;
            }

            if (isAssistantMessage(message)) {
              lastAssistantMessage = message;
            }

            if (isResultMessage(message)) {
              finalResultMessage = message;
            }
          }

          if (!streamState.emittedResponseMetadata) {
            controller.enqueue({
              type: "response-metadata",
              modelId: this.modelId,
            });
          }

          const remainingParts = closePendingStreamBlocks(streamState);
          for (const remainingPart of remainingParts) {
            if (completionMode.type === "tools" && remainingPart.type === "tool-input-end") {
              const pendingBridgeInput = pendingBridgeToolInputs.get(remainingPart.id);

              if (pendingBridgeInput !== undefined) {
                controller.enqueue(remainingPart);

                const input = normalizeToolInputJson(pendingBridgeInput.deltas.join(""));

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: remainingPart.id,
                  toolName: fromBridgeToolName(pendingBridgeInput.toolName),
                  input,
                  providerExecuted: useNativeToolExecution,
                });

                emittedToolModeToolCalls = true;
                pendingBridgeToolInputs.delete(remainingPart.id);
                continue;
              }
            }

            if (shouldBufferToolModeText && remainingPart.type === "text-end") {
              continue;
            }

            controller.enqueue(remainingPart);
          }

          if (completionMode.type === "tools" && !useNativeToolExecution) {
            const bufferedText = bufferedToolModeText.join("");
            let parsedEnvelope: unknown;

            if (finalResultMessage?.subtype === "success") {
              parsedEnvelope = parseStructuredEnvelopeFromUnknown(
                finalResultMessage.structured_output,
              );
            }

            if (parsedEnvelope === undefined && bufferedText.length > 0) {
              parsedEnvelope = parseStructuredEnvelopeFromText(bufferedText);
            }

            if (isStructuredToolEnvelope(parsedEnvelope)) {
              for (const call of parsedEnvelope.calls) {
                controller.enqueue({
                  type: "tool-call",
                  toolCallId: this.idGenerator(),
                  toolName: call.toolName,
                  input: safeJsonStringify(call.input),
                  providerExecuted: false,
                });
              }

              emittedToolModeToolCalls = parsedEnvelope.calls.length > 0;
            }

            if (isStructuredTextEnvelope(parsedEnvelope)) {
              if (!emittedToolModeToolCalls && parsedEnvelope.text.length > 0) {
                enqueueSingleTextBlock(controller, this.idGenerator, parsedEnvelope.text);
                emittedToolModeText = true;
              }
            }

            if (
              !isStructuredToolEnvelope(parsedEnvelope) &&
              !isStructuredTextEnvelope(parsedEnvelope) &&
              bufferedText.length > 0 &&
              !emittedToolModeToolCalls
            ) {
              enqueueSingleTextBlock(controller, this.idGenerator, bufferedText);
              emittedToolModeText = true;
            }

            if (!emittedToolModeToolCalls) {
              const recoveredToolCalls = recoverToolModeToolCallsFromAssistant(
                lastAssistantMessage,
                this.idGenerator,
              );

              for (const recoveredToolCall of recoveredToolCalls) {
                controller.enqueue(recoveredToolCall);
              }

              if (recoveredToolCalls.length > 0) {
                emittedToolModeToolCalls = true;
              }
            }
          }

          let finishReason = mapFinishReason(streamState.latestStopReason);
          let usage = streamState.latestUsage ?? createEmptyUsage();
          let providerMetadata: Record<string, JSONObject> | undefined;

          if (finalResultMessage !== undefined) {
            usage = mapUsage(finalResultMessage);
            finishReason = mapFinishReason(finalResultMessage.stop_reason);
            providerMetadata = buildProviderMetadata(finalResultMessage);

            if (finalResultMessage.subtype !== "success") {
              const canRecoverFromToolCallError =
                completionMode.type === "tools" &&
                !useNativeToolExecution &&
                emittedToolModeToolCalls &&
                finalResultMessage.subtype === "error_max_turns";

              const canRecoverFromStructuredOutputRetry =
                completionMode.type === "tools" &&
                !useNativeToolExecution &&
                isStructuredOutputRetryExhausted(finalResultMessage) &&
                (emittedToolModeToolCalls || emittedToolModeText);

              if (canRecoverFromToolCallError) {
                finishReason = {
                  unified: "tool-calls",
                  raw: "tool_use",
                };
              }

              if (canRecoverFromStructuredOutputRetry) {
                finishReason = {
                  unified: emittedToolModeToolCalls ? "tool-calls" : "stop",
                  raw: "error_max_structured_output_retries_recovered",
                };
              }

              if (!canRecoverFromToolCallError && !canRecoverFromStructuredOutputRetry) {
                finishReason = {
                  unified: "error",
                  raw: finalResultMessage.subtype,
                };

                controller.enqueue({
                  type: "error",
                  error: finalResultMessage.errors.join("\n"),
                });
              }
            }
          }

          const sessionId = readSessionIdFromQueryMessages({
            resultMessage: finalResultMessage,
            assistantMessage: lastAssistantMessage,
          });

          if (sessionId !== undefined) {
            this.promptSessionStates = mergePromptSessionState({
              previousSessionStates: this.promptSessionStates,
              nextSessionState: {
                sessionId,
                serializedPromptMessages: promptQueryInput.serializedPromptMessages,
              },
            });
          }

          if (
            completionMode.type === "tools" &&
            !useNativeToolExecution &&
            emittedToolModeToolCalls &&
            finishReason.unified !== "error"
          ) {
            finishReason = {
              unified: "tool-calls",
              raw: "tool_use",
            };
          }

          if (
            completionMode.type === "tools" &&
            !useNativeToolExecution &&
            !emittedToolModeToolCalls &&
            !emittedToolModeText &&
            finishReason.unified !== "error"
          ) {
            controller.enqueue({
              type: "error",
              error: EMPTY_TOOL_ROUTING_OUTPUT_TEXT,
            });

            finishReason = {
              unified: "error",
              raw: EMPTY_TOOL_ROUTING_OUTPUT_ERROR,
            };
          }

          controller.enqueue({
            type: "finish",
            usage,
            finishReason,
            providerMetadata,
          });
        } catch (error) {
          const remainingParts = closePendingStreamBlocks(streamState);
          for (const remainingPart of remainingParts) {
            if (shouldBufferToolModeText && remainingPart.type === "text-end") {
              continue;
            }

            controller.enqueue(remainingPart);
          }

          controller.enqueue({
            type: "error",
            error,
          });

          controller.enqueue({
            type: "finish",
            usage: streamState.latestUsage ?? createEmptyUsage(),
            finishReason: {
              unified: "error",
              raw: "stream-bridge-error",
            },
          });
        } finally {
          cleanupAbortListener();
          controller.close();
        }
      },
      cancel: () => {
        abortController.abort();
        cleanupAbortListener();
      },
    });

    return {
      stream,
      request: {
        body: {
          prompt,
          systemPrompt,
          completionMode: completionMode.type,
        },
      },
      response: {
        headers: undefined,
      },
    };
  }
}
