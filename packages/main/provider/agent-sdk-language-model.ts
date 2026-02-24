import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type {
  JSONObject,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Message,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type {
  Options as AgentQueryOptions,
  SDKAssistantMessage,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";

import {
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../bridge/parse-utils";
import { buildProviderMetadata, mapFinishReason, mapUsage } from "../bridge/result-mapping";
import {
  appendStreamPartsFromRawEvent,
  closePendingStreamBlocks,
  enqueueSingleTextBlock,
} from "../bridge/stream-event-mapper";
import { DEFAULT_SUPPORTED_URLS } from "../shared/constants";
import {
  createEmptyUsage,
  type StreamBlockState,
  type StreamEventState,
} from "../shared/stream-types";
import type { ToolExecutorMap } from "../shared/tool-executor";
import { safeJsonStringify } from "../shared/type-readers";
import { claudeAgentRuntime } from "./adapters/claude-agent-runtime";
import { fileIncomingSessionStore } from "./adapters/file-incoming-session-store";
import { runGenerate } from "./application/generate";
import { createAbortBridge, prepareQueryContext } from "./application/query-context";
import {
  buildIncomingSessionState,
  findIncomingSessionState,
  mergeIncomingSessionState,
  readSessionIdFromQueryMessages,
} from "./domain/incoming-session-state";
import { mergePromptSessionState, type PromptSessionState } from "./domain/prompt-session-state";
import { collectProviderSettingWarnings } from "./domain/provider-setting-warnings";
import { buildQueryEnv } from "./domain/query-env";
import {
  buildToolBridgeConfig,
  fromBridgeToolName,
  isBridgeToolName,
  normalizeToolInputJson,
} from "./domain/tool-bridge-config";
import { recoverToolModeToolCallsFromAssistant } from "./domain/tool-recovery";
import type { IncomingSessionState } from "./incoming-session-store";
import type { AgentRuntimePort } from "./ports/agent-runtime-port";
import type { IncomingSessionStorePort } from "./ports/incoming-session-store-port";

const isAssistantMessage = (message: SDKMessage): message is SDKAssistantMessage => {
  return message.type === "assistant";
};

const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === "result";
};

const isPartialAssistantMessage = (message: SDKMessage): message is SDKPartialAssistantMessage => {
  return message.type === "stream_event";
};

const isStructuredOutputRetryExhausted = (resultMessage: SDKResultMessage): boolean => {
  return resultMessage.subtype === "error_max_structured_output_retries";
};

const EMPTY_TOOL_ROUTING_OUTPUT_ERROR = "empty-tool-routing-output";
const EMPTY_TOOL_ROUTING_OUTPUT_TEXT = "Tool routing produced no tool call or text response.";

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
  private readonly runtime: AgentRuntimePort;
  private readonly sessionStore: IncomingSessionStorePort;
  private readonly providerSettingWarnings: SharedV3Warning[];
  private promptSessionStates: PromptSessionState[] = [];
  private incomingSessionStates: IncomingSessionState[] = [];

  constructor(args: {
    modelId: string;
    provider: string;
    settings: AnthropicProviderSettings;
    idGenerator: () => string;
    toolExecutors?: ToolExecutorMap;
    maxTurns?: number;
    runtime?: AgentRuntimePort;
    sessionStore?: IncomingSessionStorePort;
  }) {
    this.modelId = args.modelId;
    this.provider = args.provider;
    this.settings = args.settings;
    this.idGenerator = args.idGenerator;
    this.toolExecutors = args.toolExecutors;
    this.maxTurns = args.maxTurns;
    this.runtime = args.runtime ?? claudeAgentRuntime;
    this.sessionStore = args.sessionStore ?? fileIncomingSessionStore;
    this.providerSettingWarnings = collectProviderSettingWarnings(this.settings);
    this.supportedUrls = DEFAULT_SUPPORTED_URLS;
  }

  private async hydrateIncomingSessionState(incomingSessionKey: string): Promise<void> {
    const existingIncomingSessionState = findIncomingSessionState({
      incomingSessionKey,
      previousIncomingSessionStates: this.incomingSessionStates,
    });

    if (existingIncomingSessionState !== undefined) {
      return;
    }

    const persistedIncomingSessionState = await this.sessionStore
      .get({
        modelId: this.modelId,
        incomingSessionKey,
      })
      .catch(() => {
        return undefined;
      });

    if (persistedIncomingSessionState === undefined) {
      return;
    }

    this.incomingSessionStates = mergeIncomingSessionState({
      previousIncomingSessionStates: this.incomingSessionStates,
      nextIncomingSessionState: persistedIncomingSessionState,
    });
  }

  private async persistIncomingSessionState(
    incomingSessionState: IncomingSessionState,
  ): Promise<void> {
    this.incomingSessionStates = mergeIncomingSessionState({
      previousIncomingSessionStates: this.incomingSessionStates,
      nextIncomingSessionState: incomingSessionState,
    });

    await this.sessionStore
      .set({
        modelId: this.modelId,
        incomingSessionKey: incomingSessionState.incomingSessionKey,
        state: incomingSessionState,
      })
      .catch(() => {
        return undefined;
      });
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    return runGenerate({
      options,
      modelId: this.modelId,
      settings: this.settings,
      idGenerator: this.idGenerator,
      toolExecutors: this.toolExecutors,
      maxTurns: this.maxTurns,
      runtime: this.runtime,
      providerSettingWarnings: this.providerSettingWarnings,
      previousSessionStates: () => this.promptSessionStates,
      setPromptSessionStates: (promptSessionStates) => {
        this.promptSessionStates = promptSessionStates;
      },
      previousIncomingSessionStates: () => this.incomingSessionStates,
      hydrateIncomingSessionState: this.hydrateIncomingSessionState.bind(this),
      persistIncomingSessionState: this.persistIncomingSessionState.bind(this),
      buildPartialToolExecutorWarning,
    });
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const {
      completionMode,
      warnings,
      incomingSessionKey,
      promptQueryInput,
      prompt,
      systemPrompt,
      outputFormat,
      queryPrompt,
      toolBridgeConfig,
      useNativeToolExecution,
      effort,
      thinking,
    } = await prepareQueryContext({
      options,
      providerSettingWarnings: this.providerSettingWarnings,
      previousSessionStates: () => this.promptSessionStates,
      previousIncomingSessionStates: () => this.incomingSessionStates,
      hydrateIncomingSessionState: this.hydrateIncomingSessionState.bind(this),
      buildToolBridgeConfig: (tools) => {
        return buildToolBridgeConfig(tools, this.toolExecutors);
      },
      buildPartialToolExecutorWarning,
    });

    const { abortController, cleanupAbortListener } = createAbortBridge(options.abortSignal);

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
      effort,
      thinking,
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
          for await (const message of this.runtime.query({
            prompt: queryPrompt,
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
              const recoveredToolCalls = recoverToolModeToolCallsFromAssistant({
                assistantMessage: lastAssistantMessage,
                idGenerator: this.idGenerator,
                mapToolName: fromBridgeToolName,
              });

              for (const recoveredToolCall of recoveredToolCalls) {
                if (recoveredToolCall.type !== "tool-call") {
                  continue;
                }

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: recoveredToolCall.toolCallId,
                  toolName: recoveredToolCall.toolName,
                  input: recoveredToolCall.input,
                  providerExecuted: recoveredToolCall.providerExecuted,
                });
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
            const serializedPromptMessages = promptQueryInput.serializedPromptMessages;
            if (serializedPromptMessages !== undefined) {
              this.promptSessionStates = mergePromptSessionState({
                previousSessionStates: this.promptSessionStates,
                nextSessionState: {
                  sessionId,
                  serializedPromptMessages,
                },
              });
            }

            if (incomingSessionKey !== undefined) {
              await this.persistIncomingSessionState(
                buildIncomingSessionState({
                  incomingSessionKey,
                  sessionId,
                  promptMessages: options.prompt,
                }),
              );
            }
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
