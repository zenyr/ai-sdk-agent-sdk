import type {
  JSONObject,
  LanguageModelV3CallOptions,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type { SDKAssistantMessage, SDKResultMessage, SDKSystemMessage } from "@anthropic-ai/claude-agent-sdk";

import { parseStructuredEnvelopeFromText } from "../../bridge/parse-utils";
import { buildProviderMetadata, mapFinishReason, mapUsage } from "../../bridge/result-mapping";
import {
  appendStreamPartsFromRawEvent,
  closePendingStreamBlocks,
  enqueueSingleTextBlock,
} from "../../bridge/stream-event-mapper";
import { createEmptyUsage, type StreamBlockState, type StreamEventState } from "../../shared/stream-types";
import type { AgentSdkProviderSettings, ToolCallDelegate, ToolExecutorMap } from "../../shared/tool-executor";
import { safeJsonStringify } from "../../shared/type-readers";
import type { PromptSessionState } from "../domain/prompt-session-state";
import { buildToolBridgeConfig, fromBridgeToolName, isBridgeToolName } from "../domain/tool-bridge-config";
import { recoverToolModeToolCallsFromAssistant } from "../domain/tool-recovery";
import type { IncomingSessionState } from "../incoming-session-store";
import type { AgentRuntimePort } from "../ports/agent-runtime-port";
import type { PendingBridgeToolInputs } from "./bridge-tool-input-buffer";
import { createAbortBridge, prepareQueryContext } from "./query-context";
import { buildAgentQueryOptions } from "./query-options";
import {
  EMPTY_TOOL_ROUTING_OUTPUT_ERROR,
  EMPTY_TOOL_ROUTING_OUTPUT_TEXT,
  extractAssistantText,
  isAssistantMessage,
  isPartialAssistantMessage,
  isResultMessage,
  isStructuredOutputRetryExhausted,
  isSystemInitMessage,
  normalizeRuntimeQueryError,
} from "./runtime-message-utils";
import { persistQuerySessionState } from "./session-persistence";
import {
  appendBridgeToolCallCapture,
  finishBridgeToolCallCapture,
  startBridgeToolCallCapture,
} from "./tool-call-facade";
import {
  hasToolModeEnvelopeResolution,
  resolveToolModeEnvelopeFromText,
  resolveToolModeEnvelopeFromUnknown,
} from "./tool-mode-envelope-facade";

export const runStream = async (args: {
  options: LanguageModelV3CallOptions;
  provider: string;
  modelId: string;
  settings: AgentSdkProviderSettings;
  idGenerator: () => string;
  toolExecutors: ToolExecutorMap | undefined;
  toolCallDelegate: ToolCallDelegate | undefined;
  maxTurns: number | undefined;
  runtime: AgentRuntimePort;
  providerSettingWarnings: SharedV3Warning[];
  previousSessionStates: () => PromptSessionState[];
  setPromptSessionStates: (sessionStates: PromptSessionState[]) => void;
  previousIncomingSessionStates: () => IncomingSessionState[];
  hydrateIncomingSessionState: (incomingSessionKey: string) => Promise<void>;
  persistIncomingSessionState: (incomingSessionState: IncomingSessionState) => Promise<void>;
  buildPartialToolExecutorWarning: (missingExecutorToolNames: string[]) => SharedV3Warning;
}): Promise<LanguageModelV3StreamResult> => {
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
    options: args.options,
    provider: args.provider,
    providerSettingWarnings: args.providerSettingWarnings,
    previousSessionStates: args.previousSessionStates,
    previousIncomingSessionStates: args.previousIncomingSessionStates,
    hydrateIncomingSessionState: args.hydrateIncomingSessionState,
    buildToolBridgeConfig: tools => {
      return buildToolBridgeConfig(tools, args.toolExecutors, args.toolCallDelegate);
    },
    buildPartialToolExecutorWarning: args.buildPartialToolExecutorWarning,
  });

  const { abortController, cleanupAbortListener } = createAbortBridge(args.options.abortSignal);
  const runtimeStderrChunks: string[] = [];

  const queryOptions = buildAgentQueryOptions({
    modelId: args.modelId,
    settings: args.settings,
    allowedTools: useNativeToolExecution ? (toolBridgeConfig?.allowedTools ?? []) : [],
    mcpServers: useNativeToolExecution ? toolBridgeConfig?.mcpServers : undefined,
    resumeSessionId: promptQueryInput.resumeSessionId,
    systemPrompt,
    maxTurns: args.maxTurns,
    useNativeToolExecution,
    abortController,
    outputFormat,
    effort,
    thinking,
    includePartialMessages: true,
    onStderr: data => {
      runtimeStderrChunks.push(data);
    },
  });

  const streamState: StreamEventState = {
    blockStates: new Map<number, StreamBlockState>(),
    emittedResponseMetadata: false,
    latestStopReason: null,
    latestUsage: undefined,
  };
  const shouldBufferToolModeText = completionMode.type === "tools" && !useNativeToolExecution;

  const stream = new ReadableStream<LanguageModelV3StreamPart>({
    start: async controller => {
      let lastAssistantMessage: SDKAssistantMessage | undefined;
      let finalResultMessage: SDKResultMessage | undefined;
      let initSystemMessage: SDKSystemMessage | undefined;
      let emittedToolModeToolCalls = false;
      let emittedToolModeText = false;
      const bufferedToolModeText: string[] = [];
      const pendingBridgeToolInputs: PendingBridgeToolInputs = new Map();

      controller.enqueue({
        type: "stream-start",
        warnings,
      });

      try {
        for await (const message of args.runtime.query({
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
                startBridgeToolCallCapture({
                  pendingBridgeToolInputs,
                  part: mappedPart,
                });

                controller.enqueue({
                  type: "tool-input-start",
                  id: mappedPart.id,
                  toolName: fromBridgeToolName(mappedPart.toolName),
                  providerMetadata: mappedPart.providerMetadata,
                  providerExecuted: useNativeToolExecution,
                  dynamic: mappedPart.dynamic,
                });

                continue;
              }

              if (completionMode.type === "tools" && mappedPart.type === "tool-input-delta") {
                const hasPendingBridgeToolInput = appendBridgeToolCallCapture({
                  pendingBridgeToolInputs,
                  part: mappedPart,
                });

                if (hasPendingBridgeToolInput) {
                  controller.enqueue(mappedPart);
                  continue;
                }
              }

              if (completionMode.type === "tools" && mappedPart.type === "tool-input-end") {
                const toolCall = finishBridgeToolCallCapture({
                  pendingBridgeToolInputs,
                  part: mappedPart,
                  providerExecuted: useNativeToolExecution,
                });

                if (toolCall !== undefined) {
                  controller.enqueue(mappedPart);
                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: toolCall.toolCallId,
                    toolName: toolCall.toolName,
                    input: toolCall.input,
                    providerExecuted: toolCall.providerExecuted,
                  });

                  emittedToolModeToolCalls = true;
                  continue;
                }
              }

              if (
                shouldBufferToolModeText &&
                (mappedPart.type === "text-start" || mappedPart.type === "text-delta" || mappedPart.type === "text-end")
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

          if (isSystemInitMessage(message)) {
            initSystemMessage ??= message;
          }
        }

        if (!streamState.emittedResponseMetadata) {
          controller.enqueue({
            type: "response-metadata",
            modelId: args.modelId,
          });
        }

        const remainingParts = closePendingStreamBlocks(streamState);
        for (const remainingPart of remainingParts) {
          if (completionMode.type === "tools" && remainingPart.type === "tool-input-end") {
            const toolCall = finishBridgeToolCallCapture({
              pendingBridgeToolInputs,
              part: remainingPart,
              providerExecuted: useNativeToolExecution,
            });

            if (toolCall !== undefined) {
              controller.enqueue(remainingPart);
              controller.enqueue({
                type: "tool-call",
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                input: toolCall.input,
                providerExecuted: toolCall.providerExecuted,
              });

              emittedToolModeToolCalls = true;
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
          const structuredOutputResolution =
            finalResultMessage?.subtype === "success"
              ? resolveToolModeEnvelopeFromUnknown({
                  value: finalResultMessage.structured_output,
                  idGenerator: args.idGenerator,
                })
              : undefined;
          const bufferedTextResolution =
            (structuredOutputResolution === undefined || !hasToolModeEnvelopeResolution(structuredOutputResolution)) &&
            bufferedText.length > 0
              ? resolveToolModeEnvelopeFromText({
                  text: bufferedText,
                  idGenerator: args.idGenerator,
                })
              : undefined;
          const envelopeResolution =
            structuredOutputResolution !== undefined && hasToolModeEnvelopeResolution(structuredOutputResolution)
              ? structuredOutputResolution
              : bufferedTextResolution;

          if (envelopeResolution !== undefined) {
            for (const call of envelopeResolution.toolCalls) {
              controller.enqueue({
                type: "tool-call",
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                input: call.input,
                providerExecuted: call.providerExecuted,
              });
            }

            emittedToolModeToolCalls = envelopeResolution.toolCalls.length > 0;
          }

          if (envelopeResolution?.text !== undefined) {
            if (!emittedToolModeToolCalls && envelopeResolution.text.length > 0) {
              enqueueSingleTextBlock(controller, args.idGenerator, envelopeResolution.text);
              emittedToolModeText = true;
            }
          }

          if (
            (envelopeResolution === undefined || !hasToolModeEnvelopeResolution(envelopeResolution)) &&
            bufferedText.length > 0 &&
            !emittedToolModeToolCalls
          ) {
            enqueueSingleTextBlock(controller, args.idGenerator, bufferedText);
            emittedToolModeText = true;
          }

          if (!emittedToolModeToolCalls) {
            const recoveredToolCalls = recoverToolModeToolCallsFromAssistant({
              assistantMessage: lastAssistantMessage,
              idGenerator: args.idGenerator,
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
              isStructuredOutputRetryExhausted(finalResultMessage) &&
              (completionMode.type !== "tools" ||
                (!useNativeToolExecution && (emittedToolModeToolCalls || emittedToolModeText)));

            const canRecoverFromStructuredOutputRetryWithAssistantText =
              isStructuredOutputRetryExhausted(finalResultMessage) &&
              completionMode.type !== "tools" &&
              extractAssistantText(lastAssistantMessage).length > 0;

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

            if (canRecoverFromStructuredOutputRetryWithAssistantText) {
              const assistantText = extractAssistantText(lastAssistantMessage);
              enqueueSingleTextBlock(controller, args.idGenerator, assistantText);
              emittedToolModeText = true;
              finishReason = {
                unified: "stop",
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

        await persistQuerySessionState({
          resultMessage: finalResultMessage,
          assistantMessage: lastAssistantMessage,
          initSystemMessage,
          incomingSessionKey,
          serializedPromptMessages: promptQueryInput.serializedPromptMessages,
          promptMessages: args.options.prompt,
          previousSessionStates: args.previousSessionStates,
          setPromptSessionStates: args.setPromptSessionStates,
          persistIncomingSessionState: args.persistIncomingSessionState,
        });

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
        const runtimeQueryError = normalizeRuntimeQueryError(error, runtimeStderrChunks.join(""));
        const remainingParts = closePendingStreamBlocks(streamState);
        for (const remainingPart of remainingParts) {
          if (shouldBufferToolModeText && remainingPart.type === "text-end") {
            continue;
          }

          controller.enqueue(remainingPart);
        }

        controller.enqueue({
          type: "error",
          error: runtimeQueryError.message,
        });

        controller.enqueue({
          type: "finish",
          usage: streamState.latestUsage ?? createEmptyUsage(),
          finishReason: {
            unified: "error",
            raw: runtimeQueryError.raw,
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
};
