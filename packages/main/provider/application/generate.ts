import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type {
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3GenerateResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type {
  Options as AgentQueryOptions,
  SDKAssistantMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";

import {
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../../bridge/parse-utils";
import { buildProviderMetadata, mapFinishReason, mapUsage } from "../../bridge/result-mapping";
import {
  appendStreamPartsFromRawEvent,
  closePendingStreamBlocks,
} from "../../bridge/stream-event-mapper";
import {
  createEmptyUsage,
  type StreamBlockState,
  type StreamEventState,
} from "../../shared/stream-types";
import type { ToolExecutorMap } from "../../shared/tool-executor";
import { safeJsonStringify } from "../../shared/type-readers";
import {
  buildIncomingSessionState,
  readSessionIdFromQueryMessages,
} from "../domain/incoming-session-state";
import { mergePromptSessionState, type PromptSessionState } from "../domain/prompt-session-state";
import { buildQueryEnv } from "../domain/query-env";
import {
  buildToolBridgeConfig,
  fromBridgeToolName,
  isBridgeToolName,
  normalizeToolInputJson,
} from "../domain/tool-bridge-config";
import {
  hasToolModePrimaryContent,
  recoverToolModeContentFromAssistantText,
  recoverToolModeToolCallsFromAssistant,
} from "../domain/tool-recovery";
import type { IncomingSessionState } from "../incoming-session-store";
import type { AgentRuntimePort } from "../ports/agent-runtime-port";
import { createAbortBridge, prepareQueryContext } from "./query-context";
import {
  EMPTY_TOOL_ROUTING_OUTPUT_ERROR,
  EMPTY_TOOL_ROUTING_OUTPUT_TEXT,
  extractAssistantText,
  isAssistantMessage,
  isPartialAssistantMessage,
  isResultMessage,
  isStructuredOutputRetryExhausted,
} from "./runtime-message-utils";

export const runGenerate = async (args: {
  options: LanguageModelV3CallOptions;
  modelId: string;
  settings: AnthropicProviderSettings;
  idGenerator: () => string;
  toolExecutors: ToolExecutorMap | undefined;
  maxTurns: number | undefined;
  runtime: AgentRuntimePort;
  providerSettingWarnings: SharedV3Warning[];
  previousSessionStates: () => PromptSessionState[];
  setPromptSessionStates: (sessionStates: PromptSessionState[]) => void;
  previousIncomingSessionStates: () => IncomingSessionState[];
  hydrateIncomingSessionState: (incomingSessionKey: string) => Promise<void>;
  persistIncomingSessionState: (incomingSessionState: IncomingSessionState) => Promise<void>;
  buildPartialToolExecutorWarning: (missingExecutorToolNames: string[]) => SharedV3Warning;
}): Promise<LanguageModelV3GenerateResult> => {
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
    providerSettingWarnings: args.providerSettingWarnings,
    previousSessionStates: args.previousSessionStates,
    previousIncomingSessionStates: args.previousIncomingSessionStates,
    hydrateIncomingSessionState: args.hydrateIncomingSessionState,
    buildToolBridgeConfig: (tools) => {
      return buildToolBridgeConfig(tools, args.toolExecutors);
    },
    buildPartialToolExecutorWarning: args.buildPartialToolExecutorWarning,
  });

  const { abortController, cleanupAbortListener } = createAbortBridge(args.options.abortSignal);

  const queryOptions: AgentQueryOptions = {
    model: args.modelId,
    tools: [],
    allowedTools: toolBridgeConfig?.allowedTools ?? [],
    resume: promptQueryInput.resumeSessionId,
    systemPrompt,
    permissionMode: "dontAsk",
    settingSources: [],
    maxTurns: useNativeToolExecution ? args.maxTurns : 1,
    abortController,
    env: buildQueryEnv(args.settings),
    hooks: {},
    plugins: [],
    mcpServers: toolBridgeConfig?.mcpServers,
    outputFormat,
    effort,
    thinking,
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

  const appendRecoveredToolCall = (toolCall: {
    toolCallId: string;
    toolName: string;
    rawInput: string;
  }) => {
    recoveredToolCallsFromStream.push({
      type: "tool-call",
      toolCallId: toolCall.toolCallId,
      toolName: fromBridgeToolName(toolCall.toolName),
      input: normalizeToolInputJson(toolCall.rawInput),
      providerExecuted: useNativeToolExecution,
    });
  };

  try {
    for await (const message of args.runtime.query({
      prompt: queryPrompt,
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
    cleanupAbortListener();
  }

  const sessionId = readSessionIdFromQueryMessages({
    resultMessage: finalResultMessage,
    assistantMessage: lastAssistantMessage,
  });

  if (sessionId !== undefined) {
    const serializedPromptMessages = promptQueryInput.serializedPromptMessages;
    if (serializedPromptMessages !== undefined) {
      args.setPromptSessionStates(
        mergePromptSessionState({
          previousSessionStates: args.previousSessionStates(),
          nextSessionState: {
            sessionId,
            serializedPromptMessages,
          },
        }),
      );
    }

    if (incomingSessionKey !== undefined) {
      await args.persistIncomingSessionState(
        buildIncomingSessionState({
          incomingSessionKey,
          sessionId,
          promptMessages: args.options.prompt,
        }),
      );
    }
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
            modelId: args.modelId,
            timestamp: new Date(),
          },
        };
      }

      const recoveredToolCalls = recoverToolModeToolCallsFromAssistant({
        assistantMessage: lastAssistantMessage,
        idGenerator: args.idGenerator,
        mapToolName: fromBridgeToolName,
      });

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
            modelId: args.modelId,
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
          text: errorText,
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
        modelId: args.modelId,
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
        args.idGenerator,
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
      const nativeToolCalls = recoverToolModeToolCallsFromAssistant({
        assistantMessage: lastAssistantMessage,
        idGenerator: args.idGenerator,
        mapToolName: fromBridgeToolName,
      });

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
      if (finalResultMessage.structured_output !== undefined) {
        content = [{ type: "text", text: safeJsonStringify(finalResultMessage.structured_output) }];
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
              args.idGenerator,
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
      const recoveredToolCalls = recoverToolModeToolCallsFromAssistant({
        assistantMessage: lastAssistantMessage,
        idGenerator: args.idGenerator,
        mapToolName: fromBridgeToolName,
      });

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
      const recoveredContent = recoverToolModeContentFromAssistantText({
        assistantMessage: lastAssistantMessage,
        idGenerator: args.idGenerator,
      });

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
      modelId: args.modelId,
      timestamp: new Date(),
    },
  };
};
