import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import {
  type LanguageModelV3,
  type LanguageModelV3CallOptions,
  type LanguageModelV3Content,
  type LanguageModelV3GenerateResult,
  type LanguageModelV3StreamPart,
  type LanguageModelV3StreamResult,
  type LanguageModelV3Usage,
  type JSONObject,
} from "@ai-sdk/provider";
import {
  query,
  type Options as AgentQueryOptions,
  type SDKAssistantMessage,
  type SDKMessage,
  type SDKPartialAssistantMessage,
  type SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";

import {
  buildCompletionMode,
  buildToolInstruction,
} from "../bridge/completion-mode";
import {
  collectWarnings,
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseAnthropicProviderOptions,
} from "../bridge/parse-utils";
import { serializePrompt } from "../bridge/prompt-serializer";
import {
  buildProviderMetadata,
  mapFinishReason,
  mapUsage,
} from "../bridge/result-mapping";
import {
  appendStreamPartsFromRawEvent,
  closePendingStreamBlocks,
} from "../bridge/stream-event-mapper";
import { DEFAULT_SUPPORTED_URLS } from "../shared/constants";
import {
  createEmptyUsage,
  type StreamBlockState,
  type StreamEventState,
} from "../shared/stream-types";
import { isRecord, readString, safeJsonStringify } from "../shared/type-readers";

const isAssistantMessage = (
  message: SDKMessage,
): message is SDKAssistantMessage => {
  return message.type === "assistant";
};

const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === "result";
};

const isPartialAssistantMessage = (
  message: SDKMessage,
): message is SDKPartialAssistantMessage => {
  return message.type === "stream_event";
};

const extractAssistantText = (
  assistantMessage: SDKAssistantMessage | undefined,
): string => {
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

export class AgentSdkAnthropicLanguageModel implements LanguageModelV3 {
  readonly specificationVersion: "v3" = "v3";
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
    let outputFormat: AgentQueryOptions["outputFormat"];

    if (completionMode.type === "tools") {
      prompt = `${buildToolInstruction(
        completionMode.tools,
        options.toolChoice,
      )}\n\n${basePrompt}`;
      outputFormat = {
        type: "json_schema",
        schema: completionMode.schema,
      };
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

    const env: Record<string, string | undefined> = {
      ...process.env,
    };

    if (
      typeof this.settings.apiKey === "string" &&
      this.settings.apiKey.length > 0
    ) {
      env.ANTHROPIC_API_KEY = this.settings.apiKey;
    }

    if (
      typeof this.settings.authToken === "string" &&
      this.settings.authToken.length > 0
    ) {
      env.ANTHROPIC_AUTH_TOKEN = this.settings.authToken;
    }

    const queryOptions: AgentQueryOptions = {
      model: this.modelId,
      tools: [],
      permissionMode: "dontAsk",
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
        externalAbortSignal.removeEventListener(
          "abort",
          abortFromExternalSignal,
        );
      }
    }

    if (finalResultMessage === undefined) {
      return {
        content: [{ type: "text", text: "" }],
        finishReason: {
          unified: "error",
          raw: "agent-sdk-no-result",
        },
        usage: createEmptyUsage(),
        warnings,
      };
    }

    const usage = mapUsage(finalResultMessage);
    const providerMetadata = buildProviderMetadata(finalResultMessage);

    let content: LanguageModelV3Content[] = [];
    let finishReason = mapFinishReason(finalResultMessage.stop_reason);

    if (finalResultMessage.subtype === "success") {
      const structuredOutput = finalResultMessage.structured_output;

      if (
        completionMode.type === "tools" &&
        isStructuredToolEnvelope(structuredOutput)
      ) {
        const toolCalls = mapStructuredToolCallsToContent(
          structuredOutput.calls,
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
        isStructuredTextEnvelope(structuredOutput)
      ) {
        content = [{ type: "text", text: structuredOutput.text }];
      }

      if (content.length === 0 && completionMode.type === "json") {
        if (structuredOutput !== undefined) {
          content = [
            { type: "text", text: safeJsonStringify(structuredOutput) },
          ];
        }
      }

      if (content.length === 0) {
        const assistantText = extractAssistantText(lastAssistantMessage);
        if (assistantText.length > 0) {
          content = [{ type: "text", text: assistantText }];
        }
      }

      if (content.length === 0) {
        content = [{ type: "text", text: finalResultMessage.result }];
      }
    }

    if (finalResultMessage.subtype !== "success") {
      const errorText = finalResultMessage.errors.join("\n");
      content = [{ type: "text", text: errorText }];
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
          completionMode: completionMode.type,
        },
      },
      response: {
        modelId: this.modelId,
        timestamp: new Date(),
      },
    };
  }

  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    const completionMode = buildCompletionMode(options);
    const anthropicOptions = parseAnthropicProviderOptions(options);
    const warnings = collectWarnings(options, completionMode);

    const basePrompt = serializePrompt(options.prompt);
    let prompt = basePrompt;
    let outputFormat: AgentQueryOptions["outputFormat"];

    if (completionMode.type === "tools") {
      prompt = `${buildToolInstruction(
        completionMode.tools,
        options.toolChoice,
      )}\n\n${basePrompt}`;
      outputFormat = {
        type: "json_schema",
        schema: completionMode.schema,
      };
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
        externalAbortSignal.removeEventListener(
          "abort",
          abortFromExternalSignal,
        );
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

    const env: Record<string, string | undefined> = {
      ...process.env,
    };

    if (
      typeof this.settings.apiKey === "string" &&
      this.settings.apiKey.length > 0
    ) {
      env.ANTHROPIC_API_KEY = this.settings.apiKey;
    }

    if (
      typeof this.settings.authToken === "string" &&
      this.settings.authToken.length > 0
    ) {
      env.ANTHROPIC_AUTH_TOKEN = this.settings.authToken;
    }

    const queryOptions: AgentQueryOptions = {
      model: this.modelId,
      tools: [],
      permissionMode: "dontAsk",
      maxTurns: 1,
      abortController,
      env,
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

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: async (controller) => {
        let finalResultMessage: SDKResultMessage | undefined;

        controller.enqueue({
          type: "stream-start",
          warnings,
        });

        try {
          for await (const message of query({
            prompt,
            options: queryOptions,
          })) {
            if (isPartialAssistantMessage(message)) {
              const mappedParts = appendStreamPartsFromRawEvent(
                message.event,
                streamState,
              );

              for (const mappedPart of mappedParts) {
                controller.enqueue(mappedPart);
              }

              continue;
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
            controller.enqueue(remainingPart);
          }

          if (
            finalResultMessage?.subtype === "success" &&
            completionMode.type === "tools" &&
            isStructuredToolEnvelope(finalResultMessage.structured_output)
          ) {
            for (const call of finalResultMessage.structured_output.calls) {
              controller.enqueue({
                type: "tool-call",
                toolCallId: this.idGenerator(),
                toolName: call.toolName,
                input: safeJsonStringify(call.input),
                providerExecuted: false,
              });
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

          controller.enqueue({
            type: "finish",
            usage,
            finishReason,
            providerMetadata,
          });
        } catch (error) {
          const remainingParts = closePendingStreamBlocks(streamState);
          for (const remainingPart of remainingParts) {
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
          completionMode: completionMode.type,
        },
      },
      response: {
        headers: undefined,
      },
    };
  }
}
