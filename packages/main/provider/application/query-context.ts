import type { LanguageModelV3CallOptions, SharedV3Warning } from "@ai-sdk/provider";
import type { Options as AgentQueryOptions, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";

import { buildCompletionMode, type CompletionMode } from "../../bridge/completion-mode";
import { collectWarnings, parseAnthropicProviderOptions } from "../../bridge/parse-utils";
import { extractSystemPromptFromMessages } from "../../bridge/prompt-serializer";
import { buildPromptQueryInputWithIncomingSession } from "../domain/incoming-session-state";
import { buildMultimodalQueryPrompt } from "../domain/multimodal-prompt";
import type { PromptQueryInput, PromptSessionState } from "../domain/prompt-session-state";
import { readIncomingSessionKey } from "../domain/session-key";
import type { ToolBridgeConfig } from "../domain/tool-bridge-config";
import type { IncomingSessionState } from "../incoming-session-store";

type QueryPromptInput = string | AsyncIterable<SDKUserMessage>;

export type QueryContext = {
  completionMode: CompletionMode;
  warnings: SharedV3Warning[];
  incomingSessionKey: string | undefined;
  promptQueryInput: PromptQueryInput;
  prompt: string;
  systemPrompt: string | undefined;
  outputFormat: AgentQueryOptions["outputFormat"];
  queryPrompt: QueryPromptInput;
  toolBridgeConfig: ToolBridgeConfig | undefined;
  useNativeToolExecution: boolean;
  effort: ReturnType<typeof parseAnthropicProviderOptions>["effort"];
  thinking: ReturnType<typeof parseAnthropicProviderOptions>["thinking"];
};

export const prepareQueryContext = async (args: {
  options: LanguageModelV3CallOptions;
  providerSettingWarnings: SharedV3Warning[];
  previousSessionStates: () => PromptSessionState[];
  previousIncomingSessionStates: () => IncomingSessionState[];
  hydrateIncomingSessionState: (incomingSessionKey: string) => Promise<void>;
  buildToolBridgeConfig: (
    tools: Array<{ name: string; description?: string; inputSchema: unknown }>,
  ) => ToolBridgeConfig | undefined;
  buildPartialToolExecutorWarning: (missingExecutorToolNames: string[]) => SharedV3Warning;
}): Promise<QueryContext> => {
  const completionMode = buildCompletionMode(args.options);
  const anthropicOptions = parseAnthropicProviderOptions(args.options);
  const warnings = [
    ...collectWarnings(args.options, completionMode),
    ...args.providerSettingWarnings,
  ];

  const incomingSessionKey = readIncomingSessionKey(args.options);
  if (incomingSessionKey !== undefined) {
    await args.hydrateIncomingSessionState(incomingSessionKey);
  }

  const promptQueryInput = buildPromptQueryInputWithIncomingSession({
    promptMessages: args.options.prompt,
    incomingSessionKey,
    previousSessionStates: args.previousSessionStates(),
    previousIncomingSessionStates: args.previousIncomingSessionStates(),
  });

  const basePrompt = promptQueryInput.prompt;
  const systemPrompt = extractSystemPromptFromMessages(args.options.prompt);
  let prompt = basePrompt;
  let outputFormat: AgentQueryOptions["outputFormat"];
  const toolBridgeConfig =
    completionMode.type === "tools" ? args.buildToolBridgeConfig(completionMode.tools) : undefined;
  const useNativeToolExecution =
    completionMode.type === "tools" && toolBridgeConfig?.allToolsHaveExecutors === true;

  if (
    completionMode.type === "tools" &&
    toolBridgeConfig?.hasAnyExecutor === true &&
    !useNativeToolExecution
  ) {
    warnings.push(args.buildPartialToolExecutorWarning(toolBridgeConfig.missingExecutorToolNames));
  }

  if (completionMode.type === "json") {
    prompt = `Return only JSON that matches the required schema.\n\n${basePrompt}`;
    outputFormat = {
      type: "json_schema",
      schema: completionMode.schema,
    };
  }

  let queryPrompt: QueryPromptInput = prompt;
  const multimodalQueryPrompt = await buildMultimodalQueryPrompt({
    promptMessages: args.options.prompt,
    resumeSessionId: promptQueryInput.resumeSessionId,
    preambleText:
      completionMode.type === "json"
        ? "Return only JSON that matches the required schema."
        : undefined,
  });

  if (multimodalQueryPrompt !== undefined) {
    queryPrompt = multimodalQueryPrompt;
  }

  return {
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
    effort: anthropicOptions.effort,
    thinking: anthropicOptions.thinking,
  };
};

export const createAbortBridge = (externalAbortSignal: AbortSignal | undefined) => {
  const abortController = new AbortController();

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

  const cleanupAbortListener = () => {
    if (externalAbortSignal !== undefined) {
      externalAbortSignal.removeEventListener("abort", abortFromExternalSignal);
    }
  };

  return {
    abortController,
    cleanupAbortListener,
  };
};
