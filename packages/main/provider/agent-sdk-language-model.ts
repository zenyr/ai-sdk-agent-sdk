import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import { DEFAULT_SUPPORTED_URLS } from "../shared/constants";
import type { ToolExecutorMap } from "../shared/tool-executor";
import { claudeAgentRuntime } from "./adapters/claude-agent-runtime";
import { fileIncomingSessionStore } from "./adapters/file-incoming-session-store";
import { runGenerate } from "./application/generate";
import { runStream } from "./application/stream";
import {
  findIncomingSessionState,
  mergeIncomingSessionState,
} from "./domain/incoming-session-state";
import type { PromptSessionState } from "./domain/prompt-session-state";
import { collectProviderSettingWarnings } from "./domain/provider-setting-warnings";
import type { IncomingSessionState } from "./incoming-session-store";
import type { AgentRuntimePort } from "./ports/agent-runtime-port";
import type { IncomingSessionStorePort } from "./ports/incoming-session-store-port";

const buildPartialToolExecutorWarning = (missingExecutorToolNames: string[]): SharedV3Warning => {
  return {
    type: "compatibility",
    feature: "toolExecutors.partial",
    details:
      `toolExecutors is missing handlers for: ${missingExecutorToolNames.join(", ")}. ` +
      "Falling back to AI SDK tool loop with maxTurns=1.",
  };
};

const readSessionStoreErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }

  if (typeof error === "string" && error.length > 0) {
    return error;
  }

  return "unknown error";
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
  private readonly warnedSessionStoreFailures = new Set<string>();
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

  private reportSessionStoreFailure(args: {
    operation: "get" | "set";
    incomingSessionKey: string;
    error: unknown;
  }): void {
    const warningKey = `${args.operation}\u0000${args.incomingSessionKey}`;
    if (this.warnedSessionStoreFailures.has(warningKey)) {
      return;
    }

    this.warnedSessionStoreFailures.add(warningKey);

    const errorMessage = readSessionStoreErrorMessage(args.error);
    console.warn(
      "ai-sdk-agent-sdk: session store " +
        `${args.operation} failed (model=${this.modelId}, conversation=${args.incomingSessionKey}): ` +
        errorMessage,
    );
  }

  private async hydrateIncomingSessionState(incomingSessionKey: string): Promise<void> {
    const existingIncomingSessionState = findIncomingSessionState({
      incomingSessionKey,
      previousIncomingSessionStates: this.incomingSessionStates,
    });

    if (existingIncomingSessionState !== undefined) {
      return;
    }

    let persistedIncomingSessionState: IncomingSessionState | undefined;
    try {
      persistedIncomingSessionState = await this.sessionStore.get({
        modelId: this.modelId,
        incomingSessionKey,
      });
    } catch (error) {
      this.reportSessionStoreFailure({
        operation: "get",
        incomingSessionKey,
        error,
      });
      return;
    }

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

    try {
      await this.sessionStore.set({
        modelId: this.modelId,
        incomingSessionKey: incomingSessionState.incomingSessionKey,
        state: incomingSessionState,
      });
    } catch (error) {
      this.reportSessionStoreFailure({
        operation: "set",
        incomingSessionKey: incomingSessionState.incomingSessionKey,
        error,
      });
    }
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
    return runStream({
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
}
