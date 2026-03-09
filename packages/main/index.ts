import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";
import { createAnthropic as createLegacyAnthropic, anthropic as legacyAnthropic } from "./v2";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export { withOpenCodeCompatibility } from "./compat/open-code-compat";
export { forwardAnthropicContainerIdFromLastStep } from "./provider/forward-container";
export { VERSION } from "./shared/constants";
export type {
  AgentSdkProviderSettings,
  ToolCallDelegate,
  ToolExecutionRequest,
  ToolExecutor,
  ToolExecutorMap,
} from "./shared/tool-executor";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export const createAnthropic = (...args: Parameters<typeof createLegacyAnthropic>) => {
  return createLegacyAnthropic(...args);
};

export const anthropic = legacyAnthropic;
