import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";
import { withOpenCodeCompatibility } from "./compat/open-code-compat";
import { anthropic as coreAnthropic, createAnthropic as createCoreAnthropic } from "./provider/create-anthropic";

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
  ToolExecutor,
  ToolExecutorMap,
} from "./shared/tool-executor";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export const createAnthropic = (...args: Parameters<typeof createCoreAnthropic>) => {
  return withOpenCodeCompatibility(createCoreAnthropic(...args));
};

export const anthropic = withOpenCodeCompatibility(coreAnthropic);
