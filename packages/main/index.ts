import type { AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";
import {
  anthropic as anthropicV3,
  createAnthropic as createAnthropicV3,
} from "./provider/create-anthropic";
import { anthropic as anthropicV2, createAnthropic as createAnthropicV2 } from "./v2";

export type {
  AnthropicLanguageModelOptions,
  AnthropicMessageMetadata,
  AnthropicProvider,
  AnthropicProviderSettings,
  AnthropicToolOptions,
  AnthropicUsageIteration,
} from "@ai-sdk/anthropic";
export { forwardAnthropicContainerIdFromLastStep } from "./provider/forward-container";
export { VERSION } from "./shared/constants";
export type {
  AgentSdkProviderSettings,
  ToolExecutor,
  ToolExecutorMap,
} from "./shared/tool-executor";

export type AnthropicProviderOptions = AnthropicLanguageModelOptions;

export const isOpenCode = (): boolean => {
  const value = process.env.OPENCODE;
  return typeof value === "string" && value.length > 0 && value !== "0" && value !== "false";
};

export const createAnthropic: typeof createAnthropicV3 = isOpenCode()
  ? createAnthropicV2
  : createAnthropicV3;

export const anthropic = isOpenCode() ? anthropicV2 : anthropicV3;
