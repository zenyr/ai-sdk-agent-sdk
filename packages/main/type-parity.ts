type UpstreamAnthropic = typeof import("@ai-sdk/anthropic");
type LocalAnthropic = typeof import("./index");

import type {
  AnthropicLanguageModelOptions as UpstreamAnthropicLanguageModelOptions,
  AnthropicMessageMetadata as UpstreamAnthropicMessageMetadata,
  AnthropicProvider as UpstreamAnthropicProvider,
  AnthropicProviderSettings as UpstreamAnthropicProviderSettings,
  AnthropicToolOptions as UpstreamAnthropicToolOptions,
  AnthropicUsageIteration as UpstreamAnthropicUsageIteration,
} from "@ai-sdk/anthropic";
import type {
  AnthropicLanguageModelOptions as LocalAnthropicLanguageModelOptions,
  AnthropicMessageMetadata as LocalAnthropicMessageMetadata,
  AnthropicProvider as LocalAnthropicProvider,
  AnthropicProviderSettings as LocalAnthropicProviderSettings,
  AnthropicToolOptions as LocalAnthropicToolOptions,
  AnthropicUsageIteration as LocalAnthropicUsageIteration,
} from "./index";

type AssertNever<T extends never> = T;
type AssertAssignable<From, _To extends From> = true;

// Extensions that exist in local but not in upstream (intentional additions)
type LocalOnlyKeys = "isOpenCode";

type MissingInLocal = Exclude<keyof UpstreamAnthropic, keyof LocalAnthropic>;
type MissingInUpstream = Exclude<
  Exclude<keyof LocalAnthropic, keyof UpstreamAnthropic>,
  LocalOnlyKeys
>;

type _assertNoMissingInLocal = AssertNever<MissingInLocal>;
type _assertNoMissingInUpstream = AssertNever<MissingInUpstream>;

declare const localProvider: ReturnType<LocalAnthropic["createAnthropic"]>;
const _providerCompatibility: UpstreamAnthropicProvider = localProvider;

type _providerTypeForward = AssertAssignable<UpstreamAnthropicProvider, LocalAnthropicProvider>;
type _providerTypeBackward = AssertAssignable<LocalAnthropicProvider, UpstreamAnthropicProvider>;

type _providerSettingsForward = AssertAssignable<
  UpstreamAnthropicProviderSettings,
  LocalAnthropicProviderSettings
>;
type _providerSettingsBackward = AssertAssignable<
  LocalAnthropicProviderSettings,
  UpstreamAnthropicProviderSettings
>;

type _languageModelOptionsForward = AssertAssignable<
  UpstreamAnthropicLanguageModelOptions,
  LocalAnthropicLanguageModelOptions
>;
type _languageModelOptionsBackward = AssertAssignable<
  LocalAnthropicLanguageModelOptions,
  UpstreamAnthropicLanguageModelOptions
>;

type _messageMetadataForward = AssertAssignable<
  UpstreamAnthropicMessageMetadata,
  LocalAnthropicMessageMetadata
>;
type _messageMetadataBackward = AssertAssignable<
  LocalAnthropicMessageMetadata,
  UpstreamAnthropicMessageMetadata
>;

type _toolOptionsForward = AssertAssignable<
  UpstreamAnthropicToolOptions,
  LocalAnthropicToolOptions
>;
type _toolOptionsBackward = AssertAssignable<
  LocalAnthropicToolOptions,
  UpstreamAnthropicToolOptions
>;

type _usageIterationForward = AssertAssignable<
  UpstreamAnthropicUsageIteration,
  LocalAnthropicUsageIteration
>;
type _usageIterationBackward = AssertAssignable<
  LocalAnthropicUsageIteration,
  UpstreamAnthropicUsageIteration
>;

void _providerCompatibility;
