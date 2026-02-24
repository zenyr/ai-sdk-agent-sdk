import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import type { SharedV3Warning } from "@ai-sdk/provider";

import { isRecord } from "../../shared/type-readers";

export const collectProviderSettingWarnings = (
  settings: AnthropicProviderSettings,
): SharedV3Warning[] => {
  const warnings: SharedV3Warning[] = [];

  const headers = settings.headers;
  if (isRecord(headers) && Object.keys(headers).length > 0) {
    warnings.push({
      type: "unsupported",
      feature: "providerSettings.headers",
      details: "createAnthropic({ headers }) is not forwarded on claude-agent-sdk backend.",
    });
  }

  if (typeof settings.fetch === "function") {
    warnings.push({
      type: "unsupported",
      feature: "providerSettings.fetch",
      details: "createAnthropic({ fetch }) is not forwarded on claude-agent-sdk backend.",
    });
  }

  return warnings;
};
