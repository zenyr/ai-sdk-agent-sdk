import { describe, expect, test } from "bun:test";
import type { SharedV3Warning } from "@ai-sdk/provider";

import { collectAnthropicProviderOptionWarnings } from "../anthropic-option-warnings";

const isFeatureWarning = (warning: SharedV3Warning): warning is Extract<SharedV3Warning, { feature: string }> => {
  return "feature" in warning;
};

const isUnsupportedFeatureWarning = (
  warning: SharedV3Warning
): warning is Extract<SharedV3Warning, { type: "unsupported" }> => {
  return warning.type === "unsupported";
};

describe("collectAnthropicProviderOptionWarnings", () => {
  test("returns empty when provider options are missing", () => {
    const warnings = collectAnthropicProviderOptionWarnings({
      providerOptions: undefined,
      provider: "anthropic.messages",
    });
    expect(warnings).toEqual([]);
  });

  test("does not warn for mapped options", () => {
    const warnings = collectAnthropicProviderOptionWarnings({
      provider: "anthropic.messages",
      providerOptions: {
        anthropic: {
          effort: "low",
          thinking: { type: "enabled", budgetTokens: 1024 },
        },
      },
    });

    expect(warnings).toEqual([]);
  });

  test("warns for degraded, unsupported, and unknown options", () => {
    const warnings = collectAnthropicProviderOptionWarnings({
      provider: "my-agent",
      providerOptions: {
        "my-agent": {
          sendReasoning: true,
          cacheControl: { type: "ephemeral" },
          unknownOption: true,
        },
      },
    });

    expect(warnings.length).toBe(3);

    const features = warnings
      .filter(isFeatureWarning)
      .map(warning => warning.feature)
      .sort();

    expect(features).toEqual(["providerOptions.my-agent.cacheControl", "providerOptions.my-agent.sendReasoning"]);

    const cacheControlWarning = warnings.find(
      (warning): warning is Extract<SharedV3Warning, { type: "unsupported"; feature: string }> => {
        return isUnsupportedFeatureWarning(warning) && warning.feature === "providerOptions.my-agent.cacheControl";
      }
    );

    expect(cacheControlWarning).toBeDefined();

    if (cacheControlWarning !== undefined && typeof cacheControlWarning.details === "string") {
      expect(cacheControlWarning.details.includes("1h cache")).toBeTrue();
    }

    const otherMessages = warnings.filter(warning => warning.type === "other").map(warning => warning.message);

    expect(otherMessages.length).toBe(1);
    expect(otherMessages[0]?.includes("unknownOption")).toBeTrue();
  });
});
