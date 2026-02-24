import { describe, expect, test } from "bun:test";
import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import { collectProviderSettingWarnings } from "../domain/provider-setting-warnings";
import { buildQueryEnv } from "../domain/query-env";

describe("provider domain config helpers", () => {
  test("buildQueryEnv forwards non-empty auth settings and normalizes baseURL", () => {
    const settings: AnthropicProviderSettings = {
      apiKey: "api-key-1",
      authToken: "auth-token-1",
      baseURL: "https://example.test/v1/",
    };

    const env = buildQueryEnv(settings);

    expect(env.ANTHROPIC_API_KEY).toBe("api-key-1");
    expect(env.ANTHROPIC_AUTH_TOKEN).toBe("auth-token-1");
    expect(env.ANTHROPIC_BASE_URL).toBe("https://example.test/v1");
  });

  test("collectProviderSettingWarnings reports unsupported provider-level headers and fetch", () => {
    const settings: AnthropicProviderSettings = {
      headers: {
        "x-test-header": "value",
      },
      fetch,
    };

    const warnings = collectProviderSettingWarnings(settings);
    const unsupportedWarnings = warnings.filter((warning) => {
      return warning.type === "unsupported";
    });
    const unsupportedFeatures = unsupportedWarnings.map((warning) => {
      return warning.feature;
    });

    expect(warnings).toHaveLength(2);
    expect(unsupportedFeatures).toContain("providerSettings.headers");
    expect(unsupportedFeatures).toContain("providerSettings.fetch");
  });
});
