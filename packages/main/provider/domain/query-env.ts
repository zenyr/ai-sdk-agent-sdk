import type { AnthropicProviderSettings } from "@ai-sdk/anthropic";
import { withoutTrailingSlash } from "@ai-sdk/provider-utils";

const readNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  if (value.length === 0) {
    return undefined;
  }

  return value;
};

export const buildQueryEnv = (
  settings: AnthropicProviderSettings,
): Record<string, string | undefined> => {
  const env: Record<string, string | undefined> = {
    ...process.env,
  };

  const apiKey = readNonEmptyString(settings.apiKey);
  if (apiKey !== undefined) {
    env.ANTHROPIC_API_KEY = apiKey;
  }

  const authToken = readNonEmptyString(settings.authToken);
  if (authToken !== undefined) {
    env.ANTHROPIC_AUTH_TOKEN = authToken;
  }

  const baseURL = readNonEmptyString(settings.baseURL);
  if (baseURL !== undefined) {
    const normalizedBaseURL = withoutTrailingSlash(baseURL);
    if (normalizedBaseURL !== undefined) {
      env.ANTHROPIC_BASE_URL = normalizedBaseURL;
    }
  }

  return env;
};
