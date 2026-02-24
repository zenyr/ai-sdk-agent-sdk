import type { LanguageModelV3CallOptions } from "@ai-sdk/provider";

import { isRecord, readRecord } from "../../shared/type-readers";

const CONVERSATION_ID_HEADER = "x-conversation-id";
const LEGACY_OPENCODE_SESSION_HEADER = "x-opencode-session";
const CONVERSATION_ID_CANDIDATE_KEYS = ["conversationId", "conversationID", "conversation_id"];
const LEGACY_INCOMING_SESSION_CANDIDATE_KEYS = [
  "sessionId",
  "sessionID",
  "session_id",
  "promptCacheKey",
  "prompt_cache_key",
  "opencodeSession",
  "opencode_session",
];
const CANONICAL_PROVIDER_OPTIONS_NAMESPACES = ["agentSdk", "agent_sdk", "agent-sdk"];
const LEGACY_PROVIDER_OPTIONS_NAMESPACES = ["opencode", "anthropic"];

const readNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  if (value.length === 0) {
    return undefined;
  }

  return value;
};

const readFirstNonEmptyString = (
  source: Record<string, unknown> | undefined,
  keys: string[],
): string | undefined => {
  if (source === undefined) {
    return undefined;
  }

  for (const key of keys) {
    const value = readNonEmptyString(source[key]);
    if (value !== undefined) {
      return value;
    }
  }

  return undefined;
};

const readIncomingSessionKeyFromRecord = (
  source: Record<string, unknown> | undefined,
): string | undefined => {
  if (source === undefined) {
    return undefined;
  }

  return (
    readFirstNonEmptyString(source, CONVERSATION_ID_CANDIDATE_KEYS) ??
    readFirstNonEmptyString(source, LEGACY_INCOMING_SESSION_CANDIDATE_KEYS)
  );
};

const readCaseInsensitiveHeader = (headers: unknown, headerName: string): string | undefined => {
  if (!isRecord(headers)) {
    return undefined;
  }

  const directHeaderValue = readNonEmptyString(headers[headerName]);
  if (directHeaderValue !== undefined) {
    return directHeaderValue;
  }

  const headerGetter = headers.get;
  if (typeof headerGetter === "function") {
    try {
      const valueFromGetter = readNonEmptyString(headerGetter.call(headers, headerName));
      if (valueFromGetter !== undefined) {
        return valueFromGetter;
      }
    } catch {}
  }

  const normalizedHeaderName = headerName.toLowerCase();
  for (const [rawHeaderName, rawHeaderValue] of Object.entries(headers)) {
    if (rawHeaderName.toLowerCase() !== normalizedHeaderName) {
      continue;
    }

    const normalizedHeaderValue = readNonEmptyString(rawHeaderValue);
    if (normalizedHeaderValue !== undefined) {
      return normalizedHeaderValue;
    }
  }

  return undefined;
};

const readIncomingSessionKeyFromHeaders = (
  optionsRecord: Record<string, unknown>,
): string | undefined => {
  return (
    readCaseInsensitiveHeader(optionsRecord.headers, CONVERSATION_ID_HEADER) ??
    readCaseInsensitiveHeader(optionsRecord.headers, LEGACY_OPENCODE_SESSION_HEADER)
  );
};

const readIncomingSessionKeyFromTelemetry = (
  optionsRecord: Record<string, unknown>,
): string | undefined => {
  const telemetry = readRecord(optionsRecord, "experimental_telemetry");
  if (telemetry === undefined) {
    return undefined;
  }

  const metadata = readRecord(telemetry, "metadata");
  if (metadata === undefined) {
    return undefined;
  }

  return readIncomingSessionKeyFromRecord(metadata);
};

const readIncomingSessionKeyFromProviderOptions = (
  optionsRecord: Record<string, unknown>,
): string | undefined => {
  const providerOptions = readRecord(optionsRecord, "providerOptions");
  if (providerOptions === undefined) {
    return undefined;
  }

  for (const namespace of CANONICAL_PROVIDER_OPTIONS_NAMESPACES) {
    const canonicalOptions = readRecord(providerOptions, namespace);
    if (canonicalOptions === undefined) {
      continue;
    }

    const canonicalConversationId = readIncomingSessionKeyFromRecord(canonicalOptions);
    if (canonicalConversationId !== undefined) {
      return canonicalConversationId;
    }
  }

  for (const namespace of LEGACY_PROVIDER_OPTIONS_NAMESPACES) {
    const legacyOptions = readRecord(providerOptions, namespace);
    if (legacyOptions === undefined) {
      continue;
    }

    const legacyConversationId = readIncomingSessionKeyFromRecord(legacyOptions);
    if (legacyConversationId !== undefined) {
      return legacyConversationId;
    }
  }

  const knownNamespaces = new Set<string>([
    ...CANONICAL_PROVIDER_OPTIONS_NAMESPACES,
    ...LEGACY_PROVIDER_OPTIONS_NAMESPACES,
  ]);

  for (const [namespace, value] of Object.entries(providerOptions)) {
    if (knownNamespaces.has(namespace)) {
      continue;
    }

    const namespaceOptions = isRecord(value) ? value : undefined;
    if (namespaceOptions === undefined) {
      continue;
    }

    const discoveredSessionKey = readIncomingSessionKeyFromRecord(namespaceOptions);
    if (discoveredSessionKey !== undefined) {
      return discoveredSessionKey;
    }
  }

  return readIncomingSessionKeyFromRecord(providerOptions);
};

export const readIncomingSessionKey = (options: LanguageModelV3CallOptions): string | undefined => {
  if (!isRecord(options)) {
    return undefined;
  }

  return (
    readIncomingSessionKeyFromHeaders(options) ??
    readIncomingSessionKeyFromTelemetry(options) ??
    readIncomingSessionKeyFromProviderOptions(options)
  );
};
