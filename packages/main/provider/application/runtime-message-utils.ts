import type {
  SDKAssistantMessage,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
  SDKSystemMessage,
} from "@anthropic-ai/claude-agent-sdk";

import { isRecord, readNumber, readString, safeJsonStringify } from "../../shared/type-readers";

export const EMPTY_TOOL_ROUTING_OUTPUT_ERROR = "empty-tool-routing-output";
export const EMPTY_TOOL_ROUTING_OUTPUT_TEXT =
  "Tool routing finished without a final text response or recoverable tool call. This often means provider-side tool execution failed or no tool delegate was configured.";

export const RUNTIME_QUERY_ERROR = "runtime-query-error";

const readErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }

  if (typeof error === "string" && error.length > 0) {
    return error;
  }

  return safeJsonStringify(error);
};

const readRuntimeErrorDetail = (error: unknown, key: string): string | undefined => {
  if (!isRecord(error)) {
    return undefined;
  }

  const value = readString(error, key);
  if (value === undefined || value.length === 0) {
    return undefined;
  }

  return value;
};

const readRuntimeExitCode = (error: unknown): number | undefined => {
  if (!isRecord(error)) {
    return undefined;
  }

  return readNumber(error, "exitCode") ?? readNumber(error, "code");
};

const includesAny = (message: string, patterns: string[]): boolean => {
  return patterns.some(pattern => {
    return message.includes(pattern);
  });
};

export const normalizeRuntimeQueryError = (error: unknown, stderrText?: string): { message: string; raw: string } => {
  const stdout = readRuntimeErrorDetail(error, "stdout");
  const stderrFromError = readRuntimeErrorDetail(error, "stderr");
  const shortMessage = readRuntimeErrorDetail(error, "shortMessage");
  const exitCode = readRuntimeExitCode(error);
  const rawMessage = stderrFromError ?? stderrText ?? shortMessage ?? readErrorMessage(error);
  const normalizedMessage = [stderrFromError, stderrText, stdout, shortMessage, rawMessage]
    .filter(Boolean)
    .join("\n")
    .toLowerCase();

  if (rawMessage.trim().length === 0) {
    return {
      message:
        "Claude Code request failed before returning a result. Check Claude CLI stderr or debug logs for the underlying cause.",
      raw: RUNTIME_QUERY_ERROR,
    };
  }

  if (
    includesAny(normalizedMessage, [
      "quota",
      "usage limit",
      "credit balance",
      "billing",
      "rate limit",
      "429",
      "over limit",
      "exceeded your current quota",
    ])
  ) {
    return {
      message:
        "Claude request failed because quota appears exhausted. Wait for quota reset or switch to another account or API key.",
      raw: "runtime-query-quota-exhausted",
    };
  }

  if (
    includesAny(normalizedMessage, [
      "api key",
      "auth",
      "authentication",
      "unauthorized",
      "forbidden",
      "invalid x-api-key",
      "login",
      "logged out",
    ])
  ) {
    return {
      message:
        "Claude request failed because authentication appears invalid or expired. Re-authenticate or set a valid API key.",
      raw: "runtime-query-auth-failed",
    };
  }

  if (
    includesAny(normalizedMessage, [
      "no conversation found",
      "session isn't found",
      "session not found",
      "error opening file",
      "resume",
      "session id",
    ])
  ) {
    return {
      message:
        "Claude request failed because the prior session could not be resumed. Start a new session or clear the saved conversation state.",
      raw: "runtime-query-session-missing",
    };
  }

  if (includesAny(normalizedMessage, ["process exited with code 1", "process exited with code"])) {
    return {
      message:
        exitCode === 1
          ? "Claude Code exited before returning a result. This often happens when the current Claude session quota is exhausted, auth has expired, or resume state is broken."
          : "Claude Code exited before returning a result.",
      raw: "runtime-query-process-exit",
    };
  }

  return {
    message: rawMessage,
    raw: RUNTIME_QUERY_ERROR,
  };
};

export const isAssistantMessage = (message: SDKMessage): message is SDKAssistantMessage => {
  return message.type === "assistant";
};

export const isResultMessage = (message: SDKMessage): message is SDKResultMessage => {
  return message.type === "result";
};

export const isPartialAssistantMessage = (message: SDKMessage): message is SDKPartialAssistantMessage => {
  return message.type === "stream_event";
};

export const isSystemInitMessage = (message: SDKMessage): message is SDKSystemMessage => {
  return message.type === "system" && message.subtype === "init";
};

export const isStructuredOutputRetryExhausted = (resultMessage: SDKResultMessage): boolean => {
  return resultMessage.subtype === "error_max_structured_output_retries";
};

export const extractAssistantText = (assistantMessage: SDKAssistantMessage | undefined): string => {
  if (assistantMessage === undefined) {
    return "";
  }

  const contentBlocks = assistantMessage.message.content;
  if (!Array.isArray(contentBlocks)) {
    return "";
  }

  return contentBlocks
    .map(block => {
      if (!isRecord(block)) {
        return "";
      }

      if (block.type !== "text") {
        return "";
      }

      const textPart = readString(block, "text");
      return typeof textPart === "string" ? textPart : "";
    })
    .join("");
};
