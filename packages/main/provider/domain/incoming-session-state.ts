import type { LanguageModelV3Message } from "@ai-sdk/provider";
import { serializeMessage } from "../../bridge/prompt-serializer";
import { isRecord, readString } from "../../shared/type-readers";
import type { IncomingSessionState } from "../incoming-session-store";
import {
  buildPromptQueryInput,
  buildPromptQueryInputWithoutResume,
  buildResumePromptQueryInput,
  type PromptQueryInput,
  type PromptSessionState,
} from "./prompt-session-state";

const MAX_INCOMING_SESSION_STATES = 100;

const readPromptMessageSignature = (
  message: LanguageModelV3Message | undefined,
): string | undefined => {
  if (message === undefined) {
    return undefined;
  }

  const signature = serializeMessage(message);
  if (signature.length === 0) {
    return undefined;
  }

  return signature;
};

export const findIncomingSessionState = (args: {
  incomingSessionKey: string;
  previousIncomingSessionStates: IncomingSessionState[];
}): IncomingSessionState | undefined => {
  return args.previousIncomingSessionStates.find((incomingSessionState) => {
    return incomingSessionState.incomingSessionKey === args.incomingSessionKey;
  });
};

export const mergeIncomingSessionState = (args: {
  previousIncomingSessionStates: IncomingSessionState[];
  nextIncomingSessionState: IncomingSessionState;
}): IncomingSessionState[] => {
  const dedupedStates = args.previousIncomingSessionStates.filter((incomingSessionState) => {
    return (
      incomingSessionState.incomingSessionKey !== args.nextIncomingSessionState.incomingSessionKey
    );
  });

  return [args.nextIncomingSessionState, ...dedupedStates].slice(0, MAX_INCOMING_SESSION_STATES);
};

export const buildIncomingSessionState = (args: {
  incomingSessionKey: string;
  sessionId: string;
  promptMessages: LanguageModelV3Message[];
}): IncomingSessionState => {
  const promptMessageCount = args.promptMessages.length;
  const firstPromptMessageSignature = readPromptMessageSignature(args.promptMessages[0]);
  const lastPromptMessageSignature = readPromptMessageSignature(
    args.promptMessages[promptMessageCount - 1],
  );

  return {
    incomingSessionKey: args.incomingSessionKey,
    sessionId: args.sessionId,
    promptMessageCount,
    firstPromptMessageSignature,
    lastPromptMessageSignature,
  };
};

const isSingleUserPrompt = (promptMessages: LanguageModelV3Message[]): boolean => {
  if (promptMessages.length !== 1) {
    return false;
  }

  const firstPromptMessage = promptMessages[0];
  return firstPromptMessage !== undefined && firstPromptMessage.role === "user";
};

const buildPromptQueryInputFromIncomingSession = (args: {
  incomingSessionState: IncomingSessionState;
  promptMessages: LanguageModelV3Message[];
}): PromptQueryInput | undefined => {
  const previousPromptMessageCount = args.incomingSessionState.promptMessageCount;
  const { promptMessages } = args;

  if (promptMessages.length > previousPromptMessageCount) {
    if (previousPromptMessageCount > 0) {
      const firstPromptMessageSignature = readPromptMessageSignature(promptMessages[0]);
      if (
        args.incomingSessionState.firstPromptMessageSignature !== undefined &&
        firstPromptMessageSignature !== args.incomingSessionState.firstPromptMessageSignature
      ) {
        return undefined;
      }

      const previousLastPromptMessage = promptMessages[previousPromptMessageCount - 1];
      const previousLastPromptMessageSignature =
        readPromptMessageSignature(previousLastPromptMessage);
      if (
        args.incomingSessionState.lastPromptMessageSignature !== undefined &&
        previousLastPromptMessageSignature !== args.incomingSessionState.lastPromptMessageSignature
      ) {
        return undefined;
      }
    }

    return buildResumePromptQueryInput({
      promptMessages: promptMessages.slice(previousPromptMessageCount),
      resumeSessionId: args.incomingSessionState.sessionId,
    });
  }

  if (!isSingleUserPrompt(promptMessages)) {
    return undefined;
  }

  return buildResumePromptQueryInput({
    promptMessages,
    resumeSessionId: args.incomingSessionState.sessionId,
  });
};

export const buildPromptQueryInputWithIncomingSession = (args: {
  promptMessages: LanguageModelV3Message[];
  incomingSessionKey: string | undefined;
  previousSessionStates: PromptSessionState[];
  previousIncomingSessionStates: IncomingSessionState[];
}): PromptQueryInput => {
  const { incomingSessionKey } = args;

  if (incomingSessionKey === undefined) {
    return buildPromptQueryInput({
      promptMessages: args.promptMessages,
      previousSessionStates: args.previousSessionStates,
    });
  }

  const incomingSessionState = findIncomingSessionState({
    incomingSessionKey,
    previousIncomingSessionStates: args.previousIncomingSessionStates,
  });

  if (incomingSessionState === undefined) {
    return buildPromptQueryInputWithoutResume(args.promptMessages);
  }

  const promptQueryInputFromIncomingSession = buildPromptQueryInputFromIncomingSession({
    incomingSessionState,
    promptMessages: args.promptMessages,
  });

  if (promptQueryInputFromIncomingSession !== undefined) {
    return promptQueryInputFromIncomingSession;
  }

  return buildPromptQueryInput({
    promptMessages: args.promptMessages,
    previousSessionStates: args.previousSessionStates,
  });
};

export const readSessionIdFromQueryMessages = (args: {
  resultMessage: Record<string, unknown> | undefined;
  assistantMessage: Record<string, unknown> | undefined;
  initSystemMessage: Record<string, unknown> | undefined;
}): string | undefined => {
  if (isRecord(args.resultMessage)) {
    const sessionIdFromResult = readString(args.resultMessage, "session_id");
    if (sessionIdFromResult !== undefined) {
      return sessionIdFromResult;
    }
  }

  const sessionIdFromAssistantMessage = isRecord(args.assistantMessage)
    ? readString(args.assistantMessage, "session_id")
    : undefined;
  if (sessionIdFromAssistantMessage !== undefined) {
    return sessionIdFromAssistantMessage;
  }

  if (!isRecord(args.initSystemMessage) || args.initSystemMessage.subtype !== "init") {
    return undefined;
  }

  return readString(args.initSystemMessage, "session_id");
};
