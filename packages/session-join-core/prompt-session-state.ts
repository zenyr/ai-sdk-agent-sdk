import type { LanguageModelV3Message } from "@ai-sdk/provider";

import {
  joinSerializedPromptMessages,
  serializePromptMessages,
  serializePromptMessagesForResumeQuery,
  serializePromptMessagesWithoutSystem,
} from "./prompt-serializer";

const MAX_PROMPT_SESSION_STATES = 20;

export type PromptSessionState = {
  sessionId: string;
  serializedPromptMessages: string[];
};

export type PromptQueryInput = {
  prompt: string;
  serializedPromptMessages?: string[];
  resumeSessionId?: string;
};

const hasSerializedPromptPrefix = (prefix: string[], target: string[]): boolean => {
  if (prefix.length > target.length) {
    return false;
  }

  for (let index = 0; index < prefix.length; index += 1) {
    if (prefix[index] !== target[index]) {
      return false;
    }
  }

  return true;
};

const hasIdenticalSerializedPrompt = (source: string[], target: string[]): boolean => {
  if (source.length !== target.length) {
    return false;
  }

  for (let index = 0; index < source.length; index += 1) {
    if (source[index] !== target[index]) {
      return false;
    }
  }

  return true;
};

const findBestPromptSessionState = (args: {
  serializedPromptMessages: string[];
  previousSessionStates: PromptSessionState[];
}): PromptSessionState | undefined => {
  let bestState: PromptSessionState | undefined;

  for (const sessionState of args.previousSessionStates) {
    if (
      !hasSerializedPromptPrefix(
        sessionState.serializedPromptMessages,
        args.serializedPromptMessages,
      )
    ) {
      continue;
    }

    if (
      bestState === undefined ||
      sessionState.serializedPromptMessages.length > bestState.serializedPromptMessages.length
    ) {
      bestState = sessionState;
    }
  }

  return bestState;
};

export const mergePromptSessionState = (args: {
  previousSessionStates: PromptSessionState[];
  nextSessionState: PromptSessionState;
}): PromptSessionState[] => {
  const dedupedStates = args.previousSessionStates.filter((sessionState) => {
    if (sessionState.sessionId === args.nextSessionState.sessionId) {
      return false;
    }

    return !hasIdenticalSerializedPrompt(
      sessionState.serializedPromptMessages,
      args.nextSessionState.serializedPromptMessages,
    );
  });

  return [args.nextSessionState, ...dedupedStates].slice(0, MAX_PROMPT_SESSION_STATES);
};

export const buildPromptQueryInput = (args: {
  promptMessages: LanguageModelV3Message[];
  previousSessionStates: PromptSessionState[];
}): PromptQueryInput => {
  const serializedPromptMessages = serializePromptMessages(args.promptMessages);
  const serializedPromptMessagesForQuery = serializePromptMessagesWithoutSystem(
    args.promptMessages,
  );
  const fullPrompt = joinSerializedPromptMessages(serializedPromptMessagesForQuery);
  const previousSessionState = findBestPromptSessionState({
    serializedPromptMessages,
    previousSessionStates: args.previousSessionStates,
  });

  if (previousSessionState === undefined) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const previousPromptMessages = previousSessionState.serializedPromptMessages;
  if (!hasSerializedPromptPrefix(previousPromptMessages, serializedPromptMessages)) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const appendedPromptMessages = serializedPromptMessages.slice(previousPromptMessages.length);
  if (appendedPromptMessages.length === 0) {
    return {
      prompt: fullPrompt,
      serializedPromptMessages,
    };
  }

  const appendedSourceMessages = args.promptMessages.slice(previousPromptMessages.length);
  const appendedPromptMessagesForQuery =
    serializePromptMessagesForResumeQuery(appendedSourceMessages);

  if (appendedPromptMessagesForQuery.length === 0) {
    const fallbackAppendedPromptMessages =
      serializePromptMessagesWithoutSystem(appendedSourceMessages);

    if (fallbackAppendedPromptMessages.length === 0) {
      return {
        prompt: fullPrompt,
        serializedPromptMessages,
      };
    }

    return {
      prompt: joinSerializedPromptMessages(fallbackAppendedPromptMessages),
      serializedPromptMessages,
      resumeSessionId: previousSessionState.sessionId,
    };
  }

  return {
    prompt: joinSerializedPromptMessages(appendedPromptMessagesForQuery),
    serializedPromptMessages,
    resumeSessionId: previousSessionState.sessionId,
  };
};

export const buildPromptQueryInputWithoutResume = (
  promptMessages: LanguageModelV3Message[],
): PromptQueryInput => {
  const serializedPromptMessagesForQuery = serializePromptMessagesWithoutSystem(promptMessages);

  return {
    prompt: joinSerializedPromptMessages(serializedPromptMessagesForQuery),
  };
};

export const buildResumePromptQueryInput = (args: {
  promptMessages: LanguageModelV3Message[];
  resumeSessionId: string;
}): PromptQueryInput | undefined => {
  const appendedPromptMessagesForQuery = serializePromptMessagesForResumeQuery(args.promptMessages);

  if (appendedPromptMessagesForQuery.length > 0) {
    return {
      prompt: joinSerializedPromptMessages(appendedPromptMessagesForQuery),
      resumeSessionId: args.resumeSessionId,
    };
  }

  const fallbackPromptMessages = serializePromptMessagesWithoutSystem(args.promptMessages);
  if (fallbackPromptMessages.length === 0) {
    return undefined;
  }

  return {
    prompt: joinSerializedPromptMessages(fallbackPromptMessages),
    resumeSessionId: args.resumeSessionId,
  };
};
