import type { LanguageModelV3Message } from "@ai-sdk/provider";

import {
  buildIncomingSessionState,
  readSessionIdFromQueryMessages,
} from "../domain/incoming-session-state";
import { mergePromptSessionState, type PromptSessionState } from "../domain/prompt-session-state";
import type { IncomingSessionState } from "../incoming-session-store";

type PersistQuerySessionStateArgs = {
  resultMessage: Record<string, unknown> | undefined;
  assistantMessage: Record<string, unknown> | undefined;
  initSystemMessage: Record<string, unknown> | undefined;
  incomingSessionKey: string | undefined;
  serializedPromptMessages: string[] | undefined;
  promptMessages: LanguageModelV3Message[];
  previousSessionStates: () => PromptSessionState[];
  setPromptSessionStates: (sessionStates: PromptSessionState[]) => void;
  persistIncomingSessionState: (incomingSessionState: IncomingSessionState) => Promise<void>;
};

export const persistQuerySessionState = async (
  args: PersistQuerySessionStateArgs,
): Promise<void> => {
  const sessionId = readSessionIdFromQueryMessages({
    resultMessage: args.resultMessage,
    assistantMessage: args.assistantMessage,
    initSystemMessage: args.initSystemMessage,
  });

  if (sessionId === undefined) {
    return;
  }

  if (args.serializedPromptMessages !== undefined) {
    args.setPromptSessionStates(
      mergePromptSessionState({
        previousSessionStates: args.previousSessionStates(),
        nextSessionState: {
          sessionId,
          serializedPromptMessages: args.serializedPromptMessages,
        },
      }),
    );
  }

  if (args.incomingSessionKey === undefined) {
    return;
  }

  await args.persistIncomingSessionState(
    buildIncomingSessionState({
      incomingSessionKey: args.incomingSessionKey,
      sessionId,
      promptMessages: args.promptMessages,
    }),
  );
};
