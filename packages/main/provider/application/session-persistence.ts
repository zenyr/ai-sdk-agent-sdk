import type { LanguageModelV3Message } from "@ai-sdk/provider";
import type {
  SDKAssistantMessage,
  SDKResultMessage,
  SDKSystemMessage,
} from "@anthropic-ai/claude-agent-sdk";

import {
  buildIncomingSessionState,
  readSessionIdFromQueryMessages,
} from "../domain/incoming-session-state";
import { mergePromptSessionState, type PromptSessionState } from "../domain/prompt-session-state";
import type { IncomingSessionState } from "../incoming-session-store";

type PersistQuerySessionStateArgs = {
  resultMessage: SDKResultMessage | undefined;
  assistantMessage: SDKAssistantMessage | undefined;
  initSystemMessage: SDKSystemMessage | undefined;
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
