import type { LanguageModelV3Usage } from "@ai-sdk/provider";

export type StreamBlockState =
  | {
      kind: "text";
      id: string;
    }
  | {
      kind: "reasoning";
      id: string;
    }
  | {
      kind: "tool-input";
      id: string;
    };

export type StreamEventState = {
  blockStates: Map<number, StreamBlockState>;
  emittedResponseMetadata: boolean;
  latestStopReason: string | null;
  latestUsage: LanguageModelV3Usage | undefined;
};

export const createEmptyUsage = (): LanguageModelV3Usage => {
  return {
    inputTokens: {
      total: undefined,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: undefined,
      text: undefined,
      reasoning: undefined,
    },
  };
};
