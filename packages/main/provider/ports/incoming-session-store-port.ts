import type { IncomingSessionState } from "../incoming-session-store";

export type IncomingSessionStorePort = {
  get(args: {
    modelId: string;
    incomingSessionKey: string;
  }): Promise<IncomingSessionState | undefined>;
  set(args: {
    modelId: string;
    incomingSessionKey: string;
    state: IncomingSessionState;
  }): Promise<void>;
};
