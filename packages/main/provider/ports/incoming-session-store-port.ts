import type { IncomingSessionState } from "../incoming-session-store";

export type IncomingSessionStorePort = {
  get(args: {
    modelId: string;
    runtimeFingerprint: string;
    incomingSessionKey: string;
  }): Promise<IncomingSessionState | undefined>;
  set(args: {
    modelId: string;
    runtimeFingerprint: string;
    incomingSessionKey: string;
    state: IncomingSessionState;
  }): Promise<void>;
};
