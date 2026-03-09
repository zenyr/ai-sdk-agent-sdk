import { incomingSessionStore } from "../incoming-session-store";
import type { IncomingSessionStorePort } from "../ports/incoming-session-store-port";

export const fileIncomingSessionStore: IncomingSessionStorePort = incomingSessionStore;
