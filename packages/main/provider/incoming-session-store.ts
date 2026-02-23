import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { isRecord, readString } from "../shared/type-readers";

export type IncomingSessionState = {
  incomingSessionKey: string;
  sessionId: string;
  promptMessageCount: number;
  firstPromptMessageSignature?: string;
  lastPromptMessageSignature?: string;
};

type PersistedIncomingSessionState = {
  version: 1;
  incomingSessionKey: string;
  sessionId: string;
  promptMessageCount: number;
  firstPromptMessageSignature?: string;
  lastPromptMessageSignature?: string;
  updatedAt: number;
};

type CacheFilePathInput = {
  modelId: string;
  incomingSessionKey: string;
};

const SESSION_CACHE_ROOT_DIRNAME = "ai-sdk-agent-sdk";
const SESSION_CACHE_SUBDIR = "session-join";
const SESSION_CACHE_VERSION = "v1";
const MEMORY_CACHE_LIMIT = 100;

const readNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  if (value.length === 0) {
    return undefined;
  }

  return value;
};

const readOptionalString = (record: Record<string, unknown>, key: string): string | undefined => {
  const value = record[key];
  if (value === undefined) {
    return undefined;
  }

  return readNonEmptyString(value);
};

const readNonNegativeInteger = (value: unknown): number | undefined => {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    return undefined;
  }

  return value;
};

const encodePathSegment = (value: string): string => {
  return Buffer.from(value, "utf8").toString("base64url");
};

const readBucketKey = (encodedConversationKey: string): string => {
  if (encodedConversationKey.length >= 2) {
    return encodedConversationKey.slice(0, 2);
  }

  if (encodedConversationKey.length === 1) {
    return `${encodedConversationKey}_`;
  }

  return "__";
};

const resolveCacheHomePath = (): string => {
  const xdgCacheHome = readNonEmptyString(process.env.XDG_CACHE_HOME);
  if (xdgCacheHome !== undefined) {
    return xdgCacheHome;
  }

  return join(homedir(), ".cache");
};

const parsePersistedIncomingSessionState = (value: unknown): IncomingSessionState | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const version = value.version;
  if (version !== 1) {
    return undefined;
  }

  const incomingSessionKey = readString(value, "incomingSessionKey");
  if (incomingSessionKey === undefined) {
    return undefined;
  }

  const sessionId = readString(value, "sessionId");
  if (sessionId === undefined) {
    return undefined;
  }

  const promptMessageCount = readNonNegativeInteger(value.promptMessageCount);
  if (promptMessageCount === undefined) {
    return undefined;
  }

  return {
    incomingSessionKey,
    sessionId,
    promptMessageCount,
    firstPromptMessageSignature: readOptionalString(value, "firstPromptMessageSignature"),
    lastPromptMessageSignature: readOptionalString(value, "lastPromptMessageSignature"),
  };
};

const buildPersistedIncomingSessionState = (
  incomingSessionState: IncomingSessionState,
): PersistedIncomingSessionState => {
  return {
    version: 1,
    incomingSessionKey: incomingSessionState.incomingSessionKey,
    sessionId: incomingSessionState.sessionId,
    promptMessageCount: incomingSessionState.promptMessageCount,
    firstPromptMessageSignature: incomingSessionState.firstPromptMessageSignature,
    lastPromptMessageSignature: incomingSessionState.lastPromptMessageSignature,
    updatedAt: Date.now(),
  };
};

class IncomingSessionStore {
  private readonly cacheRootPath: string;
  private readonly memoryCache = new Map<string, IncomingSessionState>();
  private readonly writeQueueByCacheKey = new Map<string, Promise<void>>();

  constructor() {
    this.cacheRootPath = join(
      resolveCacheHomePath(),
      SESSION_CACHE_ROOT_DIRNAME,
      SESSION_CACHE_SUBDIR,
      SESSION_CACHE_VERSION,
    );
  }

  private buildCacheKey(args: CacheFilePathInput): string {
    return `${args.modelId}\u0000${args.incomingSessionKey}`;
  }

  private buildCacheFilePath(args: CacheFilePathInput): string {
    const modelPathKey = encodePathSegment(args.modelId);
    const conversationPathKey = encodePathSegment(args.incomingSessionKey);
    const bucketKey = readBucketKey(conversationPathKey);

    return join(this.cacheRootPath, modelPathKey, bucketKey, `${conversationPathKey}.json`);
  }

  private readFromMemoryCache(args: CacheFilePathInput): IncomingSessionState | undefined {
    const cacheKey = this.buildCacheKey(args);
    const cachedState = this.memoryCache.get(cacheKey);
    if (cachedState === undefined) {
      return undefined;
    }

    this.memoryCache.delete(cacheKey);
    this.memoryCache.set(cacheKey, cachedState);
    return cachedState;
  }

  private rememberInMemoryCache(args: CacheFilePathInput & { state: IncomingSessionState }): void {
    const cacheKey = this.buildCacheKey(args);
    this.memoryCache.delete(cacheKey);
    this.memoryCache.set(cacheKey, args.state);

    while (this.memoryCache.size > MEMORY_CACHE_LIMIT) {
      const oldestCacheKey = this.memoryCache.keys().next().value;
      if (typeof oldestCacheKey !== "string") {
        break;
      }

      this.memoryCache.delete(oldestCacheKey);
    }
  }

  private deleteFromMemoryCache(args: CacheFilePathInput): void {
    const cacheKey = this.buildCacheKey(args);
    this.memoryCache.delete(cacheKey);
  }

  private enqueueWrite(args: CacheFilePathInput, writer: () => Promise<void>): Promise<void> {
    const cacheKey = this.buildCacheKey(args);
    const previousWrite = this.writeQueueByCacheKey.get(cacheKey) ?? Promise.resolve();

    const nextWrite = previousWrite
      .catch(() => {
        return undefined;
      })
      .then(writer)
      .catch(() => {
        return undefined;
      })
      .then(() => {
        if (this.writeQueueByCacheKey.get(cacheKey) === nextWrite) {
          this.writeQueueByCacheKey.delete(cacheKey);
        }
      });

    this.writeQueueByCacheKey.set(cacheKey, nextWrite);
    return nextWrite;
  }

  async get(args: CacheFilePathInput): Promise<IncomingSessionState | undefined> {
    const memoryCachedState = this.readFromMemoryCache(args);
    if (memoryCachedState !== undefined) {
      return memoryCachedState;
    }

    const cacheFilePath = this.buildCacheFilePath(args);
    let cacheFileContent = "";

    try {
      cacheFileContent = await readFile(cacheFilePath, "utf8");
    } catch {
      return undefined;
    }

    let parsedCacheFile: unknown;
    try {
      parsedCacheFile = JSON.parse(cacheFileContent);
    } catch {
      return undefined;
    }

    const persistedState = parsePersistedIncomingSessionState(parsedCacheFile);
    if (persistedState === undefined) {
      return undefined;
    }

    if (persistedState.incomingSessionKey !== args.incomingSessionKey) {
      return undefined;
    }

    this.rememberInMemoryCache({
      ...args,
      state: persistedState,
    });

    return persistedState;
  }

  async set(args: CacheFilePathInput & { state: IncomingSessionState }): Promise<void> {
    this.rememberInMemoryCache(args);

    await this.enqueueWrite(args, async () => {
      const cacheFilePath = this.buildCacheFilePath(args);
      const cacheDirectoryPath = dirname(cacheFilePath);
      await mkdir(cacheDirectoryPath, { recursive: true });

      const persistedState = buildPersistedIncomingSessionState(args.state);
      const tempFilePath = `${cacheFilePath}.tmp-${process.pid}-${Date.now()}`;

      try {
        await writeFile(tempFilePath, `${JSON.stringify(persistedState)}\n`, "utf8");
        await rename(tempFilePath, cacheFilePath);
      } finally {
        await rm(tempFilePath, { force: true }).catch(() => {
          return undefined;
        });
      }
    });
  }

  async delete(args: CacheFilePathInput): Promise<void> {
    this.deleteFromMemoryCache(args);

    await this.enqueueWrite(args, async () => {
      const cacheFilePath = this.buildCacheFilePath(args);
      await rm(cacheFilePath, { force: true }).catch(() => {
        return undefined;
      });
    });
  }
}

export const incomingSessionStore = new IncomingSessionStore();
