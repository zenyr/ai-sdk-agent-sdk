import path from "node:path";

import type { FetchResponse, Runtime } from "../src/ports/runtime";

type MemoryRuntimeOptions = {
  cwd?: string;
  homeDir?: string;
  env?: Record<string, string>;
  files?: Record<string, string>;
  fetchText?: string;
  fetchStatus?: number;
};

const normalizePath = (filePath: string) => {
  return path.resolve(filePath);
};

export const createMemoryRuntime = (options: MemoryRuntimeOptions = {}) => {
  const files = new Map<string, string>();

  for (const [filePath, text] of Object.entries(options.files ?? {})) {
    files.set(normalizePath(filePath), text);
  }

  const env = new Map(Object.entries(options.env ?? {}));
  const cwd = options.cwd ?? "/workspace/project";
  const homeDir = options.homeDir ?? "/home/tester";

  const runtime: Runtime = {
    cwd: () => cwd,
    homeDir: () => homeDir,
    env: name => env.get(name),
    fileExists: async filePath => files.has(normalizePath(filePath)),
    readText: async filePath => {
      const value = files.get(normalizePath(filePath));
      if (value === undefined) {
        throw new Error(`missing file ${filePath}`);
      }

      return value;
    },
    writeText: async (filePath, text) => {
      files.set(normalizePath(filePath), text);
    },
    mkdirp: async () => {},
    fetch: async () => {
      const status = options.fetchStatus ?? 200;
      const text = options.fetchText ?? "{}";
      const response: FetchResponse = {
        ok: status >= 200 && status < 300,
        status,
        text: async () => text,
      };

      return response;
    },
  };

  return {
    runtime,
    files,
  };
};
