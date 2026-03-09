import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";

import type { FetchResponse, Runtime } from "../ports/runtime";

const toFetchResponse = async (response: Response): Promise<FetchResponse> => {
  return {
    ok: response.ok,
    status: response.status,
    text: async () => response.text(),
  };
};

export const createNodeRuntime = (): Runtime => {
  return {
    cwd: () => process.cwd(),
    homeDir: () => process.env.HOME ?? homedir(),
    env: name => process.env[name],
    fileExists: filePath => Bun.file(filePath).exists(),
    readText: filePath => Bun.file(filePath).text(),
    writeText: async (filePath, text) => {
      await mkdir(path.dirname(filePath), { recursive: true });
      await Bun.write(filePath, text);
    },
    mkdirp: async directoryPath => {
      await mkdir(directoryPath, { recursive: true });
    },
    fetch: async (url, init) => toFetchResponse(await fetch(url, init)),
  };
};
