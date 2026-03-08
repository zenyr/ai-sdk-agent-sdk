import path from "node:path";

import { applyEdits, modify, parse } from "jsonc-parser";
import { xdgConfig } from "xdg-basedir";

import type { ConfigContainerKey, ConfigScope, ConfigTarget, MatchableProvider, ProviderConfig } from "../domain/types";
import type { Runtime } from "../ports/runtime";

const globalCandidateNames = ["opencode.jsonc", "opencode.json", "config.json"];
const projectCandidateNames = ["opencode.jsonc", "opencode.json"];

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const getXdgConfigHome = (runtime: Runtime) => {
  const envHome = runtime.env("XDG_CONFIG_HOME");
  if (envHome !== undefined && envHome.length > 0) {
    return envHome;
  }

  if (typeof xdgConfig === "string" && xdgConfig.length > 0) {
    return xdgConfig;
  }

  return path.join(runtime.homeDir(), ".config");
};

const parseConfigDocument = (text: string) => {
  const value: unknown = parse(text);
  return value;
};

const detectContainerKey = (text: string): ConfigContainerKey => {
  const parsed = parseConfigDocument(text);
  if (!isRecord(parsed)) {
    return "provider";
  }

  if (isRecord(parsed.provider)) {
    return "provider";
  }

  if (isRecord(parsed.providers)) {
    return "providers";
  }

  return "provider";
};

const getInitialDocument = (format: "json" | "jsonc") => {
  if (format === "json") {
    return "{}\n";
  }

  return "{\n}\n";
};

const getFormattingOptions = () => ({
  insertSpaces: true,
  tabSize: 2,
});

export const applyProviderBlockToDocument = (
  text: string,
  containerKey: ConfigContainerKey,
  providerId: string,
  providerBlock: ProviderConfig
) => {
  const edits = modify(text, [containerKey, providerId], providerBlock, {
    formattingOptions: getFormattingOptions(),
  });

  return applyEdits(text, edits);
};

export const removeProviderBlockFromDocument = (text: string, containerKey: ConfigContainerKey, providerId: string) => {
  const edits = modify(text, [containerKey, providerId], undefined, {
    formattingOptions: getFormattingOptions(),
  });

  return applyEdits(text, edits);
};

export const resolveGlobalConfigDirectory = (runtime: Runtime) => {
  return path.join(getXdgConfigHome(runtime), "opencode");
};

const resolveExistingTarget = async (
  runtime: Runtime,
  scope: ConfigScope,
  baseDirectory: string,
  candidates: string[]
) => {
  for (const candidate of candidates) {
    const filePath = path.join(baseDirectory, candidate);
    if (await runtime.fileExists(filePath)) {
      const target: Omit<ConfigTarget, "containerKey"> = {
        scope,
        filePath,
        format: candidate.endsWith("jsonc") ? "jsonc" : "json",
        exists: true,
      };

      return target;
    }
  }

  const fallback = candidates[0];
  if (fallback === undefined) {
    throw new Error("candidate list cannot be empty");
  }

  const target: Omit<ConfigTarget, "containerKey"> = {
    scope,
    filePath: path.join(baseDirectory, fallback),
    format: fallback.endsWith("jsonc") ? "jsonc" : "json",
    exists: false,
  };

  return target;
};

export const resolveConfigTarget = async (
  runtime: Runtime,
  input: {
    scope: ConfigScope;
    cwd?: string;
    targetPath?: string;
  }
): Promise<ConfigTarget> => {
  if (input.scope === "path") {
    if (input.targetPath === undefined) {
      throw new Error("targetPath is required for path scope");
    }

    const exists = await runtime.fileExists(input.targetPath);
    const format = input.targetPath.endsWith(".json") ? "json" : "jsonc";
    const text = exists ? await runtime.readText(input.targetPath) : getInitialDocument(format);

    return {
      scope: "path",
      filePath: input.targetPath,
      format,
      exists,
      containerKey: detectContainerKey(text),
    };
  }

  if (input.scope === "global") {
    const target = await resolveExistingTarget(
      runtime,
      "global",
      resolveGlobalConfigDirectory(runtime),
      globalCandidateNames
    );
    const text = target.exists ? await runtime.readText(target.filePath) : getInitialDocument(target.format);
    return {
      ...target,
      containerKey: detectContainerKey(text),
    };
  }

  const cwd = input.cwd ?? runtime.cwd();
  let current = cwd;

  while (true) {
    const target = await resolveExistingTarget(runtime, "project", current, projectCandidateNames);
    if (target.exists) {
      const text = await runtime.readText(target.filePath);
      return {
        ...target,
        containerKey: detectContainerKey(text),
      };
    }

    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }

    current = parent;
  }

  const filePath = path.join(cwd, "opencode.jsonc");
  return {
    scope: "project",
    filePath,
    format: "jsonc",
    exists: false,
    containerKey: "provider",
  };
};

const toProviderMatch = (id: string, value: unknown): MatchableProvider | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const models = isRecord(value.models) ? value.models : undefined;
  const modelCount = models === undefined ? 0 : Object.keys(models).length;
  const npm = typeof value.npm === "string" ? value.npm : undefined;
  const name = typeof value.name === "string" ? value.name : undefined;

  return {
    id,
    modelCount,
    npm,
    name,
  };
};

const getProviderContainer = (parsed: unknown, containerKey: ConfigContainerKey) => {
  if (!isRecord(parsed)) {
    return undefined;
  }

  const container = parsed[containerKey];
  return isRecord(container) ? container : undefined;
};

export const listManagedProviders = (
  text: string,
  input: {
    containerKey: ConfigContainerKey;
    providerId: string;
    providerNpm: string;
  }
) => {
  const parsed = parseConfigDocument(text);
  const container = getProviderContainer(parsed, input.containerKey);

  if (container === undefined) {
    return [];
  }

  const exactMatch = toProviderMatch(input.providerId, container[input.providerId]);
  if (exactMatch !== undefined) {
    return [exactMatch];
  }

  return Object.entries(container)
    .map(([id, value]) => toProviderMatch(id, value))
    .filter((match): match is MatchableProvider => match !== undefined && match.npm === input.providerNpm);
};

export const readConfigDocument = async (runtime: Runtime, target: ConfigTarget) => {
  if (target.exists) {
    return runtime.readText(target.filePath);
  }

  return getInitialDocument(target.format);
};

export const ensureConfigParentDirectory = async (runtime: Runtime, target: ConfigTarget) => {
  await runtime.mkdirp(path.dirname(target.filePath));
};
