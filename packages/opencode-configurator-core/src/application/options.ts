import type { ConfiguratorDefaults, ConfiguratorOptions, SelectionPolicy } from "../domain/types";
import { CONFIGURATOR_DEFAULTS } from "../domain/types";

export type ResolvedOptions = {
  scope: "global" | "project" | "path";
  targetPath?: string;
  cwd?: string;
  providerId: string;
  providerName: string;
  providerNpm: string;
  envVars: string[];
  policy: SelectionPolicy;
  manualModelIds: string[];
  includeNoneVariant: boolean;
  modelsUrl: string;
  userAgent: string;
};

const normalizeList = (values: string[]) => {
  return [...new Set(values.map(value => value.trim()).filter(value => value.length > 0))];
};

export const resolveConfiguratorDefaults = (overrides: Partial<ConfiguratorDefaults> = {}): ConfiguratorDefaults => {
  return {
    providerId: overrides.providerId ?? CONFIGURATOR_DEFAULTS.providerId,
    providerName: overrides.providerName ?? CONFIGURATOR_DEFAULTS.providerName,
    providerNpm: overrides.providerNpm ?? CONFIGURATOR_DEFAULTS.providerNpm,
    envVars: overrides.envVars ?? [...CONFIGURATOR_DEFAULTS.envVars],
    policy: overrides.policy ?? CONFIGURATOR_DEFAULTS.policy,
  };
};

export const resolveConfiguratorOptions = (options: ConfiguratorOptions = {}): ResolvedOptions => {
  const defaults = resolveConfiguratorDefaults();

  return {
    scope: options.targetPath !== undefined ? "path" : (options.scope ?? "global"),
    targetPath: options.targetPath,
    cwd: options.cwd,
    providerId: options.providerId ?? defaults.providerId,
    providerName: options.providerName ?? defaults.providerName,
    providerNpm: options.providerNpm ?? defaults.providerNpm,
    envVars: normalizeList(options.envVars ?? defaults.envVars),
    policy: options.policy ?? defaults.policy,
    manualModelIds: normalizeList(options.manualModelIds ?? []),
    includeNoneVariant: options.includeNoneVariant ?? false,
    modelsUrl: options.modelsUrl ?? "https://models.dev/api.json",
    userAgent: options.userAgent ?? "ai-sdk-agent-sdk-opencode-configurator/0.0.0",
  };
};
