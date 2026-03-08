export type ConfigScope = "global" | "project" | "path";

export type ConfigContainerKey = "provider" | "providers";

export type SelectionPolicy = "mainstream" | "all-stable" | "latest-per-family" | "manual";

export type ModelsSourceKind = "models.dev";

export type ThinkingVariant = {
  thinking: {
    type: "enabled" | "disabled";
    budgetTokens?: number;
  };
};

export type ManifestModel = {
  id: string;
  name: string;
  family?: string;
  releaseDate: string;
  attachment: boolean;
  reasoning: boolean;
  toolCall: boolean;
  temperature: boolean;
  status?: string;
  experimental?: boolean;
  limit: {
    context: number;
    output: number;
    input?: number;
  };
  modalities?: {
    input: string[];
    output: string[];
  };
};

export type Manifest = {
  source: {
    kind: ModelsSourceKind;
    fetchedAt: string;
  };
  provider: {
    id: string;
    name: string;
    npm: string;
    env: string[];
  };
  models: ManifestModel[];
};

export type ProviderModelConfig = {
  name: string;
  family?: string;
  release_date: string;
  attachment: boolean;
  reasoning: boolean;
  tool_call: boolean;
  temperature: boolean;
  limit: {
    context: number;
    output: number;
    input?: number;
  };
  modalities?: {
    input: string[];
    output: string[];
  };
  variants?: Record<string, ThinkingVariant>;
};

export type ProviderConfig = {
  name: string;
  npm: string;
  env: string[];
  models: Record<string, ProviderModelConfig>;
};

export type ConfigTarget = {
  scope: ConfigScope;
  filePath: string;
  format: "json" | "jsonc";
  exists: boolean;
  containerKey: ConfigContainerKey;
};

export type MatchableProvider = {
  id: string;
  modelCount: number;
  npm?: string;
  name?: string;
};

export type StatusResult = {
  target: ConfigTarget;
  matches: MatchableProvider[];
};

export type PreparedConfig = {
  target: ConfigTarget;
  providerId: string;
  selectedPolicy: SelectionPolicy;
  manifest: Manifest;
  selectedModels: ManifestModel[];
  providerBlock: ProviderConfig;
  currentText: string;
  nextText: string;
};

export type PreparedRemoval = {
  target: ConfigTarget;
  providerId: string;
  currentText: string;
  nextText: string;
  existed: boolean;
};

export type ConfiguratorDefaults = {
  providerId: string;
  providerName: string;
  providerNpm: string;
  envVars: string[];
  policy: SelectionPolicy;
};

export type ConfiguratorOptions = {
  scope?: ConfigScope;
  targetPath?: string;
  cwd?: string;
  providerId?: string;
  providerName?: string;
  providerNpm?: string;
  envVars?: string[];
  policy?: SelectionPolicy;
  manualModelIds?: string[];
  includeNoneVariant?: boolean;
  modelsUrl?: string;
  userAgent?: string;
};

export type ModelsDevFetchOptions = {
  url: string;
  userAgent: string;
  providerDefaults: {
    id: string;
    name: string;
    npm: string;
    env: string[];
  };
};

export const CONFIGURATOR_DEFAULTS: ConfiguratorDefaults = {
  providerId: "agent-sdk",
  providerName: "Agent SDK",
  providerNpm: "ai-sdk-agent-sdk",
  envVars: ["ANTHROPIC_API_KEY"],
  policy: "mainstream",
};
