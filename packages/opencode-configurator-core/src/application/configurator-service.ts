import { buildProviderBlock } from "../domain/provider-block";
import { selectManifestModels } from "../domain/selection-policy";
import type { ConfiguratorOptions, PreparedConfig, PreparedRemoval, StatusResult } from "../domain/types";
import type { Runtime } from "../ports/runtime";
import { fetchAnthropicManifest } from "./models-dev";
import { resolveConfiguratorOptions } from "./options";
import {
  applyProviderBlockToDocument,
  ensureConfigParentDirectory,
  listManagedProviders,
  readConfigDocument,
  removeProviderBlockFromDocument,
  resolveConfigTarget,
} from "./provider-discovery";

export const prepareProviderConfig = async (
  runtime: Runtime,
  rawOptions: ConfiguratorOptions = {}
): Promise<PreparedConfig> => {
  const options = resolveConfiguratorOptions(rawOptions);
  const target = await resolveConfigTarget(runtime, {
    scope: options.scope,
    cwd: options.cwd,
    targetPath: options.targetPath,
  });
  const currentText = await readConfigDocument(runtime, target);
  const manifest = await fetchAnthropicManifest(runtime, {
    url: options.modelsUrl,
    userAgent: options.userAgent,
    providerDefaults: {
      id: options.providerId,
      name: options.providerName,
      npm: options.providerNpm,
      env: options.envVars,
    },
  });
  const selectedModels = selectManifestModels(manifest.models, options.policy, options.manualModelIds);

  if (selectedModels.length === 0) {
    throw new Error(`no models selected for policy ${options.policy}`);
  }

  const providerBlock = buildProviderBlock({
    manifest,
    selectedModels,
    providerName: options.providerName,
    providerNpm: options.providerNpm,
    envVars: options.envVars,
    includeNoneVariant: options.includeNoneVariant,
  });
  const nextText = applyProviderBlockToDocument(currentText, target.containerKey, options.providerId, providerBlock);

  return {
    target,
    providerId: options.providerId,
    selectedPolicy: options.policy,
    manifest,
    selectedModels,
    providerBlock,
    currentText,
    nextText,
  };
};

export const applyPreparedConfig = async (runtime: Runtime, prepared: PreparedConfig) => {
  await ensureConfigParentDirectory(runtime, prepared.target);
  await runtime.writeText(prepared.target.filePath, prepared.nextText);
};

export const readProviderStatus = async (
  runtime: Runtime,
  rawOptions: ConfiguratorOptions = {}
): Promise<StatusResult> => {
  const options = resolveConfiguratorOptions(rawOptions);
  const target = await resolveConfigTarget(runtime, {
    scope: options.scope,
    cwd: options.cwd,
    targetPath: options.targetPath,
  });
  const text = await readConfigDocument(runtime, target);

  return {
    target,
    matches: listManagedProviders(text, {
      containerKey: target.containerKey,
      providerId: options.providerId,
      providerNpm: options.providerNpm,
    }),
  };
};

export const prepareProviderRemoval = async (
  runtime: Runtime,
  rawOptions: ConfiguratorOptions = {}
): Promise<PreparedRemoval> => {
  const options = resolveConfiguratorOptions(rawOptions);
  const target = await resolveConfigTarget(runtime, {
    scope: options.scope,
    cwd: options.cwd,
    targetPath: options.targetPath,
  });
  const currentText = await readConfigDocument(runtime, target);
  const matches = listManagedProviders(currentText, {
    containerKey: target.containerKey,
    providerId: options.providerId,
    providerNpm: options.providerNpm,
  });

  if (matches.length > 1 && !matches.some(match => match.id === options.providerId)) {
    throw new Error(`multiple providers match npm ${options.providerNpm}; specify --provider-id`);
  }

  const providerId = matches[0]?.id ?? options.providerId;

  return {
    target,
    providerId,
    currentText,
    nextText: removeProviderBlockFromDocument(currentText, target.containerKey, providerId),
    existed: matches.length > 0,
  };
};

export const applyPreparedRemoval = async (runtime: Runtime, prepared: PreparedRemoval) => {
  await ensureConfigParentDirectory(runtime, prepared.target);
  await runtime.writeText(prepared.target.filePath, prepared.nextText);
};
