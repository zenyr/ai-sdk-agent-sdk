import type { Manifest, ManifestModel, ProviderConfig, ProviderModelConfig } from "./types";
import { buildModelVariants } from "./variants";

const toProviderModelConfig = (model: ManifestModel, includeNoneVariant: boolean): ProviderModelConfig => {
  const config: ProviderModelConfig = {
    name: model.name,
    release_date: model.releaseDate,
    attachment: model.attachment,
    reasoning: model.reasoning,
    tool_call: model.toolCall,
    temperature: model.temperature,
    limit: {
      context: model.limit.context,
      output: model.limit.output,
    },
  };

  if (model.limit.input !== undefined) {
    config.limit.input = model.limit.input;
  }

  if (model.family !== undefined) {
    config.family = model.family;
  }

  if (model.modalities !== undefined) {
    config.modalities = {
      input: [...model.modalities.input],
      output: [...model.modalities.output],
    };
  }

  const variants = buildModelVariants(model, includeNoneVariant);
  if (variants !== undefined) {
    config.variants = variants;
  }

  return config;
};

export const buildProviderBlock = (input: {
  manifest: Manifest;
  selectedModels: ManifestModel[];
  providerName: string;
  providerNpm: string;
  envVars: string[];
  includeNoneVariant: boolean;
}): ProviderConfig => {
  const models = Object.fromEntries(
    input.selectedModels.map(model => [model.id, toProviderModelConfig(model, input.includeNoneVariant)])
  );

  return {
    name: input.providerName,
    npm: input.providerNpm,
    env: [...input.envVars],
    models,
  };
};
