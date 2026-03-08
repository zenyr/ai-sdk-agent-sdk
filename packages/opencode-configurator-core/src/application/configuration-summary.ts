import type { PreparedConfig, PreparedRemoval, StatusResult } from "../domain/types";

export const summarizePreparedConfig = (prepared: PreparedConfig) => {
  return {
    filePath: prepared.target.filePath,
    selectedPolicy: prepared.selectedPolicy,
    modelIds: prepared.selectedModels.map(model => model.id),
    modelCount: prepared.selectedModels.length,
    variantCount: prepared.selectedModels.filter(model => model.reasoning).length,
    sourceKind: prepared.manifest.source.kind,
    fetchedAt: prepared.manifest.source.fetchedAt,
  };
};

export const summarizeStatus = (status: StatusResult) => {
  return {
    filePath: status.target.filePath,
    exists: status.target.exists,
    containerKey: status.target.containerKey,
    matches: status.matches,
  };
};

export const summarizeRemoval = (prepared: PreparedRemoval) => {
  return {
    filePath: prepared.target.filePath,
    providerId: prepared.providerId,
    existed: prepared.existed,
  };
};
