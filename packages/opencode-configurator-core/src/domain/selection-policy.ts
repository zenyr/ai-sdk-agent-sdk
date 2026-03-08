import type { ManifestModel, SelectionPolicy } from "./types";

const canonicalFamilies = ["claude-opus", "claude-sonnet", "claude-haiku"];

const normalizeFamily = (model: ManifestModel) => {
  if (typeof model.family === "string" && model.family.length > 0) {
    return model.family.toLowerCase();
  }

  const source = `${model.id} ${model.name}`.toLowerCase();

  if (source.includes("opus")) {
    return "claude-opus";
  }

  if (source.includes("sonnet")) {
    return "claude-sonnet";
  }

  if (source.includes("haiku")) {
    return "claude-haiku";
  }

  return model.id.toLowerCase();
};

const getStatusScore = (status?: string) => {
  const normalized = status?.toLowerCase();

  if (normalized === undefined || normalized === "active" || normalized === "stable") {
    return 4;
  }

  if (normalized === "beta") {
    return 3;
  }

  if (normalized === "preview") {
    return 2;
  }

  return 0;
};

const getNamePenalty = (model: ManifestModel) => {
  const source = `${model.id} ${model.name}`.toLowerCase();
  let penalty = 0;

  if (source.includes("preview")) {
    penalty += 2;
  }

  if (source.includes("legacy") || source.includes("instant")) {
    penalty += 3;
  }

  return penalty;
};

const getReleaseTime = (model: ManifestModel) => {
  const releaseTime = Date.parse(model.releaseDate);
  return Number.isFinite(releaseTime) ? releaseTime : 0;
};

const compareModels = (left: ManifestModel, right: ManifestModel) => {
  const statusDelta = getStatusScore(right.status) - getStatusScore(left.status);
  if (statusDelta !== 0) {
    return statusDelta;
  }

  const releaseDelta = getReleaseTime(right) - getReleaseTime(left);
  if (releaseDelta !== 0) {
    return releaseDelta;
  }

  const penaltyDelta = getNamePenalty(left) - getNamePenalty(right);
  if (penaltyDelta !== 0) {
    return penaltyDelta;
  }

  return left.id.localeCompare(right.id);
};

const isStableEnough = (model: ManifestModel) => {
  if (model.experimental === true) {
    return false;
  }

  const status = model.status?.toLowerCase();
  if (status === "deprecated" || status === "alpha") {
    return false;
  }

  return true;
};

const isMainstreamEligible = (model: ManifestModel) => {
  if (!isStableEnough(model)) {
    return false;
  }

  const source = `${model.id} ${model.name}`.toLowerCase();
  return !source.includes("preview") && !source.includes("legacy") && !source.includes("instant");
};

const getLatestPerFamily = (models: ManifestModel[]) => {
  const families = new Map<string, ManifestModel[]>();

  for (const model of models) {
    const family = normalizeFamily(model);
    const existing = families.get(family);

    if (existing === undefined) {
      families.set(family, [model]);
      continue;
    }

    existing.push(model);
  }

  const selected: ManifestModel[] = [];

  for (const [family, familyModels] of families.entries()) {
    const sorted = [...familyModels].sort(compareModels);
    const first = sorted[0];
    if (first !== undefined) {
      selected.push({ ...first, family });
    }
  }

  return selected.sort(compareModels);
};

export const selectManifestModels = (
  models: ManifestModel[],
  policy: SelectionPolicy,
  manualModelIds: string[] = []
) => {
  if (policy === "manual") {
    const selected = models.filter(model => manualModelIds.includes(model.id));
    return [...selected].sort(compareModels);
  }

  const stableModels = models.filter(isStableEnough).sort(compareModels);

  if (policy === "all-stable") {
    return stableModels;
  }

  const latestPerFamily = getLatestPerFamily(stableModels);

  if (policy === "latest-per-family") {
    return latestPerFamily;
  }

  const mainstream = getLatestPerFamily(stableModels.filter(isMainstreamEligible));
  const canonical = canonicalFamilies
    .map(family => mainstream.find(model => normalizeFamily(model) === family))
    .filter((model): model is ManifestModel => model !== undefined);

  if (canonical.length > 0) {
    return canonical.sort(compareModels);
  }

  return mainstream;
};

export const sortManifestModels = (models: ManifestModel[]) => {
  return [...models].sort(compareModels);
};

export const getModelFamily = (model: ManifestModel) => {
  return normalizeFamily(model);
};
