import type { Manifest, ManifestModel, ModelsDevFetchOptions } from "../domain/types";
import type { Runtime } from "../ports/runtime";

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const isStringArray = (value: unknown): value is string[] => {
  return Array.isArray(value) && value.every(item => typeof item === "string");
};

const readBoolean = (value: unknown) => {
  return typeof value === "boolean" ? value : false;
};

const readString = (value: unknown) => {
  return typeof value === "string" ? value : undefined;
};

const readNumber = (value: unknown) => {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
};

const toManifestModel = (value: unknown): ManifestModel | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const id = readString(value.id);
  const name = readString(value.name);
  const releaseDate = readString(value.release_date);
  const limit = isRecord(value.limit) ? value.limit : undefined;
  const context = limit === undefined ? undefined : readNumber(limit.context);
  const output = limit === undefined ? undefined : readNumber(limit.output);

  if (
    id === undefined ||
    name === undefined ||
    releaseDate === undefined ||
    context === undefined ||
    output === undefined
  ) {
    return undefined;
  }

  const manifestModel: ManifestModel = {
    id,
    name,
    family: readString(value.family),
    releaseDate,
    attachment: readBoolean(value.attachment),
    reasoning: readBoolean(value.reasoning),
    toolCall: readBoolean(value.tool_call),
    temperature: readBoolean(value.temperature),
    status: readString(value.status),
    experimental: typeof value.experimental === "boolean" ? value.experimental : undefined,
    limit: {
      context,
      output,
    },
  };

  if (limit !== undefined) {
    const input = readNumber(limit.input);
    if (input !== undefined) {
      manifestModel.limit.input = input;
    }
  }

  const modalities = isRecord(value.modalities) ? value.modalities : undefined;
  if (modalities !== undefined && isStringArray(modalities.input) && isStringArray(modalities.output)) {
    manifestModel.modalities = {
      input: [...modalities.input],
      output: [...modalities.output],
    };
  }

  return manifestModel;
};

export const fetchAnthropicManifest = async (runtime: Runtime, options: ModelsDevFetchOptions): Promise<Manifest> => {
  const response = await runtime.fetch(options.url, {
    headers: {
      "user-agent": options.userAgent,
      accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`models.dev fetch failed with status ${response.status}`);
  }

  const text = await response.text();
  const value = JSON.parse(text) as unknown;
  if (!isRecord(value)) {
    throw new Error("models.dev payload is not an object");
  }

  const anthropic = value.anthropic;
  if (!isRecord(anthropic)) {
    throw new Error("models.dev anthropic provider missing");
  }

  const modelsValue = anthropic.models;
  if (!isRecord(modelsValue)) {
    throw new Error("models.dev anthropic models missing");
  }

  const models = Object.values(modelsValue)
    .map(toManifestModel)
    .filter((model): model is ManifestModel => model !== undefined)
    .sort((left, right) => right.releaseDate.localeCompare(left.releaseDate));

  return {
    source: {
      kind: "models.dev",
      fetchedAt: new Date().toISOString(),
    },
    provider: {
      id: options.providerDefaults.id,
      name: options.providerDefaults.name,
      npm: options.providerDefaults.npm,
      env: [...options.providerDefaults.env],
    },
    models,
  };
};
