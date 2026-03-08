import { describe, expect, test } from "bun:test";

import { selectManifestModels } from "../src/domain/selection-policy";
import type { ManifestModel } from "../src/domain/types";

const createModel = (
  input: Partial<ManifestModel> & Pick<ManifestModel, "id" | "name" | "releaseDate">
): ManifestModel => {
  return {
    id: input.id,
    name: input.name,
    releaseDate: input.releaseDate,
    family: input.family,
    attachment: input.attachment ?? true,
    reasoning: input.reasoning ?? true,
    toolCall: input.toolCall ?? true,
    temperature: input.temperature ?? true,
    status: input.status,
    experimental: input.experimental,
    limit: input.limit ?? {
      context: 200000,
      output: 64000,
    },
    modalities: input.modalities,
  };
};

describe("selection policy", () => {
  test("mainstream keeps latest haiku sonnet opus", () => {
    const models = [
      createModel({ id: "claude-haiku-4-5", name: "Haiku 4.5", family: "claude-haiku", releaseDate: "2025-10-01" }),
      createModel({ id: "claude-haiku-4-0", name: "Haiku 4.0", family: "claude-haiku", releaseDate: "2025-01-01" }),
      createModel({ id: "claude-sonnet-4-6", name: "Sonnet 4.6", family: "claude-sonnet", releaseDate: "2025-11-01" }),
      createModel({
        id: "claude-sonnet-4-6-preview",
        name: "Sonnet 4.6 Preview",
        family: "claude-sonnet",
        releaseDate: "2025-12-01",
      }),
      createModel({ id: "claude-opus-4-5", name: "Opus 4.5", family: "claude-opus", releaseDate: "2025-11-01" }),
      createModel({ id: "claude-opus-4-0", name: "Opus 4.0", family: "claude-opus", releaseDate: "2025-03-01" }),
    ];

    const selected = selectManifestModels(models, "mainstream");

    expect(selected.map(model => model.id)).toEqual(["claude-opus-4-5", "claude-sonnet-4-6", "claude-haiku-4-5"]);
  });

  test("manual keeps requested ids only", () => {
    const models = [
      createModel({ id: "a", name: "A", releaseDate: "2025-01-01" }),
      createModel({ id: "b", name: "B", releaseDate: "2025-02-01" }),
    ];

    const selected = selectManifestModels(models, "manual", ["b"]);

    expect(selected.map(model => model.id)).toEqual(["b"]);
  });
});
