import { describe, expect, test } from "bun:test";

const readPackageJson = async () => {
  const text = await Bun.file(new URL("../package.json", import.meta.url)).text();
  const value: unknown = JSON.parse(text);
  return value;
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

describe("opencode-configurator package", () => {
  test("declares bunx bin entry", async () => {
    const packageJson = await readPackageJson();
    expect(isRecord(packageJson)).toBeTrue();

    if (!isRecord(packageJson)) {
      return;
    }

    expect(packageJson.name).toBe("ai-sdk-agent-sdk-opencode-configurator");
    expect(isRecord(packageJson.bin)).toBeTrue();

    if (!isRecord(packageJson.bin)) {
      return;
    }

    expect(packageJson.bin["ai-sdk-agent-sdk-opencode-configurator"]).toBe("./dist/cli.js");
  });
});
