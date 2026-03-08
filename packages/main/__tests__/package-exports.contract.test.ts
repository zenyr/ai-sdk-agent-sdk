import { describe, expect, test } from "bun:test";

const readPackageJson = async () => {
  const text = await Bun.file(new URL("../package.json", import.meta.url)).text();
  return JSON.parse(text) as Record<string, unknown>;
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

describe("package export contract", () => {
  test("package exports root, v2, v3, and bun entries", async () => {
    const packageJson = await readPackageJson();
    const exportsField = packageJson.exports;

    expect(isRecord(exportsField)).toBeTrue();

    if (!isRecord(exportsField)) {
      return;
    }

    expect("." in exportsField).toBeTrue();
    expect("./v2" in exportsField).toBeTrue();
    expect("./v3" in exportsField).toBeTrue();
    expect("./bun" in exportsField).toBeTrue();
    expect("./core" in exportsField).toBeFalse();
    expect("./opencode" in exportsField).toBeFalse();
  });

  test("package files and build script include v3 entry", async () => {
    const packageJson = await readPackageJson();
    const files = packageJson.files;
    const scripts = packageJson.scripts;

    expect(Array.isArray(files)).toBeTrue();
    expect(isRecord(scripts)).toBeTrue();

    if (!Array.isArray(files) || !isRecord(scripts)) {
      return;
    }

    expect(files.includes("v3.ts")).toBeTrue();
    expect(files.includes("core.ts")).toBeFalse();

    const buildScript = scripts.build;
    expect(typeof buildScript).toBe("string");

    if (typeof buildScript !== "string") {
      return;
    }

    expect(buildScript.includes("./v3.ts")).toBeTrue();
    expect(buildScript.includes("./core.ts")).toBeFalse();
    expect(buildScript.includes("./opencode.ts")).toBeFalse();
  });
});
