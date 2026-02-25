import { describe, expect, test } from "bun:test";

import { readPackageVersion, syncVersionInConstants } from "../sync-version-core";

describe("sync-version-core", () => {
  test("reads version from package.json text", () => {
    const version = readPackageVersion('{"name":"ai-sdk-agent-sdk","version":"1.2.3"}');
    expect(version).toBe("1.2.3");
  });

  test("throws when package.json is invalid JSON", () => {
    expect(() => {
      readPackageVersion("{");
    }).toThrow("failed to parse package.json");
  });

  test("throws when package version is missing", () => {
    expect(() => {
      readPackageVersion('{"name":"ai-sdk-agent-sdk"}');
    }).toThrow("package version missing from package.json");
  });

  test("updates VERSION export in constants source", () => {
    const syncResult = syncVersionInConstants({
      constantsSource: 'export const VERSION = "0.0.1";\nexport const A = 1;\n',
      version: "0.0.2",
    });

    expect(syncResult.changed).toBeTrue();
    expect(syncResult.updatedSource).toContain('export const VERSION = "0.0.2";');
  });

  test("reports unchanged when VERSION already matches", () => {
    const source = 'export const VERSION = "0.0.2";\nexport const A = 1;\n';
    const syncResult = syncVersionInConstants({
      constantsSource: source,
      version: "0.0.2",
    });

    expect(syncResult.changed).toBeFalse();
    expect(syncResult.updatedSource).toBe(source);
  });

  test("throws when VERSION marker is missing", () => {
    expect(() => {
      syncVersionInConstants({
        constantsSource: "export const A = 1;\n",
        version: "0.0.2",
      });
    }).toThrow("VERSION marker not found in constants file");
  });
});
