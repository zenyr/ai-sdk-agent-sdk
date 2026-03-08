import { describe, expect, test } from "bun:test";
import { buildRuntimeFingerprint } from "../domain/runtime-fingerprint";

describe("runtime-fingerprint", () => {
  test("normalizes trailing slash in baseURL", () => {
    const withSlash = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        baseURL: "http://localhost:5470/claude/",
      },
    });

    const withoutSlash = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        baseURL: "http://localhost:5470/claude",
      },
    });

    expect(withSlash).toBe(withoutSlash);
  });

  test("changes when baseURL changes", () => {
    const left = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        baseURL: "http://localhost:5470/claude",
      },
    });

    const right = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        baseURL: "http://localhost:5471/claude",
      },
    });

    expect(left).not.toBe(right);
  });

  test("changes when cwd changes", () => {
    const left = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        experimental_agentSdk: {
          cwd: "/tmp/a",
        },
      },
    });

    const right = buildRuntimeFingerprint({
      provider: "anthropic.messages",
      settings: {
        experimental_agentSdk: {
          cwd: "/tmp/b",
        },
      },
    });

    expect(left).not.toBe(right);
  });
});
