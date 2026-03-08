import { describe, expect, test } from "bun:test";

import {
  applyPreparedConfig,
  prepareProviderConfig,
  prepareProviderRemoval,
  readProviderStatus,
} from "../src/application/configurator-service";
import { createMemoryRuntime } from "./memory-runtime";

const modelsPayload = JSON.stringify({
  anthropic: {
    id: "anthropic",
    name: "Anthropic",
    env: ["ANTHROPIC_API_KEY"],
    npm: "@ai-sdk/anthropic",
    models: {
      "claude-haiku-4-5": {
        id: "claude-haiku-4-5",
        name: "Claude Haiku 4.5",
        family: "claude-haiku",
        release_date: "2025-10-01",
        attachment: true,
        reasoning: false,
        tool_call: true,
        temperature: true,
        limit: { context: 200000, output: 64000 },
        modalities: { input: ["text", "image", "pdf"], output: ["text"] },
      },
      "claude-sonnet-4-6": {
        id: "claude-sonnet-4-6",
        name: "Claude Sonnet 4.6",
        family: "claude-sonnet",
        release_date: "2025-11-01",
        attachment: true,
        reasoning: true,
        tool_call: true,
        temperature: true,
        limit: { context: 200000, output: 64000 },
        modalities: { input: ["text", "image", "pdf"], output: ["text"] },
      },
      "claude-opus-4-5": {
        id: "claude-opus-4-5",
        name: "Claude Opus 4.5",
        family: "claude-opus",
        release_date: "2025-11-01",
        attachment: true,
        reasoning: true,
        tool_call: true,
        temperature: true,
        limit: { context: 200000, output: 64000 },
        modalities: { input: ["text", "image", "pdf"], output: ["text"] },
      },
    },
  },
});

describe("configurator service", () => {
  test("writes provider block while preserving other config", async () => {
    const { runtime, files } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{\n  // keep me\n  "theme": "warm",\n  "provider": {\n    "other": {\n      "npm": "example"\n    }\n  }\n}\n',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    await applyPreparedConfig(runtime, prepared);

    const text = files.get("/xdg/opencode/opencode.jsonc");
    expect(text?.includes('"theme": "warm"')).toBeTrue();
    expect(text?.includes('"agent-sdk"')).toBeTrue();
    expect(text?.includes('"claude-sonnet-4-6"')).toBeTrue();
  });

  test("status finds provider by npm when id changed", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"custom-id":{"npm":"ai-sdk-agent-sdk","models":{"claude-sonnet-4-6":{}}}}}',
      },
    });

    const status = await readProviderStatus(runtime, {
      scope: "global",
      providerId: "agent-sdk",
    });

    expect(status.matches.map(match => match.id)).toEqual(["custom-id"]);
  });

  test("remove errors on ambiguous npm matches", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc": '{"provider":{"a":{"npm":"ai-sdk-agent-sdk"},"b":{"npm":"ai-sdk-agent-sdk"}}}',
      },
    });

    await expect(
      prepareProviderRemoval(runtime, {
        scope: "global",
      })
    ).rejects.toThrow("multiple providers match npm ai-sdk-agent-sdk");
  });
});
