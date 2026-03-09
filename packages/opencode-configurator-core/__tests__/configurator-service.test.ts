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

  test("preserves existing custom npm when provider id matches", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"agent-sdk":{"npm":"file:///tmp/custom-agent-sdk.js","models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.providerBlock.npm).toBe("file:///tmp/custom-agent-sdk.js");
    expect(prepared.nextText.includes('"npm": "file:///tmp/custom-agent-sdk.js"')).toBeTrue();
  });

  test("preserves existing provider options", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"agent-sdk":{"npm":"file:///tmp/custom-agent-sdk.js","options":{"setCacheKey":true,"baseURL":"https://proxy.example.test/api"},"models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.nextText.includes('"setCacheKey": true')).toBeTrue();
    expect(prepared.nextText.includes('"baseURL": "https://proxy.example.test/api"')).toBeTrue();
    expect(prepared.nextText.includes('"npm": "file:///tmp/custom-agent-sdk.js"')).toBeTrue();
  });

  test("preserves jsonc comments inside existing provider block", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc": `{
  "provider": {
    "agent-sdk": {
      // keep npm comment
      "npm": "file:///tmp/custom-agent-sdk.js",
      // keep custom field comment
      "headers": {
        "x-test": "1"
      },
      "options": {
        // keep option comment
        "baseURL": "https://proxy.example.test/api",
        "setCacheKey": false
      },
      "models": {
        "claude-haiku-4-5": {
          // keep model comment
          "name": "Claude Haiku 4.5",
          "release_date": "2025-10-01",
          "attachment": true,
          "reasoning": false,
          "tool_call": true,
          "temperature": true,
          "limit": {
            "context": 1,
            "output": 1
          }
        }
      }
    }
  }
}`,
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.nextText.includes("// keep npm comment")).toBeTrue();
    expect(prepared.nextText.includes("// keep custom field comment")).toBeTrue();
    expect(prepared.nextText.includes("// keep option comment")).toBeTrue();
    expect(prepared.nextText.includes("// keep model comment")).toBeTrue();
  });

  test("forces setCacheKey true while preserving baseURL", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"agent-sdk":{"options":{"setCacheKey":false,"baseURL":"https://proxy.example.test/api"},"models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.nextText.includes('"setCacheKey": true')).toBeTrue();
    expect(prepared.nextText.includes('"setCacheKey": false')).toBeFalse();
    expect(prepared.nextText.includes('"baseURL": "https://proxy.example.test/api"')).toBeTrue();
  });

  test("adds empty options.experimental_agentSdk block", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc": '{"provider":{"agent-sdk":{"models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.nextText.includes('"options": {')).toBeTrue();
    expect(prepared.nextText.includes('"experimental_agentSdk": {}')).toBeTrue();
  });

  test("preserves existing nested options.experimental_agentSdk config", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"agent-sdk":{"options":{"experimental_agentSdk":{"cwd":"/tmp/nested","debug":true}},"models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
    });

    expect(prepared.nextText.includes('"experimental_agentSdk": {')).toBeTrue();
    expect(prepared.nextText.includes('"cwd": "/tmp/nested"')).toBeTrue();
    expect(prepared.nextText.includes('"debug": true')).toBeTrue();
  });

  test("explicit providerNpm still overrides existing custom npm", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.jsonc":
          '{"provider":{"agent-sdk":{"npm":"file:///tmp/custom-agent-sdk.js","models":{"claude-haiku-4-5":{}}}}}',
      },
      fetchText: modelsPayload,
    });

    const prepared = await prepareProviderConfig(runtime, {
      scope: "global",
      userAgent: "test/1.0.0",
      providerNpm: "ai-sdk-agent-sdk",
    });

    expect(prepared.providerBlock.npm).toBe("ai-sdk-agent-sdk");
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
