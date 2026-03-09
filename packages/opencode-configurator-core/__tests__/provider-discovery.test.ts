import { describe, expect, test } from "bun:test";

import { resolveConfigTarget } from "../src/application/provider-discovery";
import { createMemoryRuntime } from "./memory-runtime";

describe("provider discovery", () => {
  test("global prefers existing opencode.json", async () => {
    const { runtime } = createMemoryRuntime({
      env: {
        XDG_CONFIG_HOME: "/xdg",
      },
      files: {
        "/xdg/opencode/opencode.json": '{\n  "provider": {}\n}\n',
      },
    });

    const target = await resolveConfigTarget(runtime, { scope: "global" });

    expect(target.filePath).toBe("/xdg/opencode/opencode.json");
    expect(target.format).toBe("json");
  });

  test("project finds nearest opencode file", async () => {
    const { runtime } = createMemoryRuntime({
      cwd: "/workspace/project/apps/demo",
      files: {
        "/workspace/project/opencode.jsonc": '{\n  "provider": {}\n}\n',
      },
    });

    const target = await resolveConfigTarget(runtime, { scope: "project", cwd: "/workspace/project/apps/demo" });

    expect(target.filePath).toBe("/workspace/project/opencode.jsonc");
    expect(target.containerKey).toBe("provider");
  });
});
