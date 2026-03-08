#!/usr/bin/env node

import { createInterface } from "node:readline/promises";

import {
  applyPreparedConfig,
  applyPreparedRemoval,
  type ConfiguratorOptions,
  createNodeRuntime,
  prepareProviderConfig,
  prepareProviderRemoval,
  readProviderStatus,
  summarizePreparedConfig,
  summarizeRemoval,
  summarizeStatus,
} from "../../../packages/opencode-configurator-core/index";

import { startInteractiveSetup } from "./app";

type Command = "init" | "setup" | "status" | "update" | "remove";

type ParsedCli = {
  command: Command;
  help: boolean;
  yes: boolean;
  options: ConfiguratorOptions;
};

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const parseList = (value: string) => {
  return value
    .split(",")
    .map(item => item.trim())
    .filter(item => item.length > 0);
};

const readPackageVersion = async () => {
  try {
    const packageText = await Bun.file(new URL("../package.json", import.meta.url)).text();
    const value: unknown = JSON.parse(packageText);

    if (isRecord(value) && typeof value.version === "string") {
      return value.version;
    }
  } catch {}

  return "0.0.0";
};

const formatUserAgent = async () => {
  const version = await readPackageVersion();
  const bunVersion = process.versions.bun;
  return `ai-sdk-agent-sdk-opencode-configurator/${version} bun/${bunVersion}`;
};

const confirm = async (message: string) => {
  const terminalReady = process.stdout.isTTY && process.stdin.isTTY;
  if (!terminalReady) {
    return false;
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    const answer = await rl.question(`${message} [y/N] `);
    const normalized = answer.trim().toLowerCase();
    return normalized === "y" || normalized === "yes";
  } finally {
    rl.close();
  }
};

const parseArgs = (argv: string[]): ParsedCli => {
  const args = [...argv];
  let command: Command = "init";

  if (
    args[0] === "init" ||
    args[0] === "setup" ||
    args[0] === "status" ||
    args[0] === "update" ||
    args[0] === "remove"
  ) {
    const nextCommand = args.shift();
    if (
      nextCommand === "init" ||
      nextCommand === "setup" ||
      nextCommand === "status" ||
      nextCommand === "update" ||
      nextCommand === "remove"
    ) {
      command = nextCommand;
    }
  }

  const options: ConfiguratorOptions = {};
  let help = false;
  let yes = false;

  while (args.length > 0) {
    const arg = args.shift();
    if (arg === undefined) {
      break;
    }

    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }

    if (arg === "--yes") {
      yes = true;
      continue;
    }

    if (arg === "--global") {
      options.scope = "global";
      continue;
    }

    if (arg === "--project") {
      options.scope = "project";
      continue;
    }

    if (arg === "--path") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--path requires a value");
      }

      options.targetPath = value;
      options.scope = "path";
      continue;
    }

    if (arg === "--provider-id") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--provider-id requires a value");
      }

      options.providerId = value;
      continue;
    }

    if (arg === "--provider-name") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--provider-name requires a value");
      }

      options.providerName = value;
      continue;
    }

    if (arg === "--provider-npm") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--provider-npm requires a value");
      }

      options.providerNpm = value;
      continue;
    }

    if (arg === "--env") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--env requires a value");
      }

      options.envVars = parseList(value);
      continue;
    }

    if (arg === "--policy") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--policy requires a value");
      }

      options.policy =
        value === "mainstream" || value === "all-stable" || value === "latest-per-family" || value === "manual"
          ? value
          : undefined;
      if (options.policy === undefined) {
        throw new Error(`unsupported policy ${value}`);
      }
      continue;
    }

    if (arg === "--models") {
      const value = args.shift();
      if (value === undefined) {
        throw new Error("--models requires a value");
      }

      options.manualModelIds = parseList(value);
      continue;
    }

    if (arg === "--include-none-variant") {
      options.includeNoneVariant = true;
      continue;
    }

    throw new Error(`unknown argument ${arg}`);
  }

  return {
    command,
    help,
    yes,
    options,
  };
};

const printHelp = () => {
  console.log(`ai-sdk-agent-sdk-opencode-configurator

Usage:
  bunx ai-sdk-agent-sdk-opencode-configurator [command] [options]

Commands:
  init    Initialize provider config
  update  Refresh provider models from models.dev
  remove  Remove managed provider block
  status  Show configured provider state
  setup   Alias for init

Options:
  --global                   Use global config (default)
  --project                  Use nearest project config
  --path <file>              Use explicit config file path
  --yes                      Skip TUI or confirmation and apply immediately
  --provider-id <id>         Provider id, default agent-sdk
  --provider-name <name>     Provider display name, default Agent SDK
  --provider-npm <pkg>       Package name, default ai-sdk-agent-sdk
  --env <a,b>                Env vars, default ANTHROPIC_API_KEY
  --policy <name>            mainstream | latest-per-family | all-stable | manual
  --models <a,b>             Manual model ids
  --include-none-variant     Add thinking.none variant
`);
};

const run = async () => {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    printHelp();
    return;
  }

  const runtime = createNodeRuntime();
  const userAgent = await formatUserAgent();
  const command = parsed.command === "setup" ? "init" : parsed.command;
  const terminalReady = process.stdout.isTTY && process.stdin.isTTY;
  const options: ConfiguratorOptions = {
    scope: parsed.options.scope ?? "global",
    ...parsed.options,
    userAgent,
  };

  if (command === "init" && !parsed.yes && terminalReady) {
    const exitCode = await startInteractiveSetup({
      runtime,
      options,
      userAgent,
    });
    process.exitCode = exitCode;
    return;
  }

  if ((command === "init" || command === "update" || command === "remove") && !parsed.yes && !terminalReady) {
    throw new Error("--yes is required in non-interactive mode for init, update, and remove");
  }

  if (command === "init" || command === "update") {
    const prepared = await prepareProviderConfig(runtime, options);
    const summary = summarizePreparedConfig(prepared);

    if (!parsed.yes) {
      console.log(`${command === "init" ? "Initialize" : "Update"} provider config`);
      console.log(`Target: ${summary.filePath}`);
      console.log(`Provider: ${options.providerName ?? "Agent SDK"}`);
      console.log(`Policy: ${summary.selectedPolicy}`);
      console.log(`Models: ${summary.modelCount}`);

      const approved = await confirm("Apply these changes?");
      if (!approved) {
        console.log("Cancelled");
        return;
      }
    }

    await applyPreparedConfig(runtime, prepared);
    console.log(`${command === "init" ? "Updated" : "Updated"} ${summary.filePath}`);
    console.log(`Policy: ${summary.selectedPolicy}`);
    console.log(`Models: ${summary.modelCount}`);
    for (const modelId of summary.modelIds) {
      console.log(`- ${modelId}`);
    }
    return;
  }

  if (parsed.command === "status") {
    const status = await readProviderStatus(runtime, options);
    const summary = summarizeStatus(status);
    console.log(`Target: ${summary.filePath}`);
    console.log(`Exists: ${summary.exists ? "yes" : "no"}`);
    console.log(`Container: ${summary.containerKey}`);
    if (summary.matches.length === 0) {
      console.log("Managed provider not found");
      return;
    }

    for (const match of summary.matches) {
      console.log(`- ${match.id}: ${match.modelCount} models`);
    }
    return;
  }

  const prepared = await prepareProviderRemoval(runtime, options);
  if (!prepared.existed) {
    console.log(`Provider ${prepared.providerId} not found in ${prepared.target.filePath}`);
    return;
  }

  if (!parsed.yes) {
    console.log("Remove provider config");
    console.log(`Target: ${prepared.target.filePath}`);
    console.log(`Provider id: ${prepared.providerId}`);

    const approved = await confirm("Remove this provider block?");
    if (!approved) {
      console.log("Cancelled");
      return;
    }
  }

  await applyPreparedRemoval(runtime, prepared);
  const summary = summarizeRemoval(prepared);
  console.log(`Removed ${summary.providerId} from ${summary.filePath}`);
};

void run().catch(error => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
