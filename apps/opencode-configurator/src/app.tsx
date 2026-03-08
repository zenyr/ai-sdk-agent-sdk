import { createCliRenderer } from "@opentui/core";
import { createRoot, useKeyboard } from "@opentui/react";
import type { Runtime } from "@zenyr/opencode-configurator-core";
import {
  applyPreparedConfig,
  type ConfiguratorOptions,
  fetchAnthropicManifest,
  type Manifest,
  type PreparedConfig,
  prepareProviderConfig,
  resolveConfiguratorDefaults,
  type SelectionPolicy,
  selectManifestModels,
} from "@zenyr/opencode-configurator-core";
import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/header";
import { Panel } from "./components/panel";
import { WrappedText } from "./components/wrapped-text";
import { clamp, formatModelRow, isConfirmKey } from "./lib/text";
import { ManualScreen } from "./screens/manual-screen";
import { MessageScreen } from "./screens/message-screen";
import { PathScreen } from "./screens/path-screen";
import { PolicyScreen } from "./screens/policy-screen";
import { PreviewScreen } from "./screens/preview-screen";
import { ScopeScreen } from "./screens/scope-screen";

type SetupAppProps = {
  runtime: Runtime;
  options: ConfiguratorOptions;
  userAgent: string;
  onExit(exitCode: number): void;
};

type Step = "scope" | "path" | "policy" | "manual" | "preview" | "done" | "error";

const stepOrder: Step[] = ["scope", "path", "policy", "manual", "preview"];

const stepLabels: Record<Step, string> = {
  scope: "Choose scope",
  path: "Set file path",
  policy: "Choose policy",
  manual: "Pick models",
  preview: "Review changes",
  done: "Config written",
  error: "Stopped on error",
};

const scopeOptions = [
  { value: "global", label: "global", description: "Global OpenCode config" },
  { value: "project", label: "project", description: "Nearest project config" },
  { value: "path", label: "explicit path", description: "Write exact file path" },
] as const;

const policyOptions = [
  { value: "mainstream", label: "mainstream", description: "Latest Haiku, Sonnet, Opus" },
  { value: "latest-per-family", label: "latest per family", description: "Latest model in each family" },
  { value: "all-stable", label: "all stable", description: "Keep all stable Anthropic models" },
  { value: "manual", label: "manual", description: "Pick models yourself" },
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const parseExistingModelIds = (prepared: PreparedConfig) => {
  try {
    const jsonText = prepared.currentText.replace(/\/\/.*$/gm, "");
    const parsed: unknown = JSON.parse(jsonText);
    if (!isRecord(parsed)) {
      return [];
    }

    const providerContainer = isRecord(parsed.provider)
      ? parsed.provider
      : isRecord(parsed.providers)
        ? parsed.providers
        : undefined;

    if (providerContainer === undefined) {
      return [];
    }

    const providerValue = providerContainer[prepared.providerId];
    if (!isRecord(providerValue) || !isRecord(providerValue.models)) {
      return [];
    }

    return Object.keys(providerValue.models);
  } catch {
    return [];
  }
};

const SetupApp = ({ runtime, options, userAgent, onExit }: SetupAppProps) => {
  const defaults = useMemo(() => resolveConfiguratorDefaults(), []);
  const [step, setStep] = useState<Step>(options.targetPath !== undefined ? "path" : "scope");
  const [scopeIndex, setScopeIndex] = useState(() => {
    if (options.scope === "project") {
      return 1;
    }

    if (options.targetPath !== undefined || options.scope === "path") {
      return 2;
    }

    return 0;
  });
  const [policyIndex, setPolicyIndex] = useState(0);
  const [manualCursor, setManualCursor] = useState(0);
  const [previewOffset, setPreviewOffset] = useState(0);
  const [targetPath, setTargetPath] = useState(options.targetPath ?? `${runtime.cwd()}/opencode.jsonc`);
  const [manifest, setManifest] = useState<Manifest | undefined>();
  const [manualModelIds, setManualModelIds] = useState<string[]>([]);
  const [preview, setPreview] = useState<PreparedConfig | undefined>();
  const [busyLabel, setBusyLabel] = useState("Loading models.dev...");
  const [errorMessage, setErrorMessage] = useState<string | undefined>();

  useEffect(() => {
    let active = true;

    const load = async () => {
      try {
        const loaded = await fetchAnthropicManifest(runtime, {
          url: options.modelsUrl ?? "https://models.dev/api.json",
          userAgent,
          providerDefaults: {
            id: options.providerId ?? defaults.providerId,
            name: options.providerName ?? defaults.providerName,
            npm: options.providerNpm ?? defaults.providerNpm,
            env: options.envVars ?? defaults.envVars,
          },
        });

        if (!active) {
          return;
        }

        setManifest(loaded);
        setManualModelIds(
          selectManifestModels(loaded.models, "mainstream").map((model: Manifest["models"][number]) => model.id)
        );
        setBusyLabel("");
      } catch (error) {
        if (!active) {
          return;
        }

        setErrorMessage(error instanceof Error ? error.message : "Unknown models.dev error");
        setStep("error");
      }
    };

    void load();

    return () => {
      active = false;
    };
  }, [
    defaults.envVars,
    defaults.providerId,
    defaults.providerName,
    defaults.providerNpm,
    options.envVars,
    options.modelsUrl,
    options.providerId,
    options.providerName,
    options.providerNpm,
    runtime,
    userAgent,
  ]);

  const currentScope = scopeOptions[scopeIndex]?.value ?? "global";
  const currentPolicy = policyOptions[policyIndex]?.value ?? "mainstream";
  const manualModels = manifest?.models ?? [];
  const currentStepNumber = stepOrder.indexOf(step) >= 0 ? stepOrder.indexOf(step) + 1 : undefined;
  const contextItems = [
    `Scope: ${currentScope}`,
    `Path: ${currentScope === "path" ? targetPath : (options.targetPath ?? "auto-detect")}`,
    `Policy: ${currentPolicy}`,
  ];

  const textWidth = Math.max(28, (process.stdout.columns ?? 80) - 8);
  const terminalRows = process.stdout.rows ?? 24;
  const isWide = textWidth >= 88;
  const infoColumnWidth = isWide ? Math.max(24, Math.floor((textWidth - 4) / 2)) : textWidth;
  const manualVisibleCount = Math.max(6, terminalRows - 14);
  const manualPageStart = clamp(
    manualCursor - Math.floor(manualVisibleCount / 2),
    0,
    Math.max(0, manualModels.length - manualVisibleCount)
  );
  const visibleManualModels = manualModels.slice(manualPageStart, manualPageStart + manualVisibleCount);

  const existingModelIds = useMemo(() => {
    if (preview === undefined) {
      return [];
    }

    return parseExistingModelIds(preview);
  }, [preview]);

  const previewModelIds = preview === undefined ? [] : Object.keys(preview.providerBlock.models);
  const addedModelIds = previewModelIds.filter(modelId => !existingModelIds.includes(modelId));
  const removedModelIds = existingModelIds.filter(modelId => !previewModelIds.includes(modelId));
  const allPreviewModelRows =
    preview === undefined
      ? []
      : previewModelIds.map(modelId => {
          const model = preview.providerBlock.models[modelId];
          return formatModelRow({
            id: modelId,
            reasoning: model?.variants !== undefined,
            output: model?.limit.output ?? 0,
          });
        });
  const previewVisibleCount = Math.max(4, terminalRows - (isWide ? 20 : 18));
  const boundedPreviewOffset = clamp(previewOffset, 0, Math.max(0, allPreviewModelRows.length - previewVisibleCount));
  const visiblePreviewModelRows = allPreviewModelRows.slice(
    boundedPreviewOffset,
    boundedPreviewOffset + previewVisibleCount
  );
  const hiddenPreviewModelCount = Math.max(
    0,
    allPreviewModelRows.length - (boundedPreviewOffset + visiblePreviewModelRows.length)
  );

  const buildPreview = async (policy: SelectionPolicy) => {
    setBusyLabel("Building preview...");

    try {
      const prepared = await prepareProviderConfig(runtime, {
        ...options,
        scope: currentScope,
        targetPath: currentScope === "path" ? targetPath : undefined,
        policy,
        manualModelIds: policy === "manual" ? manualModelIds : undefined,
        userAgent,
      });

      setPreview(prepared);
      setPreviewOffset(0);
      setBusyLabel("");
      setStep("preview");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Unknown preview error");
      setStep("error");
    }
  };

  const applyPreview = async () => {
    if (preview === undefined) {
      return;
    }

    setBusyLabel("Writing config...");

    try {
      await applyPreparedConfig(runtime, preview);
      setBusyLabel("");
      setStep("done");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Unknown write error");
      setStep("error");
    }
  };

  useKeyboard(key => {
    if (key.ctrl && key.name === "c") {
      onExit(1);
      return;
    }

    if (busyLabel.length > 0) {
      return;
    }

    if (key.name === "escape") {
      onExit(0);
      return;
    }

    if (step === "scope") {
      if (key.name === "up") {
        setScopeIndex(current => Math.max(0, current - 1));
      }

      if (key.name === "down") {
        setScopeIndex(current => Math.min(scopeOptions.length - 1, current + 1));
      }

      if (isConfirmKey(key.name)) {
        setStep(currentScope === "path" ? "path" : "policy");
      }

      return;
    }

    if (step === "policy") {
      if (key.name === "up") {
        setPolicyIndex(current => Math.max(0, current - 1));
      }

      if (key.name === "down") {
        setPolicyIndex(current => Math.min(policyOptions.length - 1, current + 1));
      }

      if (key.name === "backspace") {
        setStep(currentScope === "path" ? "path" : "scope");
      }

      if (isConfirmKey(key.name)) {
        if (currentPolicy === "manual") {
          setStep("manual");
          return;
        }

        void buildPreview(currentPolicy);
      }

      return;
    }

    if (step === "manual") {
      if (key.name === "up") {
        setManualCursor(current => Math.max(0, current - 1));
      }

      if (key.name === "down") {
        setManualCursor(current => Math.min(Math.max(manualModels.length - 1, 0), current + 1));
      }

      if (key.name === "space") {
        const model = manualModels[manualCursor];
        if (model === undefined) {
          return;
        }

        setManualModelIds(current => {
          if (current.includes(model.id)) {
            return current.filter(item => item !== model.id);
          }

          return [...current, model.id];
        });
      }

      if (key.name === "backspace") {
        setStep("policy");
      }

      if (isConfirmKey(key.name)) {
        void buildPreview("manual");
      }

      return;
    }

    if (step === "preview") {
      if (key.name === "up") {
        setPreviewOffset(current => Math.max(0, current - 1));
      }

      if (key.name === "down") {
        setPreviewOffset(current =>
          Math.min(Math.max(allPreviewModelRows.length - previewVisibleCount, 0), current + 1)
        );
      }

      if (key.name === "backspace") {
        setStep(currentPolicy === "manual" ? "manual" : "policy");
      }

      if (isConfirmKey(key.name)) {
        void applyPreview();
      }

      return;
    }

    if (step === "done" && isConfirmKey(key.name)) {
      onExit(0);
      return;
    }

    if (step === "error" && isConfirmKey(key.name)) {
      onExit(0);
    }
  });

  return (
    <box style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1 }}>
      <Header
        title="Agent SDK OpenCode Configurator"
        subtitle="Arrow keys move. Enter confirms. Backspace goes back. Esc quits."
        currentStepLabel={stepLabels[step]}
        currentStepNumber={currentStepNumber}
        totalSteps={stepOrder.length}
        contextItems={contextItems}
        width={textWidth}
      />

      {busyLabel.length > 0 ? (
        <Panel title="Working" tone="accent">
          <WrappedText text={busyLabel} width={textWidth} />
        </Panel>
      ) : null}

      {step === "scope" ? <ScopeScreen options={scopeOptions} selectedIndex={scopeIndex} width={textWidth} /> : null}

      {step === "path" ? (
        <PathScreen
          targetPath={targetPath}
          width={textWidth}
          onInput={setTargetPath}
          onSubmit={() => setStep("policy")}
        />
      ) : null}

      {step === "policy" ? (
        <PolicyScreen options={policyOptions} selectedIndex={policyIndex} width={textWidth} />
      ) : null}

      {step === "manual" ? (
        <ManualScreen
          models={visibleManualModels}
          selectedIds={manualModelIds}
          cursor={manualCursor}
          startIndex={manualPageStart}
          totalCount={manualModels.length}
          width={textWidth}
        />
      ) : null}

      {step === "preview" && preview !== undefined ? (
        <PreviewScreen
          prepared={preview}
          width={textWidth}
          columnWidth={infoColumnWidth}
          isWide={isWide}
          addedModelIds={addedModelIds}
          removedModelIds={removedModelIds}
          modelRows={visiblePreviewModelRows}
          hiddenModelCount={hiddenPreviewModelCount}
        />
      ) : null}

      {step === "done" && preview !== undefined ? (
        <MessageScreen
          title="Done"
          lines={[`Updated ${preview.target.filePath}`, "Press Enter to exit."]}
          width={textWidth}
          tone="success"
        />
      ) : null}

      {step === "error" ? (
        <MessageScreen
          title="Error"
          lines={[errorMessage ?? "Unknown error", "Press Enter or Esc to exit."]}
          width={textWidth}
          tone="danger"
        />
      ) : null}
    </box>
  );
};

export const startInteractiveSetup = async (input: {
  runtime: Runtime;
  options: ConfiguratorOptions;
  userAgent: string;
}) => {
  const renderer = await createCliRenderer({
    exitOnCtrlC: true,
    targetFps: 30,
  });

  return await new Promise<number>(resolve => {
    const root = createRoot(renderer);
    const finish = async (exitCode: number) => {
      root.unmount();
      if (typeof renderer.destroy === "function") {
        await renderer.destroy();
      }
      resolve(exitCode);
    };

    root.render(
      <SetupApp
        runtime={input.runtime}
        options={input.options}
        userAgent={input.userAgent}
        onExit={exitCode => void finish(exitCode)}
      />
    );
  });
};
