import { useKeyboard } from "@opentui/react";
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
import { clamp, formatModelRow, isConfirmKey } from "../lib/text";

export type Step = "scope" | "path" | "provider" | "policy" | "manual" | "preview" | "done" | "error";

export const stepOrder: Step[] = ["scope", "path", "provider", "policy", "manual", "preview"];

export const stepLabels: Record<Step, string> = {
  scope: "Choose scope",
  path: "Set file path",
  provider: "Set provider",
  policy: "Choose policy",
  manual: "Pick models",
  preview: "Review changes",
  done: "Config written",
  error: "Stopped on error",
};

export const scopeOptions = [
  { value: "global", label: "global", description: "Global OpenCode config" },
  { value: "project", label: "project", description: "Nearest project config" },
  { value: "path", label: "explicit path", description: "Write exact file path" },
] as const;

export const policyOptions = [
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

const getInitialScopeIndex = (options: ConfiguratorOptions) => {
  if (options.scope === "project") {
    return 1;
  }

  if (options.targetPath !== undefined || options.scope === "path") {
    return 2;
  }

  return 0;
};

type UseSetupFlowInput = {
  runtime: Runtime;
  options: ConfiguratorOptions;
  userAgent: string;
  onExit(exitCode: number): void;
};

export const useSetupFlow = ({ runtime, options, userAgent, onExit }: UseSetupFlowInput) => {
  const defaults = useMemo(() => resolveConfiguratorDefaults(), []);
  const [step, setStep] = useState<Step>(options.targetPath !== undefined ? "path" : "scope");
  const [scopeIndex, setScopeIndex] = useState(() => getInitialScopeIndex(options));
  const [policyIndex, setPolicyIndex] = useState(0);
  const [manualCursor, setManualCursor] = useState(0);
  const [previewOffset, setPreviewOffset] = useState(0);
  const [targetPath, setTargetPath] = useState(options.targetPath ?? `${runtime.cwd()}/opencode.jsonc`);
  const [providerId, setProviderId] = useState(options.providerId ?? defaults.providerId);
  const [providerName, setProviderName] = useState(options.providerName ?? defaults.providerName);
  const [providerField, setProviderField] = useState<"name" | "id">("name");
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
        setManualModelIds(selectManifestModels(loaded.models, "mainstream").map(model => model.id));
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
  const textWidth = Math.max(28, (process.stdout.columns ?? 80) - 8);
  const isWide = textWidth >= 88;
  const infoColumnWidth = isWide ? Math.max(24, Math.floor((textWidth - 4) / 2)) : textWidth;
  const currentStepNumber = stepOrder.indexOf(step) >= 0 ? stepOrder.indexOf(step) + 1 : undefined;
  const contextItems = [
    `Scope: ${currentScope}`,
    `Path: ${currentScope === "path" ? targetPath : (options.targetPath ?? "auto-detect")}`,
    `Provider: ${providerName}`,
    `Provider id: ${providerId}`,
    `Policy: ${currentPolicy}`,
  ];

  const policyModelCounts = useMemo(() => {
    if (manifest === undefined) {
      return {
        mainstream: 0,
        "latest-per-family": 0,
        "all-stable": 0,
        manual: manualModelIds.length,
      };
    }

    return {
      mainstream: selectManifestModels(manifest.models, "mainstream").length,
      "latest-per-family": selectManifestModels(manifest.models, "latest-per-family").length,
      "all-stable": selectManifestModels(manifest.models, "all-stable").length,
      manual: manualModelIds.length,
    };
  }, [manifest, manualModelIds.length]);

  const visibleManualRows = manualModels.map(model => ({
    id: model.id,
    name: model.name,
  }));

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
  const visiblePreviewModelRows = allPreviewModelRows;

  const selectScope = (index: number) => {
    setScopeIndex(clamp(index, 0, scopeOptions.length - 1));
  };

  const buildPreview = async (policy: SelectionPolicy) => {
    setBusyLabel("Building preview...");

    try {
      const prepared = await prepareProviderConfig(runtime, {
        ...options,
        scope: currentScope,
        targetPath: currentScope === "path" ? targetPath : undefined,
        providerId,
        providerName,
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
      setBusyLabel("");
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
      setBusyLabel("");
      setStep("error");
    }
  };

  const submitScope = (index: number) => {
    const nextIndex = clamp(index, 0, scopeOptions.length - 1);
    const nextScope = scopeOptions[nextIndex]?.value ?? "global";
    setScopeIndex(nextIndex);
    setStep(nextScope === "path" ? "path" : "provider");
  };

  const submitPath = () => {
    setStep("provider");
  };

  const submitProvider = () => {
    setStep("policy");
  };

  const submitProviderName = () => {
    setProviderField("id");
  };

  const submitProviderId = () => {
    setStep("policy");
  };

  const selectPolicy = (index: number) => {
    setPolicyIndex(clamp(index, 0, policyOptions.length - 1));
  };

  const submitPolicy = (index: number) => {
    const nextIndex = clamp(index, 0, policyOptions.length - 1);
    const policy = policyOptions[nextIndex]?.value ?? "mainstream";
    setPolicyIndex(nextIndex);

    if (policy === "manual") {
      setStep("manual");
      return;
    }

    void buildPreview(policy);
  };

  const toggleManualSelection = (modelId: string) => {
    setManualModelIds(current => {
      if (current.includes(modelId)) {
        return current.filter(item => item !== modelId);
      }

      return [...current, modelId];
    });
  };

  const submitManual = () => {
    void buildPreview("manual");
  };

  const selectManualCursor = (index: number) => {
    setManualCursor(clamp(index, 0, Math.max(manualModels.length - 1, 0)));
  };

  const goBackFromPolicy = () => {
    setStep("provider");
  };

  const goBackFromManual = () => {
    setStep("policy");
  };

  const goBackFromPreview = () => {
    setStep(currentPolicy === "manual" ? "manual" : "policy");
  };

  const handleEscape = () => {
    if (step === "preview") {
      goBackFromPreview();
      return;
    }

    if (step === "manual") {
      goBackFromManual();
      return;
    }

    if (step === "policy") {
      goBackFromPolicy();
      return;
    }

    if (step === "provider") {
      setStep(currentScope === "path" ? "path" : "scope");
      return;
    }

    if (step === "path") {
      setStep("scope");
    }
  };

  const scrollPreview = (delta: number) => {
    setPreviewOffset(current => Math.max(current + delta, 0));
  };

  useKeyboard(key => {
    if (busyLabel.length > 0) {
      return;
    }

    if (key.name === "escape") {
      handleEscape();
      return;
    }

    if (step === "scope") {
      if (key.name === "up") {
        selectScope(scopeIndex - 1);
      }

      if (key.name === "down") {
        selectScope(scopeIndex + 1);
      }

      if (isConfirmKey(key.name)) {
        submitScope(scopeIndex);
      }

      return;
    }

    if (step === "policy") {
      if (key.name === "up") {
        selectPolicy(policyIndex - 1);
      }

      if (key.name === "down") {
        selectPolicy(policyIndex + 1);
      }

      if (key.name === "backspace") {
        goBackFromPolicy();
      }

      if (isConfirmKey(key.name)) {
        submitPolicy(policyIndex);
      }

      return;
    }

    if (step === "provider") {
      if (key.name === "tab" || key.name === "down") {
        setProviderField(current => (current === "name" ? "id" : "name"));
      }

      if (key.name === "up") {
        setProviderField(current => (current === "id" ? "name" : "id"));
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

        toggleManualSelection(model.id);
      }

      if (key.name === "backspace") {
        goBackFromManual();
      }

      if (isConfirmKey(key.name)) {
        submitManual();
      }

      return;
    }

    if (step === "preview") {
      if (key.name === "up") {
        scrollPreview(-1);
      }

      if (key.name === "down") {
        scrollPreview(1);
      }

      if (key.name === "backspace") {
        goBackFromPreview();
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

  return {
    step,
    scopeIndex,
    policyIndex,
    targetPath,
    providerId,
    providerName,
    providerField,
    setTargetPath,
    setProviderId,
    setProviderName,
    setProviderField,
    manualCursor,
    preview,
    busyLabel,
    errorMessage,
    textWidth,
    isWide,
    infoColumnWidth,
    currentStepNumber,
    contextItems,
    policyModelCounts,
    visibleManualRows,
    manualModelIds,
    manualModelCount: manualModels.length,
    addedModelIds,
    removedModelIds,
    visiblePreviewModelRows,
    previewOffset,
    setPreviewOffset,
    selectScope,
    submitScope,
    submitPath,
    submitProvider,
    submitProviderName,
    submitProviderId,
    selectPolicy,
    submitPolicy,
    selectManualCursor,
    toggleManualSelection,
    submitManual,
    handleEscape,
    goBackFromPreview,
    applyPreview,
    stepLabel: stepLabels[step],
  };
};
