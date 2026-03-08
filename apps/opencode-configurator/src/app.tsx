import { createCliRenderer } from "@opentui/core";
import { createRoot } from "@opentui/react";
import type { ConfiguratorOptions, Runtime } from "@zenyr/opencode-configurator-core";
import { Header } from "./components/header";
import { Panel } from "./components/panel";
import { WrappedText } from "./components/wrapped-text";
import { policyOptions, scopeOptions, stepLabels, stepOrder, useSetupFlow } from "./hooks/use-setup-flow";
import { ManualScreen } from "./screens/manual-screen";
import { MessageScreen } from "./screens/message-screen";
import { PathScreen } from "./screens/path-screen";
import { PolicyScreen } from "./screens/policy-screen";
import { PreviewScreen } from "./screens/preview-screen";
import { ProviderScreen } from "./screens/provider-screen";
import { ScopeScreen } from "./screens/scope-screen";

type SetupAppProps = {
  runtime: Runtime;
  options: ConfiguratorOptions;
  userAgent: string;
  onExit(exitCode: number): void;
};

const SetupApp = ({ runtime, options, userAgent, onExit }: SetupAppProps) => {
  const flow = useSetupFlow({ runtime, options, userAgent, onExit });

  return (
    <box style={{ flexDirection: "column", paddingLeft: 1, paddingRight: 1 }}>
      <Header
        title="Agent SDK OpenCode Configurator"
        subtitle="Arrow keys move. Enter confirms. Backspace goes back. Esc quits."
        currentStepLabel={stepLabels[flow.step]}
        currentStepNumber={flow.currentStepNumber}
        totalSteps={stepOrder.length}
        contextItems={flow.contextItems}
        width={flow.textWidth}
      />

      {flow.busyLabel.length > 0 ? (
        <Panel title="Working" tone="accent">
          <WrappedText text={flow.busyLabel} width={flow.textWidth} />
        </Panel>
      ) : null}

      {flow.step === "scope" ? (
        <ScopeScreen
          options={scopeOptions}
          selectedIndex={flow.scopeIndex}
          width={flow.textWidth}
          onSelect={flow.submitScope}
        />
      ) : null}

      {flow.step === "path" ? (
        <PathScreen
          targetPath={flow.targetPath}
          width={flow.textWidth}
          onInput={flow.setTargetPath}
          onSubmit={flow.submitPath}
        />
      ) : null}

      {flow.step === "provider" ? (
        <ProviderScreen
          providerName={flow.providerName}
          width={flow.textWidth}
          onInput={flow.setProviderName}
          onSubmit={flow.submitProvider}
        />
      ) : null}

      {flow.step === "policy" ? (
        <PolicyScreen
          options={policyOptions}
          selectedIndex={flow.policyIndex}
          width={flow.textWidth}
          modelCounts={flow.policyModelCounts}
          recommendedValue="mainstream"
          onSelect={flow.selectPolicy}
          onSubmit={flow.submitPolicy}
        />
      ) : null}

      {flow.step === "manual" ? (
        <ManualScreen
          models={flow.visibleManualRows}
          selectedIds={flow.manualModelIds}
          cursor={flow.manualCursor}
          startIndex={flow.manualPageStart}
          totalCount={flow.manualModelCount}
          width={flow.textWidth}
          onToggle={flow.toggleManualSelection}
          onConfirm={flow.submitManual}
        />
      ) : null}

      {flow.step === "preview" && flow.preview !== undefined ? (
        <PreviewScreen
          prepared={flow.preview}
          width={flow.textWidth}
          columnWidth={flow.infoColumnWidth}
          isWide={flow.isWide}
          addedModelIds={flow.addedModelIds}
          removedModelIds={flow.removedModelIds}
          modelRows={flow.visiblePreviewModelRows}
          hiddenModelCount={flow.hiddenPreviewModelCount}
          onBack={flow.goBackFromPreview}
          onApply={() => void flow.applyPreview()}
        />
      ) : null}

      {flow.step === "done" && flow.preview !== undefined ? (
        <MessageScreen
          title="Done"
          lines={[`Updated ${flow.preview.target.filePath}`, "Press Enter to exit."]}
          width={flow.textWidth}
          tone="success"
        />
      ) : null}

      {flow.step === "error" ? (
        <MessageScreen
          title="Error"
          lines={[flow.errorMessage ?? "Unknown error", "Press Enter or Esc to exit."]}
          width={flow.textWidth}
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
