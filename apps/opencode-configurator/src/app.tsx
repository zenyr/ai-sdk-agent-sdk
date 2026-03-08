import { createCliRenderer, type ScrollBoxRenderable } from "@opentui/core";
import { createRoot } from "@opentui/react";
import type { ConfiguratorOptions, Runtime } from "@zenyr/opencode-configurator-core";
import { useEffect, useRef } from "react";
import { Header } from "./components/header";
import { KeyHelpBar } from "./components/key-help-bar";
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
  const bodyScrollRef = useRef<ScrollBoxRenderable | null>(null);

  useEffect(() => {
    const scrollbox = bodyScrollRef.current;
    if (scrollbox === null) {
      return;
    }

    if (flow.step === "manual") {
      const top = scrollbox.scrollTop;
      const viewportHeight = scrollbox.viewport.height;
      const bottom = top + Math.max(viewportHeight - 1, 0);

      if (flow.manualCursor < top + 1) {
        scrollbox.scrollTo({ x: 0, y: Math.max(flow.manualCursor - 1, 0) });
      } else if (flow.manualCursor > bottom - 1) {
        scrollbox.scrollTo({ x: 0, y: Math.max(flow.manualCursor - viewportHeight + 2, 0) });
      }

      return;
    }

    if (flow.step === "preview") {
      const maxScroll = Math.max(scrollbox.scrollHeight - scrollbox.viewport.height, 0);
      const boundedOffset = Math.min(Math.max(flow.previewOffset, 0), maxScroll);

      if (boundedOffset !== flow.previewOffset) {
        flow.setPreviewOffset(boundedOffset);
        return;
      }

      scrollbox.scrollTo({ x: 0, y: boundedOffset });

      return;
    }

    scrollbox.scrollTo({ x: 0, y: 0 });
  }, [flow.manualCursor, flow.previewOffset, flow.setPreviewOffset, flow.step]);

  const helpItems =
    flow.step === "scope"
      ? [
          { keycap: "up/down", label: "move" },
          { keycap: "enter", label: "next", onPress: () => flow.submitScope(flow.scopeIndex) },
          { keycap: "click", label: "select" },
          { keycap: "esc", label: "back", onPress: flow.handleEscape },
          { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
        ]
      : flow.step === "path"
        ? [
            { keycap: "enter", label: "next", onPress: flow.submitPath },
            { keycap: "esc", label: "back", onPress: flow.handleEscape },
            { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
          ]
        : flow.step === "provider"
          ? [
              { keycap: "enter", label: "next", onPress: flow.submitProviderId },
              { keycap: "esc", label: "back", onPress: flow.handleEscape },
              { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
            ]
          : flow.step === "policy"
            ? [
                { keycap: "up/down", label: "move" },
                { keycap: "enter", label: "next", onPress: () => flow.submitPolicy(flow.policyIndex) },
                { keycap: "click", label: "select" },
                { keycap: "backspace", label: "back", onPress: flow.handleEscape },
                { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
              ]
            : flow.step === "manual"
              ? [
                  { keycap: "up/down", label: "move" },
                  { keycap: "space", label: "toggle" },
                  { keycap: "enter", label: "next", onPress: flow.submitManual },
                  { keycap: "click", label: "toggle" },
                  { keycap: "backspace", label: "back", onPress: flow.handleEscape },
                  { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
                ]
              : flow.step === "preview"
                ? [
                    { keycap: "up/down", label: "scroll" },
                    { keycap: "enter", label: "apply", onPress: () => void flow.applyPreview() },
                    { keycap: "backspace", label: "back", onPress: flow.handleEscape },
                    { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
                  ]
                : [
                    { keycap: "enter", label: "done", onPress: () => onExit(0) },
                    { keycap: "ctrl+c", label: "exit", onPress: () => onExit(0) },
                  ];

  return (
    <box style={{ flexDirection: "column", paddingTop: 1, flexGrow: 1, flexShrink: 1 }}>
      <box style={{ flexGrow: 0, flexShrink: 0 }}>
        <Header
          title="Agent SDK OpenCode Configurator"
          currentStepLabel={stepLabels[flow.step]}
          currentStepNumber={flow.currentStepNumber}
          totalSteps={stepOrder.length}
        />
      </box>

      {flow.busyLabel.length > 0 ? (
        <box style={{ flexGrow: 0, flexShrink: 0 }}>
          <Panel title="Working" tone="accent">
            <WrappedText text={flow.busyLabel} width={flow.textWidth} />
          </Panel>
        </box>
      ) : null}

      <scrollbox
        key={flow.step}
        ref={bodyScrollRef}
        scrollY={true}
        style={{ flexDirection: "column", flexGrow: 1, flexShrink: 1, width: "100%" }}
        rootOptions={{ width: "100%" }}
        wrapperOptions={{ width: "100%", flexGrow: 1, flexShrink: 1 }}
        viewportOptions={{ width: "100%", flexGrow: 1, flexShrink: 1 }}
        contentOptions={{ width: "100%", flexGrow: 1, flexShrink: 0 }}
        verticalScrollbarOptions={{ visible: false }}
        horizontalScrollbarOptions={{ visible: false }}
      >
        {flow.step === "scope" ? (
          <ScopeScreen
            options={scopeOptions}
            selectedIndex={flow.scopeIndex}
            width={flow.textWidth}
            onSelect={flow.selectScope}
            onSubmit={flow.submitScope}
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
            providerId={flow.providerId}
            providerName={flow.providerName}
            focusedField={flow.providerField}
            width={flow.textWidth}
            onProviderNameInput={flow.setProviderName}
            onProviderIdInput={flow.setProviderId}
            onSubmitName={flow.submitProviderName}
            onSubmitId={flow.submitProviderId}
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
            totalCount={flow.manualModelCount}
            width={flow.textWidth}
            onSelect={flow.selectManualCursor}
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
          />
        ) : null}

        {flow.step === "done" && flow.preview !== undefined ? (
          <MessageScreen
            title="Done"
            lines={[`Updated ${flow.preview.target.filePath}`]}
            width={flow.textWidth}
            tone="success"
          />
        ) : null}

        {flow.step === "error" ? (
          <MessageScreen
            title="Error"
            lines={[flow.errorMessage ?? "Unknown error"]}
            width={flow.textWidth}
            tone="danger"
          />
        ) : null}
      </scrollbox>

      <box style={{ flexGrow: 0, flexShrink: 0 }}>
        <KeyHelpBar items={helpItems} />
      </box>
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
    useMouse: true,
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
