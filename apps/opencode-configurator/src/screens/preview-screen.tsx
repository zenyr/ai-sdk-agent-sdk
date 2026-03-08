import type { PreparedConfig, ProviderModelConfig } from "@zenyr/opencode-configurator-core";
import { ActionButton } from "../components/action-button";
import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type PreviewScreenProps = {
  prepared: PreparedConfig;
  width: number;
  columnWidth: number;
  isWide: boolean;
  addedModelIds: string[];
  removedModelIds: string[];
  modelRows: string[];
  hiddenModelCount: number;
  onBack(): void;
  onApply(): void;
};

export const PreviewScreen = ({
  prepared,
  width,
  columnWidth,
  isWide,
  addedModelIds,
  removedModelIds,
  modelRows,
  hiddenModelCount,
  onBack,
  onApply,
}: PreviewScreenProps) => {
  const reasoningModelCount = Object.values(prepared.providerBlock.models).filter(
    (model: ProviderModelConfig) => model.variants !== undefined
  ).length;
  const actionLabel = prepared.target.exists ? "Update existing config" : "Create new config";
  const modelCount = Object.keys(prepared.providerBlock.models).length;

  return (
    <Panel title="Preview" footer="Up/down scroll models. Enter applies config. Backspace goes back.">
      <box style={{ flexDirection: "column", marginBottom: 1 }} backgroundColor="cyan">
        <WrappedText text={actionLabel} width={width} />
        <WrappedText text={`Target file: ${prepared.target.filePath}`} width={width} />
      </box>
      <box style={{ flexDirection: isWide ? "row" : "column" }}>
        <box
          style={{ flexDirection: "column", width: columnWidth, marginBottom: isWide ? 0 : 1 }}
          backgroundColor="blue"
        >
          <text>Summary</text>
          <WrappedText text={`action: ${prepared.target.exists ? "update" : "create"}`} width={columnWidth} />
          <WrappedText text={`changes: +${addedModelIds.length} / -${removedModelIds.length}`} width={columnWidth} />
          <WrappedText text={`models kept: ${modelCount}`} width={columnWidth} />
          <WrappedText text={`reasoning enabled: ${reasoningModelCount}`} width={columnWidth} />
        </box>
        <box style={{ flexDirection: "column", width: columnWidth, marginLeft: isWide ? 2 : 0 }} backgroundColor="gray">
          <text>Source</text>
          <WrappedText text={`scope: ${prepared.target.scope}`} width={columnWidth} />
          <WrappedText text={`format: ${prepared.target.format}`} width={columnWidth} />
          <WrappedText text={`policy: ${prepared.selectedPolicy}`} width={columnWidth} />
          <WrappedText text={`source: ${prepared.manifest.source.kind}`} width={columnWidth} />
          <WrappedText text={`fetched: ${prepared.manifest.source.fetchedAt}`} width={columnWidth} />
          <WrappedText text={`provider: ${prepared.providerId}`} width={columnWidth} />
          <WrappedText text={`name: ${prepared.providerBlock.name}`} width={columnWidth} />
          <WrappedText text={`package: ${prepared.providerBlock.npm}`} width={columnWidth} />
        </box>
      </box>
      {addedModelIds.length > 0 ? (
        <box style={{ flexDirection: "column", marginTop: 1 }} backgroundColor="green">
          <text>Add</text>
          <WrappedText text={addedModelIds.join(", ")} width={width} />
        </box>
      ) : null}
      {removedModelIds.length > 0 ? (
        <box style={{ flexDirection: "column", marginTop: 1 }} backgroundColor="red">
          <text>Remove</text>
          <WrappedText text={removedModelIds.join(", ")} width={width} />
        </box>
      ) : null}
      <box style={{ flexDirection: "column", marginTop: 1 }}>
        <text>Selected models</text>
      </box>
      {modelRows.map(modelRow => (
        <box key={modelRow} style={{ flexDirection: "column" }}>
          <WrappedText text={modelRow} width={width} />
        </box>
      ))}
      {hiddenModelCount > 0 ? <WrappedText text={`... ${hiddenModelCount} more models`} width={width} /> : null}
      <box style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 1 }}>
        <ActionButton label="Back" onPress={onBack} />
        <ActionButton label="Apply config" tone="success" onPress={onApply} />
      </box>
    </Panel>
  );
};
