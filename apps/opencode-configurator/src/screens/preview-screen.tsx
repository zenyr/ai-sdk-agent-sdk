import type { PreparedConfig, ProviderModelConfig } from "@zenyr/opencode-configurator-core";
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
};

export const PreviewScreen = ({
  prepared,
  width,
  columnWidth,
  isWide,
  addedModelIds,
  removedModelIds,
  modelRows,
}: PreviewScreenProps) => {
  const reasoningModelCount = Object.values(prepared.providerBlock.models).filter(
    (model: ProviderModelConfig) => model.variants !== undefined
  ).length;
  const actionLabel = prepared.target.exists ? "Update existing config" : "Create new config";
  const modelCount = Object.keys(prepared.providerBlock.models).length;

  return (
    <Panel>
      <box
        border
        borderColor="gray"
        style={{ flexDirection: "column", marginBottom: 1, paddingLeft: 1, paddingRight: 1 }}
      >
        <WrappedText text={actionLabel} width={width} />
        <WrappedText text={`Target file: ${prepared.target.filePath}`} width={width} />
      </box>
      <box
        border
        borderColor="gray"
        style={{
          flexDirection: isWide ? "row" : "column",
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 1,
          paddingBottom: 1,
          flexShrink: 0,
        }}
      >
        <box
          style={{
            flexDirection: "column",
            width: columnWidth,
            marginBottom: isWide ? 0 : 1,
            paddingLeft: 1,
            paddingRight: 1,
            flexShrink: 0,
          }}
        >
          <text>Summary</text>
          <WrappedText text={`action: ${prepared.target.exists ? "update" : "create"}`} width={columnWidth} />
          <WrappedText text={`changes: +${addedModelIds.length} / -${removedModelIds.length}`} width={columnWidth} />
          <WrappedText text={`models kept: ${modelCount}`} width={columnWidth} />
          <WrappedText text={`reasoning enabled: ${reasoningModelCount}`} width={columnWidth} />
        </box>
        <box
          style={{
            flexDirection: "column",
            width: columnWidth,
            marginLeft: isWide ? 2 : 0,
            paddingLeft: 1,
            paddingRight: 1,
            flexShrink: 0,
          }}
        >
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
        <box
          border
          borderColor="green"
          style={{ flexDirection: "column", marginTop: 1, paddingLeft: 1, paddingRight: 1 }}
        >
          <text>
            <span fg="green">Add</span>
          </text>
          <WrappedText text={addedModelIds.join(", ")} width={width} color="gray" />
        </box>
      ) : null}
      {removedModelIds.length > 0 ? (
        <box
          border
          borderColor="red"
          style={{ flexDirection: "column", marginTop: 1, paddingLeft: 1, paddingRight: 1 }}
        >
          <text>
            <span fg="red">Remove</span>
          </text>
          <WrappedText text={removedModelIds.join(", ")} width={width} color="gray" />
        </box>
      ) : null}
      <box border borderColor="gray" style={{ flexDirection: "column", marginTop: 1, paddingLeft: 1, paddingRight: 1 }}>
        <text>Selected models</text>
        <box style={{ flexDirection: "column", flexGrow: 0, flexShrink: 0 }}>
          {modelRows.map(modelRow => (
            <box key={modelRow} style={{ flexDirection: "column", flexShrink: 0 }}>
              <WrappedText text={modelRow} width={width} />
            </box>
          ))}
        </box>
      </box>
    </Panel>
  );
};
