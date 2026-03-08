import { ActionButton } from "../components/action-button";
import { ClickableBox } from "../components/clickable-box";
import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type ManualRow = {
  id: string;
  family: string;
  reasoning: boolean;
  status: string;
  output: number;
};

type ManualScreenProps = {
  models: ManualRow[];
  selectedIds: string[];
  cursor: number;
  startIndex: number;
  totalCount: number;
  width: number;
  onToggle(modelId: string): void;
  onConfirm(): void;
};

export const ManualScreen = ({
  models,
  selectedIds,
  cursor,
  startIndex,
  totalCount,
  width,
  onToggle,
  onConfirm,
}: ManualScreenProps) => {
  return (
    <Panel
      title="Manual model picker"
      footer="Space toggles. Enter previews. Click rows to check. Backspace goes back."
    >
      <WrappedText
        text={`Selected ${selectedIds.length}. Showing ${startIndex + 1}-${Math.min(startIndex + models.length, totalCount)} of ${totalCount}.`}
        width={width}
      />
      {models.map((model: ManualRow, visibleIndex: number) => {
        const index = startIndex + visibleIndex;
        const selected = selectedIds.includes(model.id);
        const marker = selected ? "[x]" : "[ ]";
        return (
          <ClickableBox
            key={model.id}
            style={{ flexDirection: "column", marginBottom: 1 }}
            backgroundColor={index === cursor ? "cyan" : undefined}
            onPress={() => onToggle(model.id)}
          >
            <WrappedText text={`${index === cursor ? ">" : " "} ${marker} ${model.id}`} width={width} />
            <WrappedText
              text={`  family ${model.family} | status ${model.status} | ${model.reasoning ? "reasoning" : "plain"} | out ${model.output}`}
              width={width}
            />
          </ClickableBox>
        );
      })}
      <box style={{ flexDirection: "row", justifyContent: "flex-end", marginTop: 1 }}>
        <ActionButton label={`Confirm ${selectedIds.length} models`} tone="accent" onPress={onConfirm} />
      </box>
    </Panel>
  );
};
