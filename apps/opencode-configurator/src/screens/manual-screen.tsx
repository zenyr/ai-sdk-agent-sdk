import { Button } from "../components/button";
import { ClickableBox } from "../components/clickable-box";
import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";
import { truncateText } from "../lib/text";

type ManualRow = {
  id: string;
  name: string;
};

type ManualScreenProps = {
  models: ManualRow[];
  selectedIds: string[];
  cursor: number;
  totalCount: number;
  width: number;
  onSelect(index: number): void;
  onToggle(modelId: string): void;
  onConfirm(): void;
};

export const ManualScreen = ({
  models,
  selectedIds,
  cursor,
  totalCount,
  width,
  onSelect,
  onToggle,
  onConfirm,
}: ManualScreenProps) => {
  return (
    <Panel fill={true}>
      <box style={{ flexDirection: "column", flexGrow: 0, flexShrink: 0 }}>
        <WrappedText text={`Selected ${selectedIds.length}. Models ${totalCount}.`} width={width} />
      </box>
      <box style={{ flexDirection: "column", flexGrow: 1, flexShrink: 1 }}>
        {models.map((model: ManualRow, visibleIndex: number) => {
          const index = visibleIndex;
          const selected = selectedIds.includes(model.id);
          const marker = selected ? "[x]" : "[ ]";
          const titleWidth = Math.max(24, width - 6);

          return (
            <ClickableBox
              key={model.id}
              style={{ flexDirection: "column", flexGrow: 0, flexShrink: 0 }}
              active={index === cursor}
              onPress={() => {
                if (index === cursor) {
                  onToggle(model.id);
                  return;
                }

                onSelect(index);
              }}
            >
              {({ hovered, active }) => (
                <text fg={hovered || active ? "white" : "gray"}>
                  {truncateText(`${active ? ">" : " "} ${marker} ${model.name}  ${model.id}`, titleWidth)}
                </text>
              )}
            </ClickableBox>
          );
        })}
      </box>
      <box style={{ flexDirection: "row", justifyContent: "flex-end", marginTop: 1, flexGrow: 0, flexShrink: 0 }}>
        <Button label={`Confirm ${selectedIds.length} models`} onPress={onConfirm} />
      </box>
    </Panel>
  );
};
