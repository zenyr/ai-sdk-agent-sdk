import { useState } from "react";

type KeyHelpItem = {
  keycap: string;
  label: string;
  onPress?(): void;
};

type KeyHelpBarProps = {
  items: KeyHelpItem[];
};

export const KeyHelpBar = ({ items }: KeyHelpBarProps) => {
  const passiveItems = items.filter(item => item.onPress === undefined);
  const interactiveItems = items.filter(item => item.onPress !== undefined);

  return (
    <box
      backgroundColor="black"
      style={{ flexDirection: "column", marginTop: 1, paddingLeft: 1, paddingRight: 1, flexShrink: 0 }}
    >
      <box style={{ height: 1 }} />
      <box style={{ flexDirection: "row" }}>
        <box style={{ flexDirection: "row" }}>
          {passiveItems.map(item => (
            <KeyHelpChip key={`${item.keycap}:${item.label}`} item={item} />
          ))}
        </box>
        <box style={{ flexGrow: 1 }} />
        <box style={{ flexDirection: "row" }}>
          {interactiveItems.map(item => (
            <KeyHelpChip key={`${item.keycap}:${item.label}`} item={item} />
          ))}
        </box>
      </box>
      <box style={{ height: 1 }} />
    </box>
  );
};

const KeyHelpChip = ({ item }: { item: KeyHelpItem }) => {
  const [hovered, setHovered] = useState(false);
  const actionProps: Record<string, unknown> =
    item.onPress === undefined
      ? {}
      : {
          onMouseUp: (event: { button: number }) => {
            if (event.button === 0) {
              item.onPress?.();
            }
          },
          onMouseOver: () => setHovered(true),
          onMouseOut: () => setHovered(false),
        };

  return (
    <box style={{ marginRight: 2, flexShrink: 0 }} {...actionProps}>
      <text>
        <span fg="gray">{item.keycap}</span>
        <span fg={hovered ? "white" : "gray"}> {item.label}</span>
      </text>
    </box>
  );
};
