import { ClickableBox } from "../components/clickable-box";
import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type ScopeOption = {
  value: string;
  label: string;
  description: string;
};

type ScopeScreenProps = {
  options: readonly ScopeOption[];
  selectedIndex: number;
  width: number;
  onSelect(index: number): void;
  onSubmit(index: number): void;
};

export const ScopeScreen = ({ options, selectedIndex, width, onSelect, onSubmit }: ScopeScreenProps) => {
  return (
    <Panel fill={true}>
      {options.map((option, index) => (
        <ClickableBox
          key={option.value}
          style={{ flexDirection: "column", marginBottom: 1, flexShrink: 0 }}
          active={index === selectedIndex}
          onPress={() => {
            if (index === selectedIndex) {
              onSubmit(index);
              return;
            }

            onSelect(index);
          }}
        >
          {({ hovered, active }) => (
            <WrappedText
              text={`${active ? ">" : " "} ${option.label} - ${option.description}`}
              width={width}
              color={hovered || active ? "white" : "gray"}
            />
          )}
        </ClickableBox>
      ))}
    </Panel>
  );
};
