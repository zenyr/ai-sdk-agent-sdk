import { ActionButton } from "../components/action-button";
import { ClickableBox } from "../components/clickable-box";
import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type PolicyOption = {
  value: string;
  label: string;
  description: string;
};

type PolicyScreenProps = {
  options: readonly PolicyOption[];
  selectedIndex: number;
  width: number;
  modelCounts: Record<string, number>;
  recommendedValue: string;
  onSelect(index: number): void;
  onSubmit(index: number): void;
};

export const PolicyScreen = ({
  options,
  selectedIndex,
  width,
  modelCounts,
  recommendedValue,
  onSelect,
  onSubmit,
}: PolicyScreenProps) => {
  return (
    <Panel title="Choose selection policy" footer="Arrow keys move. Enter previews. Backspace goes back.">
      {options.map((option, index) => (
        <ClickableBox
          key={option.value}
          style={{ flexDirection: "column", marginBottom: 1 }}
          backgroundColor={index === selectedIndex ? "cyan" : undefined}
          onPress={() => onSubmit(index)}
        >
          <WrappedText
            text={`${index === selectedIndex ? ">" : " "} ${option.label}${option.value === recommendedValue ? " (recommended)" : ""} - ${option.description}`}
            width={width}
          />
          <WrappedText text={`Estimated models: ${modelCounts[option.value] ?? 0}`} width={width} />
        </ClickableBox>
      ))}
      <box style={{ flexDirection: "row", justifyContent: "space-between", marginTop: 1 }}>
        <ActionButton label="Select highlighted" onPress={() => onSelect(selectedIndex)} />
        <ActionButton label="Continue" tone="accent" onPress={() => onSubmit(selectedIndex)} />
      </box>
    </Panel>
  );
};
