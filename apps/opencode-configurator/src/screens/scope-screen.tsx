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
};

export const ScopeScreen = ({ options, selectedIndex, width, onSelect }: ScopeScreenProps) => {
  return (
    <Panel title="Choose target scope" footer="Arrow keys move. Enter continues.">
      {options.map((option, index) => (
        <ClickableBox
          key={option.value}
          style={{ flexDirection: "column", marginBottom: 1 }}
          backgroundColor={index === selectedIndex ? "cyan" : undefined}
          onPress={() => onSelect(index)}
        >
          <WrappedText
            text={`${index === selectedIndex ? ">" : " "} ${option.label} - ${option.description}`}
            width={width}
          />
        </ClickableBox>
      ))}
    </Panel>
  );
};
