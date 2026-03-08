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
};

export const ScopeScreen = ({ options, selectedIndex, width }: ScopeScreenProps) => {
  return (
    <Panel title="Choose target scope" footer="Arrow keys move. Enter continues.">
      {options.map((option, index) => (
        <box key={option.value} style={{ flexDirection: "column" }}>
          <WrappedText
            text={`${index === selectedIndex ? ">" : " "} ${option.label} - ${option.description}`}
            width={width}
          />
        </box>
      ))}
    </Panel>
  );
};
