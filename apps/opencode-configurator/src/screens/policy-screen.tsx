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
};

export const PolicyScreen = ({ options, selectedIndex, width }: PolicyScreenProps) => {
  return (
    <Panel title="Choose selection policy" footer="Arrow keys move. Enter previews. Backspace goes back.">
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
