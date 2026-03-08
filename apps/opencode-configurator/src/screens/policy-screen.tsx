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
            <>
              <WrappedText
                text={`${active ? ">" : " "} ${option.label}${option.value === recommendedValue ? " (recommended)" : ""} - ${option.description}`}
                width={width}
                color={hovered || active ? "white" : "gray"}
              />
              <WrappedText
                text={`Estimated models: ${modelCounts[option.value] ?? 0}`}
                width={width}
                color={hovered || active ? "white" : "gray"}
              />
            </>
          )}
        </ClickableBox>
      ))}
    </Panel>
  );
};
