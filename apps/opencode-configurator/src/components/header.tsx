import { colors } from "../lib/theme";
import { Panel } from "./panel";
import { WrappedText } from "./wrapped-text";

type HeaderProps = {
  title: string;
  subtitle: string;
  currentStepLabel: string;
  currentStepNumber?: number;
  totalSteps: number;
  contextItems: string[];
  width: number;
};

export const Header = ({
  title,
  subtitle,
  currentStepLabel,
  currentStepNumber,
  totalSteps,
  contextItems,
  width,
}: HeaderProps) => {
  const stepText =
    currentStepNumber === undefined
      ? `Status: ${currentStepLabel}`
      : `Step ${currentStepNumber}/${totalSteps} - ${currentStepLabel}`;

  return (
    <Panel tone="accent">
      <box style={{ flexDirection: "column" }}>
        <box>
          <text>{title}</text>
        </box>
        <box backgroundColor={colors.warning}>
          <text>{stepText}</text>
        </box>
      </box>
      <box style={{ marginTop: 1 }}>
        <WrappedText text={subtitle} width={width} />
      </box>
      {contextItems.length > 0 ? (
        <box style={{ flexDirection: "column", marginTop: 1 }}>
          {contextItems.map(item => (
            <WrappedText key={item} text={item} width={width} />
          ))}
        </box>
      ) : null}
    </Panel>
  );
};
