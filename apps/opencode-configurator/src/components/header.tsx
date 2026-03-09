import { Panel } from "./panel";

type HeaderProps = {
  title: string;
  currentStepLabel: string;
  currentStepNumber?: number;
  totalSteps: number;
};

export const Header = ({ title, currentStepLabel, currentStepNumber, totalSteps }: HeaderProps) => {
  const stepCount = currentStepNumber === undefined ? "Status" : `Step ${currentStepNumber}/${totalSteps}`;

  return (
    <Panel>
      <box style={{ flexDirection: "column" }}>
        <box style={{ flexDirection: "row" }}>
          <text>{title}</text>
          <box style={{ flexGrow: 1 }} />
          <text>{stepCount}</text>
        </box>
        <box style={{ marginTop: 1 }}>
          <text>
            <span fg="gray" bold>
              {currentStepLabel}
            </span>
          </text>
        </box>
      </box>
    </Panel>
  );
};
