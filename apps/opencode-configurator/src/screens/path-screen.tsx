import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type PathScreenProps = {
  targetPath: string;
  width: number;
  onInput(value: string): void;
  onSubmit(): void;
};

export const PathScreen = ({ targetPath, width, onInput, onSubmit }: PathScreenProps) => {
  return (
    <Panel title="Enter explicit config path" footer="Press Enter in the input to continue.">
      <WrappedText text="Press Enter in the input to continue." width={width} />
      <box border style={{ height: 3, paddingLeft: 1, paddingRight: 1 }}>
        <input focused={true} value={targetPath} onInput={onInput} onSubmit={onSubmit} />
      </box>
    </Panel>
  );
};
