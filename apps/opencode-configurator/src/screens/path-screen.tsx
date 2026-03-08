import { Panel } from "../components/panel";

type PathScreenProps = {
  targetPath: string;
  width: number;
  onInput(value: string): void;
  onSubmit(): void;
};

export const PathScreen = ({ targetPath, width: _width, onInput, onSubmit }: PathScreenProps) => {
  return (
    <Panel fill={true}>
      <box border style={{ height: 3, paddingLeft: 1, paddingRight: 1 }}>
        <input focused={true} value={targetPath} onInput={onInput} onSubmit={onSubmit} />
      </box>
    </Panel>
  );
};
