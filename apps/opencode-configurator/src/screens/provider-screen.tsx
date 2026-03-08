import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type ProviderScreenProps = {
  providerName: string;
  width: number;
  onInput(value: string): void;
  onSubmit(): void;
};

export const ProviderScreen = ({ providerName, width, onInput, onSubmit }: ProviderScreenProps) => {
  return (
    <Panel title="Choose provider name" footer="Edit the display name. Press Enter in the input to continue.">
      <WrappedText text="This name is what users will see in the OpenCode provider list." width={width} />
      <box border style={{ height: 3, paddingLeft: 1, paddingRight: 1, marginTop: 1 }}>
        <input focused={true} value={providerName} onInput={onInput} onSubmit={onSubmit} />
      </box>
    </Panel>
  );
};
