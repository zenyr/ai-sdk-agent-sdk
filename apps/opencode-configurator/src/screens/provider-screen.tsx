import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type ProviderScreenProps = {
  providerId: string;
  providerName: string;
  focusedField: "name" | "id";
  width: number;
  onProviderNameInput(value: string): void;
  onProviderIdInput(value: string): void;
  onSubmitName(): void;
  onSubmitId(): void;
};

export const ProviderScreen = ({
  providerId,
  providerName,
  focusedField,
  width,
  onProviderNameInput,
  onProviderIdInput,
  onSubmitName,
  onSubmitId,
}: ProviderScreenProps) => {
  return (
    <Panel fill={true}>
      <WrappedText
        text="Name is user-facing. Id is the stable config key written into the provider block."
        width={width}
      />
      <box border style={{ height: 3, paddingLeft: 1, paddingRight: 1, marginTop: 1 }}>
        <input
          focused={focusedField === "name"}
          value={providerName}
          onInput={onProviderNameInput}
          onSubmit={onSubmitName}
        />
      </box>
      <WrappedText text="Provider name" width={width} />
      <box border style={{ height: 3, paddingLeft: 1, paddingRight: 1, marginTop: 1 }}>
        <input focused={focusedField === "id"} value={providerId} onInput={onProviderIdInput} onSubmit={onSubmitId} />
      </box>
      <WrappedText text="Provider id" width={width} />
    </Panel>
  );
};
