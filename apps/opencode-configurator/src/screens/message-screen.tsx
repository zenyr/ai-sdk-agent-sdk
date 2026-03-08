import { Panel } from "../components/panel";
import { WrappedText } from "../components/wrapped-text";

type MessageScreenProps = {
  title: string;
  lines: string[];
  width: number;
  tone?: "default" | "accent" | "success" | "danger";
};

export const MessageScreen = ({ title, lines, width, tone }: MessageScreenProps) => {
  return (
    <Panel title={title} tone={tone} fill={true}>
      {lines.map(line => (
        <box key={line} style={{ flexDirection: "column" }}>
          <WrappedText text={line} width={width} />
        </box>
      ))}
    </Panel>
  );
};
