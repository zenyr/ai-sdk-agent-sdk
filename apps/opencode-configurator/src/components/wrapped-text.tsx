import { wrapText } from "../lib/text";

type WrappedTextProps = {
  text: string;
  width: number;
  color?: string;
};

export const WrappedText = ({ text, width, color }: WrappedTextProps) => {
  const seen = new Map<string, number>();

  return (
    <box style={{ flexDirection: "column" }}>
      {wrapText(text, width).map(line => {
        const count = (seen.get(line) ?? 0) + 1;
        seen.set(line, count);
        if (color === undefined) {
          return <text key={`${text}:${line}:${count}`}>{line}</text>;
        }

        return (
          <text key={`${text}:${line}:${count}`} fg={color}>
            {line}
          </text>
        );
      })}
    </box>
  );
};
