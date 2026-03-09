import { useState } from "react";

type ButtonProps = {
  label: string;
  focused?: boolean;
  onPress(): void;
};

export const Button = ({ label, focused = false, onPress }: ButtonProps) => {
  const [hovered, setHovered] = useState(false);
  const color = focused || hovered ? "white" : "gray";
  const interactiveProps: Record<string, unknown> = {
    onMouseUp: (event: { button: number }) => {
      if (event.button === 0) {
        onPress();
      }
    },
    onMouseOver: () => setHovered(true),
    onMouseOut: () => setHovered(false),
  };

  return (
    <box paddingLeft={1} paddingRight={1} style={{ flexShrink: 0 }} {...interactiveProps}>
      <text fg={color}>{label}</text>
    </box>
  );
};
