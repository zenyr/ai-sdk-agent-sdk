import { useState } from "react";

type ClickableBoxProps = {
  children: React.ReactNode | ((state: { hovered: boolean; active: boolean }) => React.ReactNode);
  active?: boolean;
  paddingLeft?: number;
  paddingRight?: number;
  onPress(): void;
  style?: {
    flexDirection?: "row" | "column";
    justifyContent?: "flex-start" | "center" | "flex-end" | "space-between" | "space-around" | "space-evenly";
    marginBottom?: number;
    marginTop?: number;
    flexGrow?: number;
    flexShrink?: number;
  };
};

export const ClickableBox = ({
  children,
  active = false,
  paddingLeft,
  paddingRight,
  onPress,
  style,
}: ClickableBoxProps) => {
  const [hovered, setHovered] = useState(false);

  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: OpenTUI box handles mouse events.
    // biome-ignore lint/a11y/useKeyWithMouseEvents: Pointer-only hover state is intentional in TUI.
    <box
      paddingLeft={paddingLeft}
      paddingRight={paddingRight}
      style={style}
      onMouseOver={() => setHovered(true)}
      onMouseOut={() => setHovered(false)}
      onMouseUp={event => {
        if (event.button === 0) {
          onPress();
        }
      }}
    >
      {typeof children === "function" ? children({ hovered, active }) : children}
    </box>
  );
};
