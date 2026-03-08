type ClickableBoxProps = {
  children: React.ReactNode;
  backgroundColor?: string;
  paddingLeft?: number;
  paddingRight?: number;
  onPress(): void;
  style?: {
    flexDirection?: "row" | "column";
    justifyContent?: "flex-start" | "center" | "flex-end" | "space-between" | "space-around" | "space-evenly";
    marginBottom?: number;
    marginTop?: number;
  };
};

export const ClickableBox = ({
  children,
  backgroundColor,
  paddingLeft,
  paddingRight,
  onPress,
  style,
}: ClickableBoxProps) => {
  const interactiveProps: Record<string, unknown> = {
    onClick: onPress,
  };

  return (
    <box
      backgroundColor={backgroundColor}
      paddingLeft={paddingLeft}
      paddingRight={paddingRight}
      style={style}
      {...interactiveProps}
    >
      {children}
    </box>
  );
};
