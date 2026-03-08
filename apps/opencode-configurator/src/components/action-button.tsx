import { ClickableBox } from "./clickable-box";

type ActionButtonProps = {
  label: string;
  tone?: "default" | "accent" | "success";
  onPress(): void;
};

const toneBackground = {
  default: "gray",
  accent: "cyan",
  success: "green",
} as const;

export const ActionButton = ({ label, tone = "default", onPress }: ActionButtonProps) => {
  return (
    <ClickableBox backgroundColor={toneBackground[tone]} paddingLeft={1} paddingRight={1} onPress={onPress}>
      <text>{label}</text>
    </ClickableBox>
  );
};
