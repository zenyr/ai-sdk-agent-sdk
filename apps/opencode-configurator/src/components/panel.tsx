import { colors } from "../lib/theme";

type PanelProps = {
  children: React.ReactNode;
  title?: string;
  footer?: string;
  tone?: "default" | "accent" | "success" | "danger";
};

const toneStyles = {
  default: {
    borderColor: colors.border,
    backgroundColor: colors.panel,
  },
  accent: {
    borderColor: colors.accent,
    backgroundColor: colors.accentSoft,
  },
  success: {
    borderColor: colors.success,
    backgroundColor: colors.successSoft,
  },
  danger: {
    borderColor: colors.danger,
    backgroundColor: colors.dangerSoft,
  },
} as const;

export const Panel = ({ children, title, footer, tone = "default" }: PanelProps) => {
  const style = toneStyles[tone];

  return (
    <box
      border
      borderColor={style.borderColor}
      backgroundColor={style.backgroundColor}
      style={{
        flexDirection: "column",
        paddingLeft: 1,
        paddingRight: 1,
        marginBottom: 1,
      }}
    >
      {title !== undefined ? (
        <box style={{ marginBottom: 1 }}>
          <text>{title}</text>
        </box>
      ) : null}
      {children}
      {footer !== undefined ? (
        <box style={{ marginTop: 1 }}>
          <text>{footer}</text>
        </box>
      ) : null}
    </box>
  );
};
