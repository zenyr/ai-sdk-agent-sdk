type PanelProps = {
  children: React.ReactNode;
  title?: string;
  tone?: "default" | "accent" | "success" | "danger";
  fill?: boolean;
};

export const Panel = ({ children, title, tone = "default", fill = false }: PanelProps) => {
  const titlePrefix =
    tone === "danger" ? "Error" : tone === "success" ? "Done" : tone === "accent" ? "Info" : undefined;

  return (
    <box
      style={{
        flexDirection: "column",
        paddingLeft: 1,
        paddingRight: 1,
        marginBottom: 1,
        flexGrow: fill ? 1 : 0,
        flexShrink: fill ? 1 : 0,
      }}
    >
      {title !== undefined ? (
        <box style={{ marginBottom: 1 }}>
          <text>
            {titlePrefix === undefined || title.startsWith(titlePrefix) ? title : `${titlePrefix} - ${title}`}
          </text>
        </box>
      ) : null}
      {children}
    </box>
  );
};
