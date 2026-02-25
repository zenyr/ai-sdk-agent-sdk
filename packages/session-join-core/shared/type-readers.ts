export const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

export const readRecord = (
  record: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined => {
  const value = record[key];
  if (!isRecord(value)) {
    return undefined;
  }

  return value;
};

export const readString = (record: Record<string, unknown>, key: string): string | undefined => {
  const value = record[key];
  if (typeof value !== "string") {
    return undefined;
  }

  return value;
};

export const safeJsonStringify = (value: unknown): string => {
  try {
    return JSON.stringify(value);
  } catch {
    return "null";
  }
};
