const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

export const readPackageVersion = (packageJsonText: string): string => {
  let parsedPackageJson: unknown;

  try {
    parsedPackageJson = JSON.parse(packageJsonText);
  } catch {
    throw new Error("failed to parse package.json");
  }

  if (!isRecord(parsedPackageJson)) {
    throw new Error("package.json must be an object");
  }

  const version = parsedPackageJson.version;
  if (typeof version !== "string" || version.length === 0) {
    throw new Error("package version missing from package.json");
  }

  return version;
};

type SyncVersionInConstantsArgs = {
  constantsSource: string;
  version: string;
};

type SyncVersionInConstantsResult = {
  updatedSource: string;
  changed: boolean;
};

export const syncVersionInConstants = (
  args: SyncVersionInConstantsArgs,
): SyncVersionInConstantsResult => {
  if (args.version.length === 0) {
    throw new Error("version must be non-empty");
  }

  const versionMarkerPattern = /^export const VERSION = .*;$/m;
  if (!versionMarkerPattern.test(args.constantsSource)) {
    throw new Error("VERSION marker not found in constants file");
  }

  const updatedSource = args.constantsSource.replace(
    versionMarkerPattern,
    `export const VERSION = "${args.version}";`,
  );

  return {
    updatedSource,
    changed: updatedSource !== args.constantsSource,
  };
};
