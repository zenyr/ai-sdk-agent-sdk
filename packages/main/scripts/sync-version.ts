const packageFile = Bun.file(new URL("../package.json", import.meta.url));
const packageJsonText = await packageFile.text();

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === "object" && value !== null;
};

const readVersion = (value: unknown): string | undefined => {
  if (!isRecord(value)) {
    return undefined;
  }

  const version = value.version;
  if (typeof version !== "string" || version.length === 0) {
    return undefined;
  }

  return version;
};

const packageJson = JSON.parse(packageJsonText);
const version = readVersion(packageJson);
if (version === undefined || version.length === 0) {
  throw new Error("package version missing from package.json");
}

const constantsFile = Bun.file(new URL("../shared/constants.ts", import.meta.url));
const currentConstants = await constantsFile.text();

const versionMarkerPattern = /^export const VERSION = .*;$/m;
if (!versionMarkerPattern.test(currentConstants)) {
  throw new Error("VERSION marker not found in constants file");
}

const updatedConstants = currentConstants.replace(
  versionMarkerPattern,
  `export const VERSION = "${version}";`,
);

if (updatedConstants !== currentConstants) {
  await Bun.write(constantsFile, updatedConstants);
}
