import { readPackageVersion, syncVersionInConstants } from "./sync-version-core";

const packageFile = Bun.file(new URL("../package.json", import.meta.url));
const packageJsonText = await packageFile.text();
const version = readPackageVersion(packageJsonText);

const constantsFile = Bun.file(new URL("../shared/constants.ts", import.meta.url));
const currentConstants = await constantsFile.text();

const syncResult = syncVersionInConstants({
  constantsSource: currentConstants,
  version,
});

if (syncResult.changed) {
  await Bun.write(constantsFile, syncResult.updatedSource);
}
