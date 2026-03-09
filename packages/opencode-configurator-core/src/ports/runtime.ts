export type FetchResponse = {
  ok: boolean;
  status: number;
  text(): Promise<string>;
};

export type Runtime = {
  cwd(): string;
  homeDir(): string;
  env(name: string): string | undefined;
  fileExists(filePath: string): Promise<boolean>;
  readText(filePath: string): Promise<string>;
  writeText(filePath: string, text: string): Promise<void>;
  mkdirp(directoryPath: string): Promise<void>;
  fetch(url: string, init?: { headers?: Record<string, string> }): Promise<FetchResponse>;
};
