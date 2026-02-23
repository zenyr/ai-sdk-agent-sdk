export const VERSION = "0.0.1-rc1";

export const DEFAULT_SUPPORTED_URLS: Record<string, RegExp[]> = {
  "image/*": [/^https?:\/\/.*/],
  "application/pdf": [/^https?:\/\/.*/],
};
