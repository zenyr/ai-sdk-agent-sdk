export const VERSION = "0.0.5-rc.0";

export const DEFAULT_SUPPORTED_URLS: Record<string, RegExp[]> = {
  "image/*": [/^https?:\/\/.*/],
  "application/pdf": [/^https?:\/\/.*/],
};
