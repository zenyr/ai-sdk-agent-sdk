import type { JSONObject } from "@ai-sdk/provider";

import { isRecord, readRecord, readString } from "../shared/type-readers";

export const forwardAnthropicContainerIdFromLastStep = ({
  steps,
}: {
  steps: Array<{
    providerMetadata?: Record<string, JSONObject>;
  }>;
}): undefined | { providerOptions?: Record<string, JSONObject> } => {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const step = steps[index];
    if (step === undefined) {
      continue;
    }

    const metadata = step.providerMetadata;
    if (metadata === undefined) {
      continue;
    }

    const anthropicMetadata = metadata.anthropic;
    if (!isRecord(anthropicMetadata)) {
      continue;
    }

    const container = readRecord(anthropicMetadata, "container");
    if (container === undefined) {
      continue;
    }

    const containerId = readString(container, "id");
    if (typeof containerId !== "string") {
      continue;
    }

    return {
      providerOptions: {
        anthropic: {
          container: {
            id: containerId,
          },
        },
      },
    };
  }

  return undefined;
};
