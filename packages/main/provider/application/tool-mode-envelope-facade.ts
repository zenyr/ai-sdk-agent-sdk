import type { LanguageModelV3Content } from "@ai-sdk/provider";

import {
  isStructuredTextEnvelope,
  isStructuredToolEnvelope,
  mapStructuredToolCallsToContent,
  parseStructuredEnvelopeFromText,
  parseStructuredEnvelopeFromUnknown,
} from "../../bridge/parse-utils";

type ToolCallContent = Extract<LanguageModelV3Content, { type: "tool-call" }>;

export type ToolModeEnvelopeResolution = {
  toolCalls: ToolCallContent[];
  text: string | undefined;
};

export const hasToolModeEnvelopeResolution = (resolution: ToolModeEnvelopeResolution): boolean => {
  return resolution.toolCalls.length > 0 || resolution.text !== undefined;
};

const emptyResolution = (): ToolModeEnvelopeResolution => {
  return {
    toolCalls: [],
    text: undefined,
  };
};

const resolveParsedToolModeEnvelope = (args: {
  parsedEnvelope: unknown;
  idGenerator: () => string;
}): ToolModeEnvelopeResolution => {
  if (isStructuredToolEnvelope(args.parsedEnvelope)) {
    const toolCalls = mapStructuredToolCallsToContent(args.parsedEnvelope.calls, args.idGenerator).filter(
      isToolCallContent
    );

    return {
      toolCalls,
      text: undefined,
    };
  }

  if (isStructuredTextEnvelope(args.parsedEnvelope)) {
    return {
      toolCalls: [],
      text: args.parsedEnvelope.text,
    };
  }

  return emptyResolution();
};

export const resolveToolModeEnvelopeFromUnknown = (args: {
  value: unknown;
  idGenerator: () => string;
}): ToolModeEnvelopeResolution => {
  return resolveParsedToolModeEnvelope({
    parsedEnvelope: parseStructuredEnvelopeFromUnknown(args.value),
    idGenerator: args.idGenerator,
  });
};

export const resolveToolModeEnvelopeFromText = (args: {
  text: string;
  idGenerator: () => string;
}): ToolModeEnvelopeResolution => {
  return resolveParsedToolModeEnvelope({
    parsedEnvelope: parseStructuredEnvelopeFromText(args.text),
    idGenerator: args.idGenerator,
  });
};
const isToolCallContent = (content: LanguageModelV3Content): content is ToolCallContent => {
  return content.type === "tool-call";
};
