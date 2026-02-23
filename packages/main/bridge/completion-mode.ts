import type {
  LanguageModelV3CallOptions,
  LanguageModelV3FunctionTool,
} from "@ai-sdk/provider";

import {
  isFunctionTool,
  isRecord,
  safeJsonStringify,
} from "../shared/type-readers";

type CompletionModeBase = {
  type: "plain-text" | "json" | "tools";
};

export type CompletionMode =
  | {
      type: "plain-text";
    }
  | {
      type: "json";
      schema: Record<string, unknown>;
    }
  | {
      type: "tools";
      schema: Record<string, unknown>;
      tools: LanguageModelV3FunctionTool[];
    };

export const buildToolInstruction = (
  tools: LanguageModelV3FunctionTool[],
  toolChoice: LanguageModelV3CallOptions["toolChoice"],
): string => {
  const toolLines = tools
    .map((toolDefinition) => {
      const description = toolDefinition.description ?? "No description";
      const schema = safeJsonStringify(toolDefinition.inputSchema);

      return `- ${toolDefinition.name}: ${description}\n  schema: ${schema}`;
    })
    .join("\n");

  let toolChoiceInstruction = "Choose tools automatically when necessary.";
  if (toolChoice?.type === "required") {
    toolChoiceInstruction = "You must return at least one tool call.";
  }

  if (toolChoice?.type === "tool") {
    toolChoiceInstruction = `You must call exactly this tool: ${toolChoice.toolName}.`;
  }

  return [
    "You are in tool routing mode.",
    toolChoiceInstruction,
    "Return strictly valid JSON and no markdown.",
    'If a tool is needed, return {"type":"tool-calls","calls":[{"toolName":"...","input":{...}}]}.',
    'If no tool is needed, return {"type":"text","text":"..."}.',
    "Available tools:",
    toolLines,
  ].join("\n");
};

export const buildToolSchema = (
  tools: LanguageModelV3FunctionTool[],
  toolChoice: LanguageModelV3CallOptions["toolChoice"],
): Record<string, unknown> => {
  const filteredTools =
    toolChoice?.type === "tool"
      ? tools.filter(
          (toolDefinition) => toolDefinition.name === toolChoice.toolName,
        )
      : tools;

  const requiresAtLeastOneCall =
    toolChoice?.type === "required" || toolChoice?.type === "tool";

  const callVariants = filteredTools.map((toolDefinition) => {
    return {
      type: "object",
      additionalProperties: false,
      required: ["toolName", "input"],
      properties: {
        toolName: { const: toolDefinition.name },
        input: toolDefinition.inputSchema,
      },
    };
  });

  return {
    type: "object",
    oneOf: [
      {
        type: "object",
        additionalProperties: false,
        required: ["type", "text"],
        properties: {
          type: { const: "text" },
          text: { type: "string" },
        },
      },
      {
        type: "object",
        additionalProperties: false,
        required: ["type", "calls"],
        properties: {
          type: { const: "tool-calls" },
          calls: {
            type: "array",
            minItems: requiresAtLeastOneCall ? 1 : undefined,
            items:
              callVariants.length > 0
                ? { oneOf: callVariants }
                : { type: "object", additionalProperties: true },
          },
        },
      },
    ],
  };
};

export const buildCompletionMode = (
  options: LanguageModelV3CallOptions,
): CompletionMode => {
  const tools = options.tools?.filter(isFunctionTool) ?? [];
  const hasToolMode = tools.length > 0 && options.toolChoice?.type !== "none";

  if (hasToolMode) {
    return {
      type: "tools",
      schema: buildToolSchema(tools, options.toolChoice),
      tools,
    };
  }

  if (options.responseFormat?.type === "json") {
    const schema = isRecord(options.responseFormat.schema)
      ? options.responseFormat.schema
      : {};

    return {
      type: "json",
      schema,
    };
  }

  return {
    type: "plain-text",
  };
};
