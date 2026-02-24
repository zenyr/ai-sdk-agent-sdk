import type { Options as AgentQueryOptions } from "@anthropic-ai/claude-agent-sdk";
import * as agentSdk from "@anthropic-ai/claude-agent-sdk";

import { buildZodRawShapeFromToolInputSchema } from "../../bridge/tool-schema-to-zod-shape";
import type { ToolExecutorMap } from "../../shared/tool-executor";
import { isRecord, safeJsonStringify } from "../../shared/type-readers";

const TOOL_BRIDGE_SERVER_NAME = "ai_sdk_tool_bridge";
const TOOL_BRIDGE_NAME_PREFIX = `mcp__${TOOL_BRIDGE_SERVER_NAME}__`;

export type ToolBridgeConfig = {
  allowedTools: string[];
  mcpServers: NonNullable<AgentQueryOptions["mcpServers"]>;
  hasAnyExecutor: boolean;
  allToolsHaveExecutors: boolean;
  missingExecutorToolNames: string[];
};

export const toBridgeToolName = (toolName: string): string => {
  return `${TOOL_BRIDGE_NAME_PREFIX}${toolName}`;
};

export const fromBridgeToolName = (toolName: string): string => {
  if (!toolName.startsWith(TOOL_BRIDGE_NAME_PREFIX)) {
    return toolName;
  }

  const mappedToolName = toolName.slice(TOOL_BRIDGE_NAME_PREFIX.length);
  return mappedToolName.length > 0 ? mappedToolName : toolName;
};

export const isBridgeToolName = (toolName: string): boolean => {
  return toolName.startsWith(TOOL_BRIDGE_NAME_PREFIX);
};

export const normalizeToolInputJson = (value: string): string => {
  const trimmedValue = value.trim();
  if (trimmedValue.length === 0) {
    return "{}";
  }

  try {
    const parsedValue: unknown = JSON.parse(trimmedValue);
    return safeJsonStringify(parsedValue);
  } catch {
    return trimmedValue;
  }
};

export const buildToolBridgeConfig = (
  tools: Array<{ name: string; description?: string; inputSchema: unknown }>,
  toolExecutors: ToolExecutorMap | undefined,
): ToolBridgeConfig | undefined => {
  if (tools.length === 0) {
    return undefined;
  }

  const createBridgeServer = agentSdk.createSdkMcpServer;
  if (typeof createBridgeServer !== "function") {
    return undefined;
  }

  const buildMcpTool = agentSdk.tool;

  const stringifyToolExecutorOutput = (value: unknown): string => {
    if (typeof value === "string") {
      return value;
    }

    return safeJsonStringify(value);
  };

  const stringifyToolExecutorError = (error: unknown): string => {
    if (error instanceof Error && error.message.length > 0) {
      return error.message;
    }

    if (typeof error === "string" && error.length > 0) {
      return error;
    }

    return safeJsonStringify(error);
  };

  const buildDisabledHandler = async () => {
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: "Provider-side execution is disabled for AI SDK bridge tools.",
        },
      ],
    };
  };

  const missingExecutorToolNames: string[] = [];
  let hasAnyExecutor = false;

  const mcpTools = tools.map((toolDefinition) => {
    const zodRawShape = buildZodRawShapeFromToolInputSchema(toolDefinition.inputSchema);
    const toolExecutor = toolExecutors?.[toolDefinition.name];

    if (toolExecutor !== undefined) {
      hasAnyExecutor = true;
    } else {
      missingExecutorToolNames.push(toolDefinition.name);
    }

    const toolHandler =
      toolExecutor === undefined
        ? buildDisabledHandler
        : async (args: unknown) => {
            const input = isRecord(args) ? args : {};

            try {
              const output = await toolExecutor(input);
              return {
                content: [
                  {
                    type: "text",
                    text: stringifyToolExecutorOutput(output),
                  },
                ],
              };
            } catch (error) {
              return {
                isError: true,
                content: [
                  {
                    type: "text",
                    text: stringifyToolExecutorError(error),
                  },
                ],
              };
            }
          };

    if (typeof buildMcpTool === "function") {
      return buildMcpTool(
        toolDefinition.name,
        toolDefinition.description ?? "No description",
        zodRawShape,
        toolHandler,
      );
    }

    return {
      name: toolDefinition.name,
      description: toolDefinition.description ?? "No description",
      inputSchema: zodRawShape,
      handler: toolHandler,
    };
  });

  const mcpServer = createBridgeServer({
    name: TOOL_BRIDGE_SERVER_NAME,
    tools: mcpTools,
  });

  return {
    allowedTools: tools.map((toolDefinition) => {
      return toBridgeToolName(toolDefinition.name);
    }),
    mcpServers: {
      [TOOL_BRIDGE_SERVER_NAME]: mcpServer,
    },
    hasAnyExecutor,
    allToolsHaveExecutors: hasAnyExecutor && missingExecutorToolNames.length === 0,
    missingExecutorToolNames,
  };
};
