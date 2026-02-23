# ADR-0004 Tooling Contract And Limitations

- Status: Accepted
- Date: 2026-02-23

## Context
- Upstream supports function tools and provider tools.
- JSON-envelope prompt routing for function tools caused unstable behavior:
  - model sometimes returned empty success output.
  - model sometimes returned legacy single-call JSON shape.
  - tool routing could stall in callers waiting for structured tool calls.

## Decision
- Use native in-process MCP bridge for AI SDK function tools.
  - Build one SDK MCP server per query with `createSdkMcpServer`.
  - Register each AI SDK function tool as MCP tool.
  - Enable only bridge MCP tool names in `allowedTools`.
  - Keep built-in Claude tools disabled (`tools: []`, isolated settings).
- Add optional provider-side tool executor mode.
  - New provider setting: `toolExecutors` map by tool name.
  - New provider setting: `maxTurns` for Agent SDK multi-turn loop when all tool executors exist.
  - In this mode, MCP bridge handler executes tool directly and returns tool result to Agent SDK.
  - Adapter does not depend on `error_max_turns` recovery for normal tool flow.
- Keep legacy JSON envelope parser as compatibility fallback for non-executor mode.
- Keep explicit empty-tool-routing guard when no tool call and no text are recoverable in non-executor mode.
- Keep `maxTurns: 1` behavior when tool executors are missing or partial.

## Consequences
- Positive:
  - tool routing matches native Claude tool-use events.
  - fewer stalls from empty structured-output responses.
  - v2 and v3 share same stable behavior via common provider path.
  - executor mode supports native Agent SDK multi-turn tool execution and streaming.
- Negative:
  - adapter now depends on MCP bridge behavior from claude-agent-sdk runtime.
  - tool mode has more bridge logic to map MCP names back to AI SDK names.
  - users must provide `toolExecutors` separately because `LanguageModelV3FunctionTool` has schema only.
  - partial executor registration falls back to legacy single-turn behavior.

## Validation
- Contract tests for v3 generate/stream:
  - executor mode with multi-turn config and provider-executed tool-call streaming.
  - partial executor fallback warning and `maxTurns: 1`.
  - native MCP tool-use recovery from `error_max_turns` in legacy mode.
  - legacy JSON fallback recovery.
  - explicit empty-output error path.
- Contract tests for v2 adapter:
  - legacy finish fields remain correct for recovered tool-call paths.
