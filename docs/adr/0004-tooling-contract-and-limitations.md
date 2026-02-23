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
- Keep legacy JSON envelope parser as compatibility fallback.
- Treat `error_max_turns` with recovered native tool-use as successful tool-call output.
- Keep explicit empty-tool-routing guard when no tool call and no text are recoverable.

## Consequences
- Positive:
  - tool routing matches native Claude tool-use events.
  - fewer stalls from empty structured-output responses.
  - v2 and v3 share same stable behavior via common provider path.
- Negative:
  - adapter now depends on MCP bridge behavior from claude-agent-sdk runtime.
  - tool mode has more bridge logic to map MCP names back to AI SDK names.

## Validation
- Contract tests for v3 generate/stream:
  - native MCP tool-use recovery from `error_max_turns`.
  - legacy JSON fallback recovery.
  - explicit empty-output error path.
- Contract tests for v2 adapter:
  - legacy finish fields remain correct for recovered tool-call paths.
