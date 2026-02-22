# ADR-0006 Error Timeout Retry Policy

- Status: Proposed
- Date: 2026-02-22

## Context
- Errors can originate from prompt conversion, Agent SDK stream, or result mapping.
- Current behavior does not fully define retry and timeout policy by error class.

## Decision
- Keep deterministic mapping for finish reasons and errors.
- Treat missing final result as explicit error (`agent-sdk-no-result`).
- Do not auto-retry by default inside provider bridge.
- Preserve cancellation through abort signal forwarding.

## Consequences
- Positive:
  - predictable behavior for callers.
  - easier debugging.
- Negative:
  - transient failures are surfaced to caller unless caller retries.

## Validation
- Tests for error subtype mapping.
- Tests for abort signal cancellation behavior.
