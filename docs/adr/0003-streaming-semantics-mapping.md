# ADR-0003 Streaming Semantics Mapping

- Status: Proposed
- Date: 2026-02-22

## Context
- Upstream provider supports native streaming semantics.
- Current local implementation builds stream parts from completed `doGenerate` output.
- This creates timing and behavior gaps.

## Decision
- Move to event-based bridge for `doStream` using Agent SDK stream events.
- Keep fallback behavior for unknown event shapes.
- Ensure finish reason and usage mapping are consistent between `doGenerate` and `doStream`.

## Consequences
- Positive:
  - closer user-visible streaming behavior.
  - easier runtime parity verification.
- Negative:
  - more event mapping logic and tests required.

## Validation
- Contract tests confirm stream lifecycle parts order.
- Contract tests confirm finish reason and usage consistency.
