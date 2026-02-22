# ADR-0004 Tooling Contract And Limitations

- Status: Proposed
- Date: 2026-02-22

## Context
- Upstream supports function tools and provider tools with rich behavior.
- Current local implementation uses JSON envelope prompting for function tools.
- Provider tools are exposed on surface but not fully executed through runtime bridge.

## Decision
- Keep function tool support as first-class path.
- For provider tools, avoid silent ignore:
  - return explicit warning for degraded paths.
  - return explicit error when behavior cannot be guaranteed.
- Document exact tool mode behavior in parity matrix.

## Consequences
- Positive:
  - less hidden mismatch.
  - predictable failure behavior.
- Negative:
  - stricter behavior may break existing implicit assumptions.

## Validation
- Tests cover function tool call path.
- Tests cover provider tool degraded or unsupported path.
