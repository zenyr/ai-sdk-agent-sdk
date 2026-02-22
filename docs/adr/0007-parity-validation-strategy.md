# ADR-0007 Parity Validation Strategy

- Status: Proposed
- Date: 2026-02-22

## Context
- Type parity alone is not enough.
- Runtime parity must be proven through contract tests.

## Decision
- Maintain dual validation:
  - Type validation (`tsc`, parity type guards)
  - Runtime contract validation (`bun test`)
- Track validation status in parity matrix and acceptance checklist.

## Consequences
- Positive:
  - lower risk of surface-only compatibility.
  - faster detection of upstream drift.
- Negative:
  - test maintenance cost increases.

## Validation
- `bunx tsc --noEmit` must pass.
- Runtime contract test suite must pass.
