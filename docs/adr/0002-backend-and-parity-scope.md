# ADR-0002 Backend And Parity Scope

- Status: Accepted
- Date: 2026-02-22

## Context
- Product intent is API compatibility with `@ai-sdk/anthropic`.
- Backend must remain `@anthropic-ai/claude-agent-sdk`.
- Full behavioral equivalence may be impossible for some features.

## Decision
- Use compatibility scope levels:
  - `full`: expected to match upstream behavior.
  - `degraded`: same API contract but reduced behavior.
  - `unsupported`: explicit warning or error.
- Never keep high-impact behavior silently ignored.

## Consequences
- Positive:
  - transparent contract for users.
  - easier triage and roadmap planning.
- Negative:
  - some upstream features will need explicit non-support handling.

## Validation
- Every public capability appears in `docs/plans/parity-matrix.md`.
- Every `degraded` or `unsupported` item references this ADR or a follow-up ADR.
