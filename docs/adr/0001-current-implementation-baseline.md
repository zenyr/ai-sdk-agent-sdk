# ADR-0001 Current Implementation Baseline

- Status: Accepted
- Date: 2026-02-22

## Context
- Project goal: provide `@ai-sdk/anthropic` API surface.
- Backend constraint: use `@anthropic-ai/claude-agent-sdk` instead of direct Anthropic HTTP calls.
- Current implementation lives in `packages/main/index.ts`.

## Decision
- Keep this ADR as the factual baseline before parity hardening.
- Track gaps by three axes:
  - export surface
  - type compatibility
  - runtime behavior

## Baseline Facts
- Root export forwards to `packages/main/index.ts`.
- Runtime exports: `VERSION`, `anthropic`, `createAnthropic`, `forwardAnthropicContainerIdFromLastStep`.
- Type parity guard exists in `packages/main/type-parity.ts`.
- Runtime uses `query()` from Agent SDK.
- `doStream` currently wraps `doGenerate` result into synthetic stream parts.
- Function tools are handled through JSON envelope prompting.
- Provider tools are not executed through real provider tool protocol.
- Many anthropic provider options are currently unsupported or degraded.

## Consequences
- Positive:
  - surface compatibility exists and compiles.
  - implementation can run with Agent SDK backend.
- Negative:
  - runtime parity is incomplete.
  - streaming semantics differ from upstream behavior.

## Validation
- `bunx tsc --noEmit` passes.
- Export key check against upstream package is required in tests.
