# Anthropic Agent SDK Parity Master Plan

## Scope
- Package: `packages/main`
- Objective: maximize `@ai-sdk/anthropic` compatibility on Agent SDK backend.

## Tracks
- Track 0: docs IA + baseline ADR gate.
- Track 1: contract lock (export/type/runtime definitions).
- Track 2: runtime generation + streaming bridge.
- Track 3: tooling behavior alignment.
- Track 4: provider option and metadata policy.
- Track 5: verification and release gate.

## Execution Order
1. Track 0
2. Track 1
3. Track 2 and Track 3 in parallel
4. Track 4
5. Track 5

## Critical Paths
- `packages/main/index.ts`
- `packages/main/type-parity.ts`
- `docs/plans/parity-matrix.md`
- `docs/plans/runtime-gap-register.md`

## Exit Criteria
- Type parity checks pass.
- Runtime contract tests pass.
- High-risk gap items are resolved or explicitly documented as unsupported.
