# ADR-0005 Option Mapping Policy

- Status: Proposed
- Date: 2026-02-22

## Context
- `AnthropicLanguageModelOptions` has many fields.
- Agent SDK backend does not expose one-to-one control for all fields.

## Decision
- Classify each option as:
  - `mapped`
  - `degraded`
  - `ignored`
  - `rejected`
- Keep classification in `docs/plans/parity-matrix.md`.
- Emit warnings for `degraded` and `ignored` runtime paths.
- Reject only when silent mismatch would be dangerous.

## Consequences
- Positive:
  - stable and explainable behavior.
  - easier compatibility docs and tests.
- Negative:
  - warning surface grows until parity improves.

## Validation
- Every known option has one classification.
- Warning tests verify expected warning behavior.
