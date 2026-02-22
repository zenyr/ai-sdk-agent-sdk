# Runtime Gap Register

## G-001 Stream Semantics Gap
- Priority: high
- Area: `doStream`
- Current: stream is generated after `doGenerate` completion.
- Target: event-based streaming bridge from Agent SDK stream events.
- Validation: stream lifecycle contract tests.
- ADR: `0003-streaming-semantics-mapping.md`

## G-002 Provider Tool Protocol Gap
- Priority: high
- Area: tool execution
- Current: provider tools are exposed on surface but not fully executed.
- Target: explicit unsupported or degraded contract per tool family.
- Validation: provider tool behavior tests.
- ADR: `0004-tooling-contract-and-limitations.md`

## G-003 Option Mapping Gap
- Priority: high
- Area: anthropic provider options
- Current: only subset is mapped.
- Progress: warning classifier added for degraded/unsupported/unknown options.
- Target: classify all options (`mapped/degraded/ignored/rejected`).
- Validation: warning and behavior tests.
- ADR: `0005-option-mapping-policy.md`

## G-004 Metadata Coverage Gap
- Priority: medium
- Area: provider metadata
- Current: fields like container/context management are mostly null.
- Target: map what is feasible, document what is unavailable.
- Validation: metadata contract tests.
- ADR: `0005-option-mapping-policy.md`

## G-005 Error Behavior Drift Risk
- Priority: medium
- Area: finish reason and failures
- Current: mapped behavior exists but policy is not fully codified.
- Target: deterministic error mapping policy and tests.
- Validation: error subtype tests + abort tests.
- ADR: `0006-error-timeout-retry-policy.md`
