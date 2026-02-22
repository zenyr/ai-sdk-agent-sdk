# Parity Matrix

Status legend:
- `full`: close to upstream behavior.
- `degraded`: same API shape, reduced behavior.
- `unsupported`: explicit warning or error.

## Export Surface

| Item | Status | Notes | Source |
|---|---|---|---|
| `anthropic` | full | callable provider exists | `packages/main/index.ts` |
| `createAnthropic` | full | exposed | `packages/main/index.ts` |
| `forwardAnthropicContainerIdFromLastStep` | degraded | helper exists, depends on metadata availability | `packages/main/index.ts` |
| `VERSION` | full | exposed | `packages/main/index.ts` |

## Type Compatibility

| Item | Status | Notes | Source |
|---|---|---|---|
| Module key parity | full | compile-time guard present | `packages/main/type-parity.ts` |
| Provider type assignability | full | forward/backward checks present | `packages/main/type-parity.ts` |
| Option/metadata/tool type assignability | full | guarded | `packages/main/type-parity.ts` |

## Runtime Behavior

| Item | Status | Notes | Source |
|---|---|---|---|
| `doGenerate` text flow | degraded | prompt serialization bridge | `packages/main/index.ts` |
| `doStream` semantics | degraded | synthetic stream from generated output | `packages/main/index.ts` |
| Function tools | degraded | JSON envelope strategy | `packages/main/index.ts` |
| Provider tools | unsupported | warning path, no full runtime protocol | `packages/main/index.ts` |
| Provider option coverage | degraded | partial mapping (`effort`, `thinking`) + explicit warning classifier | `packages/main/index.ts` |
| Metadata coverage | degraded | partial, many fields null | `packages/main/index.ts` |

## Follow-up
- Detailed per-gap tracking lives in `docs/plans/runtime-gap-register.md`.
