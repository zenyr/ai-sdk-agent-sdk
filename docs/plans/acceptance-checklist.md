# Acceptance Checklist

## Contract
- [ ] Export surface contract verified.
- [ ] Type parity guards pass.
- [ ] Runtime parity matrix updated.

## Runtime
- [ ] `doGenerate` finish reason mapping verified.
- [ ] `doStream` lifecycle behavior verified.
- [ ] Tooling behavior verified for supported and unsupported paths.
- [ ] Option warnings and degraded behavior verified.

## Quality
- [ ] `bunx tsc --noEmit` passes.
- [ ] `bun test` passes.
- [ ] Smoke checks for root exports and provider creation pass.

## Documentation
- [ ] Relevant ADR statuses updated.
- [ ] Gap register updated.
- [ ] Parity matrix updated.
