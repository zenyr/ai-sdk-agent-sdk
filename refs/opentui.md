# OpenTUI Notes

- React: `@opentui/react`. Core: `@opentui/core`.
- Entrypoint: `createCliRenderer()` -> `createRoot(renderer).render(<App />)`.
- App-local `tsconfig.json` needs `jsxImportSource: "@opentui/react"`.
- `input`/`select` need `focused={true}` for input.
- Simple wizard > rich widget guesswork. Custom `<text>` rows + explicit state more predictable.
- `useKeyboard()` covers arrows, Enter, Space, Escape, Ctrl+C.
- In this repo, workspace-relative imports resolve more reliably than package-name imports for new app pkg.
- `ScrollBox` not overlay by default. Vertical bar is sibling of wrapper/content, so it steals width unless hidden.
- If width > visible scrollbar, hide scrollbar, drive scroll position yourself.
- Body-level scroll can work better than per-screen scroll, but step change should remount/reset scroll state.
- Selection UI: keep row single-line, truncate hard. Rich meta rows destabilize terminal flex layout.
- Hover/click works, keyboard still primary/reliable path.
- Keep v1 simple: text boxes, single input, keyboard nav, summary preview.
