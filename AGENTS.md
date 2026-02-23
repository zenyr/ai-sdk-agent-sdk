## Core policy

- No data loss. Keep full intent and technical details.
- Concision over English grammar. Short, clear, direct.
- Internal writing can be fragments if meaning is exact.

## Language policy

- Code comments, docs, commit/PR text: plain layman's English.
- User-facing conversation: always same language as user.

## TypeScript coding style

- Prefer `const` function expressions over function statements.
- No `as` casting.
- No non-null assertion (`!`).
- Use type guard utility functions aggressively.
- Handle nullable/unknown values with explicit checks.

## Runtime and package manager

- Default to Bun, not Node.js.
- Use `bun <file>` instead of `node <file>` or `ts-node <file>`.
- Use `bun test` instead of `jest` or `vitest`.
- Use `bun build <file.html|file.ts|file.css>` instead of `webpack` or `esbuild`.
- Use `bun install` instead of `npm/yarn/pnpm install`.
- Use `bun run <script>` instead of `npm/yarn/pnpm run <script>`.
- Use `bunx <package> <command>` instead of `npx <package> <command>`.
- Bun auto-loads `.env`; do not add dotenv.

## APIs and libraries

- Use `Bun.serve()` for HTTP/WebSocket/routes. Do not use `express`.
- Use `bun:sqlite` for SQLite. Do not use `better-sqlite3`.
- Use `Bun.redis` for Redis. Do not use `ioredis`.
- Use `Bun.sql` for Postgres. Do not use `pg` or `postgres.js`.
- Use built-in `WebSocket`. Do not use `ws`.
- Prefer `Bun.file` over `node:fs` read/write helpers.
- Prefer `Bun.$` shell API over execa.

## Testing

- Run tests with `bun test`.

```ts
import { test, expect } from "bun:test";

test("hello world", () => {
  expect(1).toBe(1);
});
```

## Frontend

- Use HTML imports with `Bun.serve()`. Do not use `vite`.
- HTML can import `.tsx/.jsx/.js` directly; Bun bundles automatically.
- CSS via `<link>` is bundled by Bun CSS bundler.

```ts
import index from "./index.html";

Bun.serve({
  routes: {
    "/": index,
    "/api/users/:id": {
      GET: (req) => new Response(JSON.stringify({ id: req.params.id })),
    },
  },
  websocket: {
    open: (ws) => ws.send("Hello, world!"),
    message: (ws, message) => ws.send(message),
    close: () => {},
  },
  development: {
    hmr: true,
    console: true,
  },
});
```

```html
<html>
  <body>
    <h1>Hello, world!</h1>
    <script type="module" src="./frontend.tsx"></script>
  </body>
</html>
```

```tsx
import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

const Frontend = () => <h1>Hello, world!</h1>;

const root = createRoot(document.body);
root.render(<Frontend />);
```

- Run dev server with `bun --hot ./index.ts`.
- For more Bun API details, read `node_modules/bun-types/docs/**.mdx`.
