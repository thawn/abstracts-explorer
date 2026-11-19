# Plan: Migrate the Web UI from JavaScript to TypeScript

Status: implemented

> **Implementation note:** all modules are now TypeScript and strictly typed
> except `clustering.ts`, which carries an explicit `// @ts-nocheck` while the
> Plotly-heavy code is typed incrementally. Tests remain `.js` and run against
> the `.ts` sources via `@swc/jest`; vendor libraries are accessed through lazy
> proxies so the existing global-based mocks keep working.

## Goal

Convert the browser frontend in
`src/abstracts_explorer/web_ui/static/` from plain JavaScript ES modules to
TypeScript, bundled with esbuild, while preserving current behavior, tests,
container deployment, and packaging.

## Decisions

- **Bundler:** esbuild. Produces a single ESM bundle at
  `static/dist/app.js` (with code splitting for large libraries).
- **Vendor libraries:** import `plotly.js-dist-min`, `marked`, `dompurify`,
  `katex`, and `marked-katex-extension` from npm instead of loading them as
  global `<script>` tags.
- **Compiled output:** `static/dist/` is generated and gitignored. It is built
  locally, in CI, in the Docker build, and by a Hatch build hook for wheels.
- **Container deployment:** build assets in a dedicated Node Docker build
  stage. The final runtime image stays Node-free.
- **Compatibility:** no backwards-compatibility constraints (project is alpha),
  but behavior and test coverage (>= 80% JS) must be preserved.

## Current state (baseline)

- 18 source JS files (~7.6k lines): `static/app.js`, 10 `modules/*.js`,
  7 `modules/utils/*.js`.
- 15 Jest test files (~4.5k lines), jsdom + native ESM with `transform: {}`
  (no transpilation), importing `.js` files directly.
- No bundler/build step. Flask serves `static/` directly; `index.html` loads
  `app.js` as a native ES module and loads 5 vendor libraries as globals.
- 41 functions assigned to `window` in `app.js`; 30 inline `onclick=` handlers
  in `index.html`.
- 84 references to global vendor objects (`Plotly`, `marked`, `DOMPurify`,
  `katex`, `markedKatex`) across 10 test files.
- CI: `.github/workflows/js-tests.yml` path-filters on `**/*.js`.
- Docker: multi-stage `Dockerfile`; builder runs `uv sync`, final image copies
  `.venv` and `src`. Node is not used in the image.
- Packaging: Hatchling packages the whole `src/abstracts_explorer/` tree.

## Target layout

```text
src/abstracts_explorer/web_ui/
  static/
    app.ts                      # entry (was app.js)
    modules/*.ts                # 10 feature modules
    modules/utils/*.ts          # 7 utility modules
    dist/app.js                 # esbuild output (gitignored)
    vendor/                     # tailwind.css, katex css + fonts, fontawesome css
  types/
    globals.d.ts                # Window typings for inline handlers
```

- Imports become extensionless (`'./utils/x'`) with
  `moduleResolution: "bundler"`.
- `index.html`:
  - Remove the 5 vendor `<script>` tags and the `app.js` module script.
  - Keep the small inline `window.urlConference` script.
  - Add `<script type="module" src="{{ url_for('static', filename='dist/app.js') }}"></script>`.
- Keep `attachToWindow()` because server-rendered inline `onclick=` handlers
  still require global functions.

## Tooling

### `tsconfig.json`

- `strict: true`
- `target: "ES2022"`, `module: "ESNext"`, `moduleResolution: "bundler"`
- `noEmit: true` (esbuild emits)
- `lib: ["DOM", "ES2022"]`
- Include `static/**/*.ts`, `types/**/*.d.ts`
- Vendor typings: rely on shipped types where available (marked, dompurify,
  katex); add `@types/plotly.js` and a `declare module 'plotly.js-dist-min'`
  shim if needed.

### `scripts/build.mjs`

esbuild configuration:

- entry point: `static/app.ts`
- `bundle: true`, `format: "esm"`, `splitting: true`, `outdir: "static/dist"`
- lazy/separate chunk for Plotly to keep initial payload small
- `sourcemap: true` in dev only
- `minify: true` for production builds
- an esbuild `--watch` path exposed via npm script

### `package.json` scripts

```json
{
  "build:js": "node scripts/build.mjs",
  "watch:js": "node scripts/build.mjs --watch",
  "typecheck": "tsc --noEmit",
  "build": "npm run build:js && npm run build:tailwind"
}
```

Dependencies to add (dev): `esbuild`, `typescript`, `@types/plotly.js`
(and `@swc/jest` or `esbuild-jest` for TS tests).

Remove `install:vendor:*` steps that only copied JS bundles imported from npm
(keep Tailwind and any retained CSS/font assets). Update `postinstall`.

### Other

- `.gitignore`: add `static/dist/`.
- `types/globals.d.ts`: declare `Window` extensions (`urlConference`,
  attached handler functions, any remaining globals).

## Migration order (keep every phase green)

1. Leaf utilities: `constants`, `dom-utils`, `sort-utils`, `cluster-utils`,
   `ui-utils`, `api-utils`, `markdown-utils`.
2. `state.ts`.
3. `paper-card.ts`.
4. `search.ts`, `tabs.ts`, `filters.ts`.
5. `interesting-papers.ts`, `chat.ts`, `clustering.ts`.
6. `app.ts` entry, `index.html` update, window typings.
7. Convert JSDoc annotations to real TS types/interfaces for domain objects
   (papers, clusters, API payloads).

Each phase: run `npm run typecheck`, `npm test`, and `npm run build` before
moving on.

## Tests

This is the largest effort.

- Add a Jest transform for TS (`@swc/jest` or `esbuild-jest`); keep jsdom.
- Update `testMatch` and `collectCoverageFrom` to include `.ts`.
- Replace global vendor mocks with module mocks:
  - Prefer `moduleNameMapper` pointing at shared manual mocks under
    `tests/__mocks__/` that export `jest.fn()` implementations.
  - Use `jest.unstable_mockModule` + dynamic `await import()` where
    module-level mocking is required.
- Simplify `tests/setup.js` (drop `global.DOMPurify` / `global.marked` stubs).
- Convert the 15 test files to `.ts`.
- Preserve >= 80% coverage.

## Deployment and CI

### `Dockerfile`

- Add `FROM node:20-slim AS frontend-builder`:
  - install deps (`npm ci`), run `npm run build:js`.
- In the `python-builder` stage, copy `static/dist` from `frontend-builder`
  before `uv sync`.
- Final runtime stage unchanged (no Node).

### `.dockerignore`

- Ignore `node_modules/` and `static/dist/` in the build context; the dist
  output is copied from the build stage, not the context.

### Wheel / sdist

- Add `hatch_build.py` with a custom build hook
  (`[tool.hatch.build.hooks.custom]`) that runs `npm run build:js` when Node
  is available and fails loudly if the bundle is missing. This ensures
  `uv build` ships `static/dist`.

### Workflows

- `.github/workflows/js-tests.yml`:
  - Add path filters: `src/**/*.ts`, `tsconfig.json`, `scripts/**`.
  - Run `npm run typecheck`, `npm test`, and `npm run build`.
- `.github/workflows/docker.yml`:
  - Add `src/**/*.ts`, `tsconfig.json`, `scripts/**` to trigger paths.

## Documentation

Update references to the frontend architecture and commands:

- `AGENTS.md` (layout, code style, testing, commands)
- `docs/web_ui.md`
- `docs/docker.md` (Node build stage; runtime Node-free)
- `.github/instructions/project.instructions.md`
- `README.md` / `docs/contributing.md` if they mention frontend commands

## Risks and mitigations

- **Plotly bundle size:** use esbuild code splitting / lazy-loading for the
  clustering chunk.
- **marked / katex / marked-katex-extension interop:** validate ESM/CJS interop
  under esbuild early (Phase 1).
- **Jest ESM mocking order:** budget time for `unstable_mockModule` plumbing.
- **Missing bundle in Node-less environments:** guarded by the Hatch build hook
  and the Docker build stage.
- **Source maps in production:** do not ship them in the image/wheel.

## Verification checklist

- `npm run typecheck`
- `npm run build`
- `npm test` (>= 80% coverage)
- Docker image build + smoke test (`/health`, UI loads bundle without console
  errors)
- Existing `uv run black src/ tests/`, `uv run ruff check src/ tests/ --fix`,
  `uv run mypy src/ --ignore-missing-imports`, and `uv run pytest` still pass

## Suggested commit message

```text
commit message:

refactor: migrate web UI from JavaScript to TypeScript

Convert static/ ES modules to .ts and bundle with esbuild, importing
vendor libraries from npm instead of global script tags. Add tsconfig,
build/watch scripts, Window typings for inline handlers, and TS-aware
Jest config with shared vendor mocks. Build assets in a Node Docker
build stage and a Hatch build hook while keeping the runtime image
Node-free; update CI path filters and docs.
```
