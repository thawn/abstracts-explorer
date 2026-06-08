# Design — Client-side XSS Sanitization for Markdown Rendering

**Date:** 2026-06-08
**Branch:** `fix/xss-markdown-sanitization`
**Status:** Approved (design), pending implementation
**Addresses:** Code-review Critical findings #1 (stored/reflected XSS via unsanitized `marked.parse()`) and #2 (prompt-injection → XSS chain). See `code-review/05-web-flask-frontend.md` and `code-review/02-embeddings-rag-evaluation.md`.

---

## 1. Problem

The web UI renders untrusted text as HTML with **no sanitization**:

- `marked.parse()` / `marked.parseInline()` output is assigned to `innerHTML` at three sinks:
  - `static/modules/utils/markdown-utils.js:38` (`renderMarkdownWithLatex`) — renders **paper abstracts** (`paper-card.js:78,81,87,238`).
  - `static/modules/utils/markdown-utils.js:18` (`renderInlineMarkdownWithLatex`) — renders **paper titles** (`paper-card.js:143,199`, `clustering.js:1263`).
  - `static/modules/chat.js:310` — renders **LLM chat output** by calling `marked.parse(text)` directly (bypasses the helpers).
- DOMPurify is **confirmed absent** from the entire non-vendor source tree; `marked` has no sanitize configuration.
- The reverse-proxy CSP uses `script-src 'self' 'unsafe-inline' 'unsafe-eval'`, so attribute-based payloads such as `<img src=x onerror=...>` **execute** and are not blocked. `app.py` sets no security headers of its own — protection lives only in the proxy.

The data rendered is attacker-influenceable: paper titles/abstracts are scraped from external conference sites, and LLM chat output is steerable via prompt injection embedded in those same abstracts. The RAG system prompt (`rag.py:454-455`) additionally **instructs the model to emit raw `<a href='#paper-1'>` HTML**, normalizing HTML in the model's output. Donated chat transcripts (`/api/donate-chat`) persist attacker content, turning reflected XSS into **stored** XSS.

## 2. Goal

Neutralize HTML/script injection at every markdown render sink while preserving legitimate markdown, KaTeX math rendering, and `#paper-N` citation anchors. Remove the instruction that tells the LLM to emit raw HTML.

## 3. Scope

**In scope (this PR):**
- Sanitize all three markdown render sinks.
- Vendor DOMPurify following the existing `install:vendor` pipeline.
- Harden the RAG citation prompt to use markdown links instead of raw `<a>`.
- Unit tests pinning sanitization behavior and KaTeX preservation; a Python test for the prompt.

**Out of scope (separate PRs, noted in the code review):**
- Removing `'unsafe-inline'` from the CSP / eliminating the ~40 inline `window` event handlers.
- Path-traversal in `/api/clusters/cached` (Critical #3).
- Shared global `RAGChat` singleton cross-user bleed (Critical #4).

## 4. Approach (selected: A — centralized DOMPurify)

Sanitize at the existing render-helper choke point so a single fix covers titles, abstracts, and (after routing the one bypass through it) chat output. Keep the prompt hardening as defense-in-depth behind the sanitizer.

Rejected alternatives:
- **B — marked built-in sanitize / strip HTML.** Modern `marked` removed the `sanitize` option (docs explicitly recommend DOMPurify); the KaTeX extension intentionally injects HTML, so a blanket strip would break math.
- **C — prompt change + server-side scrub only.** Does not fix the actual sink (rendered abstracts/titles stay vulnerable); a prompt is not a security boundary. Its prompt half is folded into A.

## 5. Detailed changes (file by file)

### 5.1 Vendor DOMPurify
- **`package.json`**
  - Add `dompurify` (`^3`) to `devDependencies`.
  - Add script `install:vendor:dompurify`: `mkdir -p src/abstracts_explorer/web_ui/static/vendor && cp node_modules/dompurify/dist/purify.min.js src/abstracts_explorer/web_ui/static/vendor/purify.min.js`.
  - Append `&& npm run install:vendor:dompurify` to the `install:vendor` chain.
- **`src/abstracts_explorer/web_ui/static/vendor/purify.min.js`** — committed vendored asset (the pre-commit hook regenerates vendor on web-file changes, so `npm install` + `npm run install:vendor` must run before committing).
- **`src/abstracts_explorer/web_ui/templates/index.html`** — add `<script src="{{ url_for('static', filename='vendor/purify.min.js') }}"></script>` next to the other vendor `<script>` tags (must load before `app.js`). No CSP change required — it is a same-origin `'self'` script.

### 5.2 Sanitize at the choke point — `static/modules/utils/markdown-utils.js`
- Introduce a single internal `sanitize(html)` wrapper calling `DOMPurify.sanitize(html, SANITIZE_CONFIG)`.
- `renderInlineMarkdownWithLatex`: `return sanitize(marked.parseInline(text))`.
- `renderMarkdownWithLatex`: `return sanitize(marked.parse(text))`.
- Preserve the existing `try/catch` fallback (escapes text on parse failure).
- **`SANITIZE_CONFIG`:** DOMPurify default HTML profile — strips `<script>`, `on*` handler attributes, and `javascript:`/`data:`-script URLs, while allowing standard markdown tags, `<a href="#paper-N">` anchors, and KaTeX `output:'html'` spans (class/style/aria-hidden are permitted by default). A unit test pins KaTeX survival; if a specific tag/attr is stripped, extend via `ADD_TAGS`/`ADD_ATTR` (kept minimal).

### 5.3 Fix the bypass — `static/modules/chat.js`
- Import `renderMarkdownWithLatex` from `./utils/markdown-utils.js`.
- Replace `marked.parse(text)` at line 310 with `renderMarkdownWithLatex(text)`, so assistant output is sanitized identically to abstracts.

### 5.4 Harden the source — `src/abstracts_explorer/rag.py`
- In `_build_base_instructions` (lines 454-455), replace the raw-HTML citation instruction:
  - From: `<a href='#paper-1'>Paper-1</a>, <a href='#paper-2'>Paper-2</a>, etc.`
  - To: markdown links `[Paper-1](#paper-1)`, `[Paper-2](#paper-2)`, etc.
- `marked` renders these to identical `<a href="#paper-1">` anchors, so citation behavior is unchanged.

## 6. Data flow (after fix)

```
untrusted text (scraped abstract / LLM output)
  → marked.parse / parseInline (+ KaTeX extension)
  → DOMPurify.sanitize(SANITIZE_CONFIG)      ← choke point, removes script/on*/js: URLs
  → innerHTML
```

Malicious `<img onerror=…>`, `<script>…</script>`, and `javascript:` URLs are neutralized at the choke point regardless of whether the source is a scraped paper or steered LLM output.

## 7. Testing

**JavaScript (Jest + jsdom).** New tests for `markdown-utils` (and confirm `tests/setup.js` exposes a real `DOMPurify` global under jsdom, the same way `marked`/`markedKatex` are exposed):
- `<img src=x onerror=alert(1)>` → rendered without the `onerror` attribute.
- `<script>alert(1)</script>` → stripped.
- `[x](javascript:alert(1))` → `href` neutralized (no `javascript:`).
- `**bold**` and `[Paper-1](#paper-1)` → preserved (bold renders; anchor `href="#paper-1"` retained).
- `$E=mc^2$` → renders `.katex` markup and it survives sanitization.
- Run existing `chat.test.js` / `paper-card.test.js` to confirm no regression (chat now routes through the helper).

**Python (pytest).** In `test_rag.py`: assert base instructions contain the markdown citation form (`[Paper-1](#paper-1)`) and contain no raw `<a ` tag.

## 8. Verification criteria (definition of done)

- All three sinks pass through DOMPurify; no remaining direct `marked.parse`/`parseInline` → `innerHTML` outside `markdown-utils.js`.
- New JS tests pass (XSS payloads neutralized, markdown + KaTeX + citation anchors preserved).
- `test_rag.py` prompt assertion passes.
- `npm test` and the targeted `pytest` selection are green; `ruff`/`mypy` clean on the touched Python.
- Vendored `purify.min.js` present and referenced; pre-commit vendor-freshness hook passes.

## 9. Risks & rollback

- **Primary risk:** DOMPurify over-stripping KaTeX output. Mitigated by a pinned KaTeX round-trip test before merge; if it fails, extend `SANITIZE_CONFIG` with the minimal required `ADD_TAGS`/`ADD_ATTR`.
- **Surface:** 4 source files + 1 vendored asset + tests. Isolated and cleanly revertible.
- **No behavior change** for legitimate content (citations, math, formatting render the same).

## 10. Out-of-scope follow-ups (tracked from the review)

1. Tighten CSP by removing `'unsafe-inline'` and the inline `onclick`/`onchange` handlers.
2. `/api/clusters/cached` path traversal (remove or hard-validate the `file` param).
3. Replace the global `RAGChat`/embeddings singletons with per-request resources.
