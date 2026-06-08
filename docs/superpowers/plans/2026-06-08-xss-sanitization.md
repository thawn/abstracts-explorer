# Client-side XSS Sanitization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sanitize every markdown render sink in the web UI with DOMPurify and stop the RAG prompt from emitting raw HTML, closing the stored/reflected XSS chain (code-review Critical #1 + #2).

**Architecture:** Vendor DOMPurify as a global script (matching the existing `install:vendor` pipeline), sanitize at the single `markdown-utils.js` choke point, route the one bypass (`chat.js`) through that helper, and change the RAG citation instruction from raw `<a>` to markdown links. Untrusted text flows `marked` → **DOMPurify.sanitize** → `innerHTML`.

**Tech Stack:** Vanilla ES modules, `marked` + `marked-katex-extension` + KaTeX (vendored globals), DOMPurify (new vendored global), Jest + jsdom (JS tests), Python/pytest + `pydantic-ai` (RAG), `uv` toolchain.

**Branch:** `fix/xss-markdown-sanitization` (already created off `develop`). Spec: `docs/superpowers/specs/2026-06-08-xss-sanitization-design.md`.

---

## File Structure

- **Modify** `package.json` — add `dompurify` dependency + `install:vendor:dompurify` script, wire it into the `install:vendor` chain.
- **Create** `src/abstracts_explorer/web_ui/static/vendor/purify.min.js` — vendored DOMPurify (generated, committed).
- **Modify** `src/abstracts_explorer/web_ui/templates/index.html` — load `purify.min.js` before `app.js`.
- **Modify** `src/abstracts_explorer/web_ui/static/modules/utils/markdown-utils.js` — sanitize both render helpers (the choke point).
- **Modify** `src/abstracts_explorer/web_ui/static/modules/chat.js` — render assistant output via the sanitizing helper instead of `marked.parse`.
- **Modify** `src/abstracts_explorer/web_ui/tests/setup.js` — provide a passthrough `DOMPurify` global so existing tests are unaffected.
- **Create** `src/abstracts_explorer/web_ui/tests/markdown-sanitization.test.js` — real-DOMPurify security tests (XSS neutralized; markdown/KaTeX/anchors preserved).
- **Modify** `src/abstracts_explorer/rag.py` — markdown citation links instead of raw `<a>`.
- **Modify** `tests/test_rag.py` — assert the prompt uses markdown links and no raw `<a>`.

---

## Task 1: Vendor DOMPurify

**Files:**
- Modify: `package.json`
- Create: `src/abstracts_explorer/web_ui/static/vendor/purify.min.js` (generated)
- Modify: `src/abstracts_explorer/web_ui/templates/index.html:24`

- [ ] **Step 1: Add `dompurify` to dependencies in `package.json`**

In the `"dependencies"` block (currently lines 45-51), add the `dompurify` entry so it reads:

```json
    "dependencies": {
        "@fortawesome/fontawesome-free": "^7.1.0",
        "dompurify": "^3.2.6",
        "katex": "^0.16.25",
        "marked": "^17.0.1",
        "marked-katex-extension": "^5.1.6",
        "plotly.js-dist-min": "^3.3.1"
    }
```

- [ ] **Step 2: Add the vendor-copy script and wire it into the chain**

In `"scripts"`, update the `install:vendor` chain (line 14) to append the new step, and add the new script after `install:vendor:plotly` (line 19):

```json
        "install:vendor": "npm run install:vendor:fontawesome && npm run install:vendor:marked && npm run install:vendor:katex && npm run install:vendor:marked-katex && npm run install:vendor:plotly && npm run install:vendor:dompurify && npm run build:tailwind",
        "install:vendor:plotly": "mkdir -p src/abstracts_explorer/web_ui/static/vendor && cp node_modules/plotly.js-dist-min/plotly.min.js src/abstracts_explorer/web_ui/static/vendor/plotly.min.js",
        "install:vendor:dompurify": "mkdir -p src/abstracts_explorer/web_ui/static/vendor && cp node_modules/dompurify/dist/purify.min.js src/abstracts_explorer/web_ui/static/vendor/purify.min.js"
```

(Note: add a trailing comma after the `plotly` line since `dompurify` now follows it.)

- [ ] **Step 3: Install the dependency**

Run: `npm install`
Expected: completes without error; `node_modules/dompurify/dist/purify.min.js` exists and `package-lock.json` is updated.

Verify: `ls node_modules/dompurify/dist/purify.min.js` → prints the path.

- [ ] **Step 4: Generate the vendored asset**

Run: `npm run install:vendor:dompurify`
Expected: creates `src/abstracts_explorer/web_ui/static/vendor/purify.min.js`.

Verify: `test -s src/abstracts_explorer/web_ui/static/vendor/purify.min.js && echo OK` → prints `OK`.

- [ ] **Step 5: Load DOMPurify in the template**

In `src/abstracts_explorer/web_ui/templates/index.html`, insert a script tag immediately after the marked-katex line (line 24), before the Plotly block:

```html
    <!-- Marked-KaTeX extension for LaTeX in Markdown -->
    <script src="{{ url_for('static', filename='vendor/marked-katex-extension.min.js') }}"></script>

    <!-- DOMPurify for sanitizing rendered markdown (XSS protection) -->
    <script src="{{ url_for('static', filename='vendor/purify.min.js') }}"></script>

    <!-- Plotly for cluster visualization -->
    <script src="{{ url_for('static', filename='vendor/plotly.min.js') }}"></script>
```

- [ ] **Step 6: Commit**

The pre-commit hook regenerates vendor assets on web-file changes; since versions are locked, only `purify.min.js` is new. If the hook reports drift, run `npm run install:vendor` and `git add` the regenerated files, then re-commit.

```bash
git add package.json package-lock.json src/abstracts_explorer/web_ui/static/vendor/purify.min.js src/abstracts_explorer/web_ui/templates/index.html
git commit -m "$(cat <<'EOF'
build: vendor DOMPurify for markdown sanitization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Sanitize the markdown choke point

**Files:**
- Modify: `src/abstracts_explorer/web_ui/tests/setup.js:9`
- Create: `src/abstracts_explorer/web_ui/tests/markdown-sanitization.test.js`
- Modify: `src/abstracts_explorer/web_ui/static/modules/utils/markdown-utils.js`

- [ ] **Step 1: Add a passthrough `DOMPurify` global to the shared test setup**

In `src/abstracts_explorer/web_ui/tests/setup.js`, after the `global.fetch = jest.fn();` line (line 9), add:

```javascript
// Passthrough DOMPurify so modules that sanitize markdown work in tests that
// don't exercise sanitization. The dedicated security test overrides this with
// the real DOMPurify.
global.DOMPurify = { sanitize: (html) => html };
```

- [ ] **Step 2: Write the failing security test**

Create `src/abstracts_explorer/web_ui/tests/markdown-sanitization.test.js`:

```javascript
/**
 * Security tests: markdown render helpers must sanitize XSS while preserving
 * legitimate markdown, KaTeX math, and #paper-N citation anchors.
 *
 * Unlike other suites, this one wires the REAL marked + marked-katex + DOMPurify
 * so it verifies actual sanitization behavior (not a passthrough mock).
 */

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import DOMPurifyImport from 'dompurify';

// DOMPurify's default export is already an instance in a DOM env (jsdom);
// fall back to the factory form just in case.
const DOMPurify = typeof DOMPurifyImport.sanitize === 'function'
    ? DOMPurifyImport
    : DOMPurifyImport(window);

global.marked = marked;
global.markedKatex = markedKatex;
global.DOMPurify = DOMPurify;

import {
    renderMarkdownWithLatex,
    renderInlineMarkdownWithLatex,
    configureMarkedWithKatex,
} from '../static/modules/utils/markdown-utils.js';

describe('Markdown sanitization (XSS)', () => {
    it('strips event-handler attributes from images', () => {
        const html = renderMarkdownWithLatex('<img src=x onerror="alert(1)">');
        expect(html).not.toContain('onerror');
    });

    it('strips <script> tags', () => {
        const html = renderMarkdownWithLatex('hello <script>alert(1)</script> world');
        expect(html).not.toContain('<script>');
    });

    it('neutralizes javascript: URLs in links', () => {
        const html = renderMarkdownWithLatex('[click](javascript:alert(1))');
        expect(html).not.toContain('javascript:');
    });

    it('sanitizes inline-rendered titles too', () => {
        const html = renderInlineMarkdownWithLatex('<img src=x onerror="alert(1)">Title');
        expect(html).not.toContain('onerror');
    });

    it('preserves legitimate markdown formatting', () => {
        const html = renderMarkdownWithLatex('This is **bold** text');
        expect(html).toContain('<strong>');
        expect(html).toContain('bold');
    });

    it('preserves #paper-N citation anchors', () => {
        const html = renderMarkdownWithLatex('See [Paper-1](#paper-1)');
        expect(html).toContain('href="#paper-1"');
        expect(html).toContain('Paper-1');
    });

    it('preserves KaTeX math output', () => {
        configureMarkedWithKatex();
        const html = renderMarkdownWithLatex('Energy: $E=mc^2$');
        expect(html).toContain('katex');
    });
});
```

- [ ] **Step 3: Run the new test to verify it fails**

Run: `npm test -- markdown-sanitization`
Expected: FAIL — e.g. the `onerror` and `<script>` assertions fail because `markdown-utils.js` does not sanitize yet (it returns raw `marked` output).

- [ ] **Step 4: Implement sanitization in `markdown-utils.js`**

Replace the entire contents of `src/abstracts_explorer/web_ui/static/modules/utils/markdown-utils.js` with:

```javascript
/**
 * Markdown Rendering Utilities
 *
 * Renders markdown (with LaTeX) and sanitizes the resulting HTML with DOMPurify
 * before it is inserted into the DOM, preventing XSS from untrusted content
 * (scraped paper abstracts/titles and LLM chat output).
 */

/**
 * Sanitize an HTML string with DOMPurify.
 *
 * Uses DOMPurify's secure defaults, which strip <script>, on* event-handler
 * attributes, and javascript:/data: script URLs while preserving standard
 * markup, <a href="#paper-N"> citation anchors, and KaTeX (output: 'html')
 * spans (class/style/aria-hidden are retained).
 *
 * @param {string} html - Untrusted HTML produced by marked.
 * @returns {string} Sanitized HTML safe for innerHTML.
 */
function sanitizeHtml(html) {
    return DOMPurify.sanitize(html);
}

/**
 * Render inline markdown with LaTeX support (no block-level wrappers).
 * Use this for titles and other inline content where block elements like <p>
 * are not desired.
 * @param {string} text - Inline markdown text to render
 * @returns {string} Sanitized HTML without block-level wrappers
 */
export function renderInlineMarkdownWithLatex(text) {
    if (!text) return '';

    try {
        // parseInline avoids wrapping in <p>; sanitize before it reaches innerHTML
        return sanitizeHtml(marked.parseInline(text));
    } catch (e) {
        console.warn('Markdown inline parsing error:', e);
        // Fallback to escaped HTML if markdown parsing fails
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

/**
 * Render markdown with LaTeX support.
 * @param {string} text - Markdown text to render
 * @returns {string} Sanitized HTML
 */
export function renderMarkdownWithLatex(text) {
    if (!text) return '';

    try {
        // Render with the globally loaded marked (+ KaTeX), then sanitize
        return sanitizeHtml(marked.parse(text));
    } catch (e) {
        console.warn('Markdown parsing error:', e);
        // Fallback to escaped HTML if markdown parsing fails
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

/**
 * Configure marked with KaTeX extension
 * Should be called once during initialization
 */
export function configureMarkedWithKatex() {
    if (typeof markedKatex !== 'undefined' && typeof marked !== 'undefined') {
        marked.use(markedKatex({
            throwOnError: false,
            nonStandard: true,
            output: 'html'
        }));
    }
}
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `npm test -- markdown-sanitization`
Expected: PASS (all 7 assertions green).

- [ ] **Step 6: Run the full JS suite to confirm no regression**

Run: `npm test`
Expected: PASS. Existing `utils.test.js` markdown tests use `toContain('Test')` with the passthrough `DOMPurify` from `setup.js`, so they remain green.

- [ ] **Step 7: Commit**

```bash
git add src/abstracts_explorer/web_ui/tests/setup.js src/abstracts_explorer/web_ui/tests/markdown-sanitization.test.js src/abstracts_explorer/web_ui/static/modules/utils/markdown-utils.js
git commit -m "$(cat <<'EOF'
fix(security): sanitize rendered markdown with DOMPurify

Closes the XSS sink: paper titles/abstracts rendered via the markdown
helpers are now passed through DOMPurify before innerHTML.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Route chat output through the sanitizing helper

**Files:**
- Modify: `src/abstracts_explorer/web_ui/static/modules/chat.js:12` (imports) and `:310` (render)

- [ ] **Step 1: Import the sanitizing helper**

In `src/abstracts_explorer/web_ui/static/modules/chat.js`, add an import after line 12 (`import { formatPaperCard } from './paper-card.js';`):

```javascript
import { renderMarkdownWithLatex } from './utils/markdown-utils.js';
```

- [ ] **Step 2: Replace the direct `marked.parse` call**

At `chat.js:310`, in `addChatMessage`, change the assistant branch from `marked.parse(text)` to the helper:

```javascript
    // Render markdown for assistant messages, escape HTML for user messages
    const contentHtml = isUser
        ? `<p class="whitespace-pre-wrap">${escapeHtml(text)}</p>`
        : `<div class="markdown-content">${renderMarkdownWithLatex(text)}</div>`;
```

- [ ] **Step 3: Run the chat tests to confirm no regression**

Run: `npm test -- chat`
Expected: PASS. `chat.test.js` mocks `global.marked.parse` as identity; `renderMarkdownWithLatex` calls that mock then the passthrough `DOMPurify.sanitize`, so rendered content is unchanged in tests. The real sink is now covered by `markdown-sanitization.test.js` (which tests `renderMarkdownWithLatex` directly).

- [ ] **Step 4: Commit**

```bash
git add src/abstracts_explorer/web_ui/static/modules/chat.js
git commit -m "$(cat <<'EOF'
fix(security): render chat output via sanitizing markdown helper

LLM chat output no longer calls marked.parse directly; it now goes
through renderMarkdownWithLatex, so it is DOMPurify-sanitized like
abstracts. Removes the last unsanitized markdown sink.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Harden the RAG citation prompt

**Files:**
- Modify: `tests/test_rag.py` (add a test after `test_build_base_instructions_contains_defaults`, ~line 302)
- Modify: `src/abstracts_explorer/rag.py:449-456`

- [ ] **Step 1: Write the failing test**

In `tests/test_rag.py`, add this method to the same test class as `test_build_base_instructions_contains_defaults` (immediately after it, ~line 303):

```python
    def test_build_base_instructions_uses_markdown_citation_links(
        self, mock_embeddings_manager, mock_database
    ):
        """Citations must use markdown links, not raw HTML anchors (XSS hardening)."""
        chat = RAGChat(mock_embeddings_manager, mock_database)
        instructions = chat._build_base_instructions()
        assert "[Paper-1](#paper-1)" in instructions
        assert "<a href" not in instructions
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_rag.py -k markdown_citation_links -v`
Expected: FAIL — instructions still contain `<a href='#paper-1'>`.

- [ ] **Step 3: Update the prompt in `rag.py`**

In `src/abstracts_explorer/rag.py`, change the return of `_build_base_instructions` (lines 449-456) so the citation lines use markdown:

```python
        return (
            "You are an AI assistant helping researchers analyze conference data. "
            "Use the available tools to search for papers, analyze topics, and understand trends. "
            "Present the information in a clear, easy-to-understand format. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
            "When referencing specific papers, cite them using local markdown links: "
            "[Paper-1](#paper-1), [Paper-2](#paper-2), etc."
        )
```

- [ ] **Step 4: Run the new + existing instruction tests to verify they pass**

Run: `uv run pytest tests/test_rag.py -k build_base_instructions -v`
Expected: PASS — both `test_build_base_instructions_contains_defaults` (still finds "AI assistant"/"conference data"/"Today's date") and the new markdown-citation test pass.

- [ ] **Step 5: Format/lint the touched Python**

Run: `uv run black src/abstracts_explorer/rag.py tests/test_rag.py && uv run ruff check src/abstracts_explorer/rag.py tests/test_rag.py`
Expected: black reports formatted/unchanged; ruff reports no errors. (Pre-commit also enforces this.)

- [ ] **Step 6: Commit**

```bash
git add src/abstracts_explorer/rag.py tests/test_rag.py
git commit -m "$(cat <<'EOF'
fix(security): use markdown citation links in RAG prompt

Stop instructing the model to emit raw <a> HTML; markdown links render
to the same #paper-N anchors. Defense-in-depth behind the sanitizer
(code-review finding #2).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full verification and PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run the full JS suite**

Run: `npm test`
Expected: PASS (all suites, including `markdown-sanitization.test.js`).

- [ ] **Step 2: Run the RAG test module**

Run: `uv run pytest tests/test_rag.py -v`
Expected: PASS.

- [ ] **Step 3: Push the branch**

Run: `git push -u origin fix/xss-markdown-sanitization`
Expected: branch published to `origin`.

- [ ] **Step 4: Open the PR against `develop`**

```bash
gh pr create --base develop --head fix/xss-markdown-sanitization \
  --title "fix(security): sanitize rendered markdown to close XSS chain" \
  --body "$(cat <<'EOF'
## Summary
Closes the stored/reflected XSS chain (code-review Critical findings #1 + #2):
untrusted paper titles/abstracts and LLM chat output were rendered to `innerHTML`
via `marked` with no sanitizer.

## Changes
- Vendor **DOMPurify** via the existing `install:vendor` pipeline; load it in `index.html`.
- Sanitize at the `markdown-utils.js` choke point (`renderMarkdownWithLatex` / `renderInlineMarkdownWithLatex`) — covers paper titles + abstracts everywhere.
- Route `chat.js` (the one bypass) through the sanitizing helper.
- Harden the RAG prompt (`rag.py`) to cite with markdown links instead of raw `<a>` HTML.

## Tests
- New `markdown-sanitization.test.js`: `<img onerror>` / `<script>` / `javascript:` neutralized; `**bold**`, `#paper-N` anchors, and KaTeX preserved (real DOMPurify).
- New `test_rag.py` assertion: prompt uses markdown links, no raw `<a>`.

## Out of scope (tracked separately)
- CSP `'unsafe-inline'` removal, `/api/clusters/cached` path traversal (#3), global `RAGChat` singleton (#4).

Spec: `docs/superpowers/specs/2026-06-08-xss-sanitization-design.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR created; prints the PR URL.

- [ ] **Step 5: Report the PR URL to the user.**

---

## Notes for the implementer

- **Vendor/hook coupling:** committing web files triggers the pre-commit hook, which runs `install:vendor`. Ensure `npm install` (Task 1 Step 3) ran first so `node_modules/dompurify` exists, or the hook will fail to regenerate `purify.min.js`.
- **DOMPurify config:** the implementation uses DOMPurify's secure defaults. If the KaTeX assertion in Task 2 fails (some span/attr stripped), extend `sanitizeHtml` with the minimal `ADD_ATTR`/`ADD_TAGS` needed and re-run — do not broaden beyond what KaTeX requires.
- **Do not** widen scope to the CSP, path-traversal, or RAG-singleton items — they are separate PRs.
