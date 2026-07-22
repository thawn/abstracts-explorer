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
